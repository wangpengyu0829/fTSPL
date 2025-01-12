import os
import sys
import time
import torch
import numpy
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from sklearn import metrics
from nilearn import connectome
from torch.utils.data import DataLoader
from data_h5 import get_train, get_test

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import dense_to_sparse

from build_model import build_model

cudnn.benchmark = True
cudnn.fastest = True

import warnings
warnings.filterwarnings("ignore")  
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 超参数
parser = argparse.ArgumentParser()
parser.add_argument('--train_path',        type=str,   default='./data/ABIDE/train'  )
parser.add_argument('--test_path',         type=str,   default='./data/ABIDE/test'   )
parser.add_argument('--net',               type=str,   default=''      )
parser.add_argument('--lr',                type=float, default=5e-4                ) # 5e-4 6e-4
parser.add_argument('--annealStart',       type=int,   default=0                   )
parser.add_argument('--annealEvery',       type=int,   default=200                 )
parser.add_argument('--epochs',            type=int,   default=50                  ) # 50
parser.add_argument('--workers',           type=int,   default=0                   )
parser.add_argument('--BN',                type=int,   default=8                   ) # 16
parser.add_argument('--test_BN',           type=int,   default=4                   )
parser.add_argument('--display',           type=int,   default=10                  )
parser.add_argument('--exp',               type=str,   default='fMRI'              )

parser.add_argument('--dataset_name', type=str, default="ABIDE", 
                    choices=['PPMI', 'HIV', 'BP', 'ABCD', 'PNC', 'ABIDE'])
parser.add_argument('--view', type=int, default=1)
parser.add_argument('--node_features', type=str, default='adj',
                    choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 'diff_matrix', 'eigenvector', 'eigen_norm'])
parser.add_argument('--pooling', type=str, default='concat',
                    choices=['sum', 'concat', 'mean'])     
parser.add_argument('--model_name', type=str, default='gcn')
parser.add_argument('--gcn_mp_type', type=str, default='edge_node_concate',
                    choices=['weighted_sum', 'bin_concate', 'edge_weight_concate', 'edge_node_concate', 'node_concate'])
parser.add_argument('--gat_mp_type', type=str, default='attention_weighted',
                    choices=['attention_weighted', 'attention_edge_weighted', 'sum_attention_edge', 'edge_node_concate', 'node_concate'])
                    
parser.add_argument('--enable_nni', action='store_true')
parser.add_argument('--n_GNN_layers', type=int, default=2)
parser.add_argument('--n_MLP_layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--gat_hidden_dim', type=int, default=8)
parser.add_argument('--edge_emb_dim', type=int, default=256)
parser.add_argument('--bucket_sz', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=1e-4) # 1e-4
parser.add_argument('--dropout', type=float, default=0.5) # 0.5

parser.add_argument('--seed', type=int, default=112078)
parser.add_argument('--diff', type=float, default=0.2)
parser.add_argument('--mixup', type=int, default=1)
args = parser.parse_args()

#opt.manualSeed = random.randint(1, 10000)
args.manualSeed = 101
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
print("Random Seed: ", args.manualSeed)


'''读取CLIP模型'''
def load_clip_to_cpu():
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    model = torch.jit.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe', "vision_depth": 0, "language_depth": 0, 
                      "vision_ctx": 0, "language_ctx": 0, "maple_length": 2}
    model = clip.build_model(model.state_dict(), design_details)

    for p in model.parameters(): # 冻结clip的参数
     	p.requires_grad = False
    return model

'''文本编码器'''
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer                   # 读取transformer模型
        self.position_embedding = clip_model.positional_embedding   # 位置编码
        self.ln_final = clip_model.ln_final                         # layer norm
        self.text_projection = clip_model.text_projection           # 映射层
        self.dtype = clip_model.dtype                               # 数据类型

    def forward(self, texts, tokens, compound_prompts_text):
        x = texts + self.position_embedding.type(self.dtype)        # 文本 + 位置编码
        x = x.permute(1, 0, 2)                                      # NLD -> LND
        combined = [x, compound_prompts_text, 0]                    # 文本与提示组成输入
        outputs = self.transformer(combined)
        x = outputs[0]                                              # 提取输出的第一个向量
        x = x.permute(1, 0, 2)                                      # LND -> NLD
        x = self.ln_final(x).type(self.dtype)                       # batchsize tokensize dim 16 77 512
        x_h = x[torch.arange(x.shape[0]), tokens.squeeze(1).argmax(dim=-1)] @ self.text_projection # 选出x每行的最大值 16 512
        return x_h, x
        
'''投影层'''   
class project_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(in_dim, out_dim))

    def forward(self, x):
        x = x @ self.proj
        return x

'''多模态提示'''
class multi_modal_prompt_learner(nn.Module):
    def __init__(self, clip_model, batch_size, prompt_length, prompt_depth):
        super().__init__()

        self.batch_size = batch_size
        self.dtype = clip_model.dtype
        self.token_embed = clip_model.token_embedding
        self.proj = nn.Linear(128, 768).type(self.dtype)

        # 第一层视觉/本文提示
        self.prompt_length = prompt_length
        self.ctx_text = nn.Parameter(torch.empty(self.prompt_length, 512, dtype=self.dtype))
        nn.init.normal_(self.ctx_text, std=0.02)
        self.ctx_img = nn.Parameter(torch.empty(self.prompt_length, 768, dtype=self.dtype))
        nn.init.normal_(self.ctx_img, std=0.02)

        print('Multi-modal Prompt Learning')
        print(f"Number of MaPLe context words (tokens): {prompt_length}")

        # 第二到K层视觉/本文提示 max=12, but will create 11 such shared prompts
        self.prompt_length = prompt_length
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 512, dtype=self.dtype))
                                                       for _ in range(prompt_depth - 1)])
        self.compound_prompts_img = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 768, dtype=self.dtype))
                                                      for _ in range(prompt_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        for single_para in self.compound_prompts_img:
            nn.init.normal_(single_para, std=0.02)

    def forward(self, fmri, token):
        ctx_text = self.ctx_text
        ctx_text = ctx_text.unsqueeze(0).expand(fmri.size()[0], -1, -1)
        ctx_img = self.ctx_img
        imgs = self.proj(fmri)
        texts = self.token_embed(token.squeeze(1)).type(self.dtype) # batch_size 77 512
        prefix = texts[:, :1, :]
        suffix = texts[:, 1+self.prompt_length:, :]
        texts = torch.cat([prefix, ctx_text, suffix], dim=1) # 提示插入文本特征
        return imgs, texts, ctx_img, self.compound_prompts_text, self.compound_prompts_img
        #  报告  初始视觉提  示文本提示  视觉提示
        
# 参数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# 创建路径
def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True

# 学习率衰减        
def adjust_learning_rate(optimizer, init_lr, epoch):
    lrd = init_lr / epoch 
    old_lr = optimizer.param_groups[0]['lr']
    lr = old_lr - lrd
    if lr < 0: lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Top-K 检索准确率
def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# 敏感度和特异性  
def calculate_confusion_matrix(preds, labels): 
    # 计算混淆矩阵的四个指标：TP, TN, FP, FN
    TP = torch.sum((preds == 1) & (labels == 1)).item()
    TN = torch.sum((preds == 0) & (labels == 0)).item()
    FP = torch.sum((preds == 1) & (labels == 0)).item()
    FN = torch.sum((preds == 0) & (labels == 1)).item()
    return TP, TN, FP, FN

def sensitivity_specificity(preds, labels):
    # 计算敏感性和特异性
    TP, TN, FP, FN = calculate_confusion_matrix(preds, labels)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity
        
# 生成 batch 图数据 
def dense_to_ind_val(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)
    index = (torch.isnan(adj)==0).nonzero(as_tuple=True)
    edge_attr = adj[index]
    return torch.stack(index, dim=0), edge_attr
    
def graph_data(fmri, corr):
    data_list = []
    for i in range(corr.size()[0]):
        edge_index, edge_attr = dense_to_ind_val(corr[i])
        feat_corr = torch.corrcoef(fmri[i])
        data_list.append(Data(x=torch.cat((corr[i], feat_corr[1:117,0:1]), dim=1), edge_index=edge_index, edge_attr=edge_attr))
        # data_list.append(Data(x=corr[i], edge_index=edge_index, edge_attr=edge_attr))
    graph_batch = Batch.from_data_list(data_list)
    return graph_batch

# 读取数据
create_exp_dir(args.exp)
train_dataset = get_train(args.train_path)
test_dataset = get_test(args.test_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.BN, shuffle=False, num_workers=args.workers, drop_last=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_BN, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

# 构建模型
clip_model = load_clip_to_cpu().cuda()

# 多模态提示
MMP = multi_modal_prompt_learner(clip_model, args.BN, 2, 2).cuda() # 2, 2
MMP.train()

# 图卷积网络(2层)
GCN = build_model(args, device=torch.device('cuda'), model_name='gcn', num_features=117, num_nodes=116) # 117 116
GCN.train()

## 投影层
# PJ = project_layer(512, 768).cuda()
# PJ.train()
##

image_encoder = clip_model.visual.cuda()
text_encoder = TextEncoder(clip_model).cuda()

combined_params = list(MMP.parameters()) + list(GCN.parameters())
# combined_params = list(MMP.parameters()) + list(GCN.parameters()) + list(PJ.parameters())

# 优化器
# optimizer = optim.Adam(combined_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
optimizer = torch.optim.SGD(combined_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

best_epoch = {'Epoch':0, 'ACC':0, 'AUC':0, 'F1':0}

for epoch in range(args.epochs):
    print('Model Train')
    start_time = time.time()
    S_loss_align = 0.0
    S_loss_class = 0.0
    S_loss = 0.0
    r_acc = 0.0
    total = 0
    correct = 0
    ganIterations = 0

    # if epoch+1 > args.annealStart:  # 调整学习率
        # adjust_learning_rate(optimizer, args.lr, args.annealEvery)
        
    for i, data_train in enumerate(train_loader):

        fmri, corr, token, gt = data_train 
        fmri, corr, token, gt = fmri.cuda(), corr.cuda(),token.cuda(), gt.cuda()
        gt = gt.squeeze(-1).long()
        
        img, text, ctx_img, deep_prompts_text, deep_prompts_img = MMP(fmri, token)

        img_head, img_feat = image_encoder(img, ctx_img, deep_prompts_img)
        text_head, _ = text_encoder(text, token, deep_prompts_text)
        
        graph_batch = graph_data(img_feat.float(), corr.float())
        predict = GCN(graph_batch)
        
        optimizer.zero_grad()
        
        # 计算对齐损失
        img_head = img_head / img_head.norm(dim=-1, keepdim=True)
        text_head = text_head / text_head.norm(dim=-1, keepdim=True)

        logits = img_head @ text_head.t()
        
        scores1 = logits/0.01     # 0.01
        scores2 = scores1.transpose(0, 1)
        
        bz = img_head.size(0)
        labels = torch.arange(bz).type_as(img_head).long()
        
        i2t_acc1, i2t_acc5 = precision_at_k(scores1, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = precision_at_k(scores2, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.
        
        loss_1 = F.cross_entropy(scores1, labels)
        loss_2 = F.cross_entropy(scores2, labels)
        loss_align = (loss_1 + loss_2) / 2
        
        # 计算分类损失
        loss_class = F.cross_entropy(predict, gt)
        
        loss = 0.6*loss_align + 0.4*loss_class # (0.5 0.5) (0.6 0.4)
        
        loss.backward()
        
        S_loss_align += loss_align.item()
        S_loss_class += loss_class.item()
        S_loss += loss.item()
        
        # 检索准确率
        r_acc += acc1
        
        # 分类准确率
        total += gt.size(0)
        correct += (torch.max(predict.data, 1)[1] == gt).sum().item()
        
        # 反向传播
        optimizer.step()
        ganIterations += 1 # 迭代次数加1
        
        # 损失展示
        if ganIterations % args.display == 0:
            print('[%d/%d][%d/%d] | Total_Loss: %f | Align_Loss: %f | Class_Loss: %f ' 
            % (epoch+1, args.epochs, i+1, len(train_loader), S_loss*args.BN, S_loss_align*args.BN, S_loss_class*args.BN))  
            S_loss = 0.0
            S_loss_align = 0.0
            S_loss_class = 0.0
            
    train_acc = 100.0*correct/total
    print('[%d/%d] | R_Acc: %.4f | C_ACC: %.4f' % (epoch+1, args.epochs, r_acc/ganIterations, train_acc))  
    total_time = time.time() - start_time
    print('Total-Time: {:.2f} '.format(total_time))     
    
    
    # 模型测试, SSIM and PSNR
    print('Model Test')
    MMP.eval()
    GCN.eval()
    preds, trues, preds_prob = [], [], []

    with torch.no_grad():
        total_t = 0
        correct_t = 0
        for j, data_test in enumerate(test_loader):
        
            fmri, corr, token, gt = data_test
            fmri, corr, token, gt = fmri.cuda(), corr.cuda(),token.cuda(), gt.cuda()
            gt = gt.squeeze(-1).long()
            
            img, text, ctx_img, _, deep_prompts_img = MMP(fmri, token)
            _, img_feat = image_encoder(img, ctx_img, deep_prompts_img)
            #_, text_feat = text_encoder(text, token, deep_prompts_text)
            
            graph_batch = graph_data(img_feat.float(), corr.float())
            predict = GCN(graph_batch)
            
            pred = predict.max(dim=1)[1]
            total_t += gt.size(0)
            correct_t += (pred == gt).sum().item()
            
            preds += pred.cpu().tolist()
            preds_prob += torch.exp(predict)[:, 1].cpu().tolist()
            trues += gt.cpu().tolist()
            
        P = torch.tensor(preds)   
        T = torch.tensor(trues)
        test_sens, test_spec = sensitivity_specificity(P, T)
        test_acc = 100.0*correct_t/total_t    
        test_auc = metrics.roc_auc_score(trues, preds_prob)
        test_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])
         
    print('[%d/%d] | test_acc: %.4f | test_macro: %.4f | test_auc: %.4f | test_sens: %.4f | test_spec: %.4f' % (epoch+1, args.epochs, test_acc, test_macro*100, test_auc*100, test_sens*100, test_spec*100))
    MMP.train()
    GCN.train()
    
    # 保存最佳模型
    if test_acc > best_epoch['ACC']:
         # torch.save(netG.state_dict(), '%s/netG_epoch%d.pth' % (args.exp, epoch+1))
         best_epoch['ACC'] = test_acc
         best_epoch['AUC'] = test_auc
         best_epoch['F1'] = test_macro
         best_epoch['Epoch'] = epoch+1
         
print(best_epoch)

