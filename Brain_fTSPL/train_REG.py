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

from sklearn.metrics import r2_score
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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 超参数
parser = argparse.ArgumentParser()
parser.add_argument('--train_path',        type=str,   default='./data/HCP/train'  )
parser.add_argument('--test_path',         type=str,   default='./data/HCP/test'   )
parser.add_argument('--net',               type=str,   default=''      )
parser.add_argument('--lr',                type=float, default=1e-3                ) # 5e-4
parser.add_argument('--epochs',            type=int,   default=100                 ) # 100
parser.add_argument('--workers',           type=int,   default=0                   )
parser.add_argument('--BN',                type=int,   default=16                   ) # 32
parser.add_argument('--test_BN',           type=int,   default=174                 )
parser.add_argument('--display',           type=int,   default=10                  )
parser.add_argument('--exp',               type=str,   default='fMRI'              )

parser.add_argument('--view', type=int, default=1)
parser.add_argument('--node_features', type=str, default='adj',
                    choices=['identity', 'degree', 'degree_bin', 'LDP', 'node2vec', 'adj', 'diff_matrix', 'eigenvector', 'eigen_norm']) # 'adj'
parser.add_argument('--pooling', type=str, default='mean',
                    choices=['sum', 'concat', 'mean']) # 'concat'
parser.add_argument('--model_name', type=str, default='gcn')
parser.add_argument('--gcn_mp_type', type=str, default='weighted_sum',
                    choices=['weighted_sum', 'bin_concate', 'edge_weight_concate', 'edge_node_concate', 'node_concate']) # 'edge_node_concate'
parser.add_argument('--gat_mp_type', type=str, default='node_concate',
                    choices=['attention_weighted', 'attention_edge_weighted', 'sum_attention_edge', 'edge_node_concate', 'node_concate']) # 'edge_node_concate'
                    
parser.add_argument('--enable_nni', action='store_true')
parser.add_argument('--n_GNN_layers', type=int, default=2)
parser.add_argument('--n_MLP_layers', type=int, default=1)
parser.add_argument('--num_heads', type=int, default=2)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--gat_hidden_dim', type=int, default=8)
parser.add_argument('--edge_emb_dim', type=int, default=256)
parser.add_argument('--bucket_sz', type=float, default=0.05)
parser.add_argument('--weight_decay', type=float, default=1e-4) # 1e-4
parser.add_argument('--dropout', type=float, default=0.5)       # 0.5

parser.add_argument('--seed', type=int, default=112078)
parser.add_argument('--diff', type=float, default=0.2)
parser.add_argument('--mixup', type=int, default=1)

parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL', help='Name of model to train')
args = parser.parse_args()

#opt.manualSeed = random.randint(1, 10000)
args.manualSeed = 101
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
print("Random Seed: ", args.manualSeed)

    
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

# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer                   # 读取transformer
        self.position_embedding = clip_model.positional_embedding   # 位置编码
        self.ln_final = clip_model.ln_final                         # LN 层
        self.text_projection = clip_model.text_projection           # 映射层
        self.dtype = clip_model.dtype                               # 数据类型

    def forward(self, texts, tokens, text_prompts):
        x = texts + self.position_embedding.type(self.dtype)        # 文本 + 位置编码
        x = x.permute(1, 0, 2)                                      # NLD -> LND
        combined = [x, text_prompts, 0]                             # 文本与提示组成输入
        outputs = self.transformer(combined)
        x = outputs[0]                                              # 提取输出的第一个向量
        x = x.permute(1, 0, 2)                                      # LND -> NLD
        x = self.ln_final(x)                                        # 16 77 512
        x_cls = x[torch.arange(x.shape[0]), tokens.squeeze(1).argmax(dim=-1)] @ self.text_projection # 选出x每行的最大值 16 512
        return x_cls, x

# 多模态提示
class multi_modal_prompt_learner(nn.Module):
    def __init__(self, clip_model, prompt_length, prompt_depth):
        super().__init__()
        
        self.dtype = clip_model.dtype
        print(self.dtype)
        self.token_embed = clip_model.token_embedding
        
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
        self.text_prompts = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 512, dtype=self.dtype)) for _ in range(prompt_depth - 1)])
        self.img_prompts = nn.ParameterList([nn.Parameter(torch.empty(self.prompt_length, 768, dtype=self.dtype)) for _ in range(prompt_depth - 1)])
        for single_para in self.text_prompts:
            nn.init.normal_(single_para, std=0.02)
        for single_para in self.img_prompts:
            nn.init.normal_(single_para, std=0.02)

    def forward(self, fmri, token):
    
        ctx_text = self.ctx_text.repeat(fmri.shape[0], 1, 1)
        #ctx_text = ctx_text.unsqueeze(0).expand(fmri.shape[0], -1, -1)
        ctx_img = self.ctx_img
        
        imgs = fmri                                                 # batch_size 82 100
        texts = self.token_embed(token.squeeze(1)).type(self.dtype) # batch_size 77 512
        prefix = texts[:, :1, :]
        suffix = texts[:, 1+self.prompt_length:, :]
        texts = torch.cat([prefix, ctx_text, suffix], dim=1)        # 提示插入文本特征
        return imgs, texts, ctx_img, self.text_prompts, self.img_prompts
        
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
def adjust_learning_rate(optimizer, init_lr):
    lr = init_lr / 2 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(optimizer.param_groups[0]['lr'])

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

# 回归任务评分     
def regress_score(predict, target):
    # ss_res = torch.sum((target - predict) ** 2)
    # ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    # r2_score = 1 - ss_res / ss_tot
    r2_sc = r2_score(target.cpu().numpy(), predict.cpu().numpy())
    mse_sc = F.mse_loss(predict, target)
    mae_sc = F.l1_loss(predict, target)
    pea_sc = torch.corrcoef(torch.cat([predict.permute(1, 0), target.permute(1, 0)], dim=0))
    return r2_sc, mse_sc, mae_sc, pea_sc[0,1]
     
# 生成 batch 图数据 
def dense_to_ind_val(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)
    index = (torch.isnan(adj)==0).nonzero(as_tuple=True)
    edge_attr = adj[index]
    return torch.stack(index, dim=0), edge_attr

def graph_data(f_latent, f_roi, corr):
    data_list = []
    f_roi = f_roi.reshape(f_roi.shape[0], 82, 768*5)
    for i in range(corr.shape[0]):
        corr_r = torch.corrcoef(f_roi[i]) # 82 82 
        corr_ori = torch.corrcoef(f_latent[i])
        corr_c = torch.sum(corr_ori[:1, 1:].reshape(82, 5), dim=1) # 82       
        edge_index, edge_attr = dense_to_ind_val(0.9*corr[i] + 0.1*corr_r)
        data_list.append(Data(x=torch.cat((corr[i], corr_c.unsqueeze(1)), dim=1), edge_index=edge_index, edge_attr=edge_attr))
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
MMP = multi_modal_prompt_learner(clip_model, 2, 2).cuda() # 2, 2
MMP.train()

# 图卷积网络(2层)
GCN = build_model(args, device=torch.device('cuda'), model_name='gcn', num_features=83, num_nodes=82) # 117 116
GCN.train()

image_encoder = clip_model.visual.cuda()
text_encoder = TextEncoder(clip_model).cuda()

combined_params = list(MMP.parameters()) + list(GCN.parameters())

# 优化器
optimizer = torch.optim.SGD(combined_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

best_epoch = {'Epoch':0, 'R2':0, 'MSE':0, 'MAE':0, 'PEA':0}

for epoch in range(args.epochs):
    print('Model Train')
    start_time = time.time()
    S_loss_align = 0.0
    S_loss_regress = 0.0
    S_loss = 0.0
    r_acc = 0.0
    mse = 0.0
    ganIterations = 0

    if epoch+1 == 50:  # 调整学习率
        adjust_learning_rate(optimizer, 1e-3)
        
    for i, data_train in enumerate(train_loader):

        fmri, corr, token, gt = data_train 
        fmri = fmri[:, :, :100]

        fmri, corr, token, gt = fmri.cuda(), corr.cuda(),token.cuda(), gt.cuda()
        gt = F.sigmoid(((gt - 116.81)/10.63).float())
        
        # gt = ((gt - 84.20)/(150.71-84.20)).float()
        # gt = (gt/200).float()
        
        img, text, ctx, text_prompts, img_prompts = MMP(fmri, token)
        
        img_latent, img_roi, img_head = image_encoder(img, ctx, img_prompts)
        text_head, _ = text_encoder(text, token, text_prompts)
        graph_batch = graph_data(img_latent.float(), img_roi.float(), corr.float())  
        
        predict = GCN(graph_batch)

        optimizer.zero_grad()
        
        # 计算对齐损失
        img_head = img_head / img_head.norm(dim=-1, keepdim=True)
        text_head = text_head / text_head.norm(dim=-1, keepdim=True)
        
        logits = img_head @ text_head.t()
        
        scores1 = logits/0.1     # 0.1
        scores2 = scores1.transpose(0, 1)
        
        bz = img_head.size(0)
        labels = torch.arange(bz).type_as(img_head).long()
        
        print(scores1)
        
        i2t_acc1, i2t_acc5 = precision_at_k(scores1, labels, top_k=(1, 2))
        t2i_acc1, t2i_acc5 = precision_at_k(scores2, labels, top_k=(1, 2))
        acc1 = (i2t_acc1 + t2i_acc1) / 2
        acc5 = (i2t_acc5 + t2i_acc5) / 2
        
        loss_1 = F.cross_entropy(scores1, labels)
        loss_2 = F.cross_entropy(scores2, labels)
        loss_align = (loss_1 + loss_2) / 2
        
        # 计算回归损失
        loss_regress = F.mse_loss(predict, gt)

        # 总损失
        loss = 0.1*loss_align + 0.9*loss_regress # (0.1 0.9)
        loss.backward()
        
        # 记录损失
        S_loss_align += 0.1*loss_align.item()
        S_loss_regress += 0.9*loss_regress.item()
        S_loss += loss.item()
        
        # 检索准确率
        r_acc += acc1
        
        # 回归误差
        mse += loss_regress
        
        # 反向传播
        optimizer.step()
        ganIterations += 1 # 迭代次数加1
        
        # 损失展示
        if ganIterations % args.display == 0:
            print('[%d/%d][%d/%d] | Total_Loss: %f | Align_Loss: %f | Regress_Loss: %f ' 
            % (epoch+1, args.epochs, i+1, len(train_loader), S_loss*args.BN, S_loss_align*args.BN, S_loss_regress*args.BN))  
            S_loss = 0.0
            S_loss_align = 0.0
            S_loss_regress = 0.0
            
    print('[%d/%d] | R_Acc: %.4f | MSE: %.4f' % (epoch+1, args.epochs, r_acc/ganIterations, mse/ganIterations))  
    total_time = time.time() - start_time
    print('Total-Time: {:.2f} '.format(total_time))     
    
    
    # 模型测试, SSIM and PSNR
    print('Model Test')
    MMP.eval()
    GCN.eval()
    R2, MSE, MAE, PEA = [], [], [], []

    with torch.no_grad():
        total_t = 0
        correct_t = 0
        for j, data_test in enumerate(test_loader):
        
            fmri, corr, token, gt = data_test
            fmri_ = fmri[:, :, 0:100]
            print(fmri_.size())
            fmri, corr, token, gt = fmri.cuda(), corr.cuda(), token.cuda(), gt.cuda()
            gt = F.sigmoid(((gt - 116.81)/10.63).float())
            # gt = ((gt - 84.20)/(150.71-84.20)).float()
            # gt = (gt/200).float()
            
            img, text, ctx_img, deep_prompts_text, deep_prompts_img = MMP(fmri, token)
            _, img_feat = image_encoder(img, ctx_img, deep_prompts_img)
            # _, text_feat = text_encoder(text, token, deep_prompts_text)
            
            graph_batch = graph_data(img_feat.float(), corr.float())
            predict = GCN(graph_batch)
            
            tr2, tmse, tmae, tpea = regress_score(predict, gt)
            
            R2.append(tr2)
            MSE.append(tmse.cpu().numpy())
            MAE.append(tmae.cpu().numpy())
            PEA.append(tpea.cpu().numpy())
        
        test_r2 = np.mean(R2)    
        test_mse = np.mean(MSE)
        test_mae = np.mean(MAE)
        test_pea = np.mean(PEA)
        
    print('[%d/%d] | test_r2: %.4f | test_mse: %.4f | test_mae: %.4f | test_pea: %.4f' % (epoch+1, args.epochs, test_r2, test_mse, test_mae, test_pea))
    MMP.train()
    GCN.train()
    
    # 保存最佳模型
    if test_pea > best_epoch['PEA']:
         # torch.save(netG.state_dict(), '%s/netG_epoch%d.pth' % (args.exp, epoch+1))
         best_epoch['R2'] = test_r2
         best_epoch['MSE'] = test_mse
         best_epoch['MAE'] = test_mae
         best_epoch['PEA'] = test_pea
         best_epoch['Epoch'] = epoch+1
         
print(best_epoch)


