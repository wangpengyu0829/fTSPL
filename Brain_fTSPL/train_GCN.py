import os
import sys
import time
import torch
import numpy
import argparse
import random
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn

from sklearn import metrics
from torch.utils.data import DataLoader
from data_h5 import get_train, get_test

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
# from torch_geometric.utils import dense_to_sparse

from build_model import build_model

cudnn.benchmark = True
cudnn.fastest = True

import warnings
warnings.filterwarnings("ignore")  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--train_path',        type=str,   default='./data/ABIDE/train'  )
parser.add_argument('--test_path',         type=str,   default='./data/ABIDE/test'   )
parser.add_argument('--net',               type=str,   default=''      )
parser.add_argument('--lr',                type=float, default=0.001               )
parser.add_argument('--epochs',            type=int,   default=50                  )
parser.add_argument('--workers',           type=int,   default=0                   )
parser.add_argument('--BN',                type=int,   default=16                  )
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
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.5)

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

'''参数量'''
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

'''创建路径'''
def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True

def dense_to_ind_val(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)
    index = (torch.isnan(adj)==0).nonzero(as_tuple=True)
    edge_attr = adj[index]
    return torch.stack(index, dim=0), edge_attr

# 生成 batch 图数据     
def graph_data(fmri, corr):
    data_list = []
    
    for i in range(corr.size()[0]):
        edge_index, edge_attr = dense_to_ind_val(corr[i])
        data_list.append(Data(x=corr[i], edge_index=edge_index, edge_attr=edge_attr))
        
    graph_batch = Batch.from_data_list(data_list)
    return graph_batch
        
# 读取数据
create_exp_dir(args.exp)
train_dataset = get_train(args.train_path)
test_dataset = get_test(args.test_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.BN, shuffle=False, num_workers=args.workers, drop_last=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_BN, shuffle=False, num_workers=args.workers, drop_last=False, pin_memory=True)

# 构建模型
device = torch.device('cuda')
GCN = build_model(args, device, model_name='gcn', num_features=116, num_nodes=116)
GCN.train()

# 优化器
# optimizer = optim.Adam(GCN.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
optimizer = torch.optim.SGD(GCN.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


for epoch in range(args.epochs):
    print('Model Train')
    start_time = time.time()
    S_loss = 0.0
    total = 0
    correct = 0
    ganIterations = 0
        
    for i, data_train in enumerate(train_loader):

        fmri, corr, token, gt = data_train 
        fmri, corr, token, gt = fmri.cuda(), corr.cuda(),token.cuda(), gt.cuda()
        gt = gt.squeeze(-1).long()
        
        graph_batch = graph_data(fmri.float(), corr.float())
        predict = GCN(graph_batch)

        optimizer.zero_grad()
        
        loss = F.cross_entropy(predict, gt)
        
        loss.backward()

        S_loss += loss.item()
        
        # 分类准确率
        total += gt.size(0)
        correct += (torch.max(predict.data, 1)[1] == gt).sum().item()
        
        # 反向传播
        optimizer.step()
        ganIterations += 1 # 迭代次数加1
        
        # 损失展示
        if ganIterations % args.display == 0:
            print('[%d/%d][%d/%d] | Total_Loss: %f ' % (epoch+1, args.epochs, i+1, len(train_loader), S_loss*args.BN))
            S_loss = 0.0
            
    train_acc = 100.0*correct/total
    print('[%d/%d] | ACC: %.4f' % (epoch+1, args.epochs, train_acc))
    total_time = time.time() - start_time
    print('Total-Time: {:.2f} '.format(total_time))
    
    # 模型测试, SSIM and PSNR
    print('Model Test')
    GCN.eval()
    preds, trues, preds_prob = [], [], []

    with torch.no_grad():
        total_t = 0
        correct_t = 0
        for j, data_test in enumerate(test_loader):
        
            fmri, corr, token, gt = data_test
            fmri, corr, token, gt = fmri.cuda(), corr.cuda(),token.cuda(), gt.cuda()
            gt = gt.squeeze(-1).long()

            graph_batch = graph_data(fmri.float(), corr.float())
            predict = GCN(graph_batch)

            pred = predict.max(dim=1)[1]
            total_t += gt.size(0)
            correct_t += (pred== gt).sum().item()
            
            preds += pred.cpu().tolist()
            preds_prob += torch.exp(predict)[:, 1].cpu().tolist()
            trues += gt.cpu().tolist()
        
        test_acc = 100.0*correct_t/total_t    
        test_auc = metrics.roc_auc_score(trues, preds_prob)
        test_macro = metrics.f1_score(trues, preds, average='macro', labels=[0, 1])
        
    print('[%d/%d] | test_acc: %.4f | test_macro: %.4f | test_auc: %.4f' % (epoch+1, args.epochs, test_acc, test_macro*100, test_auc*100))
    GCN.train()
    
    # # 保存最佳模型
    # if psnr_avg > best_epoch['psnr']:
    #     torch.save(netG.state_dict(), '%s/netG_epoch%d.pth' % (opt.exp, epoch+1))
    #     best_epoch['psnr'] = psnr_avg
    #     best_epoch['epoch'] = epoch+1 


