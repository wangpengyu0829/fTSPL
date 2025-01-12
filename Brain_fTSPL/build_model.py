import torch
from gat import GAT
from gcn import GCN
from mlp import MLP
from typing import List
from brainnn import BrainNN


def build_model(args, device, model_name, num_features, num_nodes):
    if model_name == 'gcn':
    # num_classes ∑÷¿‡2£¨‘§≤‚1
        model = BrainNN(args, GCN(num_features, args, num_nodes, num_classes=2),
                        MLP(2 * num_nodes, args.hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2)).to(device)
    elif model_name == 'gat':
        model = BrainNN(args, GAT(num_features, args, num_nodes, num_classes=2),
                        MLP(2 * num_nodes, args.gat_hidden_dim, args.n_MLP_layers, torch.nn.ReLU, n_classes=2)).to(device)
    else:
        raise ValueError(f"ERROR: Model variant \"{args.variant}\" not found!")
    return model

