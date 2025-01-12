import torch
import numpy as np 

from itertools import permutations
from collections import defaultdict
from torch.nn import functional as F
from torch_geometric.utils import to_dense_adj



class BrainNN(torch.nn.Module):
    def __init__(self, args, gnn, discriminator=lambda x, y: x @ y.t()):
        super(BrainNN, self).__init__()
        self.gnn = gnn
        self.pooling = args.pooling
        self.discriminator = discriminator

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        g = self.gnn(x, edge_index, edge_attr, batch)
        
        # return F.log_softmax(g, dim=-1) # ·ÖÀà
        # return F.sigmoid(g)
        return g # Ô¤²â
