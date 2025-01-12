import os
import torch
import random
import deepdish as dd
import torch.utils.data as data

from clip import clip
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse


def is_h5(x):
    if x.endswith('.h5'):
        return True
    else:
        return False
        
def fmri_norm(x):
    mean = x.mean(0)
    std = x.std(0)
    norm_x = (x - mean) / std
    return norm_x

def _np2Tensor(sample):
    tensor = torch.from_numpy(sample).half() # numpy 转化为 tensor
    return tensor

'''读取训练数据'''
class get_train(data.Dataset):
    def __init__(self, data_path):

        self.data_path = data_path
        print('********* Train dir *********')
        print(self.data_path)
        self.list_data = self._scan()

    '''遍历图像，获取名称集合'''
    def _scan(self):
        # 遍历 groudtruth 路径中的 h5 文件，其名字形成列表
        list_data = sorted([os.path.join(self.data_path, x) for x in os.listdir(self.data_path) if is_h5(x)])
        random.shuffle(list_data)
        return list_data

    def __getitem__(self, idx):
        time, corr, text, label, file = self._load_file(idx)  # 获取图像
         # 转化为 tensor
        time_ts = _np2Tensor(time)   # 116x100
        corr_ts = _np2Tensor(corr)   # 116x116
        label_ts = _np2Tensor(label)   # 1
        token_ts = clip.tokenize(text, truncate=True)   # 77
        return time_ts, corr_ts, token_ts, label_ts

    def __len__(self):
        return len(self.list_data)

    '''依次读取每个 patch 的图像的索引'''
    def _get_index(self, idx):
        return idx % len(self.list_data)   # 余数

    '''依次读取每个 patch 的图像'''
    def _load_file(self, idx):
        idx = self._get_index(idx)     # 选取 idx
        file = self.list_data[idx]   # 选取训练图像名
        temp = dd.io.load(file)
        time = temp['timeseires'][()]
        time = fmri_norm(time)
        corr = temp['corr'][()]
        text = temp['text']
        label = temp['label'][()]
        return time, corr, text, label, file
    
    
'''读取测试数据'''
class get_test(data.Dataset):
    def __init__(self, data_path):

        self.data_path = data_path
        print('********* Test dir *********')
        print(self.data_path)
        self.list_data = self._scan()

    '''遍历图像，获取名称集合'''
    def _scan(self):
        # 遍历 groudtruth 路径中的 h5 文件，其名字形成列表
        list_data = sorted([os.path.join(self.data_path, x) for x in os.listdir(self.data_path) if is_h5(x)])
        return list_data

    def __getitem__(self, idx):
        time, corr, text, label, file = self._load_file(idx)  # 获取图像
         # 转化为 tensor
        time_ts = _np2Tensor(time)   # 116x100
        corr_ts = _np2Tensor(corr)   # 116x116
        label_ts = _np2Tensor(label)   # 1
        token_ts = clip.tokenize(text, truncate=True)   # 77
        return time_ts, corr_ts, token_ts, label_ts

    def __len__(self):
        return len(self.list_data)

    '''依次读取每个 patch 的图像的索引'''
    def _get_index(self, idx):
        return idx % len(self.list_data)   # 余数

    '''依次读取每个 patch 的图像'''
    def _load_file(self, idx):
        idx = self._get_index(idx)     # 选取 idx
        file = self.list_data[idx]   # 选取训练图像名
        temp = dd.io.load(file)
        time = temp['timeseires'][()]
        time = fmri_norm(time)
        corr = temp['corr'][()]
        text = temp['text']
        label = temp['label'][()]
        return time, corr, text, label, file





