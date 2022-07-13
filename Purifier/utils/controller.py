import torch
import torch.cuda
import torch.backends.cudnn
from torch.utils.data import Dataset
import numpy as np
import random
import os

PYTORCH_NO_CUDA_MEMORY_CACHING=1
def set_seed(seed=9699): # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    #但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    #cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class myDataset(Dataset):
    """
    be used to transfer class list to class Dataset 
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        if self.transform != None:
            image = self.transform(image)
        return image, label


def doc_maker(path:str):
    if not os.path.exists(path):
        os.makedirs(path)

def summer(names:list):
    concat = ''
    for name in names:
        concat += name
    return concat

