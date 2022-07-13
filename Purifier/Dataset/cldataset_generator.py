from utils.controller import myDataset
from torch.utils.data import Dataset

class CLDataset():
    def __init__(self, config:dict):
        self.train_transform = config['Global']['train_transform']
        self.test_transform = config['Global']['test_transform']
    
    def __call__(self, dataset:Dataset, train:bool=True):
        if train == True:
            return myDataset(list(dataset), self.train_transform)
        elif train == False:
            return myDataset(list(dataset), self.test_transform)
        