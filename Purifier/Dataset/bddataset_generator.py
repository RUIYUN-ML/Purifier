import os
import sys
sys.path.append(os.path.abspath('Purifier'))
from Attacks.Basic import Container
from utils.controller import myDataset
from torch.utils.data import Dataset
from tqdm import tqdm
import random

'''
The procession to generate backdoor trainset or testset:

1.Make a list to save your attack class:[attack1,...,attackN].
2.Send the list to Container class to generate your attack container.
3.Send the container to this BDDataset class.
4.Send the target dataset for instance CIFAR10 to BDDataset to generate your backdoor dataset dict. 
'''

class BDDataset():
    def __init__(self, container:Container, config:dict):
        self.target = config['Attack']['target']
        self.train_transform = config['Global']['train_transform']
        self.test_transform = config['Global']['test_transform']
        self.container = container
    
    def __call__(self, dataset:Dataset, ratio:float, train:bool=True):
        
        if train == False:
            assert ratio == 1., 'Wrong bddataset setting in backdoor testset generating!'
        elif train ==True:
            assert ratio != 0, 'Wrong bddataset setting in backdoor trainset generating!'

        length = len(dataset)
        perm = random.sample(range(length), int(ratio*length))
        bddataset = dict()

        for attack_idx in range(len(self.container)):
            bddataset[self.container.attack[attack_idx]._name_] = list()

        for attack_idx in range(len(self.container)):
            with tqdm(range(length),'Trigger generating') as tbar:
                
                for idx, (data, label) in enumerate(dataset):
                    
                    if idx in perm:
                        if train == True:
                            bdsamples = self.container(data)
                            if idx == perm[0]:
                                self.container.sample_ploter(bdsamples)
                            del bdsamples['Clean']
                            for name, sample in bdsamples.items():
                                bddataset[name].append((sample, self.target))
                        
                        elif train == False:
                            if label == self.target:
                                tbar.update(1)
                                continue
                            else:
                                bdsamples = self.container(data)
                                if idx == perm[0]:
                                    self.container.sample_ploter(bdsamples)
                                del bdsamples['Clean']
                                for name, sample in bdsamples.items():
                                    bddataset[name].append((sample, self.target))
                        
                        tbar.update(1)
                    
                    else:
                        tbar.update(1)
                        continue

        
        if train == True:
            for attack_name, bdset in bddataset.items():
                bddataset[attack_name] = myDataset(bdset+list(dataset), transform=self.train_transform)

        elif train == False:
            for attack_name, bdset in bddataset.items():
                bddataset[attack_name] = myDataset(bdset, transform=self.test_transform)

        '''
        {
            'Attack 1':Dataset1,
            ...
            'Attack N':DatasetN
        }
        ''' 
        return bddataset



