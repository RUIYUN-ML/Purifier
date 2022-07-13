from cProfile import label
import os
import sys
sys.path.append(os.path.abspath('Purifier'))
from Attacks.Basic import basic_attacker
from torchvision.transforms import Normalize
import torch.nn as nn
import torch
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
import PIL.Image
import numpy as np
from utils.controller import myDataset

class PGD_attack():
    def __init__(self, config:dict):
        if isinstance(config['Global']['test_transform'].transforms[-1], Normalize):
            self.mean = config['Global']['test_transform'].transforms[-1].mean
            self.std = config['Global']['test_transform'].transforms[-1].std
        else:
            self.mean, self.std = None, None
        self._name_ = 'PGD'
        self.epsilon = config['PGD']['epsilon']
        self.alpha = config['PGD']['alpha']
        self.iter_max = config['PGD']['iter_max']
        self.target = config['Attack']['target']
        self.normalize(device=config['Global']['device'])
        self.loss = nn.CrossEntropyLoss()

    def normalize(self, device):
        if self.mean != None and self.std != None: 
            mean = torch.tensor(self.mean, device = device).view(1,3,1,1)
            std = torch.tensor(self.std, device = device).view(1,3,1,1)
            shape = mean.shape
            device = mean.device
            self.epsilon = torch.full(shape, self.epsilon, device=device).sub(mean).div(std)
            self.alpha = torch.full(shape, self.alpha, device=device).sub(mean).div(std)
        else:
            pass

    def adv_gen(self, model:nn.Module,trigger:torch.Tensor ,x:torch.Tensor, label:torch.Tensor=None):
        x = x.to(trigger.device)
        label = label.to(trigger.device)
        x_adv = (x+trigger).detach().requires_grad_(True)

        for idx in range(1, self.iter_max+1):
            if self.target == None:
                loss = self.loss(model(x_adv), label)
            else:
                target = torch.full(label.shape, self.target, device = label.device)
                loss = self.loss(model(x_adv), target)
            update = grad(loss, x_adv)[0]
            sign_update = self.alpha*torch.sign(update)
            mask_update = torch.zeros(sign_update.shape, device=sign_update.device)
            mask_update[:,:,0:4,0:4] = sign_update[:,:,0:4,0:4]

            x_adv = x_adv + mask_update if self.target == None else x_adv - mask_update
            x_adv.clamp_(min = x - self.epsilon, max = x + self.epsilon)
            x_adv.clamp_(0.,1.)
            x_adv.detach_().requires_grad_(True)
        
        return (x_adv-x)[0].unsqueeze(dim=0)



class CLattack(basic_attacker):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._name_='CL'
        self.PGD = PGD_attack(config)


    def trainset_split(self, trainset:Dataset):
        perm = list()
        for idx, (_, label) in enumerate(trainset):
            if label == self.config['Attack']['target']:
                pass
            else:
                perm.append(idx)
        splitset = Subset(trainset, perm)
        length = len(splitset)
        poisoned_length = int(0.5*length)
        splitset,_ = random_split(splitset, [poisoned_length, length-poisoned_length])
        return splitset
        

    def adv_gen(self, model:nn.Module, trainset:Dataset):
        model.eval()
        splitset = self.trainset_split(trainset)
        splitset = myDataset(list(splitset), transforms.ToTensor())
        splitloader = DataLoader(splitset, batch_size=self.config['Global']['test_batch_size'])
        mask = torch.zeros((1,3,32,32), device=self.config['Global']['device'])
        mask[:,:,0:4,0:4] = torch.rand((1,3,32,32), device=mask.device)[:,:,0:4,0:4]

        for idx, (data, label) in enumerate(splitloader):
            mask = self.PGD.adv_gen(model, mask, data, label)
        
        unloader = transforms.ToPILImage()
        mask = unloader(mask.squeeze())
        torch.save(mask, '../Purifier/Attacks/trigger/{}/CL.pth'.format(self.config['Global']['dataset']))
    
    def make_trigger(self, sample:PIL.Image):
        loader = transforms.ToTensor()
        unloader = transforms.ToPILImage()
        cl_mask = torch.load('../Purifier/Attacks/trigger/{}/CL.pth'.format(self.config['Global']['dataset']))
        cl_img = (loader(sample)+loader(cl_mask)).clamp_(0., 1.)
        return unloader(cl_img)
