import os
import sys
sys.path.append('..')
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
sys.path.append('../Purifier/Models/')
from Purifier.Models.ResNet.ChannelResNet import channel_resnet18
from Purifier.Models.ResNet.CubicResNet import cubic_resnet18
from Purifier.Models.ResNet.FeatureResNet import feature_resnet18
from Purifier.Models.ResNet.ResNet import resnet18
from Purifier.Models.ResNet.prun_ResNet import Prun_resnet

from Purifier.Models.VGG.ChannelVGG import channel_vgg16
from Purifier.Models.VGG.CubicVGG import cubic_vgg16
from Purifier.Models.VGG.FeatureVGG import feature_vgg16
from Purifier.Models.VGG.VGG import vgg16
from Purifier.Models.VGG.prun_VGG import prun_vgg16

from Purifier.Attacks.BadNets import BadNets_attack
from Purifier.Attacks.Blend import Blend_attack
from Purifier.Attacks.SIG import SIG_attack
from Purifier.Attacks.Trojan import Trojan_attack

def dataset_picker(config:dict)->dict:
    dataset = dict()

    if config['Global']['dataset'] == 'CIFAR10':

        trainset = datasets.CIFAR10(
            root = '/home/data/',
            transform = None,
            train = True
        )
        testset = datasets.CIFAR10(
            root = '/home/data/',
            transform = None,
            train = False
        )
    
    elif config['Global']['dataset'] == 'MNIST':
        trainset = datasets.MNIST(
            root = '/home/data/',
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3)
            ]),
            train = True
        )
        testset = datasets.MNIST(
            root = '/home/data/',
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3)
            ]),
            train = False
        )

    elif config['Global']['dataset'] == 'SVHN':

        trainset = datasets.SVHN(
            root='/home/data/SVHN/',
            split = 'train',
            transform = None
        )
        testset = datasets.SVHN(
            root='/home/data/SVHN/',
            split = 'test',
            transform = None
        )
    
    elif config['Global']['dataset'] == 'GTSRB':
        train_dataset_root = '/home/data/GTSRB/Final_Training/Images'
        trainset = datasets.ImageFolder(train_dataset_root, transform=transforms.Compose([
                transforms.Resize((32,32))
            ]))
        testset = MyDataset(txt='/home/data/GTSRB/Final_Test/test_dataset.txt', transform=transforms.Compose([
                transforms.Resize((32,32))
            ]))

    else:
        raise NotImplementedError('The picker doesn\'t difine the dataset you want!')

    dataset['train'] = trainset
    dataset['test'] = testset

    return dataset

def model_picker(config:dict)->nn.Module:
    if config['Global']['model'] == 'ResNet18':
        if config['Global']['dataset'] == 'GTSRB':
            model = resnet18(num_classes=43).to(config['Global']['device'])
        else:
            model = resnet18(num_classes=10).to(config['Global']['device'])
    
    elif config['Global']['model'] == 'Prune_ResNet18':
        if config['Global']['dataset'] == 'GTSRB':
            model = Prun_resnet(num_classes=43).to(config['Global']['device'])
        else:
            model = Prun_resnet(num_classes=10).to(config['Global']['device'])

    elif config['Global']['model'] == 'Channel_ResNet18':
        if config['Global']['dataset'] == 'GTSRB':
            model = channel_resnet18(num_classes=43).to(config['Global']['device'])
        else:
            model = channel_resnet18(num_classes=10).to(config['Global']['device'])

    elif config['Global']['model'] == 'Cubic_ResNet18':
        if config['Global']['dataset'] == 'GTSRB':
            model = cubic_resnet18(num_classes=43).to(config['Global']['device'])
        else:
            model = cubic_resnet18(num_classes=10).to(config['Global']['device'])
    elif config['Global']['model'] == 'Feature_ResNet18':
        if config['Global']['dataset'] == 'GTSRB':
            model = feature_resnet18(num_classes=43).to(config['Global']['device'])
        else:
            model = feature_resnet18(num_classes=10).to(config['Global']['device'])
    elif config['Global']['model'] == 'VGG16':
        if config['Global']['dataset'] == 'GTSRB':
            model = vgg16(num_classes=43).to(config['Global']['device'])
        else:
            model = vgg16(num_classes=10).to(config['Global']['device'])
        
    elif config['Global']['model'] == 'Channel_VGG16':
        if config['Global']['dataset'] == 'GTSRB':
            model = channel_vgg16(num_classes=43).to(config['Global']['device'])
        else:
            model = channel_vgg16(num_classes=10).to(config['Global']['device'])

    elif config['Global']['model'] == 'Cubic_VGG16':
        if config['Global']['dataset'] == 'GTSRB':
            model = cubic_vgg16(num_classes=43).to(config['Global']['device'])
        else:
            model = cubic_vgg16(num_classes=10).to(config['Global']['device'])

    elif config['Global']['model'] == 'Feature_VGG16':
        if config['Global']['dataset'] == 'GTSRB':
            model = feature_vgg16(num_classes=43).to(config['Global']['device'])
        else:
            model = feature_vgg16(num_classes=10).to(config['Global']['device'])
    
    elif config['Global']['model'] == 'Prune_VGG16':
        if config['Global']['dataset'] == 'GTSRB':
            model = prun_vgg16(num_classes=43).to(config['Global']['device'])
        else:
            model = prun_vgg16(num_classes=10).to(config['Global']['device'])

    return model

    
    

def optimizer_picker(config:dict, model:nn.Module)->SGD: 

    if config['opt'] == 'SGD':
        lr = config['lr']
        weight_decay = config['weight_decay']
        momentum = config['momentum']
        optimizer = SGD(model.parameters(), lr, momentum, weight_decay)
    else:
        raise NotImplementedError('You haven\'t define this opt!')
    
    return optimizer

def scheduler_picker(config:dict, opt:SGD)->MultiStepLR:

    if config['stl'] == 'MultiStepLR':
        milestones = config['milestones']
        gamma = config['gamma']
        scheduler = MultiStepLR(opt, milestones, gamma)
    else:
        raise NotImplementedError('You haven\'t define this stl!')
    
    return scheduler


def backdoor_train_auxiliary_picker(config:dict):
    model = model_picker(config)
    optimizer = optimizer_picker(config['Global'], model)
    scheduler = scheduler_picker(config['Global'], optimizer)
    
    return {
        'model':model,
        'opt':optimizer,
        'stl':scheduler
    }

def finetune_train_auxiliary_picker(config:dict):
    model = model_picker(config)
    optimizer = optimizer_picker(config['Finetune'], model)
    scheduler = scheduler_picker(config['Finetune'], optimizer)
    
    return {
        'model':model,
        'opt':optimizer,
        'stl':scheduler
    }

def cas_train_auxiliary_picker(config:dict):
    model = model_picker(config)
    optimizer = optimizer_picker(config['CAS'], model)
    scheduler = scheduler_picker(config['CAS'], optimizer)
    
    return {
        'model':model,
        'opt':optimizer,
        'stl':scheduler
    }

def nad_train_auxiliary_picker(config:dict):
    model = model_picker(config)
    optimizer = optimizer_picker(config['NAD'], model)
    scheduler = scheduler_picker(config['NAD'], optimizer)
    
    return {
        'model':model,
        'opt':optimizer,
        'stl':scheduler
    }

def isolate_train_auxiliary_picker(config:dict):
    model = model_picker(config)
    optimizer = optimizer_picker(config['ISO'], model)
    scheduler = scheduler_picker(config['ISO'], optimizer)
    
    return {
        'model':model,
        'opt':optimizer,
        'stl':scheduler
    }

def unlearn_train_auxiliary_picker(config:dict):
    model = model_picker(config)
    optimizer = optimizer_picker(config['UN'], model)
    scheduler = scheduler_picker(config['UN'], optimizer)
    
    return {
        'model':model,
        'opt':optimizer,
        'stl':scheduler
    }


class MyDataset(Dataset):  
    def __init__(self, txt, transform=None, target_transform=None):  
        super(MyDataset, self).__init__()  
        fh = open(txt, 'r')  
        imgs = []
        for line in fh: 
            line = line.strip('\n')
            line = line.rstrip('\n') 
            words = line.split()  
            imgs.append((words[0], int(words[1]))) 

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  
        fn, label = self.imgs[index]  
        img = Image.open(fn).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img) 
            if img.size != (32,32):
                print(index)
        return img, label 

    def __len__(self): 
        return len(self.imgs)