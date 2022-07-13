import os
from torchvision import transforms
x = transforms.Normalize
config = {
    'Global':{
        'dataset':'MNIST',
        'model':'ResNet18',
        'in_channels':3,
        
        'opt':'SGD',
        'lr':1e-1,
        'weight_decay':1e-4,
        'momentum':0.9,
        
        'stl':'MultiStepLR',
        'milestones':[70,120],
        'gamma':0.1,

        'epoch_num':80,
        'train_batch_size':64,
        'test_batch_size':100,
        'root_savepath':'../Purifier/Checkpoints/',

        'train_transform':transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        ]),

        'test_transform':transforms.Compose([transforms.ToTensor()]),

        'device':'cuda'
    },

    'Attack':{
        'target':5,
        'ratio':0.1,
        'root_savepath':'../Purifier/Visualization/'
    },

    'PGD':{
        'epsilon':16/256,
        'alpha':4/256,
        'iter_max':20,
    },

    'CAS':{
        'opt':'SGD',
        'lr':1e-2,
        'weight_decay':1e-4,
        'momentum':0.9,
        
        'stl':'MultiStepLR',
        'milestones':[20],
        'gamma':0.0,

        'epoch_num':20,
    },

    'Finetune':{
        'opt':'SGD',
        'lr':1e-2,
        'weight_decay':1e-4,
        'momentum':0.9,
        
        'stl':'MultiStepLR',
        'milestones':[20],
        'gamma':0.0,

        'epoch_num':20
    },

    'NAD':{        
        'opt':'SGD',
        'lr':1e-1,
        'weight_decay':1e-4,
        'momentum':0.9,
        
        'stl':'MultiStepLR',
        'milestones':[2,4,6,8,10],
        'gamma':0.5,

        'epoch_num':10,
    },

    'ISO':{
        'opt':'SGD',
        'lr':1e-1,
        'weight_decay':1e-4,
        'momentum':0.9,
        
        'stl':'MultiStepLR',
        'milestones':[70,120],
        'gamma':0.1,

        'epoch_num':20,
    },

    'UN':{
        'opt':'SGD',
        'lr':5e-4,
        'weight_decay':1e-4,
        'momentum':0.9,
        
        'stl':'MultiStepLR',
        'milestones':[10,120],
        'gamma':0.2,

        'epoch_num':20,
    }


}

