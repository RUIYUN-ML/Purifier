import os
import sys
path = os.path.abspath('Purifier')
sys.path.append(path)
from utils.unnormalizer import unnormalize
from utils.controller import doc_maker
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

'''
This file provides method to visualize the clean samples and backdoor samples for every backdoor attack
The input samples has below structure:
samples = {
    'Clean':Tensor(3*32*32),
    'Attack 1':Tensor(3*32*32),
    ...
    'Attack N':Tensor(3*32*32)
}
'''

class backdoor_samples():
    def __init__(self, config:dict) -> None:
        self.root_savepath = config['Attack']['root_savepath']
        if isinstance(config['Global']['test_transform'].transforms[-1], transforms.Normalize):
            self.mean = config['Global']['test_transform'].transforms[-1].mean
            self.std = config['Global']['test_transform'].transforms[-1].std
        else:
            self.mean = (0.,0.,0.)
            self.std = (1.,1.,1.)
            
        self.dataset = config['Global']['dataset']

        assert isinstance(self.mean, tuple), 'Maybe revise the vital parameters \'mean\''
        assert isinstance(self.std, tuple), 'Maybe revise the vital parameters \'std\''

    def show(self, samples:dict):
        backdoor_plot(samples, self.dataset, self.root_savepath, self.mean, self.std)
        
def backdoor_plot(samples:dict, dataset:str, root_savepath:str, mean:tuple, std:tuple):
    fig = plt.figure()

    shape = (2, 3)
    loader = transforms.ToTensor()
    for idx, (names, sample) in enumerate(samples.items()):
        if idx == 0:
            continue
        sample = loader(sample)
        assert len(sample.shape) == 3,'Wrong shape of samples in visualization!'
        ax = fig.add_subplot(shape[0],shape[1], idx)
        show_sample = unnormalize(sample, mean, std)
        ax.imshow(show_sample)
        ax.set_title('{}'.format(names))
    
    path = root_savepath + '{}/{}/'.format(dataset,'Samples_Contrast')
    doc_maker(path)

    plt.savefig(path+'{}.png'.format('Samples'))
    


