import os
import sys
sys.path.append(os.path.abspath('Purifier'))
from Attacks.Basic import basic_attacker
import PIL.Image
import numpy as np
import torch
from torchvision import transforms

class SIG_attack(basic_attacker):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._name_ = 'SIG'
    
    def make_trigger(self, sample:PIL.Image)->PIL.Image:
        alpha = 0.2
        loader = transforms.ToTensor()
        unloader = transforms.ToPILImage()
        width, height = sample.width, sample.height
        signal_mask = np.load('../Purifier/Attacks/trigger/{}/SIG.npy'.format(self.config['Global']['dataset']))
        signal_mask = torch.tensor(signal_mask/256.)
        blend_img = (1 - alpha) * loader(sample) + alpha * signal_mask.reshape((1, width, height))  # FOR CIFAR10
        blend_img = blend_img.clamp(0., 1.)
        blend_img = unloader(blend_img)
        return blend_img
