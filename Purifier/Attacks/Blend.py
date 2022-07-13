import os
import sys
sys.path.append(os.path.abspath('Purifier'))
from Attacks.Basic import basic_attacker
import PIL.Image
from torchvision import transforms
import torch

class Blend_attack(basic_attacker):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._name_ = 'Blend'
    
    def make_trigger(self, sample:PIL.Image)->PIL.Image:
        loader = transforms.ToTensor()
        unloader = transforms.ToPILImage()        
        alpha = 0.2
        width, height = sample.width, sample.height
        sample = loader(sample)
        mask = torch.rand(sample.shape)
        blend_img = (1 - alpha) * sample + alpha * mask
        blend_img = blend_img.clamp(0.,1.)

        return unloader(blend_img)