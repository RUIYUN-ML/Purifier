import os
from random import sample
import sys
sys.path.append(os.path.abspath('Purifier'))

from utils.hooker import hook
from Train.backdoortrainer import backdoortrain
from Dataset.cldataset_generator import CLDataset
from Attacks.Basic import basic_attacker
import numpy as np
import torch
from torch.autograd import grad
from torchvision import transforms
import PIL.Image


class Trojan_attack(basic_attacker):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._name_ = 'Trojan'
    
    def reference_model_train(self, dataset_picker, backdoor_train_auxiliary_picker):
        dataset = dataset_picker(self.config)
        cldataset = CLDataset(self.config)
        clean_trainset = cldataset(dataset['train'])
        clean_testset = cldataset(dataset['test'])

        auxiliary = backdoor_train_auxiliary_picker(self.config)

        reference_train_process = backdoortrain(self.config)
        reference_train_process.setting_update(auxiliary)
        reference_train_process.train(auxiliary['model'], clean_trainset, clean_testset, None, None)
    
    def neroun_reverse(self, model_picker):
        model = model_picker(self.config).to(self.config['Global']['device'])
        PATH = self.config['Global']['root_savepath']+'/{}/{}/'.format(self.config['Global']['dataset'], self.config['Global']['model'])
        model.load_state_dict(torch.load(PATH+'clean.pth')['model'])
        model.eval()
        unloader = transforms.ToPILImage()
        myhook = hook()
        neroun_idx = 0
        weight_value = 0.
        
        for name, params in model.named_parameters():
            if name == 'layer4.1.conv2.weight':
                shape = params.shape
                weight_value = torch.sum(params.view(shape[0],-1), dim = 1)
                neroun_idx = torch.max(weight_value, dim=0)[1]
                model.layer4.register_forward_hook(myhook.forward_hook)

            elif name == 'layer6.1.weight':
                shape = params.shape
                weight_value = torch.sum(params.view(shape[0],-1), dim = 1)
                neroun_idx = torch.max(weight_value, dim=0)[1]
                model.layer6.register_forward_hook(myhook.forward_hook)
        
        mask = torch.zeros((1,3,32,32), device=weight_value.device)
        mask[:,:, 25:28, 25:28] = torch.rand(1,3,32,32)[:,:,25:28,25:28]
        mask.detach_().requires_grad_(True)

        for idx in range(150):
            model(mask)
            if self.config['Global']['model'] == 'ResNet18':
                now = myhook.feature_map_block['output'][:, int(neroun_idx), :, :]
            elif self.config['Global']['model'] == 'VGG16':
                now = myhook.feature_map_block['output'][:, int(neroun_idx)]

            mask = self.trigger_update(mask, now, 10.)
            mask_show = unloader(mask.squeeze())
        
        trigger = unloader(mask.squeeze())
        torch.save(trigger, '../Purifier/Attacks/trigger/{}/Trojan.pth'.format(self.config['Global']['dataset']))

    
    def trigger_update(self, mask:torch.Tensor, now:torch.Tensor, target:float):
        
        loss = torch.pow(torch.norm(torch.max(now)-target), 2)
        
        update = grad(loss, mask)[0]
        mask.detach_()
        mask[:,:, 25:28, 25:28] -= 0.05*torch.sign(update[:,:, 25:28, 25:28])
        print(loss.item())
        mask.clamp_max_(1.).clamp_min_(0.)
        mask.detach_().requires_grad_(True)

        return mask

    def make_trigger(self, sample:PIL.Image)->PIL.Image:
        unloader = transforms.ToPILImage()
        width, height = sample.width, sample.height
        trojan_mask = np.load('../Purifier/Attacks/trigger/{}/Trojan.npz'.format(self.config['Global']['dataset']))['x']
        trojan_mask = np.transpose(trojan_mask, (1, 2, 0))
        trojan_img = np.clip((np.asarray(sample)+np.asarray(trojan_mask)).astype('uint8'), 0, 255)
        return PIL.Image.fromarray(trojan_img)
        
        

        


