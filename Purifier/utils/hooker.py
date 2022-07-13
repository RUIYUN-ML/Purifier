import torch
import torch.nn as nn



class hook():
    def __init__(self):
        self.grad_block = dict()
        self.feature_map_block = dict()
    
    def backward_hook(self, module:nn.Module, grad_in:torch.Tensor, grad_out:torch.Tensor):
        self.grad_block['grad_in'] = grad_in
        self.grad_block['grad_out'] = grad_out
    
    def forward_hook(self, module:nn.Module, input:torch.Tensor, output:torch.Tensor):
        self.feature_map_block['input'] = input
        self.feature_map_block['output'] = output