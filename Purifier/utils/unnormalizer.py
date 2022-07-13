import torch
from torch import Tensor
from torchvision import transforms
import PIL.Image

def unnormalize(data:Tensor, mean:tuple, std:tuple)->PIL.Image:
    if len(data.shape) == 4:
        data = data[0]
    unloader = transforms.ToPILImage()
    mean = torch.tensor(mean, device=data.device).reshape(3, 1, 1)
    std = torch.tensor(std, device=data.device).reshape(3, 1, 1)
    redata = data.mul(std).add(mean).detach().cpu()
    redata = unloader(redata)

    return redata

