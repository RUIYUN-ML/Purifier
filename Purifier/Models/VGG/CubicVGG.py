import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import sys
sys.path.append('../Purifier/')
from Purifier.config import config




def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


class Conv_FC_Block(tnn.Module):

    def __init__(self, chann_in, chann_out, k_size, p_size):
        super(Conv_FC_Block, self).__init__()
        self.conv = tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size)
        self.bn = tnn.BatchNorm2d(chann_out)
        self.relu = tnn.ReLU()
        self.channel_out = chann_out
        if config['Global']['dataset'] == 'GTSRB':
            self.fc = tnn.Linear(32*32, 43)
        else:
            self.fc = tnn.Linear(32*32, 10)
        

    def forward(self, x, label=None):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        fc_in = torch.mean(out.view(out.shape[0], out.shape[1], -1), dim=1)
        fc_out = self.fc(fc_in.view(out.shape[0], -1))
        if self.training:
            N, C, H, W = out.shape
            mask = self.fc.weight[label, :]
            out = out * mask.view(N, 1, H, W)
        else:
            N, C, H, W = out.shape
            pred_label = torch.max(fc_out, dim=1)[1]
            mask = self.fc.weight[pred_label, :]
            out = out * mask.view(N, 1, H, W)
        return out, fc_out

def vgg_conv_block_channel_reg(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [Conv_FC_Block(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    # layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.ModuleList(layers)

def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16_cubic(tnn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_cubic, self).__init__()

        # # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block_channel_reg([3,64],[64,64],[3,3],[1,1],2,2)
        self.max_pooling = tnn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        

        # # FC layers
        # self.layer6 = vgg_fc_layer(1 * 1 * 512, 512)
        # self.layer7 = vgg_fc_layer(512, 512)
        # Conv blocks (BatchNorm + ReLU activation added in each block)


    

        # FC layers
        self.layer6 = vgg_fc_layer(1 * 1 * 512, 512)
        self.layer7 = vgg_fc_layer(512, 512)

        # Final layer
        self.layer8 = tnn.Linear(512, num_classes)

    def forward(self, x, y=None, eval=False):
        if eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()
        
        extra_output = []
        out = x
        for layer in self.layer1:
            out, layer1_out = layer(out, y)
            extra_output.append(layer1_out)
        out = self.max_pooling(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return {
            'normal':out,
            'auxiliary':extra_output
        }


def cubic_vgg16(**kwargs):
    return VGG16_cubic(**kwargs)
    