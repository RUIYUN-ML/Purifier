import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class NoisyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NoisyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.neuron_mask = Parameter(torch.Tensor(num_features))
        self.neuron_noise = Parameter(torch.Tensor(num_features))
        self.neuron_noise_bias = Parameter(torch.Tensor(num_features))
        init.ones_(self.neuron_mask)
        init.zeros_(self.neuron_noise)
        init.zeros_(self.neuron_noise_bias)
        self.is_perturbed = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.neuron_noise, a=-eps, b=eps)
            init.uniform_(self.neuron_noise_bias, a=-eps, b=eps)
        else:
            init.zeros_(self.neuron_noise)
            init.zeros_(self.neuron_noise_bias)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        if self.is_perturbed:
            coeff_weight = self.neuron_mask + self.neuron_noise
            coeff_bias = 1.0 + self.neuron_noise_bias
        else:
            coeff_weight = self.neuron_mask
            coeff_bias = 1.0
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight, self.bias * coeff_bias,
            bn_training, exponential_average_factor, self.eps)


class NoisyBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(NoisyBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.neuron_mask_fc = Parameter(torch.Tensor(num_features))
        self.neuron_noise_fc = Parameter(torch.Tensor(num_features))
        self.neuron_noise_bias_fc = Parameter(torch.Tensor(num_features))
        init.ones_(self.neuron_mask_fc)
        init.zeros_(self.neuron_noise_fc)
        init.zeros_(self.neuron_noise_bias_fc)
        self.is_perturbed = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.neuron_noise_fc, a=-eps, b=eps)
            init.uniform_(self.neuron_noise_bias_fc, a=-eps, b=eps)
        else:
            init.zeros_(self.neuron_noise_fc)
            init.zeros_(self.neuron_noise_bias_fc)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        if self.is_perturbed:
            coeff_weight = self.neuron_mask_fc + self.neuron_noise_fc
            coeff_bias = 1.0 + self.neuron_noise_bias_fc
        else:
            coeff_weight = self.neuron_mask_fc
            coeff_bias = 1.0
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight, self.bias * coeff_bias,
            bn_training, exponential_average_factor, self.eps)


class NoisyBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(NoisyBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.neuron_mask_fc = Parameter(torch.Tensor(num_features))
        self.neuron_noise_fc = Parameter(torch.Tensor(num_features))
        self.neuron_noise_bias_fc = Parameter(torch.Tensor(num_features))
        init.ones_(self.neuron_mask_fc)
        init.zeros_(self.neuron_noise_fc)
        init.zeros_(self.neuron_noise_bias_fc)
        self.is_perturbed = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.neuron_noise_fc, a=-eps, b=eps)
            init.uniform_(self.neuron_noise_bias_fc, a=-eps, b=eps)
        else:
            init.zeros_(self.neuron_noise_fc)
            init.zeros_(self.neuron_noise_bias_fc)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        if self.is_perturbed:
            coeff_weight = self.neuron_mask_fc + self.neuron_noise_fc
            coeff_bias = 1.0 + self.neuron_noise_bias_fc
        else:
            coeff_weight = self.neuron_mask_fc
            coeff_bias = 1.0
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight, self.bias * coeff_bias,
            bn_training, exponential_average_factor, self.eps)


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d
        else:
            self._norm_layer = norm_layer
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self._norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def presnet18(num_classes=10, norm_layer=nn.BatchNorm2d):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, norm_layer)


def resnet34(num_classes=10, norm_layer=nn.BatchNorm2d):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, norm_layer)


def resnet50(num_classes=10, norm_layer=nn.BatchNorm2d):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, norm_layer)


def resnet101(num_classes=10, norm_layer=nn.BatchNorm2d):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer)


def resnet152(num_classes=10, norm_layer=nn.BatchNorm2d):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, norm_layer)


def test():
    net = presnet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
