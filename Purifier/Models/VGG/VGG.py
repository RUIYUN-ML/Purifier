import torch.nn as tnn


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


def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16(tnn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.max_pooling = tnn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers
        self.layer6 = vgg_fc_layer(1 * 1 * 512, 512)
        self.layer7 = vgg_fc_layer(512, 512)

        # Final layer
        self.layer8 = tnn.Linear(512, num_classes)

    def forward(self, x):
        
        self.hook = list()
        out = self.layer1(x)
        out = self.layer2(out)
        activation1 = out
        self.hook.append(activation1)
        out = self.layer3(out)
        activation2 = out
        self.hook.append(activation2)
        out = self.layer4(out)
        activation3 = out
        self.hook.append(activation3)
        out = self.layer5(out)
        activation4 = out
        self.hook.append(activation4)
      
        out = out.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out




def vgg16(**kwargs):
    return VGG16(**kwargs)
