import torch
import sys
sys.path.append('..')
from Purifier.Models.ResNet.PerbResNet import presnet18, NoisyBatchNorm2d, NoisyBatchNorm1d
from Purifier.Attacks.BadNets import BadNets_attack
from Purifier.Attacks.CL import CLattack
from Purifier.Attacks.Blend import Blend_attack
from Purifier.Attacks.SIG import SIG_attack
from Purifier.Attacks.Trojan import Trojan_attack
from Purifier.utils.picker import*
from config import config
import pandas as pd


model = presnet18(10, NoisyBatchNorm2d)
model_dict = model.state_dict()
device = 'cuda'

pretrained_dict = torch.load('../Purifier/Checkpoints/MNIST/ResNet18/Blend.pth')['model']

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

data_dict = dataset_picker(config)
trainset = data_dict['train']
testset = data_dict['test']
attacker = Blend_attack(config)

from torchvision import transforms
train_transform = transforms.Compose([
transforms.RandomHorizontalFlip(),
transforms.RandomCrop(32, padding=4),
transforms.ToTensor(),
])
test_transform = transforms.Compose([
transforms.ToTensor(),
])

class myDataset(Dataset):
    """
    be used to transfer class list to class Dataset 
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        if self.transform != None:
            image = self.transform(image)
        return image, label

trigger_set = []
for (data, label) in testset:
    if label != 5:
        trigger_set.append((attacker.make_trigger(data), 5))

testset = list(testset)
testset = myDataset(testset, test_transform)
trigger_set = myDataset(trigger_set, test_transform)

from torch.utils.data import DataLoader, random_split, RandomSampler

testloader = DataLoader(testset, 500)
triggerloader = DataLoader(trigger_set, 500)
model.to('cuda')
model.eval()


clean_acc = 0
for idx, (data, label) in enumerate(testloader):
    data, label = data.to('cuda'), label.to("cuda")
    output = model(data)
    pred = output.argmax(dim=1,keepdim=True)
    clean_acc+=pred.eq(label.view_as(pred)).sum().item()

clean_acc/=len(testset)
print('Clean : {}%'.format(100*clean_acc))


trigger_acc = 0
for idx, (data, label) in enumerate(triggerloader):
    data, label = data.to('cuda'), label.to("cuda")
    output = model(data)
    pred = output.argmax(dim=1,keepdim=True)
    trigger_acc+=pred.eq(label.view_as(pred)).sum().item()

trigger_acc/=len(trigger_set)
print('Trigger : {}%'.format(100*trigger_acc))



length = len(trainset)
p = 0.1
_, val_trainset = random_split(trainset, [int((1-p)*length), length-int((1-p)*length)])
val_trainset = list(val_trainset)
val_trainset = myDataset(val_trainset, train_transform)



parameters = list(model.named_parameters())
mask_params = [v for n, v in parameters if "neuron_mask" in n]
mask_optimizer = torch.optim.SGD(mask_params, lr = 0.2, momentum=0.9)
noise_params = [v for n, v in parameters if "neuron_noise" in n]
noise_optimizer = torch.optim.SGD(noise_params, lr = 0.25)

def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=0.4)



def mask_train(model, criterion, mask_opt, noise_opt, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if 0.4 > 0.0:
            reset(model, rand_init=True)
            for _ in range(1):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if 0.4 > 0.0:
            include_noise(model)
            output_noise = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)
        loss = 0.2 * loss_nat + (1 - 0.2) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

random_sampler = RandomSampler(data_source=val_trainset, replacement=True,
                                num_samples=500 )
clean_val_loader = DataLoader(val_trainset, batch_size=128,
                                shuffle=False, sampler=random_sampler, num_workers=0)

criterion = torch.nn.CrossEntropyLoss().to(device)


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

import time

ANP_time = 0.

for i in range(4):
    lr = mask_optimizer.param_groups[0]['lr']
    t1 = time.time()
    train_loss, train_acc = mask_train(model=model, criterion=criterion, data_loader=clean_val_loader,
                                        mask_opt=mask_optimizer, noise_opt=noise_optimizer)
    ANP_time+= time.time()-t1
    cl_test_loss, cl_test_acc = test(model=model, criterion=criterion, data_loader=testloader)
    po_test_loss, po_test_acc = test(model=model, criterion=criterion, data_loader=triggerloader)

    print('{} \t {:.3f} \t  {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
        (i + 1) * 500, lr,  train_loss, train_acc, po_test_loss, po_test_acc,
        cl_test_loss, cl_test_acc))

def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)

save_mask_scores(model.state_dict(), os.path.join('../Purifier/', 'mask_values.txt'))

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values
mask_values = read_data('../Purifier/mask_values.txt')
mask_values = sorted(mask_values, key=lambda x: float(x[2]))




import numpy as np
def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


def evaluate_by_number(model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    nb_max = int(np.ceil(pruning_max))
    nb_step = int(np.ceil(pruning_step))
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        print('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
    return results


def evaluate_by_threshold(model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
    start = 0
    for threshold in thresholds:
        idx = start
        for idx in range(start, len(mask_values)):
            if float(mask_values[idx][2]) <= threshold:
                pruning(model, mask_values[idx])
                start += 1
            else:
                break
        layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        print('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{:.2f} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
    return results




model = presnet18(10, NoisyBatchNorm2d)
model_dict = model.state_dict()
device = 'cuda'
model.to(device)
pretrained_dict = torch.load('../Purifier/Checkpoints/MNIST/ResNet18/Blend.pth')['model']

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()


clean_acc = 0
for idx, (data, label) in enumerate(testloader):
    data, label = data.to('cuda'), label.to("cuda")
    output = model(data)
    pred = output.argmax(dim=1,keepdim=True)
    clean_acc+=pred.eq(label.view_as(pred)).sum().item()

clean_acc/=len(testset)
print('Clean : {}%'.format(100*clean_acc))


trigger_acc = 0
for idx, (data, label) in enumerate(triggerloader):
    data, label = data.to('cuda'), label.to("cuda")
    output = model(data)
    pred = output.argmax(dim=1,keepdim=True)
    trigger_acc+=pred.eq(label.view_as(pred)).sum().item()

trigger_acc/=len(trigger_set)
print('Trigger : {}%'.format(100*trigger_acc))
pruning_by = 'threshold'


if pruning_by == 'threshold':
    t1 = time.time()
    results = evaluate_by_threshold(
        model, mask_values, pruning_max=0.9, pruning_step=0.05,
        criterion=criterion, clean_loader=testloader, poison_loader=triggerloader
    )
    t2 = time.time()-t1
else:
    results = evaluate_by_number(
        model, mask_values, pruning_max=0.9, pruning_step=0.05,
        criterion=criterion, clean_loader=testloader, poison_loader=triggerloader
    )

print(ANP_time, t2)