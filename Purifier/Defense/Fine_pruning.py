import sys
import os
sys.path.append('../Purifier/')
from utils.picker import *
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import copy
from Models.ResNet.ResNet import resnet18
from Models.VGG.VGG import vgg16

class Fine_pruning():
    def __init__(self, config) -> None:
        self.config = config
    
    def setting_update(self, train_auxiliary_settings:dict):
        self.opt = train_auxiliary_settings['opt']
        self.stl = train_auxiliary_settings['stl']
        self.loss = nn.CrossEntropyLoss()
    
    def make_loader(self, dataset:Dataset, train:bool):
        if train == True:
            return DataLoader(dataset, batch_size=self.config['Global']['train_batch_size'], shuffle=True)
        else:
            return DataLoader(dataset, batch_size=self.config['Global']['test_batch_size'])
    

    def model_loader(self, model:nn.Module, PATH:str, attack_name=None):

        if attack_name == None:
            PATH += 'clean.pth'
        else:
            PATH += '{}.pth'.format(attack_name)

        model_dict = torch.load(PATH)['model']
        mymodel_dict = model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if k in mymodel_dict}
        mymodel_dict.update(model_dict)
        model.load_state_dict(mymodel_dict)

        return model
    
    def trainset_split(self, trainset:Dataset):
        length = 0.05*len(trainset)
        trainset, _ = random_split(trainset, [int(length), len(trainset)-int(length)])
        return trainset
    

    def frozen(self, model:nn.Module):
        for params in model.parameters():
            params.requires_grad_(False)
    
    def defrozen(self, model:nn.Module):
        for params in model. parameters():
            params.requires_grad_(True)
    
    def prune(self, model:nn.Module, trainset:Dataset, clean_testset:Dataset, trigger_testset:Dataset = None, attack_name = None):
        
        model.eval()
        if 'ResNet' in self.config['Global']['model']:
            model_copy = resnet18(num_classes=10).to('cuda') if self.config['Global']['dataset'] != 'GTSRB' else resnet18(num_classes=43).to('cuda')
        elif 'VGG' in self.config['Global']['model']:
            model_copy = vgg16(num_classes=10).to('cuda') if self.config['Global']['dataset'] != 'GTSRB' else vgg16(num_classes=43).to('cuda')
        model_copy.eval()

        trainset = self.trainset_split(trainset)
        trainloader = self.make_loader(trainset, True)
        clean_testloader = self.make_loader(clean_testset, False)

        if 'ResNet' in self.config['Global']['model']:
            PATH = config['Global']['root_savepath']+'/{}/{}/'.format(config['Global']['dataset'], 'ResNet18')
        elif 'VGG' in self.config['Global']['model']:
            PATH = config['Global']['root_savepath']+'/{}/{}/'.format(config['Global']['dataset'], 'VGG16')

        model = self.model_loader(model, PATH, attack_name).to('cuda')
        model_copy = self.model_loader(model_copy, PATH, attack_name).to('cuda')

        PATH+='Finepruing.pth'

        if trigger_testset != None:  
            '''
            When you train a clean model for trojan or cl, set the trigger testset=None
            '''
            assert isinstance(trigger_testset, Dataset), 'Wrong datatype of trigger testset!'
            trigger_testloader = self.make_loader(trigger_testset, False)
        else:
            trigger_testloader = None

        loss_record = list()
        acc = {
            'clean_acc':list(),
            'backdoor_asr':list()
        }
        TRAIN_BATCH_NUM = len(trainloader)

        """
        Trian the cas:
        """
        self.frozen(model_copy)

        pointer = list()
        def forward_hook(module, input, output):
            pointer.append(output)
        
        if 'ResNet' in self.config['Global']['model']:
            hook = model_copy.layer4.register_forward_hook(forward_hook)
        
        elif 'VGG' in self.config['Global']['model']:
            hook = model_copy.layer5.register_forward_hook(forward_hook)
        
        for idx, (data, label) in enumerate(trainloader):
            data = data.to(self.config['Global']['device'])
            model_copy(data)
        
        print('pointer length : {}'.format(len(pointer)))

        pointer = torch.cat(pointer, dim=0)
        activation = torch.mean(pointer, dim=[0, 2, 3])
        seq_sort = torch.argsort(activation)
   
        pruning_mask = torch.ones(seq_sort.shape[0], dtype=bool, device=self.config['Global']['device'])

        hook.remove()

        p1 = self.config['Global']['dataset'] == 'MNIST' or 'CIFAR10' or'SVHN'
        p2 = self.config['Global']['dataset'] == 'GTSRB'
        for index in range(pruning_mask.shape[0]):
            num_pruned = index

            if index:
                channel = seq_sort[index-1]
                pruning_mask[channel] = False
            
            if num_pruned<=416:
                continue
            
            print("Pruned {} filters".format(num_pruned))

            if 'ResNet' in self.config['Global']['model']:
                model.layer4[1].conv2 = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, 3, stride=1, padding=1, bias=False
                )
                model.layer4[1].bn2 = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned)
                model.fc = nn.Linear(pruning_mask.shape[0]-num_pruned, 10) if p1 else nn.Linear(pruning_mask.shape[0]-num_pruned, 43)
            
            elif 'VGG' in self.config['Global']['model']:
                model.layer5[2][0] = nn.Conv2d(
                pruning_mask.shape[0], pruning_mask.shape[0] - num_pruned, 3, padding=1, bias=False
                )
                model.layer5[2][1] = nn.BatchNorm2d(pruning_mask.shape[0] - num_pruned)
                model.layer6[0] = nn.Linear(pruning_mask.shape[0]-num_pruned, 10) if p1 else nn.Linear(pruning_mask.shape[0]-num_pruned, 43)

            for name, module in model._modules.items():
                if 'ResNet' in self.config['Global']['model']:
                    if 'layer4' == name:
                        module[1].conv2.weight.data = model_copy.layer4[1].conv2.weight.data[pruning_mask]
                        module[1].bn2.weight.data = model_copy.layer4[1].bn2.weight.data[pruning_mask]
                        module[1].bn2.bias.data = model_copy.layer4[1].bn2.bias.data[pruning_mask]
                        module[1].bn2.running_mean.data = model_copy.layer4[1].bn2.running_mean.data[pruning_mask]
                        module[1].bn2.running_var.data = model_copy.layer4[1].bn2.running_var.data[pruning_mask]
                    
                    elif 'fc' == name:
                        module.weight.data = model_copy.fc.weight.data[:, pruning_mask]
                        module.bias.data = model_copy.fc.bias.data
                
                elif 'VGG' in self.config['Global']['model']:
                    if 'layer5' == name:
                        module[2][0].weight.data = model_copy.layer5[2][0].weight.data[pruning_mask]
                        module[2][1].weight.data = model_copy.layer5[2][1].weight.data[pruning_mask]
                        module[2][1].bias.data = model_copy.layer5[2][1].bias.data[pruning_mask]
                    
                    elif 'layer6' == name:
                        module[0].weight.data = model_copy.layer6[0].weight.data[:,pruning_mask]
                        module[0].bias.data = model_copy.layer6[0].bias.data
            
            model.eval()
            model = model.to('cuda')
            prune_acc_dict = self.test(model, clean_testloader, trigger_testloader, prun_mask=pruning_mask)
            for name, item in prune_acc_dict.items():
                print('{} : {}'.format(name, item))
            
            if prune_acc_dict['clean_acc']<=0.91*len(clean_testset):
                self.defrozen(model)
                for epoch_idx in range(1, self.config['Finetune']['epoch_num']+1):
                    model.train()
                    epoch_loss = 0.
                    print('******************Begin the {}th of finetuning of pruning*******************'.format(epoch_idx))
                    for batch_idx, (data, label) in enumerate(trainloader):
                        self.opt.zero_grad()
                        data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                        output = model(data, pruning_mask.to('cuda'))
                        loss = self.loss(output, label)
                        loss.backward()
                        epoch_loss+=loss.item()
                        self.opt.step()
                    
                    loss_record.append(epoch_loss/TRAIN_BATCH_NUM)
                    self.stl.step()

                    model.eval()
                    acc_dict = self.test(model, clean_testloader, trigger_testloader, prun_mask=pruning_mask)

                    print('Avg loss : {}'.format(loss_record[-1]))
                    for name, item in acc_dict.items():
                        acc[name].append(item)
                        print('{} : {}'.format(name, item))
                    
                    torch.save({
                        'model':model.state_dict(),
                        'acc':acc,
                        'loss':loss_record
                    }, PATH)
                break
        return acc

    
    def test(self, model:nn.Module, clean_testloader, trigger_testloader=None, prun_mask=None):
        assert model.training == False, 'The model is training mode in testing!'
        
        clean_acc = 0.
        
        with torch.no_grad():
            for idx, (data, label) in enumerate(clean_testloader):
                data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                output = model(data, prun_mask)
                pred = output.argmax(dim=1,keepdim=True)
                clean_acc+=pred.eq(label.view_as(pred)).sum().item()
        
        if trigger_testloader == None:
            return {
                'clean_acc':clean_acc,
                'backdoor_asr':None
                }
        else:
            backdoor_asr = 0.
            with torch.no_grad():
                for idx, (data, label) in enumerate(trigger_testloader):
                    data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                    output = model(data, prun_mask)
                    pred = output.argmax(dim=1,keepdim=True)
                    backdoor_asr+=pred.eq(label.view_as(pred)).sum().item()
            return {
                'clean_acc':clean_acc,
                'backdoor_asr':backdoor_asr
                }















from Attacks.CL import CLattack
from Attacks.SIG import SIG_attack
from Attacks.Trojan import Trojan_attack
from Attacks.BadNets import BadNets_attack
from Attacks.Blend import Blend_attack
from Attacks.Basic import Container
from Dataset.bddataset_generator import BDDataset
from Dataset.cldataset_generator import CLDataset
from utils.picker import *
from config import config
import time

x = list()
time_list = list()
for i in range(1):
    x.append(list())
    if __name__ == '__main__':
        attack = [BadNets_attack(config),Trojan_attack(config),Blend_attack(config),CLattack(config),SIG_attack(config)]
        dataset = dataset_picker(config)
        for idx in attack:
            print('---------------------------------------Start the Fine pruning defense of {}-----------------------------------------'.format(idx._name_))
            auxiliary_finetune = finetune_train_auxiliary_picker(config)
            container = Container(config, [idx])
            bddataset = BDDataset(container, config=config)
            cldataset = CLDataset(config)
            
            cltrainset = cldataset(dataset['train'], True)
            cltestset = cldataset(dataset['test'], False)
            bdtestset = bddataset(dataset['test'], 1., False)
            if idx._name_ :
                length = len(bdtestset[idx._name_])
            trainer = Fine_pruning(config)
            trainer.setting_update(auxiliary_finetune)
            
            time1 = time.time()
            acc = trainer.prune(auxiliary_finetune['model'], cltrainset, cltestset, bdtestset['{}'.format(idx._name_)], idx._name_)
            time2 = time.time()
            time_list.append(time2-time1)
            x[-1].append(acc)

for i in range(len(x[0])):
    asr = sum([x[k][i]['backdoor_asr'][-1] for k in range(len(x))])/len(x)
    acc = sum([x[k][i]['clean_acc'][-1] for k in range(len(x))])/len(x)
    print('asr : {}  acc : {}'.format(asr/length, acc/len(cltestset)))

print('Avgtime:{}'.format(sum(time_list)/len(time_list)))