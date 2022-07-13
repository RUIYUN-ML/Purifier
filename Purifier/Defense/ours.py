
import os
import sys
sys.path.append('../Purifier/')
from utils.controller import doc_maker
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import copy

class block_train():
    def __init__(self, config) -> None:
        self.config = config



    def setting_update(self, train_auxiliary_settings:dict):
        self.opt = train_auxiliary_settings['opt']
        self.stl = train_auxiliary_settings['stl']
        self.loss = nn.CrossEntropyLoss()

    
    def finetune_update(self, finetune_auxiliary_settings:dict):
        self.finetune_opt = finetune_auxiliary_settings['opt']
        self.finetune_stl = finetune_auxiliary_settings['stl']



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

        model_dict = torch.load(PATH)
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
        if 'ResNet' in self.config['Global']['model']:
            layer_name = list()
            layer_name.append('layer4.0.fc.weight')
            layer_name.append('layer4.0.fc.bias')
            layer_name.append('layer4.1.fc.weight')
            layer_name.append('layer4.1.fc.bias')
            layer_name.append('layer1.0.fc.weight')
            layer_name.append('layer1.0.fc.bias')
            layer_name.append('layer1.1.fc.weight')
            layer_name.append('layer1.1.fc.bias')
            layer_name.append('layer3.0.fc.weight')
            layer_name.append('layer3.0.fc.bias')
            layer_name.append('layer3.1.fc.weight')
            layer_name.append('layer3.1.fc.bias')
            layer_name.append('layer2.0.fc.weight')
            layer_name.append('layer2.0.fc.bias')
            layer_name.append('layer2.1.fc.weight')
            layer_name.append('layer2.1.fc.bias')
            # layer_name.append('fc.weight')
            # layer_name.append('fc.bias')

            for name, params in model.named_parameters():
                if name in layer_name:
                    params.requires_grad_(True)
                else:
                    params.requires_grad_(False)
        
        elif 'VGG' in self.config['Global']['model']:
            layer_name = list()
            layer_name.append('layer5.0.fc.weight')
            layer_name.append('layer5.0.fc.bias')
            layer_name.append('layer5.1.fc.weight')
            layer_name.append('layer5.1.fc.bias')
            layer_name.append('layer5.2.fc.weight')
            layer_name.append('layer5.2.fc.bias')
            layer_name.append('layer1.0.fc.weight')
            layer_name.append('layer1.0.fc.bias')
            layer_name.append('layer1.1.fc.weight')
            layer_name.append('layer1.1.fc.bias')
            layer_name.append('layer1.2.fc.weight')
            layer_name.append('layer1.2.fc.bias')
            layer_name.append('layer3.0.fc.weight')
            layer_name.append('layer3.0.fc.bias')
            layer_name.append('layer3.1.fc.weight')
            layer_name.append('layer3.1.fc.bias')
            layer_name.append('layer3.2.fc.weight')
            layer_name.append('layer3.2.fc.bias')

            for name, params in model.named_parameters():
                if name in layer_name:
                    params.requires_grad_(True)
                else:
                    params.requires_grad_(False)



    def defrozen(self, model:nn.Module):
        for name, params in model.named_parameters():
            params.requires_grad_(True)
        
    


    def train(self, model:nn.Module, model_copy:nn.Module,trainset:Dataset, clean_testset:Dataset, trigger_testset:Dataset = None, attack_name = None):
        
        trainset = self.trainset_split(trainset)

        print('The size of small train set : {}'.format(len(trainset)))
        
        trainloader = self.make_loader(trainset, True)
        clean_testloader = self.make_loader(clean_testset, False)
        
        if 'ResNet' in self.config['Global']['model']:
            PATH = self.config['Global']['root_savepath']+'/{}/{}/'.format(self.config['Global']['dataset'], 'ResNet18')
        elif 'VGG' in self.config['Global']['model']:
            PATH = self.config['Global']['root_savepath']+'/{}/{}/'.format(self.config['Global']['dataset'], 'VGG16')
        model = self.model_loader(model, PATH, attack_name)
        
        PATH+='CAS.pth'

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

        self.frozen(model)
        for epoch_idx in range(1, self.config['CAS']['epoch_num']+1):
            
            print('***************************************begin the {}th epoch of CAS********************************************'.format(epoch_idx))
            epoch_loss = 0.
            
                
            for batch_idx, (data, label) in enumerate(trainloader):
                self.opt.zero_grad()
                data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                output = model(data, label, eval=False)
                
                loss_normal = 1*self.loss(output['normal'], label)
                loss_ours = 1*sum([self.loss(idx, label) for idx in output['auxiliary']])
                
                loss = loss_normal+loss_ours
                loss.backward()
                epoch_loss += loss.item()
                self.opt.step()
                    
                
            epoch_loss/=TRAIN_BATCH_NUM
            loss_record.append(epoch_loss)
            self.stl.step()
            
            acc_dict = self.test(model, clean_testloader, trigger_testloader)

            print('Avg loss : {}'.format(loss_record[-1]))
            for name, item in acc_dict.items():
                acc[name].append(item)
                print('{} : {}'.format(name, item))
            
            torch.save({
                'model':model.state_dict(),
                'acc':acc,
                'loss':loss_record
            }, PATH)
        
        """
        Finetune
        """

        model_copy.load_state_dict(torch.load(PATH)['model'])
        self.defrozen(model_copy)

        for epoch_idx in range(1, self.config['Finetune']['epoch_num']+1):
            
            print('***************************************begin the {}th epoch of finetune********************************************'.format(epoch_idx))
            epoch_loss = 0.
        
                
            for batch_idx, (data, label) in enumerate(trainloader):
                self.finetune_opt.zero_grad()
                data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                output = model_copy(data, label, eval=False)
                loss_normal = 1*self.loss(output['normal'], label)
                loss_ours = 1*sum([self.loss(idx, label) for idx in output['auxiliary']])
                
                loss = loss_normal+loss_ours
                loss.backward()
                epoch_loss += loss.item()
                self.finetune_opt.step()
                    
                
            epoch_loss/=TRAIN_BATCH_NUM
            loss_record.append(epoch_loss)
            self.finetune_stl.step()
            
            acc_dict = self.test(model_copy, clean_testloader, trigger_testloader)

            print('Avg loss : {}'.format(loss_record[-1]))
            for name, item in acc_dict.items():
                acc[name].append(item)
                print('{} : {}'.format(name, item))
            
            torch.save({
                'model':model_copy.state_dict(),
                'acc':acc,
                'loss':loss_record
            }, PATH)
        return acc



    def test(self, model:nn.Module, clean_testloader, trigger_testloader=None):
        
        clean_acc = 0.
        
        with torch.no_grad():
            for idx, (data, label) in enumerate(clean_testloader):
                data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                output = model(data, eval=True)['normal']
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
                    output = model(data, eval=True)['normal']
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
for exp_idx in range(1):
    x.append(list())
    if __name__ == '__main__':
        attack = [BadNets_attack(config)]
        dataset = dataset_picker(config)
        for idx in attack:
            print('---------------------------------------Start the CAS defense of {}-----------------------------------------'.format(idx._name_))
            auxiliary = cas_train_auxiliary_picker(config)
            auxiliary_copy = finetune_train_auxiliary_picker(config)
            container = Container(config, [idx])
            bddataset = BDDataset(container, config=config)
            cldataset = CLDataset(config)
            trainer = block_train(config)
            trainer.setting_update(auxiliary)
            trainer.finetune_update(auxiliary_copy)
            cltrainset = cldataset(dataset['train'], True)
            cltestset = cldataset(dataset['test'], False)
            bdtestset = bddataset(dataset['test'], 1., False)
            if idx._name_ :
                length = len(bdtestset[idx._name_])
            time1 = time.time()
            acc = trainer.train(auxiliary['model'], auxiliary_copy['model'], cltrainset, cltestset, bdtestset['{}'.format(idx._name_)], idx._name_)
            time2 = time.time()
            time_list.append(time2-time1)
            x[-1].append(acc)
print([x[k][0]['backdoor_asr'][-1] for k in range(len(x))])
print([x[k][0]['clean_acc'][-1] for k in range(len(x))])

for i in range(len(x[0])):
    asr = sum([x[k][i]['backdoor_asr'][-1] for k in range(len(x))])/len(x)
    acc = sum([x[k][i]['clean_acc'][-1] for k in range(len(x))])/len(x)
    print('asr : {}  acc : {}'.format(asr/length, acc/len(cltestset)))

print('Avgtime:{}'.format(sum(time_list)/len(time_list)))
'''
1. Check the correctness of the requires_grad of the model!
2. Adjust the super-parameters
3. Achieve the visualization of middle layers' activations. 
'''