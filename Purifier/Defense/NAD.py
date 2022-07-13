import os
import sys
sys.path.append(os.path.abspath('Purifier'))
from utils.controller import doc_maker
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import copy
import torch.nn.functional as F


class NAD():
    def __init__(self, config) -> None:
        self.config = config
    
    def finetune_setting_update(self, train_auxiliary_settings:dict):
        self.opt = train_auxiliary_settings['opt']
        self.stl = train_auxiliary_settings['stl']
        self.loss = nn.CrossEntropyLoss()

    def close_setting_update(self, close_auxiliary_settings:dict):
        self.nad_opt = close_auxiliary_settings['opt']
        self.nad_stl = close_auxiliary_settings['stl']
    
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
        for name, params in model.named_parameters():
            params.requires_grad_(False)
    

    def defrozen(self, model:nn.Module):
        for name, params in model.named_parameters():
            params.requires_grad_(True)
    
    def finetune(self, teacher_model:nn.Module, trainset:Dataset, clean_testset:Dataset, trigger_testset:Dataset = None, attack_name = None):
        trainset = self.trainset_split(trainset)
        print('The size of small train set : {}'.format(len(trainset)))
        trainloader = self.make_loader(trainset, True)
        clean_testloader = self.make_loader(clean_testset, False)

        PATH = self.config['Global']['root_savepath']+'/{}/{}/'.format(self.config['Global']['dataset'], self.config['Global']['model'])
        teacher_model = self.model_loader(teacher_model, PATH, attack_name)

        loss_record = list()
        acc = {
            'clean_acc':list(),
            'backdoor_asr':list()
        }
        TRAIN_BATCH_NUM = len(trainloader)
        PATH+='Finetune.pth'

        if trigger_testset != None:  
            '''
            When you train a clean model for trojan or cl, set the trigger testset=None
            '''
            assert isinstance(trigger_testset, Dataset), 'Wrong datatype of trigger testset!'
            trigger_testloader = self.make_loader(trigger_testset, False)
        else:
            trigger_testloader = None

        self.defrozen(teacher_model)
        for epoch_idx in range(1, self.config['Finetune']['epoch_num']+1):  
            teacher_model.train()
            print('**********************Begin the {}th finetune***********************'.format(epoch_idx))
            epoch_loss = 0.
            for batch_idx, (data, label) in enumerate(trainloader):
                data, label = data.to('cuda'), label.to('cuda')
                self.opt.zero_grad()
                output = teacher_model(data)
                loss = self.loss(output, label)
                loss.backward()
                self.opt.step()
                epoch_loss+=loss.item()
            
            self.stl.step()
            epoch_loss/=TRAIN_BATCH_NUM
            loss_record.append(epoch_loss)
            teacher_model.eval()

            acc_dict = self.test(teacher_model, clean_testloader, trigger_testloader)

            print('Avg loss : {}'.format(loss_record[-1]))
            for name, item in acc_dict.items():
                acc[name].append(item)
                print('{} : {}'.format(name, item))
            
            torch.save({
                'model':teacher_model.state_dict(),
                'acc':acc,
                'loss':loss_record
            }, PATH)

        return trainset
    


    def nad_train(self, teacher_model:nn.Module, student_model:nn.Module, trainset:Dataset, clean_testset:Dataset, trigger_testset:Dataset = None, attack_name = None):
        trainloader = self.make_loader(trainset, True)
        clean_testloader = self.make_loader(clean_testset, False)

        PATH = self.config['Global']['root_savepath']+'/{}/{}/'.format(self.config['Global']['dataset'], self.config['Global']['model'])
        Teacher_PATH = PATH +'Finetune.pth'
        teacher_model.load_state_dict(torch.load(Teacher_PATH)['model'])
        student_model = self.model_loader(student_model, PATH, attack_name)
        self.frozen(teacher_model)
        self.defrozen(student_model)

        

        loss_record = list()
        acc = {
            'clean_acc':list(),
            'backdoor_asr':list()
        }
        TRAIN_BATCH_NUM = len(trainloader)
        PATH+='NAD.pth'

        if trigger_testset != None:  
            '''
            When you train a clean model for trojan or cl, set the trigger testset=None
            '''
            assert isinstance(trigger_testset, Dataset), 'Wrong datatype of trigger testset!'
            trigger_testloader = self.make_loader(trigger_testset, False)
        else:
            trigger_testloader = None

        


        def attention_map(input:torch.Tensor):
            am = torch.pow(torch.abs(input), 2)
            am = torch.sum(am, dim=1, keepdim=True)
            norm = torch.norm(am, dim=(2,3), keepdim=True)
            am = torch.div(am, norm+1e-6)
            return am

        k = [1000, 1000, 1000, 1000]
        for epoch_idx in range(1, self.config['NAD']['epoch_num']+1):  
            print('**********************Begin the {}th NAD***********************'.format(epoch_idx))
            epoch_loss = 0.
            for batch_idx, (data, label) in enumerate(trainloader):
                
                
                data, label = data.to('cuda'), label.to('cuda')
                self.nad_opt.zero_grad()
                teacher_model(data)
                output = student_model(data)
                
                nadloss = list()
                for i in range(len(teacher_model.hook)):
                    x = student_model.hook[i]
                    y = teacher_model.hook[i]
                    x_map = attention_map(x)
                    y_map = attention_map(y)
                    nadloss.append(k[3-i]*F.mse_loss(x_map, y_map))
                
                loss = self.loss(output, label) + sum(nadloss)
                loss.backward()
                self.nad_opt.step()
                epoch_loss+=loss.item()
            
            self.nad_stl.step()
            epoch_loss/=TRAIN_BATCH_NUM
            loss_record.append(epoch_loss)
            teacher_model.eval()

            acc_dict = self.test(teacher_model, clean_testloader, trigger_testloader)

            print('Avg loss : {}'.format(loss_record[-1]))
            for name, item in acc_dict.items():
                acc[name].append(item)
                print('{} : {}'.format(name, item))
            
            torch.save({
                'model':teacher_model.state_dict(),
                'acc':acc,
                'loss':loss_record
            }, PATH)
        return acc






    def test(self, model:nn.Module, clean_testloader, trigger_testloader=None):
        assert model.training == False, 'The model is training mode in testing!'
        
        clean_acc = 0.
        
        with torch.no_grad():
            for idx, (data, label) in enumerate(clean_testloader):
                data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                output = model(data)
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
                    output = model(data)
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
            print('---------------------------------------Start the NAD defense of {}-----------------------------------------'.format(idx._name_))
            auxiliary_finetune = finetune_train_auxiliary_picker(config)
            auxiliary_nad = nad_train_auxiliary_picker(config)
            container = Container(config, [idx])
            bddataset = BDDataset(container, config=config)
            
            cldataset = CLDataset(config)

            cltrainset = cldataset(dataset['train'], True)
            cltestset = cldataset(dataset['test'], False)
            bdtestset = bddataset(dataset['test'], 1., False)
            if idx._name_ :
                length = len(bdtestset[idx._name_])
            trainer = NAD(config)
            trainer.finetune_setting_update(auxiliary_finetune)
            trainer.close_setting_update(auxiliary_nad)
            time1 = time.time()
            subtrainset = trainer.finetune(auxiliary_finetune['model'], cltrainset, cltestset, bdtestset['{}'.format(idx._name_)], idx._name_)
            acc = trainer.nad_train(auxiliary_finetune['model'], auxiliary_nad['model'], subtrainset, cltestset, bdtestset['{}'.format(idx._name_)], idx._name_)
            time2 = time.time()
            time_list.append(time2-time1)
            x[-1].append(acc)

for i in range(len(x[0])):
    asr = sum([x[k][i]['backdoor_asr'][-1] for k in range(len(x))])/len(x)
    acc = sum([x[k][i]['clean_acc'][-1] for k in range(len(x))])/len(x)
    print('asr : {}  acc : {}'.format(asr/length, acc/len(cltestset)))

print('Avgtime:{}'.format(sum(time_list)/len(time_list)))