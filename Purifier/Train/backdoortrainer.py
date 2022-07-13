import os
import sys
sys.path.append(os.path.abspath('Purifier'))
from utils.controller import doc_maker
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class backdoortrain():
    '''
    The method is used to train the clean or backdoor models
    '''
    def __init__(self, config:dict) -> None:
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




    def train(self, model:nn.Module, trainset:Dataset, clean_testset:Dataset, trigger_testset:Dataset = None, attack_name = None):
        trainloader = self.make_loader(trainset, True)
        clean_testloader = self.make_loader(clean_testset, False)
        PATH = self.config['Global']['root_savepath']+'/{}/{}/'.format(self.config['Global']['dataset'], self.config['Global']['model'])
        doc_maker(PATH)

        if attack_name == None:
            PATH += 'clean.pth'
        else:
            PATH += '{}.pth'.format(attack_name)

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

            
        for epoch_idx in range(1, self.config['Global']['epoch_num']):
            
            print('***************************************begin the {}th epoch of train********************************************'.format(epoch_idx))
            epoch_loss = 0.
        
            model.train()
            
            
                
            for batch_idx, (data, label) in enumerate(trainloader):
                self.opt.zero_grad()
                data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                loss = self.loss(model(data), label)
                loss.backward()
                epoch_loss += loss.item()
                self.opt.step()
                    
                
            epoch_loss/=TRAIN_BATCH_NUM
            loss_record.append(epoch_loss)
            self.stl.step()
            
            model.eval()
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
            



    def test(self, model:nn.Module, clean_testloader, trigger_testloader=None):
        CLEAN_TEST_BATCH_NUM = len(clean_testloader)
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
            TRIGGER_TEST_BATCH_NUM = len(trigger_testloader)
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

