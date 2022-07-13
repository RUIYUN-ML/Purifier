import sys
sys.path.append('../Purifier/')
from config import config
from utils.picker import *
import torch
from torch.utils.data import Dataset, DataLoader, Subset



class ABL():
    def __init__(self, config:dict) -> None:
        self.config = config



    def setting_update(self, isolate_auxiliary_settings:dict):
        self.opt = isolate_auxiliary_settings['opt']
        self.stl = isolate_auxiliary_settings['stl']
        self.loss = nn.CrossEntropyLoss()

    def finetune_update(self, finetune_auxiliary_settings:dict):
        self.finetune_opt = finetune_auxiliary_settings['opt']
        self.finetune_stl = finetune_auxiliary_settings['stl']
    
    def unlearn_update(self, unlearn_auxiliary_settings:dict):
        self.unlearn_opt = unlearn_auxiliary_settings['opt']
        self.unlearn_stl = unlearn_auxiliary_settings['stl']


    def make_loader(self, dataset:Dataset, train:bool):
        if train == True:
            return DataLoader(dataset, batch_size=self.config['Global']['train_batch_size'], shuffle=True)
        else:
            return DataLoader(dataset, batch_size=self.config['Global']['test_batch_size'])



    def data_isolate(self, model:nn.Module, trainset:Dataset, clean_testset:Dataset, trigger_testset:Dataset = None, attack_name = None):
        trainloader = self.make_loader(trainset, True)
        clean_testloader = self.make_loader(clean_testset, False)
        PATH = self.config['Global']['root_savepath']+'/{}/{}/'.format(self.config['Global']['dataset'], self.config['Global']['model'])
        PATH+='Isolate_{}.pth'.format(attack_name)

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

        for epoch_idx in range(1, self.config['ISO']['epoch_num']):
            
            print('***************************************begin the {}th epoch of isolate********************************************'.format(epoch_idx))
            epoch_loss = 0.
        
            model.train()
            
                
            for batch_idx, (data, label) in enumerate(trainloader):
                self.opt.zero_grad()
                data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                output = model(data)
                
                loss = self.loss(output, label)
                loss = torch.sign(loss-0.5)*loss
                
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
    
    def data_split(self, model:nn.Module, trainset:Dataset):
        model.eval()

        loss_record = list()
        trainloader = DataLoader(trainset, batch_size=1)
        for idx, (data, label) in enumerate(trainloader):
            data, label = data.to('cuda'), label.to('cuda')
            output = model(data)
            loss = self.loss(output, label)
            loss_record.append(loss.item())
        
        loss_idx = torch.argsort(torch.tensor(loss_record))
        ratio = 0.01
        perm = loss_idx[0:int(len(loss_idx)*ratio)]

        poisoned_dataset = Subset(trainset, perm)
        
        fullset = set(range(len(trainset)))
        permset = set(perm)
        cleanset = fullset-permset
        clean_dataset = Subset(trainset, list(cleanset))

        return {
            'clean':clean_dataset,
            'poisoned':poisoned_dataset
        }
    


    def finetune(self, model:nn.Module, trainset:Dataset, clean_testset:Dataset, trigger_testset:Dataset = None, attack_name = None):
        
        trainloader = self.make_loader(trainset, True)
        clean_testloader = self.make_loader(clean_testset, False)
        PATH = self.config['Global']['root_savepath']+'/{}/{}/'.format(self.config['Global']['dataset'], self.config['Global']['model'])
        model.load_state_dict(torch.load(PATH+'Isolate_{}.pth'.format(attack_name))['model'])
        PATH+='ABL_finetune_{}.pth'.format(attack_name)

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

        for epoch_idx in range(1, self.config['Finetune']['epoch_num']):
            
            print('***************************************begin the {}th epoch of ABL finetune********************************************'.format(epoch_idx))
            epoch_loss = 0.
        
            model.train()
            
                
            for batch_idx, (data, label) in enumerate(trainloader):
                self.finetune_opt.zero_grad()
                data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                output = model(data)
                
                loss = self.loss(output, label)
                
                loss.backward()
                epoch_loss += loss.item()
                self.finetune_opt.step()
            
            epoch_loss/=TRAIN_BATCH_NUM
            loss_record.append(epoch_loss)
            self.finetune_stl.step()
            
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
        

    def unlearning(self, model:nn.Module, trainset:Dataset, clean_testset:Dataset, trigger_testset:Dataset = None, attack_name = None):
        trainloader = self.make_loader(trainset, True)
        clean_testloader = self.make_loader(clean_testset, False)
        PATH = self.config['Global']['root_savepath']+'/{}/{}/'.format(self.config['Global']['dataset'], self.config['Global']['model'])
        model.load_state_dict(torch.load(PATH+'ABL_finetune_{}.pth'.format(attack_name))['model'])
        PATH+='ABL_unlearning_{}.pth'.format(attack_name)

        target_num = 0.
        for (data, label) in trainset:
            if label == 5:
                target_num+=1
        print('The target domain {}'.format(target_num/len(trainset)))

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

        for epoch_idx in range(1, self.config['UN']['epoch_num']):
            
            print('***************************************begin the {}th epoch of ABL finetune********************************************'.format(epoch_idx))
            epoch_loss = 0.
        
            model.train()
            
                
            for batch_idx, (data, label) in enumerate(trainloader):
                self.unlearn_opt.zero_grad()
                data, label = data.to(self.config['Global']['device']), label.to(self.config['Global']['device'])
                output = model(data)
                
                loss = self.loss(output, label)
                
                (-1.*loss).backward()
                epoch_loss += loss.item()
                self.unlearn_opt.step()
            
            epoch_loss/=TRAIN_BATCH_NUM
            loss_record.append(epoch_loss)
            self.unlearn_stl.step()
            
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


x = list()
for i in range(1):
    x.append(list())
    if __name__ == '__main__':
        attack = [BadNets_attack(config),Trojan_attack(config), Blend_attack(config), CLattack(config),SIG_attack(config)]
        dataset = dataset_picker(config)
        for idx in attack:
            print('---------------------------------------Start the ABL defense of {}-----------------------------------------'.format(idx._name_))
            auxiliary_isolate = isolate_train_auxiliary_picker(config)
            auxiliary_finetune = finetune_train_auxiliary_picker(config)
            auxiliary_unlearn = unlearn_train_auxiliary_picker(config)
            container = Container(config, [idx])
            bddataset = BDDataset(container, config=config)
            cldataset = CLDataset(config)

            cltrainset = cldataset(dataset['train'], True)
            cltestset = cldataset(dataset['test'], False)
            bdtrainset = bddataset(dataset['train'], config['Attack']['ratio'], True)
            bdtestset = bddataset(dataset['test'], 1., False)

            if idx._name_ :
                length = len(bdtestset[idx._name_])

            trainer = ABL(config)
            trainer.setting_update(auxiliary_isolate)
            trainer.finetune_update(auxiliary_finetune)
            trainer.unlearn_update(auxiliary_unlearn)

            trainer.data_isolate(auxiliary_isolate['model'], bdtrainset['{}'.format(idx._name_)], cltestset, bdtestset['{}'.format(idx._name_)], idx._name_)
            splitset = trainer.data_split(auxiliary_isolate['model'], bdtrainset['{}'.format(idx._name_)])
            trainer.finetune(auxiliary_finetune['model'], splitset['clean'], cltestset, bdtestset['{}'.format(idx._name_)], idx._name_)
            acc = trainer.unlearning(auxiliary_unlearn['model'], splitset['poisoned'], cltestset, bdtestset['{}'.format(idx._name_)], idx._name_)
            x[-1].append(acc)

for i in range(len(x[0])):
    asr = sum([x[k][i]['backdoor_asr'][-1] for k in range(len(x))])/len(x)
    acc = sum([x[k][i]['clean_acc'][-1] for k in range(len(x))])/len(x)
    print('asr : {}  acc : {}'.format(asr/length, acc/len(cltestset)))