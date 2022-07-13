import sys
sys.path.append('..')
from Attacks.CL import CLattack
from Attacks.SIG import SIG_attack
from Attacks.Trojan import Trojan_attack
from Attacks.BadNets import BadNets_attack
from Attacks.Blend import Blend_attack
from Attacks.Basic import Container
from config import config
from Dataset.bddataset_generator import BDDataset
from Dataset.cldataset_generator import CLDataset
from Purifier.utils.picker import *
from Train.backdoortrainer import backdoortrain
import torch

attack = [BadNets_attack(config) ,Trojan_attack(config), Blend_attack(config), SIG_attack(config)]

dataset = dataset_picker(config)
print(dataset['test'].transform)
for idx in attack:
    print('---------------------------------------Start the training of {}-----------------------------------------'.format(idx._name_))
    auxiliary = backdoor_train_auxiliary_picker(config)

    if idx._name_ == 'CL':
        auxiliary1 = backdoor_train_auxiliary_picker(config)
        auxiliary1['model'].load_state_dict(torch.load('../Purifier/Checkpoints/{}/{}/clean.pth'.format(config['Global']['dataset'], config['Global']['model']))['model'])
        idx.adv_gen(auxiliary1['model'], dataset['train'])

    elif idx._name_ == 'Refool':
        idx.make_reflection()

    container = Container(config, [idx])
    bddataset = BDDataset(container, config=config)
    cldataset = CLDataset(config)
    trainer = backdoortrain(config)
    trainer.setting_update(auxiliary)
    bdtrainset = bddataset(dataset['train'], config['Attack']['ratio'], True)
    bdtestset = bddataset(dataset['test'], 1., False)
    cltestset = cldataset(dataset['test'], False)
    trainer.train(auxiliary['model'], bdtrainset['{}'.format(idx._name_)], cltestset, bdtestset['{}'.format(idx._name_)], idx._name_)

