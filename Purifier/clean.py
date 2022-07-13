from Attacks.CL import CLattack
from Attacks.SIG import SIG_attack
from Attacks.Trojan import Trojan_attack
from Attacks.BadNets import BadNets_attack
from Attacks.Blend import Blend_attack
from Attacks.Basic import Container
from config import config
from Dataset.bddataset_generator import BDDataset
from Dataset.cldataset_generator import CLDataset
from utils.picker import *
from Train.backdoortrainer import backdoortrain
import torch



dataset = dataset_picker(config)

auxiliary = backdoor_train_auxiliary_picker(config)

cldataset = CLDataset(config)
trainer = backdoortrain(config)
trainer.setting_update(auxiliary)
cltestset = cldataset(dataset['test'], False)
cltrainset = cldataset(dataset['train'], True)
trainer.train(auxiliary['model'], cltrainset, cltestset)

