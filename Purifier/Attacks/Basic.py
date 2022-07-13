import os
import sys
sys.path.append(os.path.abspath('Purifier'))

from Visualtools.backdoor_ploter import backdoor_samples
import PIL.Image

class basic_attacker():
    def __init__(self, config:dict) -> None:
        self.config = config
    
    def make_trigger(self):
        raise NotImplementedError('You haven\'t achieve {} method!'.format(self._name_))
    

class Container():
    def __init__(self, config:dict, attack:list) -> None:
        self.config = config
        self.attack = attack

    def __call__(self, sample:PIL.Image):
        samples = dict()
        samples['Clean'] = sample

        for attack in self.attack:
            samples['{}'.format(attack._name_)] = attack.make_trigger(sample)

        '''
        {
            'Clean':...,
            'Attack 1':...,
            ...
        }
        '''
        return samples
    
    def __len__(self):
        return len(self.attack)

    def sample_ploter(self, samples:dict):
        shower = backdoor_samples(self.config)
        shower.show(samples)
