import os
import sys
sys.path.append(os.path.abspath('Purifier'))
from Attacks.Basic import basic_attacker
import PIL.Image
import copy


class BadNets_attack(basic_attacker):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._name_ = 'BadNets'
    
    def make_trigger(self, sample:PIL.Image)->PIL.Image:
            
        data = copy.deepcopy(sample)
        width, height = data.width, data.height
        value_255 = tuple([255]*self.config['Global']['in_channels'])
        value_0 = tuple([0]*self.config['Global']['in_channels'])
        # right bottom
        data.putpixel((width-1,height-1),value_255)
        data.putpixel((width-1,height-2),value_0)
        data.putpixel((width-1,height-3),value_255)
        
        data.putpixel((width-2,height-1),value_0)
        data.putpixel((width-2,height-2),value_255)
        data.putpixel((width-2,height-3),value_0)
        
        data.putpixel((width-3,height-1),value_255)
        data.putpixel((width-3,height-2),value_0)
        data.putpixel((width-3,height-3),value_0)

        # left top
        data.putpixel((1,1),value_255)
        data.putpixel((1,2),value_0)
        data.putpixel((1,3),value_255)
        
        data.putpixel((2,1),value_0)
        data.putpixel((2,2),value_255)
        data.putpixel((2,3),value_0)

        data.putpixel((3,1),value_255)
        data.putpixel((3,2),value_0)
        data.putpixel((3,3),value_0)

        # right top
        data.putpixel((width-1,1),value_255)
        data.putpixel((width-1,2),value_0)
        data.putpixel((width-1,3),value_255)

        data.putpixel((width-2,1),value_0)
        data.putpixel((width-2,2),value_255)
        data.putpixel((width-2,3),value_0)

        data.putpixel((width-3,1),value_255)
        data.putpixel((width-3,2),value_0)
        data.putpixel((width-3,3),value_0)

        # left bottom
        data.putpixel((1,height-1),value_255)
        data.putpixel((2,height-1),value_0)
        data.putpixel((3,height-1),value_255)

        data.putpixel((1,height-2),value_0)
        data.putpixel((2,height-2),value_255)
        data.putpixel((3,height-2),value_0)

        data.putpixel((1,height-3),value_255)
        data.putpixel((2,height-3),value_0)
        data.putpixel((3,height-3),value_0)
        
        return data
        