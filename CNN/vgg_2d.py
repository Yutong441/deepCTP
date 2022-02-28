# from pytorch source code: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg19_bn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast

def make_layers(in_channels, cfg: List[Union[str, int]], batch_norm: bool =
        False):
    layers: List[nn.Module] = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'M0':
            layers += [nn.Identity()]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M0', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M0', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M0', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M0', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
