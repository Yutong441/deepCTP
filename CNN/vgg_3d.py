# from pytorch source code: https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html#vgg19_bn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast

class VGG(nn.Module):
    def __init__(self, features: nn.Module, n_classes= 1000,
            init_weights = True, add_sigmoid=False, times_max=1, **kwargs):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, n_classes),
        )
        if init_weights:
            self._initialize_weights()
        self.add_sigmoid = add_sigmoid
        self.times_max = times_max

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.add_sigmoid: x = F.sigmoid (x)*self.times_max
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(in_channels, cfg: List[Union[str, int]], batch_norm: bool =
        False):
    layers: List[nn.Module] = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        elif v == 'M0':
            layers += [nn.Identity()]
        else:
            v = cast(int, v)
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M0', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M0', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M0', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M0', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_type, in_channels, **kwargs) -> VGG:
    kwargs['init_weights'] = False
    model_type = model_type.split ('_')
    bn= True if len (model_type) == 2 else False
    vgg_layers = make_layers(in_channels, cfgs[model_type[0]], batch_norm= bn)
    model = VGG(vgg_layers, **kwargs)
    return model
