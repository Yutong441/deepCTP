import re
import torch
from torch import nn
from CNN.resRNN import RNN_decoder
import CNN.vgg_2d as vgg2

class vggCRNN (nn.Module):
    def __init__(self, in_channels, n_classes, add_sigmoid=False, times_max=1,
            n_decoder=2, step=4, dropout=0, model_type='vgg19', *args, **kwargs):
        super().__init__()
        vgg_type = re.sub ('CRNN', '', model_type)
        vgg_class = re.sub ('_bn', '', vgg_type)
        bn = True if '_bn' in vgg_type else False
        self.features = vgg2.make_layers (in_channels, vgg2.cfgs [vgg_class],
                batch_norm=bn)
        self.decoder = RNN_decoder (512, n_classes, n_decoder=n_decoder,
                step=step, dropout=dropout)
        self.add_sigmoid= add_sigmoid
        self.times_max = times_max
        
    def forward(self, x):
        ''' `x`: [B, C, D, H, W] '''
        B, C, D, H, W = x.shape
        x = x.permute (2,0,1,3,4) # [D, B, C, H, W]
        encode_x = self.features(x.reshape(-1, C, H, W) )
        x = self.decoder(encode_x, B)
        if self.add_sigmoid: x = torch.sigmoid (x)*self.times_max
        return x
