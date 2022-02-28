# 2D over each slice, followed by 1D across slices
import numpy as np
import torch
from torch import nn
from CNN.resnet_2d import ResNetEncoder, ResNetBasicBlock, ResNetBottleNeckBlock

class Conv1dAuto(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        K = self.kernel_size[0]
        self.padding =  int (np.floor (K/2))

class conv1d_block (nn.Module):
    def __init__ (self, in_channels, out_channels):
        super().__init__()
        out_3 = out_channels - 2*(out_channels//3)
        #self.conv1 = Conv1dAuto (in_channels, out_channels//3, kernel_size=3) 
        #self.conv2 = Conv1dAuto (in_channels, out_channels//3, kernel_size=5) 
        #self.conv3 = Conv1dAuto (in_channels, out_3, kernel_size=7) 
        self.conv = Conv1dAuto (in_channels, out_channels, kernel_size=5) 
        self.bn = nn.Sequential (nn.BatchNorm1d (out_channels), nn.ReLU (True))

    def forward (self, x):
        #x1 = self.conv1 (x) # [B, C, L]
        #x2 = self.conv2 (x)
        #x3 = self.conv3 (x)
        #return self.bn (torch.cat ([x1, x2, x3], axis=1))
        return self.bn (self.conv (x))

class CNN1d_decoder (nn.Module):
    def __init__(self, in_channels, n_classes, n_conv=2, n_decoder=2, step=4,
            dropout=0):
        super().__init__()
        self.avg2d = nn.AdaptiveAvgPool2d((1, 1))
        conv_list = []
        for i in range (n_conv):
            conv_list.append (conv1d_block (in_channels//(step**(i)), 
                in_channels//(step**(i+1))))
        self.conv_layers = nn.Sequential (*conv_list)
        self.avg1d = nn.AdaptiveAvgPool1d(1)

        # linear layers
        linear_list = []
        if n_decoder >=2:
            for i in range(0,n_decoder-1):
                linear_list.extend ([nn.Linear(in_channels//(step**(i+n_conv)), 
                        in_channels//(step**(i+1+n_conv))), nn.ReLU (True) ])
        div = (step**(n_conv + n_decoder-1))
        linear_list.append (nn.Linear (in_channels//div, n_classes))
        self.linear= nn.Sequential (*linear_list)

    def forward (self, x, batch_size):
        avg = self.avg2d (x) # [B*L, C, 1, 1]
        BxL, C, _, _ = avg.shape
        L = BxL//batch_size
        avg = avg.reshape(L, batch_size, C) # [L, B, C]
        cnn_out = self.conv_layers (avg.permute (1,2,0)) #B, C, L
        cnn_out = self.avg1d (cnn_out)
        B, C, _ = cnn_out.shape
        return self.linear (cnn_out.reshape(B, C))  # B, C_{out}

class CNN2_5d (nn.Module):
    def __init__(self, in_channels, n_classes, add_sigmoid=False, times_max=1,
            n_conv=2, n_decoder=2, step=4, dropout=0, deepths=[2,2,2,2],
            block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, deepths=deepths, block=block)
        N_in = self.encoder.blocks[-1].blocks[-1].expanded_channels
        self.decoder = CNN1d_decoder (N_in, n_classes, n_decoder=n_decoder,
                n_conv=n_conv, step=step, dropout=dropout)
        self.add_sigmoid= add_sigmoid
        self.times_max = times_max
        
    def forward(self, x):
        ''' `x`: [B, C, D, H, W] '''
        B, C, D, H, W = x.shape
        x = x.permute (0,2,1,3,4) # [B, D, C, H, W]
        encode_x = self.encoder(x.reshape(-1, C, H, W) )
        x = self.decoder(encode_x, B)
        if self.add_sigmoid: x = torch.sigmoid (x)*self.times_max
        return x

def resnet2_5d_n (in_channels, n_classes, model_type='resnet18', **kwargs):
    if model_type == 'resnet2.5d_18':
        return CNN2_5d (in_channels, n_classes, block=ResNetBasicBlock,
                deepths=[2, 2, 2, 2], **kwargs)
    elif model_type == 'resnet2.5d_34':
        return CNN2_5d (in_channels, n_classes, block=ResNetBasicBlock,
                deepths=[3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet2.5d_50':
        return CNN2_5d (in_channels, n_classes, block=ResNetBottleNeckBlock,
                deepths=[3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet2.5d_101':
        return CNN2_5d (in_channels, n_classes, block=ResNetBottleNeckBlock,
                deepths=[3, 4, 23, 3], **kwargs)
