import torch
from torch import nn
from CNN.resnet_2d import ResNetEncoder, ResNetBasicBlock, ResNetBottleNeckBlock

class RNN2layer (nn.Module):
    def __init__ (self, in_channels, out_channels, hidden_size=None):
        super ().__init__ ()
        if hidden_size is None: hidden_size = in_channels//2
        self.gru1 = nn.GRU(in_channels, hidden_size)
        self.gru2 = nn.GRU(in_channels+hidden_size, out_channels)

    def forward (self, x):
        output1, h_n1 = self.gru1(x)
        output1_residual = torch.cat((x, output1),dim=2)
        return self.gru2(output1_residual)

class RNN_decoder (nn.Module):
    def __init__(self, in_channels, n_classes, n_decoder=2, step=4, dropout=0):
        super().__init__()
        assert n_decoder >=2, 'number of decoder layers should be at least 2'
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.rnn = nn.GRU(in_channels, in_channels//step, bias=True,
                batch_first=False, dropout=dropout, num_layers=2)
        linear_list = []
        if n_decoder >=3:
            for i in range(1,n_decoder-1):
                linear_list.extend ([nn.Linear(in_channels//(step**i), 
                        in_channels//(step**(i+1))), nn.ReLU (True) ])
        div = step**(n_decoder-1)
        linear_list.append (nn.Linear (in_channels//div, n_classes))
        self.linear= nn.Sequential (*linear_list)

    def forward (self, x, batch_size):
        avg = self.avg (x) # [L*B, C, 1, 1]
        LxB, C, _, _ = avg.shape
        L = LxB//batch_size
        rnn_out = self.rnn (avg.reshape(L, batch_size, C)) #L, B, C
        return self.linear (rnn_out[0] [-1])  # B, C_{out}

class resCRNN (nn.Module):
    def __init__(self, in_channels, n_classes, add_sigmoid=False, times_max=1,
            n_decoder=2, step=4, dropout=0, deepths=[2,2,2,2],
            block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, deepths=deepths, block=block)
        N_in = self.encoder.blocks[-1].blocks[-1].expanded_channels
        self.decoder = RNN_decoder (N_in, n_classes, n_decoder=n_decoder,
                step=step, dropout=dropout)
        self.add_sigmoid= add_sigmoid
        self.times_max = times_max
        
    def forward(self, x):
        ''' `x`: [B, C, D, H, W] '''
        B, C, D, H, W = x.shape
        x = x.permute (2,0,1,3,4) # [D, B, C, H, W]
        encode_x = self.encoder(x.reshape(-1, C, H, W) )
        x = self.decoder(encode_x, B)
        if self.add_sigmoid: x = torch.sigmoid (x)*self.times_max
        return x

def resCRNN_n (in_channels, n_classes, model_type='resnet18', **kwargs):
    if model_type == 'resCRNN18':
        return resCRNN (in_channels, n_classes, block=ResNetBasicBlock,
                deepths=[2, 2, 2, 2], **kwargs)
    elif model_type == 'resCRNN34':
        return resCRNN (in_channels, n_classes, block=ResNetBasicBlock,
                deepths=[3, 4, 6, 3], **kwargs)
    elif model_type == 'resCRNN50':
        return resCRNN (in_channels, n_classes, block=ResNetBottleNeckBlock,
                deepths=[3, 4, 6, 3], **kwargs)
    elif model_type == 'resCRNN101':
        return resCRNN (in_channels, n_classes, block=ResNetBottleNeckBlock,
                deepths=[3, 4, 23, 3], **kwargs)
