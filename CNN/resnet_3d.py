# from https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from functools import partial

class Conv3dAuto(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0]//2, 
                self.kernel_size[1]//2, self.kernel_size [2]//2) 
        # dynamic add padding based on the kernel_size

class Coral (nn.Linear):  
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Coral, self).__init__(in_features, out_features, bias, device,
                dtype)
        self.weight = Parameter(torch.empty((1, in_features), **factory_kwargs))
        if bias: self.bias = Parameter(torch.empty(out_features,
            **factory_kwargs))
        else: self.register_parameter('bias', None)
        self.reset_parameters()

    def forward (self, input):
        return F.linear (input, torch.cat ([self.weight]*self.out_features, 0),
                self.bias)
        
def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = \
                in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1,
            *args, **kwargs):
        conv = partial(Conv3dAuto, kernel_size=3, bias=False)
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv3d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm3d(self.expanded_channels)) if self.should_apply_shortcut else None
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), 
            nn.BatchNorm3d(out_channels))

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )
    
class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation_func(self.activation),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, dropout=0,
            block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs,
                downsampling=downsampling),
            *[block(out_channels * block.expansion, out_channels, 
                downsampling=1, *args, **kwargs) for _ in range(n - 1)],
            nn.Dropout (dropout)
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    Args:
        `in_channels`: number of channels in the input image
        `block_sizes`: number of channels in each layer
        `deepths`: number of resnet blocks
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], 
            deepths=[2,2,2,2], activation='relu', block=ResNetBasicBlock,
            dropout=0, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        # first layer
        self.gate = nn.Sequential(
            nn.Conv3d(in_channels, self.blocks_sizes[0], kernel_size=7,
                stride=2, padding=3, bias=False),
            nn.BatchNorm3d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0],
                activation=activation, block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, dropout=dropout, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

def decoder_conv (in_features, n_conv, step=2, activation='relu'):
    if n_conv == 0: return nn.Identity ()
    else:
        layer_list = []
        for i in range (n_conv):
            layer_list.extend([
                nn.Conv3d(in_features//(step**i), 
                    in_features//(step**(i+1)), kernel_size=3,
                    stride=2, padding=3, bias=False),
                nn.BatchNorm3d(in_features//(step**(i+1))),
                activation_func(activation)])
        return nn.Sequential (*layer_list)

def decoder_linear (in_features: int, n_classes: int, n_decoder: int=1,
        step=4, coral=False, dropout=0, **kwargs):
    layer_list = []
    if n_decoder > 1:
        for i in range (n_decoder-1):
            layer_list.extend ([
                nn.Linear(in_features//(step**i), in_features//(step**(i+1))),
                nn.ReLU (True)])
            if dropout != 0: layer_list.append (nn.Dropout(dropout))

    div = (step**(n_decoder-1))
    if not coral:
        layer_list.append (nn.Linear (in_features//div, n_classes))
    else: layer_list.append (Coral (in_features//div, n_classes))
    return torch.nn.Sequential (*layer_list)

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and
    maps the output to the correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes, step=2, n_conv=0, dropout=0,
            n_pool = 0, **kwargs):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.decconv = decoder_conv (in_features, n_conv, step=step)
        #self.decoder = decoder_linear (in_features//(step**(n_conv+n_pool)), 
        #        n_classes, step=step, dropout=dropout, **kwargs)
        self.decoder = nn.Linear (in_features//(step**(n_conv+n_pool)),
            n_classes)
        self.n_pool = n_pool
        self.step = step

    def forward(self, x):
        if type (x) == torch.Tensor:
            x = self.decconv (x)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
        elif type (x) == list:
            avg_x = self.decconv (x[0])
            avg_x = self.avg (avg_x)
            x = torch.cat ([avg_x.view (avg_x.size(0),-1), x[1]], dim=1)
        if self.n_pool >0:
            for i in range (self.n_pool):
                x = F.avg_pool1d (x.unsqueeze(1), kernel_size=4, stride=
                        self.step, padding=1).squeeze ()
        x = self.decoder(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, 
            add_sigmoid: bool=False, times_max: int=1,
            depths: list=[2,2,2,2], block=ResNetBasicBlock, extra_features:
            int=0, dropout=0, **kwargs):
        '''
        Args:
            `in_channels`: number of input channels
            `n_classes`: number of output channels
            `add_sigmoid`: whether to add sigmoid activation at the end
            `times_max`: multiple the output by a constant
            `coral`: whether the output nodes all share the same weights
            `depths`: number of conv layers in each of the 4 resnet blocks
            `block`: either `ResNetBasicBlock` or `ResNetBottleNeckBlock`
            `extra_features`: whether to concat other information after average
            pooling
        '''
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, deepths=depths, block=block,
                dropout=dropout)
        deep_chan = self.encoder.blocks[-1].blocks[-2].expanded_channels + \
                extra_features
        self.decoder = ResnetDecoder(deep_chan, n_classes, dropout=dropout,
                **kwargs)
        self.add_sigmoid = add_sigmoid
        self.times_max = times_max
        
    def forward(self, x):
        '''
        Args: `x`: either a pytorch tensor or a list of 2 tensors
        '''
        if type (x) == torch.Tensor:
            x = self.encoder(x)
            x = self.decoder(x)
        elif type (x) == list:
            encoder_x = self.encoder (x[0])
            x = self.decoder ([encoder_x, x[1]])
        if self.add_sigmoid: x = torch.sigmoid (x)*self.times_max
        return x

def resnet3d_n (in_channels, n_classes, model_type='resnet18', **kwargs):
    if model_type == 'resnet18':
        return ResNet(in_channels, n_classes, block=ResNetBasicBlock,
                depths=[2, 2, 2, 2], **kwargs)
    elif model_type == 'resnet34':
        return ResNet(in_channels, n_classes, block=ResNetBasicBlock,
                depths=[3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet50':
        return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock,
                depths=[3, 4, 6, 3], **kwargs)
    elif model_type == 'resnet101':
        return ResNet(in_channels, n_classes, block=ResNetBottleNeckBlock,
                depths=[3, 4, 23, 3], **kwargs)
