# from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
from collections import OrderedDict
import torch
import torch.nn as nn
from CNN.resnet_3d import ResnetDecoder, activation_func

class UNet_3d (nn.Module):
    def __init__(self, in_channels=3, n_classes=7, init_features=32,
            add_sigmoid=False, times_max=1, coral=False, n_conv=0, n_decoder=2,
            dropout=0, step=4, extra_features=0, resblock=False, **kwargs):
        super(UNet_3d, self).__init__()

        self.add_sigmoid = add_sigmoid
        self.times_max = times_max
        out_channels = in_channels

        features = init_features
        _block = ResUnetBlock if resblock else basic_block
        self.encoder1 = _block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = _block(features*8, features*16,
                name="bottleneck")

        self.classifier = ResnetDecoder (features*16+extra_features, 
                n_classes, n_conv=n_conv, n_decoder=n_decoder, coral=coral,
                dropout=dropout, step=step)

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = _block((features * 8) * 2, features * 8,
                name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = _block((features * 4) * 2, features * 4,
                name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = _block((features * 2) * 2, features * 2,
                name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = _block(features * 2, features, name="dec1")
        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        pred_class = self.classifier (bottleneck)
        if self.add_sigmoid: 
            pred_class = torch.sigmoid (pred_class)*self.times_max

        if self.training:
            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat((dec4, enc4), dim=1)
            dec4 = self.decoder4(dec4)
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3), dim=1)
            dec3 = self.decoder3(dec3)
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2), dim=1)
            dec2 = self.decoder2(dec2)
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1), dim=1)
            dec1 = self.decoder1(dec1)
            out_img = torch.sigmoid(self.conv(dec1))
            return out_img, pred_class
        else: return pred_class

def basic_block(in_channels, features, name):
    return nn.Sequential(
        OrderedDict(
            [(name + "conv1",
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "norm1", nn.BatchNorm3d(num_features=features)),
                (name + "relu1", nn.ReLU(inplace=True)),
                (name + "conv2",
                    nn.Conv3d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "norm2", nn.BatchNorm3d(num_features=features)),
                (name + "relu2", nn.ReLU(inplace=True)),
            ]
        )
    )

class ResUnetBlock (nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', layers=3,
            name=None):
        super().__init__()
        self.inp_layer = nn.Conv3d (in_channels=in_channels,
                out_channels=out_channels, kernel_size=3, padding=1,
                bias=False)
        blocks = []
        for i in range(layers-1):
            blocks.extend ([
                nn.Conv3d (in_channels=out_channels, out_channels=out_channels,
                    kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels), activation_func(activation)])
        self.blocks = nn.Sequential (*blocks)
    
    def forward(self, x):
        first_out = self.inp_layer (x)
        return first_out + self.blocks (first_out)
