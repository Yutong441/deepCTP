import torch
from torchsummary import summary
from torchvision import models

def check_param (model):
    for name, param in model.named_parameters(): 
        if param.requires_grad: print ('{}, {}'.format (name, param.shape))

# ----------Resnet----------
from CNN.resnet_3d import resnet_n
from utils.load_weights import load_resnet
model = resnet_n (7,7, 'resnet50')
load_resnet (model, 'resnet50')
resnet50 = models.resnet50(pretrained=True)

# check model parameter shape
summary (model, (7,20,128,128))
summary (resnet50, (3,244,244))

# pytorch model parameter names
check_param (resnet50)
check_param (model)

# examples
model.encoder.blocks[0].blocks[0].blocks[0][0].weight.mean()
resnet_model.layer1[0].conv1.weight.mean()
model.encoder.blocks[1].blocks[0].blocks[0][0].weight.mean()
resnet_model.layer2[0].conv1.weight.mean()

# ----------VGG----------
from CNN.vgg_3d import vgg
from CNN.load_weights import load_vgg_layer
model = vgg('vgg13_bn', 7)
vgg13 = models.vgg13_bn(pretrained=True)

load_vgg_layer (model, 'vgg13_bn')
summary (model, (7,21,128,128))

check_param (model)
check_param (vgg13)

# examples
model.features[3].weight.mean()
vgg13.features[3].weight.mean()

# --------------------Unet--------------------
from CNN.unet_3d import UNet_3d
model = UNet_3d (in_channels=7, n_classes=7, n_decoder=3)
check_param (model)

from CNN.unet_2d import UNet
unet = UNet (in_channels=3)
check_param (unet)

# --------------------Load resnet18--------------------
import torch
from CNN.resnet_3d import resnet_n
model = resnet_n (7,1, 'resnet18')
model_sd = torch.load ('results/CTP_re_186/CTP_model',
        map_location=torch.device ('cpu'))
model.load_state_dict (model_sd)

# --------------------RCNN--------------------
from CNN.resRNN import resCRNN
model = resCRNN (7,1)
from CNN.load_weights import load_resnet 
load_resnet (model, 'resnet18', img3D=False)

# --------------------resnet2.5d--------------------
from CNN.resnet_2_5d import CNN2_5d
model = CNN2_5d (7,1)
from torchsummary import summary
summary (model, (7,21,128,128))

# --------------------vggCRNN--------------------
from CNN.vggRNN import vggCRNN
model = vggCRNN(7,1, model_type='vggCRNN13_bn')
from torchsummary import summary
summary (model, (7,21,128,128))
check_param (model)
