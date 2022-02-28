import re
import utils.config as cg
import CNN.load_weights as lw

def choose_models (cfg):
    if 'resnet2.5d' in cfg ['model_type']: 
        from CNN.resnet_2_5d import resnet2_5d_n
        return resnet2_5d_n (**cg.get_model_args (cfg))
    elif 'resnet' in cfg ['model_type']:
        from CNN.resnet_3d import resnet3d_n
        return resnet3d_n (**cg.get_model_args (cfg))
    elif 'unet' in cfg ['model_type']: 
        from CNN.unet_3d import UNet_3d
        return UNet_3d (**cg.get_model_args (cfg))
    elif 'resCRNN' in cfg ['model_type']: 
        from CNN.resRNN import resCRNN_n 
        return resCRNN_n (**cg.get_model_args (cfg))
    elif 'vggCRNN' in cfg ['model_type']:
        from CNN.vggRNN import vggCRNN
        return vggCRNN (**cg.get_model_args (cfg))
    elif 'vgg' in cfg ['model_type']:
        from CNN.vgg_3d import vgg
        return vgg (**cg.get_model_args (cfg))

def load_net (model, cf):
    if 'unet' in cf ['model_type']:
        lw.load_unet (model, grad=cf ['train_weight'])
    elif 'resCRNN' in cf ['model_type']:
        restype='resnet'+re.sub ('resCRNN', '', cf['model_type'])
        lw.load_resnet (model, resnet_type = restype, grad=cf ['train_weight'],
                img3D=False)
    elif 'resnet2.5d' in cf ['model_type']:
        restype=re.sub ('2\.5d_', '', cf['model_type'])
        lw.load_resnet (model, resnet_type = restype, grad=cf ['train_weight'],
                img3D=False)
    elif 'resnet' in cf ['model_type']:
        lw.load_resnet (model, resnet_type=cf ['model_type'], 
                    grad=cf ['train_weight'])
    elif 'vggCRNN' in cf ['model_type']:
        vgg_type = re.sub ('CRNN', '', cf ['model_type'])
        lw.load_vgg_layer (model, vgg_type=vgg_type, 
                grad = cf ['train_weight'], img3D=False)
    elif 'vgg' in cf ['model_type']:
        lw.load_vgg_layer (model, vgg_type=cf ['model_type'], 
                grad = cf ['train_weight'])
