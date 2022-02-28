import argparse
import utils.config as cg
from CNN.resnet_3d import ResNet
from utils.train_utils import train_test
from utils.test_utils import test_model

cg.config ['save_prefix']='results/DWI_pt_fw/DWI'
cg.config ['resume_training']= False
cg.config ['input_channels'] = 6
cg.config ['predict_class'] = 6
cg.config ['pretrained'] = True
cg.config ['train_weight'] = False

cg.config ['x_features'] = ['TICIScaleGrade', 'timeSinceStroke',
        'timeToTreatment']
cg.config ['y_features'] = ['mRS', 'pred', 'TICIScaleGrade',
        'timeSinceStroke', 'timeToTreatment']
model = ResNet (cg.config ['input_channels'], cg.config ['predict_class'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='train, test')
    args = parser.parse_args()
    if args.mode == 'train': train_test (cg, model)
    else:
        import data_raw.ISLES2016 as isles
        show_maps = isles.show_all_maps
        test_model (cg, model, show_maps)
