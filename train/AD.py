# detect Alzheimer and dementia
import argparse
import utils.config as cg
from CNN.resnet_3d import ResNet
from utils.train_utils import train_test
from utils.test_utils import test_model

cg.config ['root'] = 'data/OASIS3/'
cg.config ['save_prefix']='results/AD/AD'
cg.config ['select_channels'] = None
cg.config ['select_depths'] = 20
cg.config ['downsize'] = [128, 128]
cg.config ['select_num'] = 10
cg.config ['batch_size'] = 8
cg.config ['lr'] = 0.001

cg.config ['input_channels'] = 1
cg.config ['predict_class'] = 2
cg.config ['outcome_col'] = 'ylabel'
cg.config ['num_classes'] = 2
cg.config ['pos_label'] = 1
cg.config ['level'] = 10

cg.config ['resume_training']= False
cg.config ['pretrained'] = False
cg.config ['train_weight'] = True
cg.config ['x_features'] = ['M.F', 'i.Age', 'mmse', 'cdr', 'commun',
        'homehobb','memory', 'judgment', 'orient', 'apoe', 'perscare',
        'sumbox','height', 'weight']
cg.config ['y_features'] = ['ylabel', 'pred']+ cg.config ['x_features']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='train, test')
    args = parser.parse_args()
    model = ResNet (cg.config ['input_channels'], cg.config ['predict_class'])
    if args.mode == 'train': train_test (cg, model)
    else:
        import data_raw.OASIS3 as oasis
        show_maps = oasis.show_img
        test_model (cg, model, show_maps)
