import argparse
import utils.config as cg
from CNN.resnet_3d import ResNet
from utils.train_utils import train_test
from utils.test_utils import test_model

cg.config ['num_epochs'] = 150
cg.config ['root'] = 'data/CTP/'
cg.config ['data_folder'] = 'NCCT_noskull_pro'
cg.config ['save_prefix']='results/NCCT/NCCT'
cg.config ['select_channels'] = None
cg.config ['select_depths'] = 22
cg.config ['common_shape'] = [128, 128]
cg.config ['downsize'] = None
cg.config ['select_num'] = None
cg.config ['batch_size'] = 6
cg.config ['lr'] = 0.005

cg.config ['input_channels'] = 1
cg.config ['predict_class'] = 7
cg.config ['outcome_col'] = 'mRS_3m'
cg.config ['num_classes'] = 7
cg.config ['pos_label'] = 2
cg.config ['level'] = 10

cg.config ['resume_training']= False
cg.config ['pretrained'] = True
cg.config ['train_weight'] = True
cg.config ['x_features'] = ["Sex", "Age", "HTN", "Diabetes", "HC", "Smoke",
        "AF", "HF", "MI.Angina", "Side", "TotalDT2", "TotalDT3", "TotalDT4",
        "TotalDT6", "TotalDT8", "TotalDT10", "CoreCBF30", "MismCBF30",
        "CoreCBF35", "MismCBF35", "CoreCBF40", "MismCBF40", "CoreCBF45",
        "MismCBF45", "CoreCBV50", "MismCBV50", "CoreCBV55", "MismCBV55",
        "CoreCBV60", "MismCBV60", "CoreCBV65", "MismCBV65", "CoreAbsCBV",
        "MismAbsCBV", "Total", "Core", "Core_no_zero", "Penumbra", "Mismatch",
        "DT8"]
cg.config ['y_features'] = ['mRS_3m', 'pred', 'ICH']+cg.config ['x_features']

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
