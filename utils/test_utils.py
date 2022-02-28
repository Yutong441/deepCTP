import os
import glob
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.dataloader import dataloader
from utils.config import print_config
from utils.train_utils import check_accuracy
import utils.img_disp as idisp
from utils.deep_features import deep_feature, dim_red
from utils.metric import model_accuracy
from utils.gradcam import save_gradcam
from CNN.all_models import choose_models
from CNN.cum_link import OrdinalLogisticModel
from data_raw.npz2csv import npz2csv

def test_model (cg, show_maps):
    '''
    Args:
        `cg`: the utils.config file
        `show_maps`: a function to display the images
    '''
    model = choose_models (cg.config)
    if cg.config['loss_type'] == 'cum_link': 
        model = OrdinalLogisticModel (model, cg.config)

    if not torch.cuda.is_available (): device = 'cpu'
    else: 
        device = cg.config['device']
        model = model.to(device)

    loader_arg, data_arg = cg.get_data_args(cg.config)
    eval_set = dataloader (cg.config['root'], "train", transformation=None,
            **loader_arg)
    test_set = dataloader (cg.config['root'], "test", **loader_arg)
    eval_loader = DataLoader(dataset=eval_set, shuffle=False, **data_arg)
    test_loader = DataLoader(dataset=test_set, shuffle=False, **data_arg)
    model.load_state_dict(torch.load(cg.config['save_prefix']+'_model', 
        map_location=torch.device (device)))

    #acc_metric = []
    #model.eval()
    #fig_dir = cg.config['save_prefix']+'_tmp/'
    #if not os.path.exists (fig_dir): os.mkdir (fig_dir)
    #for mode, loader in zip (['train', 'test'], [eval_loader, test_loader]):
    #    acc_metric.append (deep_feature (model, loader, cg.config, device,
    #        mode))
    #    ytrue, ypred= check_accuracy(loader, model, cg.config, device,
    #            final_val=False)
    #    np.savez_compressed (cg.config['save_prefix']+'_tmp/ypred_{}.npz'.format (
    #        mode), ypred)

    #N = eval_set.__len__()
    #save_metric (acc_metric, fig_dir, [N, test_set.__len__()])
    #idisp.loss_history (cg.config ['save_prefix'], N, cg.config['batch_size'])
    #idisp.metric_history (cg.config ['save_prefix'], cg.config['eval_every'])

    #idisp.get_images (test_loader, test_set, model, cg.config, device, show_maps)
    #idisp.get_images (test_loader, test_set, model, cg.config, device,
    #        show_maps, accuracy='most')
    #activations = np.load (cg.config['save_prefix']+'_act_train.npz')['arr_0']
    #dim_red (activations, 'train', cg.config, non_imaging=True)
    #dim_red (activations, 'train', cg.config, non_imaging=False)
    #save_html (cg.config)
    #npz2csv (os.path.dirname (cg.config ['save_prefix']))
    #model_accuracy (os.path.dirname (cg.config ['save_prefix']), 
    #        cg.config ['root']+'/'+cg.config ['label_dir'])
    if cg.config ['save_gradcam']: save_gradcam (model, test_set, cg.config)

def save_metric (acc_metric, fig_dir, sample_sizes):
    '''
    Save accuracy metric into html and csv. 
    Args:
        `acc_metric`: a list of 2 elements: one for train set, the other for
        the test set. Each element contains a dataframe of one row. The columns
        correspond to each metric tested.
        `fig_dir`: where to save the html
        `sample_sizes`: a list of 2 integers: one for train set, the other for
        the test set
    '''
    acc_metric = pd.concat (acc_metric, axis=0)
    acc_metric.index = ['train (n={})'.format (sample_sizes[0]), 
            'test (n={})'.format (sample_sizes[1])]

    if not os.path.exists (fig_dir): os.mkdir (fig_dir)
    acc_metric.to_html (fig_dir+'/acc_metric.html', col_space='150px',
            justify='left')
    acc_metric.to_csv (fig_dir+'/acc_metric.csv')

def bold_text (txt):
    return '<font size="+2"><b>'+re.sub ('#', '', txt)+'</b></font>'

def save_html (cfg):
    '''
    Create an html file to summarise all the results. The file will be saved in
    the directory specified by the `save_prefix` attribute in config ending in
    '_results.html'.
    '''
    with open(cfg['save_prefix']+'_results.html', 'w') as f:
        #------------------------------ 
        f.write ('<html>\n <body> \n <h1>Parameters</h1> \n')
        params = print_config (cfg, join_str= False)
        param = [bold_text (par) if '#' in par else par for par in params]
        f.write('\n <br>'.join(param))
        with open(cfg['save_prefix']+'_tmp/acc_metric.html') as am:
            metrics = am.readlines ()
        f.write ('\n')
        f.write ('<h1>Metrics</h1> \n')
        f.writelines (metrics)

        #------------------------------ 
        f.write ('<h1>Training history</h1> \n')
        fig_dir = os.path.basename (cfg['save_prefix'])
        f.write ('<img src="{}_tmp/loss.png" width="500"> \n'.format (
            fig_dir))
        f.write ('<br> \n')
        f.write ('<img src="{}_tmp/metric.png" width="800"> \n'.format (
            fig_dir))

        #------------------------------ 
        for lab in ['bad', 'good']:
            f.write ('<h1>{} images</h1> \n'.format (lab))
            figs = glob.glob (cfg['save_prefix']+'_tmp/{}_img_*.png'.format
                    (lab))
            txt_dir = cfg['save_prefix']+'_tmp/{}_img_log.txt'.format(lab)
            with open (txt_dir) as bi: img_acc = bi.readlines ()
            for i, fig in enumerate (figs):
                f.write ('<h3>'+img_acc[i]+'</h3> \n')
                img_path = '/'.join (fig.split('/')[-2:])
                act_path = re.sub ('img', 'act', img_path)
                f.write ('<img src="{}" width="130"> \n'.format (act_path))
                f.write ('<br>')
                f.write ('<img src="{}" width="600"> \n'.format (img_path))

        #------------------------------ 
        f.write ('<h1>Dimensionality reduction</h1>')
        for lab in ['img', 'nimg']:
            title = 'Imaging features only' if lab=='img' else \
                    'Non-imaging features included'
            f.write ('<h2>{}</h2> \n'.format (title))
            dr_path = os.path.dirname (img_path) + '/dimred_{}.png'.format (lab)
            f.write ('<img src="{}" width="800"> \n'.format (dr_path))
            f.write ('<br>')

        f.write ('<body>\n <html> \n')
