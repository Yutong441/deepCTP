import os
import shutil
import glob
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotnine as pn
from sklearn.decomposition import PCA
from utils.train_utils import check_accuracy

def get_activation (self, input, output, save_dir):
    '''
    This is the function to put on the forward hook of the model.
    It will save the activations of a given layer to the `save_dir` directory
    as a compressed npz file.
    '''
    num = len (os.listdir (save_dir))
    save_file = save_dir + 'act_batch_{}.npz'.format (num)
    if type(output) == tuple: # if it is RNN
        output = output [0][-1]
    np.savez_compressed (save_file, output = output.detach().cpu().squeeze())

def add_dim (xx, dims=2):
    if len (xx.shape) < dims: return xx [np.newaxis]
    else: return xx

def load_np (arr_path):
    arr = np.load (arr_path)
    arr_name = list (arr)[0]
    return arr [arr_name]

def deep_feature (model, loader, cfg, device, mode):
    '''
    Extract the output of the second to last layer while evaluating model
    accuracy
    '''
    save_dir = cfg['save_prefix']+'_act/'
    if os.path.exists (save_dir): shutil.rmtree(save_dir)
    os.mkdir (save_dir)
    get_act = partial (get_activation, save_dir=save_dir)

    model.eval ()
    if hasattr (model, 'predictor'):
        handle = model.predictor.decoder.avg.register_forward_hook(get_act)
    elif 'CRNN' in cfg ['model_type']:
        handle = model.decoder.rnn.register_forward_hook(get_act)
    elif cfg ['model_type'] == 'unet' or 'vgg' in cfg ['model_type']:
        try: handle = model.classifier[-4].register_forward_hook (get_act)
        except: handle = model.avg.register_forward_hook (get_act)
    else: handle = model.decoder.avg.register_forward_hook(get_act)

    acc = check_accuracy(loader, model, cfg, device)
    handle.remove ()
    N = len (os.listdir (save_dir))
    all_acts = np.concatenate ([add_dim (load_np (
        save_dir+'/act_batch_{}.npz'.format (i))) for i in
        range(N)], axis=0)
    shutil.rmtree(save_dir)
    np.savez_compressed (cfg['save_prefix']+'_act_{}.npz'.format (mode), all_acts)
    return acc

def dim_red (activations, mode, cfg, non_imaging=False, ncols=2):
    '''
    Perform dimensionality reduction on imaging and/or non-imaging features
    Args:
        `activations`: imaging features extracted from CNN
        `mode`: 'train' or 'test'
        `cfg`: config file
        `non_imaging`: whether to include non-imaging features in dim red
    '''
    labs = pd.read_csv (cfg['root']+'/'+cfg['label_dir']+'/{}.csv'.format (
        mode), index_col=[0])
    ypred = np.load (cfg ['save_prefix']+'_tmp/ypred_{}.npz'.format (
        mode)) ['arr_0']
    labs ['pred'] = ypred
    labs = labs [cfg['y_features']]
    xlabs = labs [cfg['x_features']]
    if non_imaging:
        activations = np.concatenate ([activations, xlabs.to_numpy()], axis=1)
        nimg_feature = 'nimg'
    else: nimg_feature= 'img'

    pca = PCA (n_components=2)
    activations = activations [:, np.nanstd (activations, 0)!=0]
    activations = (activations - np.nanmean(activations, 0, keepdims=True
        ))/np.nanstd (activations, 0, keepdims=True)
    contain_na = np.sum (np.isnan (activations))
    if contain_na ==0:
        act_red = pca.fit_transform (activations)
        labs ['PC1'] = act_red [:,0]
        labs ['PC2'] = act_red [:,1]

        for i, fea in enumerate (cfg['y_features']):
            p = (pn.ggplot (labs, pn.aes (x='PC1', y='PC2', color=fea))+
                    pn.geom_point (size=5)+pn.theme_bw ())
            p.save (cfg ['save_prefix']+'_tmp/dimred_{}_{}.png'.format
                    (nimg_feature, fea))

        combine_dimred (cfg, nimg_feature, ncols=ncols)

def combine_dimred (cfg, nimg_feature, ncols=2):
    img_names= glob.glob (cfg['save_prefix']+'_tmp/dimred_{}_*.png'.format (nimg_feature))
    N = len (img_names)
    nrows = N//ncols + N%ncols
    fig, ax = plt.subplots (nrows, ncols, figsize=(ncols*4.8, nrows*4.8))
    for i, one_name in enumerate(sorted (img_names)):
        img = plt.imread (one_name)
        ax[i//ncols, i%ncols].imshow (img)
        #os.remove (one_name)
    [axi.set_axis_off () for axi in ax.ravel ()]
    plt.savefig (cfg['save_prefix']+'_tmp/dimred_{}.png'.format (nimg_feature),
            dpi=400, bbox_inches='tight')
    plt.close ()
