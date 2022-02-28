"""
process ISLES 2016 data
The downloaded dataset consists of one folder per patient
Each folder contains the image for each map (ADC or PWI)
"""
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import os
import glob
import re

def all_maps_per_case (folder):
    all_maps = []
    all_files = os.listdir (folder)
    for one_file in all_files:
        if '4DPWI' not in one_file: #ignore raw 4D data
            subfolder = folder+'/'+one_file
            filename = glob.glob (subfolder+'/*.nii')
            img = nib.load (filename[0]) # 192, 192, 19
            # rotate the image to conventional view point
            img = np.rot90 (img.get_fdata (), axes=(1,0)) 
            # normalize each map to [0, 1]
            img -= img.min()
            img /= img.max()
            all_maps.append (img)
        else: continue
    return np.stack (all_maps, -1)

def show_all_maps (img, level, width=2, fig_dir=None):
    '''
    Args:
        `img`: [height, width, depth]
    '''
    one_level = img [:,:,level]
    N = one_level.shape[-1]
    length = int (np.ceil (N/width))
    #maps = ['ADC', 'MTT', 'rCBF', 'rCBV', 'Tmax', 'TTP', 'core']
    maps = ['MIP', 'CBFA', 'CBVA', 'TTP', 'TTDA', 'MTTA', 'PMBA']
    fig, ax = plt.subplots (width, length, figsize=(length*4.8, width*4.8),
            squeeze=False)
    for i in range (N):
        ax[i//length, i%length].imshow (one_level [:,:,i])
        ax[i//length, i%length].set_title (maps [i])
    [axi.set_axis_off () for axi in ax.ravel ()]
    if fig_dir is None: plt.show ()
    else: fig.savefig (fig_dir, bbox_inches='tight',dpi=400)

def all_maps_all_cases (folder, save_path):
    all_cases = os.listdir (folder)
    for one_case in all_cases:
        img = all_maps_per_case (folder+'/'+one_case)
        np.savez_compressed (save_path+'/'+one_case, img)

def train_test_split (folder, test_folder, split_ratio=0.25):
    all_cases = os.listdir (folder)
    N = len (all_cases)
    test_cases = list (np.random.choice (N, int (N*split_ratio), replace=False))
    for i in test_cases:
        os.rename (folder+'/'+all_cases[i], test_folder+'/'+all_cases[i])

def convert_TICI (x):
    if x == '4': return 5
    elif x == '3': return 4
    elif x == '2a': return 2
    elif x == '2b': return 3
    else: return float (x)

if __name__ == '__main__':
    os.mkdir ('data/ISLES_2016/train')
    os.mkdir ('data/ISLES_2016/test')
    all_maps_all_cases ('data/ISLES_2016/raw_data', 'data/ISLES_2016/train')
    np.random.seed (100)
    train_test_split ('data/ISLES_2016/train', 'data/ISLES_2016/test')

    labels = pd.read_csv('data/ISLES_2016/labels/all.csv')
    labels.index = labels.case
    labels ['TICIScaleGrade'] = [convert_TICI (i) for i in labels[
        'TICIScaleGrade']]
    all_tests = os.listdir ('data/ISLES_2016/test/')
    labels.loc [[re.sub ('.npz', '', i) for i in all_tests]].to_csv (\
            'data/ISLES_2016/labels/test.csv')
    all_trains = os.listdir ('data/ISLES_2016/train/')
    labels.loc [[re.sub ('.npz', '', i) for i in all_trains]].to_csv (\
            'data/ISLES_2016/labels/train.csv')
