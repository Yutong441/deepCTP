import os
import re
import glob

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

def one_img (data_dir, dirname, depth=100, offset=10):
    '''
    Args:
        `depth`: how many central levels to select
        `offset`: reduce the starting point by how much
    '''
    img_sets = os.listdir (data_dir+'/'+dirname)
    img_name = glob.glob (data_dir+'/'+dirname+'/'+img_sets[0]+'/*.nii.gz')
    img = nib.load (img_name[0])
    img = np.rot90 (img.get_fdata (), axes=(0,1))
    img -= img.min()
    img /= img.max()
    start = (img.shape[2] - depth)//2 - offset
    print (start)
    print (img.shape)
    return img [:,:,start:start+depth] # removed slices at the periphery

def show_img (img, level, fig_dir=None):
    plt.imshow (np.squeeze (img)[:,:,level], cmap='gray') # axial plane
    plt.axis ('off')
    if fig_dir is None: plt.show ()
    else: plt.savefig (fig_dir, bbox_inches='tight', pad_inches=0.05)

def show_multi_level (img, levels, width=2):
    length = int (np.ceil (len(levels)/width))
    for i, level in enumerate (levels):
        plt.subplot (width, length, i+1)
        plt.imshow (img [:,:,int(level)], cmap='gray')
        plt.title ('level = {}'.format (int (level)))
        plt.axis ('off')
    plt.show()

def all_imgs_all_cases (folder, save_path):
    all_cases = os.listdir (folder)
    for one_case in all_cases:
        img = one_img (folder, one_case)
        np.savez_compressed (save_path+'/'+one_case, img)

def train_test_split (folder, test_folder, split_ratio=0.25):
    all_cases = os.listdir (folder)
    N = len (all_cases)
    test_cases = list (np.random.choice (N, int (N*split_ratio), replace=False))
    for i in test_cases:
        os.rename (folder+'/'+all_cases[i], test_folder+'/'+all_cases[i])

def equal_class (df, col, ID_col):
    df.index = np.arange (len (df))
    num_norm = np.sum (df [col] == 0)
    num_abnorm = np.sum (df [col] == 1)
    diff = abs (num_norm - num_abnorm)

    if num_norm is not num_abnorm:
        if num_norm > num_abnorm:
            df_norm = df[df [col]==0]
            sel_df = df_norm.iloc [np.random.choice (num_norm, diff, replace=False)]
        else:
            df_abnorm = df[df [col]==1]
            sel_df = ab_abnorm.iloc [np.random.choice (num_abnorm, diff,
                replace=False)]

        dfd = df.drop( sel_df.index )
        dfd.index = dfd [ID_col]
        return dfd
    else: return df

if __name__ == '__main__':
    os.mkdir ('data/OASIS3/train')
    os.mkdir ('data/OASIS3/test')
    all_imgs_all_cases ('data/OASIS3/MR_ID/', 'data/OASIS3/train/')
    np.random.seed (100)
    train_test_split ('data/OASIS3/train', 'data/OASIS3/test', split_ratio=0.2)

    labels = pd.read_csv('data/OASIS3/labels/all.csv')
    labels ['M.F'] = (labels ['M.F'] == 'M').astype (float)
    labels.index = labels.experiment_id

    all_tests = os.listdir ('data/OASIS3/test/')
    np.random.seed (100)
    test_df = equal_class (labels.loc [[re.sub ('.npz', '', i) for i in
        all_tests]], 'ylabel', 'experiment_id')
    test_df.to_csv ('data/OASIS3/labels/test.csv')

    all_trains = os.listdir ('data/OASIS3/train/')
    np.random.seed (100)
    train_df = equal_class (labels.loc [[re.sub ('.npz', '', i) for i in
        all_trains]], 'ylabel', 'experiment_id')
    train_df.to_csv ('data/OASIS3/labels/train.csv')
    # shutil.rmtree ('data/OASIS3/MR_ID')
