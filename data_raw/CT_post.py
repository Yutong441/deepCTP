# post processing for CT images
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from data_raw.CTP import load_nib, save_nib, makedir

def find_shapes (root, modality, save_dir):
    '''
    Obtain the shape of tensors stored as nii.gz in a directory
    (`$root/$modality`). Save the results as a csv (`save_dir`).
    Examples:
    >>> import argparse
    >>> from data_raw.CTP import makedir
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--dir', type=str)
    >>> parser.add_argument('--mode', type=str, default='CTP,CTA,NCCT')
    >>> parser.add_argument('--save')
    >>> args = parser.parse_args()
    >>> makedir (args.save)
    >>> for i in args.mode.split (','): 
    >>>     find_shapes (args.dir, i, args.save)
    '''
    data_dir = root+'/'+modality
    all_files = sorted (os.listdir (data_dir))
    shape_list = [re.sub ('.nii.gz', '', i)+','+str((load_nib (
        data_dir+'/'+i)).shape)[1:-1]+'\n' for i in all_files]
    with open (save_dir+'/'+modality+'_shape.csv', 'w') as f:
        f.writelines (shape_list)

def trim_to_shape (img, to_shape):
    shape_diff = np.array (img.shape)[::-1][:2] - np.array (to_shape) 
    tos= shape_diff//2+shape_diff%2
    return img [..., shape_diff[0]//2:(img.shape[::-1][0] - tos[0]),
            shape_diff[1]//2:(img.shape[::-1][1] - tos[1])]

def sel_central (num_slices, central_slices):
    if num_slices > central_slices:
        start_from = (num_slices - central_slices)//2
        end_from = start_from + (num_slices - central_slices)%2
        return np.arange (start_from, num_slices-end_from)
    else: return np.arange (num_slices)

def standardise_CT (img, to_shape, downsize=None, extreme=0, thres=100,
        central_slices=None, select_brain_slices=None):
    '''
    Take slices, trim to a particular shape, then downsample the array
    Args:
        `to_shape`: trim to shape
        `downsize`: interpolated to shape
        `extreme`: remove the top and bottom n slices, which may not contain
        brain tissues or may contain unwanted masks.
        `thres`: beyond a particular number of slices, sample the slices at 5
        intervals, because some slices were 1mm thick, most others 5mm thick.
        `central_slices`: how many slices in the center to select
        `select_brain_slices`: how many brain slices to choose

    Examples:
    >>> # for non-contrast CT images
    >>> img_post = standardise_CT (img, [512, 512], extreme=1, thres=100,
            central_slices=24)
    >>> # for CT perfusion
    >>> img_post = standardise_CT (img, [323, 323], extreme=0, thres=100,
            central_slices=None)
    >>> # for CT angiogram
    >>> img_post = standardise_CT (img, [512, 512], thres=65, 
            select_brain_slices=120)

    output dimensions:
    CTP: 323 x 323 x 21 x 7
    NCCT: 512 x 512 x [22-24]
    CTA: 512 x 512 x [14-24]
    '''
    if select_brain_slices is not None:
        img = select_brain_level (img, select_brain_slices)
    end_at = img.shape[2] - extreme
    if len (img.shape) == 3: 
        img = torch.tensor (img.copy()).unsqueeze (-1)
    else: img= torch.tensor (img)

    img = img.permute (3,2,0,1) [:,extreme:end_at]
    if img.shape[1] > thres: img = img [:,::5]
    if central_slices is not None:
        img = img [:,sel_central (img.shape[1], central_slices)]

    if img.shape[::-1][:2] != to_shape:
        padded = trim_to_shape (img, to_shape)
    else: padded = img
    if downsize is not None: padded = F.interpolate (padded, downsize)
    print (padded.shape)
    return np.array (padded.permute (2,3,1,0).squeeze())

def select_brain_level (img, num_slices=120, thres=200):
    '''
    Select the slices containing the brain.
    The center of the brain is assumed to be the slice containing the largest
    number of foreground pixels.
    Args:
        `img`: skull-stripped image
        `num_slices`: how many slices to collect in total
        `thres`: only attempt correcting z stack orientation if the number of
        slices is sufficiently large
    '''
    brain_central = np.argmax (np.sum (img >0, axis=(0,1)))
    start_from = max (brain_central-num_slices//2, 0)
    end_at = min (start_from + num_slices, img.shape[2])
    img_br = img [...,start_from:end_at]
    if start_from < img.shape[2]//2 and img.shape[2] > thres:
        return img_br [...,::-1]
    else: return img_br

def select_brain_level_couple (skull, noskull, num_slices=120, thres=200):
    brain_central = np.argmax (np.sum (noskull >0, axis=(0,1)))
    start_from = max (brain_central-num_slices//2, 0)
    end_at = min (start_from + num_slices, skull.shape[2])
    skull_br = skull [...,start_from:end_at]
    noskull_br = noskull [...,start_from:end_at]
    if start_from < skull.shape[2]//2 and skull.shape[2] > thres:
        return skull_br [...,::-1], noskull_br [...,::-1]
    else: return skull_br, noskull_br

def process_CTA (skull, noskull):
    skull, noskull = select_brain_level_couple (skull, noskull, num_slices=120)
    skull_post = standardise_CT (skull, [512, 512], thres=65)
    noskull_post = standardise_CT (noskull, [512, 512], thres=65)
    return skull_post, noskull_post

def process_all (root, mode, save_dir=None):
    if save_dir is None:
        save_dir = root+'/'+mode+'_pro'
        makedir (save_dir)

    for i in sorted (os.listdir (root+'/'+mode)):
        img = load_nib (root+'/'+mode+'/'+i)
        if mode == 'CTP':
            img_post = standardise_CT (img, [323, 323], extreme=0, thres=100,
                    central_slices=None)
        if 'NCCT' in mode:
            img_post = standardise_CT (img, [512, 512], extreme=1, thres=100,
                    central_slices=24)
        save_nib (img_post, save_dir+'/'+i)

def process_all_CTA (root, save_dir=None):
    if save_dir is None:
        skull_dir = root+'/CTA_skull_pro'
        noskull_dir = root+'/CTA_noskull_pro'
        makedir (skull_dir)
        makedir (noskull_dir)

    for i in sorted (os.listdir (root+'/CTA_skull')):
        skull = load_nib (root+'/CTA_skull/'+i)
        noskull = load_nib (root+'/CTA_noskull/'+i)
        skull, noskull = process_CTA (skull, noskull)
        save_nib (skull, skull_dir+'/'+i)
        save_nib (noskull, noskull_dir+'/'+i)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--mode', type=str,
            default='CTP,CTA,NCCT_skull,NCCT_noskull')
    args = parser.parse_args()
    for i in args.mode.split (','): 
        if i != 'CTA': process_all (args.dir, i)
        else: process_all_CTA (args.dir)

#import numpy as np
#from data_raw.CTP import load_nib
#from data_raw.ISLES2016 import show_all_maps
#import matplotlib.pyplot as plt
#img = load_nib ('data/CTP/CTP_pro/CTP029.nii.gz')
#show_all_maps (img, 10, fig_dir='results/example.png')
#plt.show ()
#activations = np.load ('results/CTP_weight/CTP_act_test.npz')['arr_0']
