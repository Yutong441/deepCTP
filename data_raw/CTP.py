'''
Preprocess CTP images.

==========Background==========
When the CTP images are pulled from PACS, the images from each patient are
stored in each corresponding folder. Each folder has subfolders corresponding
to image modalities. For CTP images, the modalities could be the original 4D
raw data, CTA, or each of the reconstructed perfusion maps. Each subfolder
contains slices of CT images stored as dicom

patient1
----mode1
--------slide1.dicom
--------slide2.dicom
--------slide3.dicom
----mode2
--------slide1.dicom
--------slide2.dicom
--------slide3.dicom


==========Method==========
1. For each patient, identify the image modalities of interest
2. For each modality, stack the images as a 3D tensor [H, W, D]
3. For CTP, stack the perfusion maps as a 4D tensor [H, W, D, C]
4. For CTP, remove heatmap bars
5. For CTA/NCCT, strip the skull
6. Extract dicom metadata, rename the folders
7. Normalize each tensor on a scale from 0 to 1

==========Usage==========
python data_raw/CTP.py --dir=<data directory> \
        --mode='CTP,CTA,NCCT' \
        --window='0,100' \
        --start_from='0,0,0' \
        --mode_regex='data_raw/query_name.json'

Args:
    `mode`: there should not be spaces between each imaging modality.
    `window`: CT windowing, 2 values (min and max) separated by comma. The
    default is the brain window from 0 to 100
    `start_from`: discard the CT slices below a certain level, which contains
    neck rather than brain structures. The number of values should match the
    number of modalities. The values entered will only be effective for CTA,
    CTP and NCCT.
    `mode_regex`: a json file containing the regular expression that matches
    the filenames of images belonging to a particular modality

==========Output==========
Each nii.gz file stores 3D images of a patient

mode1
----patient1.nii.gz
----patient2.nii.gz
mode2
----patient1.nii.gz
----patient2.nii.gz

If the modality is CTA or NCCT, then the output folder looks like:

mode1_skull
----patient1.nii.gz
----patient2.nii.gz
mode1_noskull
----patient1.nii.gz
----patient2.nii.gz

This contains both the original and skull stripped images.

At the end of directory, there is a document called 'missing_studies.csv' that
indicates in which patients the images are missing

NB: the tensor stored in each nii.gz file may not have exactly the same
dimensions.
'''
import re
import os
import glob
import json
import shutil
import numpy as np
import pandas as pd

import nibabel as nib
import pydicom as dicom
import matplotlib.pyplot as plt
import skimage
from skimage import morphology
from skimage import measure
from scipy import ndimage

# ----------File management----------
def load_nib (img_path):
    img = nib.load (img_path)
    return img.get_fdata ()

def save_nib (img, img_path):
    img_nib = nib.Nifti1Image (img, None)
    img_nib.to_filename(img_path)

def append_str (str_name, append_name):
    filename = os.path.basename (str_name)
    first_name = filename.split('.')[0]
    return re.sub ('/'+first_name+'.', 
            '/'+first_name+'_'+append_name+'.', str_name)

def makedir (directory):
    if not os.path.exists (directory): os.mkdir (directory)

# ----------Disambiguation----------
def discard_first (img_list):
    '''
    Sometimes the first CT slice is in a different plane and has a different
    shape as the rest of the slices
    '''
    if str (img_list [0].shape) != str (img_list[1].shape):
        return img_list [1:]
    else: return img_list

def choose_modality_name (root, ori_patient, modality_name):
    if len (modality_name) == 1:
        return glob.glob (root+'/'+ori_patient+'/*'+modality_name[0]+'*')
    else:
        for i in modality_name:
            mod_dir = glob.glob (root+'/'+ori_patient+'/*'+i+'*')
            if len(mod_dir) >=1: break
        return mod_dir

def choose_modality (root, ori_patient, modality):
    '''
    If more than one match for a particular modality is found, choose the one
    with the largest number of slices
    '''
    if len (modality) > 1:
        img_list = []
        for i in sorted(modality):
            img_dir = root+'/'+ori_patient+'/'+os.path.basename(i)+'/*.dcm'
            img_list.append (glob.glob (img_dir))
        img_len = np.array ([len (i) for i in img_list])
        max_index = np.argmax (img_len == np.max (img_len))
        print ('choosig {}'.format (sorted (modality) [max_index]))
        return img_list [max_index]
    else: 
        img_dir = root+'/'+ori_patient+'/'+os.path.basename(modality[0])+'/*.dcm'
        return glob.glob (img_dir)

def missing_study (root, ori_patient, ori_modality, save_path):
    with open (root+'/missing_studies.csv', 'a') as f: 
        patient_name = get_patient_name (root, ori_patient)
        f.write ('{},{},{}\n'.format (ori_patient, patient_name, ori_modality))
        if save_path is None: return None

def remove_missing (root, ori_patient, ori_modality):
    missed = pd.read_csv (root+'/missing_studies.csv')
    missed = missed [(missed['ori_ID'] != ori_patient) | (missed [
            'modality'] != ori_modality)]
    missed.to_csv (root+'/missing_studies.csv', index=False)

def get_patient (root, modality):
    missed = pd.read_csv (root+'/missing_studies.csv')
    missed = missed [missed ['modality'] == modality]
    if len(missed)>0:
        return list (missed ['ori_ID'])
    else: 
        patients = os.listdir (root)
        patients.remove ('missing_studies.csv')
        return patients

def get_patient_name (root, ori_patient):
    mode_paths = os.listdir(root+'/'+ori_patient)
    img_paths = glob.glob (root+'/'+ori_patient+'/'+mode_paths[0]+'/*.dcm')
    ds = dicom.dcmread (img_paths[0])
    return ds.PatientName

# ----------Stacking 2D slices----------
def original_CT (img_path, modality):
    ds = dicom.dcmread(img_path)
    if 'NCCT' in modality or 'CTA' in modality:
        return ds.pixel_array*ds.RescaleSlope + ds.RescaleIntercept
    else: return ds.pixel_array

def stack_3d (root, ori_patient, ori_modality, modality_dict_path=None,
        window=None, save_path=None):
    if modality_dict_path is None:
        modality_dict = {
                'CTA': ['DynMulti4D__1_5__H20f', 'AngioArchCoW'],
                'NCCT': ['[Head|Brain]*__5_0__', '5_mm_brain', 
                    'HeadStandard', 'StdBrain']
                }
    else:
        with open (modality_dict_path, 'r') as f: 
            modality_dict = json.load (f)

    if ori_modality in modality_dict.keys():
        modality_name = modality_dict [ori_modality]
    else: modality_name = [ori_modality]
    modality = choose_modality_name (root, ori_patient, modality_name)

    if len (modality) >=1:
        img_paths= choose_modality (root, ori_patient, modality)
        if len (img_paths) >1: # do not include images with just one slice
            dss = [original_CT (i, ori_modality) for i in sorted (img_paths)]
            dss = np.stack(discard_first (dss), axis=-1)
            if window is not None: dss = dss.clip (window[0], window[1])

            if save_path is None: return dss
            else: save_nib (dss, save_path+'.nii.gz')
            remove_missing (root, ori_patient, ori_modality)
        else: missing_study (root, ori_patient, ori_modality, save_path) 
    else: missing_study (root, ori_patient, ori_modality, save_path) 
    # if no studies can be found

def normalize (img):
    img -= img.min ()
    return img/img.max ()

def rm_heatbar (img, coord=[288, 98, 300, 230]):
    img [coord[1]:coord[3], coord[0]:coord[2]] = 0
    return img

# ----------CTP----------
def stack_consensus (img_list):
    '''
    Sometimes some CT perfusion maps contain a slightly different number of
    slices. This function finds out the smallest number of slices and truncate
    the maps with extra slices.
    Args:
        `img_list`: a list of 3D numpy arrays
    '''
    depths = np.array ([i.shape[2] for i in img_list])
    min_depth = min (depths)
    return np.stack ([img_list[i][:,:,(depths[i]-min_depth):] for i in range (
        len (img_list))], axis=-1)

def stack_3d_CTP (root, ori_patient, start_from=0, save_path=None):
    CTP_map = ['MIP', 'CBFA', 'CBVA', 'TTP', 'TTDA', 'MTTA', 'PMBA']
    maps = []
    for one_map in CTP_map:
        dss = stack_3d (root, ori_patient, 'VPCT_RGB_Axial_'+one_map, 
                window=None)
        if dss is not None:
            ds = rm_heatbar (dss.sum (2)) # sum along the color axis  
            maps.append (normalize (ds[:,:,start_from:])) 
    if len (maps) == len (CTP_map):
        maps = stack_consensus (maps)
        if save_path is None: return maps
        else: save_nib (maps, save_path+'.nii.gz')
        remove_missing (root, ori_patient, 'CTP') 
    else: missing_study (root, ori_patient, 'CTP', save_path) 

# ----------Strip skulls----------
def skull_strip (img_path, save_mask=False, window=[0, 100], sigma=0):
    '''
    Smart histogram analysis
    https://www-sciencedirect-com.ezp.lib.cam.ac.uk/science/article/pii/S0010482512000157?via%3Dihub
    '''
    ori_img = load_nib (img_path)
    if sigma != 0: img = skimage.filters.gaussian (ori_img, sigma=sigma) 
    else: img = ori_img

    lab_list = []
    for i in range (img.shape[2]):
        # thresholding and erosion
        mask = (img [:,:,i] > window[0]).astype(float) + \
                (img [:,:,i] < window[1]).astype(float)
        mask = morphology.binary_erosion (mask == 2)

        # largest connected component
        labels = measure.label(mask)
        lab_count = np.bincount(labels.flat)
        if len (lab_count) == 1: LCC= np.zeros (labels.shape)
        else: LCC = (labels == np.argmax (lab_count [1:]) +1)

        # dilation and fill holes
        LCC = morphology.binary_dilation (LCC)
        lab_list.append (ndimage.binary_fill_holes (LCC))

    masks = (np.stack (lab_list, axis=-1)).astype(int)
    save_nib ((ori_img*masks).clip(window[0], window[1]), \
            append_str (img_path, 'noskull'))
    if save_mask: save_nib (masks, append_str (img_path, 'mask'))

def RGB (img, channel=0):
    out_img = np.zeros ([img.shape[0], img.shape[1], 3])
    out_img [:,:,channel]=img
    return out_img

def disp_mask (img_path, mask_path=None, nlevels=6, ncol=3, disp_window=None):
    '''
    Example:
    >>> skull_strip ('data/CTP/NCCT/CTP001.nii.gz')
    >>> disp_mask ('data/CTP/NCCT/CTP001.nii.gz', nlevels=9, 
            disp_window=[0,100])
    '''
    ori_img = load_nib (img_path)
    if disp_window is not None:
        ori_img = ori_img.clip (disp_window[0], disp_window[1])
    if mask_path is None: mask_path= append_str (img_path, 'mask')
    mask = load_nib (mask_path)
    levels = np.linspace(0, ori_img.shape[-1]-1, nlevels)
    nrow = int (np.ceil (nlevels/ncol))
    for index, i in enumerate (levels):
        plt.subplot (nrow, ncol, index+1)
        plt.imshow (ori_img[:,:,int(i)], cmap='gray')
        plt.imshow (RGB (mask [:,:,int(i)]), alpha=0.2)
        plt.axis ('off')
    plt.show()

def CT_postprocessing (save_file, skull_dir, noskull_dir, ID, window,
        start_from):
    if os.path.exists (save_file):
        skull_strip (save_file, window=window)
        standardise (save_file, append_str (save_file, 'skull'),
                window=window, start_from=start_from)
        standardise (append_str (save_file, 'noskull'), 
                append_str (save_file, 'noskull'), start_from=start_from)

        os.rename (append_str (save_file, 'noskull'),
                noskull_dir+'/'+str(ID)+'.nii.gz')
        os.rename (append_str (save_file, 'skull'),
                skull_dir+'/'+str(ID)+'.nii.gz')
    else: print ('cannot find {}'.format (save_file))

# ----------Save images----------
def standardise (img_path, save_path, window=None, start_from=None):
    '''Perform windowing, Remove neck slices and Normalize'''
    img = load_nib (img_path)
    if window is not None: img = img.clip (window[0], window[1])
    if start_from is not None: img = img [:,:,start_from:]
    save_nib (normalize (img), save_path)

def save_3d_mode (root, modality, save_dir=None, window=None, start_from=0,
        modality_dict_path=None):
    # create directories
    if save_dir is None:
        save_dir = os.path.dirname (root)+'/'+modality
        makedir (save_dir)
    if 'CT' in modality and modality != 'CTP': 
        skull_dir = save_dir+'_skull'
        noskull_dir = save_dir+'_noskull'
        makedir (skull_dir)
        makedir (noskull_dir)
    if not os.path.exists (root+'/missing_studies.csv'):
        with open (root+'/missing_studies.csv', 'w') as f:
            f.write ('ori_ID,ID,modality\n')

    patients = get_patient (root, modality)
    #patients = os.listdir(root)
    #patients.remove ('missing_studies.csv')
    for i in sorted (patients):
        ID= get_patient_name (root, i)
        print ('analysing {}, {}'.format (i, ID))
        save_path = save_dir+'/'+str(ID)
        if modality == 'CTP': stack_3d_CTP (root, i,
                start_from=start_from, save_path=save_path) 
        else: stack_3d (root, i, modality, save_path=save_path,
                modality_dict_path=modality_dict_path)

        # save both original and skull stripped images for CTA and NCCT
        if 'CT' in modality and modality != 'CTP': 
            CT_postprocessing (save_path+'.nii.gz', skull_dir, noskull_dir,
                    ID, window, start_from)

    if 'CT' in modality and modality != 'CTP': shutil.rmtree(save_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--mode', type=str, default='CTP,CTA,NCCT')
    parser.add_argument('--window', type=str, default='0,100')
    parser.add_argument('--start_from', type=str, default='0,0,0')
    parser.add_argument('--mode_regex', type=str,
            default='data_raw/query_name.json')
    args = parser.parse_args()
    if args.window != 'None': 
        window = [int(i) for i in args.window.split(',')]
    else: window = None
    for index, i in enumerate (args.mode.split (',')): 
        save_3d_mode (args.dir, i, window=window,
                start_from=int (args.start_from.split(',')[index]),
                modality_dict_path=args.mode_regex)
