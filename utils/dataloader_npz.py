import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torchvision import transforms

def padding_to_shape (x, to_shape):
    '''
    Args:
        `x`: pytorch tensor. The last two dimensions to be padded
        `to_shape`: the shape of the final padded tensor
    '''
    shape_diff = np.array (to_shape) - np.array (x.shape)[::-1][:2]
    pad_dim = [shape_diff[0]//2, shape_diff[0]//2+shape_diff[0]%2, 
            shape_diff[1]//2, shape_diff[1]//2+shape_diff[1]%2]
    return F.pad (x, pad_dim)

def random_depth (img, depths, random=True):
    if random: start_depth = np.random.choice (img.shape[1] - depths, 1)
    else: start_depth = (img.shape[1] - depths)//2
    return img [:, start_depth:(start_depth + depths)]

class dataloader (torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, transformation='default',
            outcome_col='mRS', num_classes=6, common_shape=[256, 256],
            downsize=None, select_channels=None, select_depths=None,
            select_num=None):
        '''
        Args:
            `root_dir`: top-level directory for a particular dataset, which
            should contains `train`, `test` and `labels` folder
            `mode`: 'train', 'test' or 'validate'
            `num_classes`: number of prediction output class; if None, it will
            be inferred from the data
            `common_shape`: the height and width for all images. If an image
            does not have this shape, zero padding is used. 
            `select_channels`: which channels to be in the input
            `select_depths`: an integer, number of central slices to select
            `select_num`: number of images to select
        '''
        self.root_dir = root_dir
        self.common_shape = common_shape
        self.annotations = pd.read_csv(root_dir+'/labels/'+mode+'.csv',
                index_col=[0])
        if select_num is not None:
            self.annotations = self.annotations [:select_num]

        self.ylabels = torch.tensor (self.annotations[outcome_col])
        if num_classes is None:
            num_classes = int (max(np.unique (ylabels)))
        #self.ylabels = F.one_hot (ylabels, num_classes=num_classes)
        self.mode = mode
        self.channels = select_channels
        self.depths = select_depths
        self.downsize = downsize

        if transformation == 'default':
            transformation = torch.nn.Sequential (
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomAffine (0, translate=(0.05, 0.05),
                        scale=(0.95, 1.05)),
                    transforms.ColorJitter (brightness=0.1),
                    transforms.GaussianBlur (3)
                    )
            transformation = torch.jit.script (transformation)
        if mode == 'test': transformation = None
        self.transform = transformation

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.index[index]
        filename = self.root_dir+'/'+self.mode+'/'+img_id+'.npz'
        img = torch.tensor (np.load (filename)['arr_0'])
        if (len (img.shape)==3): img = img.unsqueeze (3)

        # make sure every image has the same shape
        img = padding_to_shape (img.permute (3, 2, 0, 1), self.common_shape)
        if self.downsize is not None: img = F.interpolate (img, self.downsize) 
        if self.channels is not None: img = img [self.channels]
        if self.depths is not None: 
            img = random_depth (img, self.depths, self.transform)

        # perform image augmentation
        if self.transform is not None:
            old_shape = img.shape
            new_shape = [-1, 1, *self.common_shape]
            img = self.transform(img.reshape(new_shape))
            img = img.reshape (old_shape)
            # randomly flip z axis
            if np.random.rand(1) > 0.5: img = img [:,::-1]
        return (img.type (torch.float32), self.ylabels[index])
