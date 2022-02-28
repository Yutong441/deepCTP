import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
import ML.GP as gp

def load_img (root, img_paths):
    img_list = []
    for one_img in img_paths:
        # maxmimum intensity projection
        img = nib.load (root+'/'+one_img+'.nii.gz')
        img = np.max(img.get_fdata(), axis=2)
        img -= img.min ()
        img /= img.max ()
        img_list.append (img)
    return np.stack (img_list, axis=0)

def load_data (label_path, ycol, img_path):
    train_y = pd.read_csv (label_path+'/train_y.csv', index_col=[0])
    test_y = pd.read_csv (label_path+'/test_y.csv', index_col=[0])

    train_x = (load_img (img_path, train_y.index)).astype ('float32')
    test_x = (load_img (img_path, test_y.index)).astype ('float32')
    train_y = train_y[ycol].values.astype ('float32')[:,np.newaxis]
    test_y = test_y[ycol].values.astype ('float32')[:,np.newaxis]

    train_df = {'x': train_x, 'y': train_y}
    test_df = {'x': test_x, 'y': test_y}
    return train_df, test_df

def AffineScalar (shift, scale):
    scale_fun = tfp.bijectors.Scale (scale)
    return scale_fun (tfp.bijectors.Shift (shift))

def get_conv_kernel (img_shape, patch_shape = [6,6]):
    f64 = lambda x: np.array(x, dtype=np.float32)
    positive_with_min = lambda: tfp.bijectors.Shift(shift=f64(1e-4)
            )(tfp.bijectors.Softplus())
    constrained = lambda: AffineScalar(shift=f64(1e-4), 
            scale=f64(100.0))(tfp.bijectors.Sigmoid())
    max_abs_1 = lambda: AffineScalar(shift=f64(-0.5), 
            scale=f64(4.0))(tfp.bijectors.Sigmoid())
    #max_abs_1 = tfp.bijectors.Sigmoid 

    conv_k = gpflow.kernels.Convolutional(gpflow.kernels.SquaredExponential(),
            img_shape, patch_shape)
    conv_k.base_kernel.lengthscales = gpflow.Parameter(1.0, 
            transform=positive_with_min())
    # Weight scale and variance are non-identifiable. We also need to prevent
    # variance from shooting off crazily.
    conv_k.base_kernel.variance = gpflow.Parameter(1.0, transform=constrained())
    conv_k.weights = gpflow.Parameter(conv_k.weights.numpy(), 
            transform=max_abs_1())
    return conv_k

def train_convGP (train_df, maxiter=1000, patch_shape = [3,3], batch_size=5):
    conv_k = get_conv_kernel (train_df ['x'].shape [1:], patch_shape)
    N = patch_shape[0]*patch_shape[1]
    patches = conv_k.get_patches(train_df['x'])
    conv_f = gpflow.inducing_variables.InducingPatches(
        np.unique(patches.numpy().reshape(-1, N), axis=0))
    C = len (np.unique(train_df['y']))
    invlink = gpflow.likelihoods.RobustMax(C)  
    likelihood = gpflow.likelihoods.MultiClass(C, invlink=invlink)  
    num, h, w = train_df ['x'].shape
    batched_set = tf.data.Dataset.from_tensor_slices(
            (train_df['x'].reshape([num,-1]), train_df['y']) 
            ).batch(batch_size)
    return gp.optimisation (conv_k, likelihood, conv_f, C, 
            iter (batched_set), maxiter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='data/CTP/labels_186')
    parser.add_argument('--img', type=str, default='data/CTP/CTA_noskull_pro')
    parser.add_argument('--ycol', type=str, default='mRS_3m')
    parser.add_argument('--poslab', type=float, default=2.)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    gpflow.config.set_default_float(np.float32)
    train_df, test_df = load_data (args.label, args.ycol, args.img)
    if len (tf.config.list_physical_devices ('GPU')) >=1:
        device = '/GPU:0'
    else: device = '/CPU:0'

    with tf.device (device):
        model = train_convGP (train_df)
        df = gp.test_model (model, train_df, test_df, pos_label=args.poslab)

    with open (args.save, 'w') as f: 
        f.write ('imaging features: {}\n'.format (args.img))
        f.write ('y feature: {}\n'.format (args.ycol))
        f.write ('positive label: {}\n'.format (args.poslab))
        f.write (' \n')
        f.write (df.to_string ())
