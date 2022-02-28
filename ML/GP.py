import argparse
import os
import numpy as np
import pandas as pd
import gpflow
from gpflow.utilities import set_trainable, print_summary
from gpflow.ci_utils import ci_niter
from sklearn import metrics 

# --------------------Data processing--------------------
def load_data (label_path, ycol, img_path=None, join_img=False):
    '''
    Args:
        `label_path`: path to the csv storing x and y variables
        `ycol`: name of the y variable
        `img_path`: path to the npz file storing imaging features extracted by
        deep neural network
    Returns:
        `train_df`: a dictionary of x and y for training dataset
        `test_df`: a dictionary of x and y for testing dataset
    '''
    train_y = pd.read_csv (label_path+'/train_y.csv', index_col=[0]
            )[ycol].values.astype (float)[:,np.newaxis]
    test_y = pd.read_csv (label_path+'/test_y.csv', index_col=[0]
            )[ycol].values.astype (float)[:,np.newaxis]

    if img_path is not None:
        train_x = np.load (img_path+'/CTP_act_train.npz'
                )['arr_0'].astype(np.float64)
        test_x = np.load (img_path+'/CTP_act_test.npz'
                )['arr_0'].astype(np.float64)
        no_na = np.logical_and (train_x.std (axis=0) !=0, 
                test_x.std (axis=0) != 0)
        train_x = train_x[:, no_na]
        test_x = test_x[:, no_na]
        if join_img:
            train_x_nimg = pd.read_csv (label_path+'/train_x.csv', 
                    index_col=[0]).values
            test_x_nimg = pd.read_csv (label_path+'/test_x.csv', 
                    index_col=[0]).values
            train_x = pd.concat ([train_x, train_x_nimg], axis=1)
            test_x = pd.concat ([test_x, test_x_nimg], axis=1)
    else:
        train_x = pd.read_csv (label_path+'/train_x.csv', index_col=[0]).values
        test_x = pd.read_csv (label_path+'/test_x.csv', index_col=[0]).values

    train_df = {'x': normalize (train_x), 'y': train_y}
    test_df = {'x': normalize (test_x), 'y': test_y}
    return train_df, test_df

def normalize (xx):
    return (xx - xx.mean (0))/xx.std (0)

def one_hot (label, num_class):
    assert max (label) <= num_class - 1, \
            'num_class should be larger than the maximum label'
    uniq_vec = np.arange (0,num_class)
    return (label.squeeze ()[:,None] == uniq_vec[None]).astype(float)

def label2ordinal (label: np.ndarray, num_class: int):
    label_hot = one_hot (label, num_class)
    return np.cumsum (label_hot [:,::-1], axis=1)[:,::-1][:,1:]

def ordinal2label(pred: np.ndarray):
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) 

# --------------------Model training--------------------
def optimisation (kernel, likelihood, Z, C, train_tuple, maxiter):
    model = gpflow.models.SVGP( kernel=kernel, likelihood=likelihood,
            inducing_variable=Z, num_latent_gps=C, whiten=True, q_diag=True)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss_closure(train_tuple, compile=True), 
        model.trainable_variables, options=dict(maxiter=ci_niter(maxiter)))
    return model

def get_kernel (kernel_type, indiv_ls, train_df):
    kernel_fun = getattr (gpflow.kernels, kernel_type)
    if indiv_ls: 
        ls = np.ones ([train_df['x'].shape[1]])
        return kernel_fun (lengthscales =ls)+ gpflow.kernels.White(variance=0.01)
    else: 
        return kernel_fun ()+ gpflow.kernels.White(variance=0.01)

def train_GPC (train_df, kernel_type='Matern52', maxiter=1000, indiv_ls=False):
    kernel = get_kernel (kernel_type, indiv_ls, train_df)
    unique_y = np.unique(train_df[ 'y'])
    C = len (unique_y)
    invlink = gpflow.likelihoods.RobustMax(C)  
    likelihood = gpflow.likelihoods.MultiClass(C, invlink=invlink)  

    Z = train_df['x'][::10].copy()  # inducing inputs
    return optimisation (kernel, likelihood, Z, C, (train_df['x'], 
        train_df['y']), maxiter)

def train_GPR (train_df, kernel_type, maxiter=1000, indiv_ls=False):
    kernel = get_kernel (kernel_type, indiv_ls, train_df)
    train_tuple = (train_df['x'], train_df['y'])
    Z = train_df['x'][::10].copy()  # inducing inputs
    model = gpflow.models.SGPR(data=train_tuple, kernel=kernel, 
            mean_function=None, inducing_variable=Z)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables,
            options=dict(maxiter=ci_niter(maxiter)))
    return model

def train_GPR_ordinal (train_df, kernel_type, maxiter=1000, indiv_ls=False):
    unique_y = np.unique(train_df[ 'y'])
    bin_edges = np.array(np.arange(unique_y.size+1), dtype=float)
    bin_edges = bin_edges - bin_edges.mean()
    likelihood = gpflow.likelihoods.Ordinal(bin_edges)

    kernel = get_kernel (kernel_type, indiv_ls, train_df)
    train_tuple = (train_df['x'], train_df['y'])
    model = gpflow.models.VGP(data=train_tuple, kernel=kernel, 
            likelihood=likelihood)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables,
            options=dict(maxiter=ci_niter(maxiter)))
    return model

def train_cumlink (train_df, kernel_type='Matern52', maxiter=1000,
        indiv_ls=False):
    kernel = get_kernel (kernel_type, indiv_ls, train_df)
    C = len (np.unique(train_df[ 'y'])) - 1
    kernel = gpflow.kernels.SharedIndependent (kernel, output_dim=C)

    likelihood = gpflow.likelihoods.Bernoulli()
    Z = train_df['x'][::10].copy()  
    iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
        gpflow.inducing_variables.InducingPoints(Z))
    train_y = label2ordinal (train_df['y'], C+1)
    return optimisation (kernel, likelihood, iv, C, (train_df['x'],
            train_y), maxiter)

# --------------------Model testing--------------------
def model_metrics (ytrue, ypred, pos_label=2):
    '''
    Args:
        `ytrue`: a 1D numpy array of the true y variable labels
        `ypred`: a 1D numpy array of the predicted y variable labels
        `pos_label`: value above which a sample is labelled as positive. If
        None, then the y variable is assumed to be continuous
    Return:
        `all_metrics`: a pandas dataframe containing one row. Each column
        indicates each metric.
    '''
    labels= list (np.unique (ytrue))
    ypred, ytrue = ypred.squeeze(), ytrue.squeeze ()
    if pos_label is not None or len (labels) == 2:
        ypred = (ypred >= pos_label).astype (float)
        ytrue = (ytrue >= pos_label).astype(float)
        tn, fp, fn, tp= metrics.confusion_matrix (ytrue, ypred, labels=
                [0,1]).ravel()
        fpr, tpr, _ = metrics.roc_curve (ytrue, ypred)
        all_metrics = [[metrics.auc (fpr, tpr), tp/(tp+fn),
                tn/(tn+fp), (tn+tp)/len(ytrue)]]
        all_metrics = pd.DataFrame (all_metrics, columns=['AUC',
            'sensitivity', 'specificity', 'accuracy'])
    else:
        all_metrics = [np.mean ((ytrue - ypred)**2)]
        all_metrics = pd.DataFrame (all_metrics, columns = ['MSE'])
    return all_metrics

def prediction (model, x_df, num_class):
    mu, var = model.predict_y (x_df)
    mu = mu.numpy ()
    if len (mu.squeeze().shape) == 2: 
        if mu.shape[1] == num_class - 1: return ordinal2label (mu)
        elif mu.shape[1] == num_class: return mu.argmax (1)
    else: return mu

def test_model (model, train_df, test_df, pos_label=None):
    C = len (np.unique (train_df ['y']))
    mu = prediction(model, test_df['x'], C)
    test_metric = model_metrics (test_df['y'], mu, pos_label)

    mu = prediction(model, train_df['x'], C)
    train_metric = model_metrics (train_df['y'], mu, pos_label)
    all_metrics = pd.concat ([train_metric, test_metric], axis=0)
    all_metrics.index = ['train', 'test']
    return all_metrics 

#train_df, test_df= load_data ('data/CTP/labels_186', 'mRS_bool',
#        img_path='logs/CTP_re')

def train_test (label, ycol, img_path=None, poslab=2., save=None,
        indiv_ls=False, join_img=False):
    with open (save, 'a') as f:
        f.write (' \n')
        if not join_img:
            if img_path is None: f.write ('# Non-imaging only')
            else: f.write ('# Imaging features only')
        else: f.write ('# Imaging and non-imaging features')

    train_df, test_df = load_data (label, ycol, img_path=img_path)
    all_modes = ['numeric', 'categorical','cumlink','ordinal']
    if len (np.unique (train_df['y'])) ==2: all_modes = ['categorical']
    for mode in all_modes:
        df_list = []
        for kern in ['RBF', 'Matern12', 'Matern32', 'Matern52']:
            if mode =='categorical':
                model = train_GPC (train_df, kernel_type=kern,
                        indiv_ls=indiv_ls)
            elif mode == 'ordinal':
                model = train_GPR_ordinal (train_df, kernel_type=kern,
                        indiv_ls=indiv_ls)
            elif mode == 'numeric':
                model = train_GPR (train_df, kernel_type=kern,
                        indiv_ls=indiv_ls)
            elif mode == 'cumlink': 
                model = train_cumlink (train_df, kernel_type=kern,
                    indiv_ls=indiv_ls)
            df = test_model (model, train_df, test_df, pos_label=poslab)
            order_col = ['kernel'] + list (df.columns)
            df ['kernel']= [kern]*len(df)
            df_list.append (df [order_col])

        all_df = pd.concat (df_list, axis=0)
        if save is not None:
            with open (save, 'a') as f:
                f.write ('\n## {}\n'.format (mode))
                f.write (all_df.to_string ())
                f.write (' \n')
        else: print (mode); print (df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=str, default='data/CTP/labels_186')
    parser.add_argument('--img', type=str, default='None')
    parser.add_argument('--ycol', type=str, default='mRS_3m')
    parser.add_argument('--poslab', type=float, default=2.)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--indiv_ls', type=str, default='True')
    args = parser.parse_args()

    if args.img == 'None': args.img = None
    indiv_ls = args.indiv_ls == 'True'
    if args.save is not None: 
        with open (args.save, 'w') as f: 
            f.write ('imaging features: {}\n'.format (args.img))
            f.write ('y feature: {}\n'.format (args.ycol))
            f.write ('positive label: {}\n'.format (args.poslab))
            f.write ('individual lengthscales: {}\n'.format (args.indiv_ls))
            f.write (' \n')
    train_test (args.label, args.ycol, args.img, args.poslab, args.save,
            indiv_ls)
    if args.img is not None:
        train_test (args.label, args.ycol, None, args.poslab, args.save,
                indiv_ls)
        train_test (args.label, args.ycol, args.img, args.poslab, args.save,
                indiv_ls, join_img=True)
