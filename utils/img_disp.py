import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from medcam import medcam

from utils.train_utils import check_accuracy
from utils.metric import reverse_one_hot

def loss_history (save_prefix, train_num, batch_size):
    '''Plot loss per epoch'''
    loss = pd.read_csv(save_prefix+'_loss.csv', index_col=[0])
    loss.index = np.arange(len(loss))/np.ceil (train_num/batch_size)
    loss.plot ()
    plt.xlabel ('epoch')
    plt.ylabel ('loss')
    plt.savefig (save_prefix+'_tmp/loss.png',bbox_inches='tight')

def metric_history (save_prefix, eval_every):
    '''Plot metric changes per epoch'''
    train= pd.read_csv (save_prefix+'_metric_train.csv', index_col=[0])
    test = pd.read_csv (save_prefix+'_metric_test.csv',  index_col=[0])
    xax = np.arange(len (train))*eval_every
    N = len (train.columns)
    nrow = N//2+N%2
    fig, ax = plt.subplots (ncols=2, nrows=nrow, squeeze=False,
            figsize=(10, 4.8*nrow), frameon=False)
    for i in range (N):
        ax [i//2, i%2].plot (xax, train.iloc[:,i], color='blue', label='train')
        ax [i//2, i%2].plot (xax, test.iloc[:,i], color='red', label='test')
        ax [i//2, i%2].set_xlabel ('epoch')
        ax [i//2, i%2].set_ylabel (train.columns[i])
        ax [i//2, i%2].legend ()
    fig.savefig(save_prefix+'_tmp/metric.png',bbox_inches='tight')
    plt.clf ()

def get_images (loader, loader_set, model, cfg, device, show_maps,
        accuracy='least'):
    '''
    Show images on which the classification was the least or most accurate.
    Args:
        `loader`: pytorch dataloader
        `loader_set`: the pytorch dataset used to create the data loader
        `model`: pytorch model
        `cfg`: the config dictionary
        `device`: where to perform model testing, 'cuda:0' or 'cpu'
        `show_maps`: a function to display the images
        `level`: which plane to display
        `accuracy`: 'least' if choosing the images with lowest accuracy; or
        'most' otherwise
    '''
    model.eval ()
    ytrue, ypred= check_accuracy(loader, model, cfg, device, final_val=False)
    errors = (ytrue-ypred)**2
    if accuracy == 'least':
        error_sorted = -np.sort (-errors) #error values from max to min
        img_label = 'bad'
    else: 
        error_sorted = np.sort (errors) #error values from min to max
        img_label = 'good'

    if hasattr (model, 'predictor'): cfg['gcam']='predictor.'+cfg['gcam']
    model = medcam.inject(model, backend="gcam", layer=cfg['gcam'],
            save_maps=False, label='best', replace=True)

    torch.backends.cudnn.enabled=False
    show_acc = []
    for i in range (cfg['show_num']):
        # where is the nth largest/smallest value in the `errors` array
        show_index = np.where (errors == error_sorted[i])[0][0]
        #prevent re-selecting the same image if ties occur
        errors [show_index] = error_sorted[-1] 

        # save original image
        img, lab = loader_set.__getitem__ (show_index)
        fig_dir = cfg['save_prefix']+'_tmp/{}_img_{}.png'.format (img_label, i)
        show_maps (np.array (img.permute (2,3,1,0)), 
                level=cfg['level'], fig_dir= fig_dir)

        # prediction information
        show_acc.append ('Truth: {}; Predicted: {} \n'.format (ytrue[show_index],
            ypred[show_index]))

        # save activation map
        plt.clf ()
        act_dir = cfg['save_prefix']+'_tmp/{}_act_{}.png'.format (img_label, i)
        act = model (img.unsqueeze (0).to(device))
        plt.imshow (img [0,cfg['level']], cmap='gray')
        plt.imshow (act [0,0,cfg['level']].detach().cpu().numpy(), alpha=0.5)
        plt.axis ('off')
        plt.savefig(act_dir, bbox_inches='tight')

    # write prediction information
    log_dir = cfg['save_prefix']+'_tmp/{}_img_log.txt'.format (img_label)
    with open (log_dir, 'w') as f: f.writelines (show_acc)
