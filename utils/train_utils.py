import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import logging
import os
import json

from utils.metric import AUC_MSE
from utils.config import print_config
from utils.dataloader import dataloader, to_device
from utils.loss import get_loss_fun
from utils.metric import reverse_one_hot, ordinal2label
from CNN.all_models import choose_models, load_net
from CNN.cum_link import OrdinalLogisticModel

def train_loop (model, cf, loaders):
    '''
    Perform training, testing and checkpointing.
    args:
        `model`: pytorch model
        `cf`: config, see the `utils/config.py` file for explanation
        `loader`: list of dataloaders for train, eval and test sets
    '''
    save_dir=os.path.dirname (cf['save_prefix'])
    if not os.path.exists (save_dir): os.mkdir (save_dir)

    if 'cum_link' in cf['loss_type']: model = OrdinalLogisticModel (model, cf)
    criterion = get_loss_fun (cf)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
        model.parameters()), lr=cf['lr'], weight_decay=cf['L2'])
    scheduler= torch.optim.lr_scheduler.StepLR (optimizer, cf[
        'step_size'], cf['gamma'])
    CP = checkpoint (cf['initial_metric'], cf['metric'], cf['tolerance'],
            cf['better'])
    model, logger = initialise_logs (model, cf)
    log_config (cf['save_prefix']+'_log', cf)

    model.to(cf['device'])
    for epoch in range(cf['num_epochs']):
        logger.info ('epoch {}/{}'.format (epoch, cf['num_epochs']))
        logger.info ('The learning rate is {}'.format (scheduler.get_last_lr ()))
        model.eval()

        if epoch % cf['eval_every'] == 0:
            for mode, loader in zip (['train', 'test'], loaders[1:]):
                epoch_metric = check_accuracy(loader, model, cf, cf['device'])
                epoch_metric.to_csv (cf['save_prefix']+'_metric_'+mode+'.csv',
                        mode='a', header=False)
                if mode == 'test':
                    CP.update (float (epoch_metric[CP.metric].iloc[-1]))
                    if CP.balance == CP.tolerance: 
                        logger.info ('saving model')
                        torch.save(model.state_dict(), cf['save_prefix']+'_model')
                    else: logger.info('{} epochs left'.format (CP.balance))

        if CP.balance == 0: break
        model.train()
        loss_list = []
        for imgs, labels in loaders[0]:
            imgs = to_device (imgs, cf['device'])
            labels = to_device (labels, cf['device'])
            if 'unet' in cf ['model_type']: labels= (imgs, labels)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().cpu().numpy())

        (pd.DataFrame (loss_list)).to_csv (cf['save_prefix']+'_loss.csv',
                header=False, mode='a')
        scheduler.step ()
        print ('----------')

def check_accuracy(loader, model, cfg, device='cpu', final_val=True):
    with torch.no_grad():
        y_list, ypred_list = [], []
        for x, y in loader:
            y_list.append (y)
            ypred = model(to_device (x, device))
            ypred_list.append (ypred.to('cpu'))

    all_y = np.array (torch.cat (y_list, axis=0))
    all_ypred = np.array (torch.cat (ypred_list, axis=0))
    if final_val:
        auc_mse = AUC_MSE (all_ypred, all_y, pos_label=cfg ['pos_label'],
                loss_type=cfg['loss_type'])
        return pd.DataFrame (auc_mse [np.newaxis], columns=cfg ['all_metrics'])
    else:
        if 'ordinal' in cfg['loss_type']: all_ypred = ordinal2label (all_ypred)
        else: all_ypred = reverse_one_hot (all_ypred)
        return all_y, all_ypred

class checkpoint:
    def __init__ (self, initial_metric=0, metric='AUC', tolerance=20,
            better='pos'):
        '''
        Stop the training process after a certain number of epochs with failure
        of improvement.
        Args:
            `initial_metric`: expected initial value for the metric of interest
            `metric`: on which metric to determine model improvement
            `tolerance`: after how many epochs of failure of improvement would
            the training stops
            `better`: whether a higher value in the metric of interest is
            better; either 'pos' or 'neg'
        '''
        self.metric = metric
        self.metric_val = initial_metric
        self.tolerance = tolerance
        self.balance = tolerance
        self.better = 1 if better == 'pos' else -1

    def update (self, metric):
        # if the outcome is better than the previously established benchmark
        if metric*self.better >= self.metric_val*self.better:
            self.metric_val = metric
            self.balance = self.tolerance 
        # if the outcome is worse
        else:  self.balance -= 1

def initialise_logs (model, cf):
    '''
    If `resume_training` is False, then files storing new metric will be
    created. If not, then the model will be loaded with the previous weights.
    '''
    if not cf['resume_training']:
        (pd.DataFrame (columns=cf['all_metrics'])).to_csv (
                cf['save_prefix']+'_metric_train.csv')
        (pd.DataFrame (columns=cf['all_metrics'])).to_csv (
                cf['save_prefix']+'_metric_test.csv')
        (pd.DataFrame (columns=['loss'])).to_csv (cf['save_prefix']+'_loss.csv')
        if os.path.isfile(cf['save_prefix']+'_log'): 
            os.remove (cf['save_prefix']+'_log')
    else: model.load_state_dict(torch.load(cf['save_prefix']+'_model'))
    # load the resnet weights later to prevent the requires_grad attribute from
    # being washed out

    logger = logging.getLogger('log_all') 
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(cf['save_prefix']+'_log'))
    logger.info ("================================================")
    return model, logger

def log_config(filename, cfg):
    with open(filename, 'a') as f:
        f.write(print_config (cfg))
        f.write("\n================================================\n")

def train_test (cg):
    model = choose_models (cg.config)
    if cg.config ['pretrained']: load_net (model, cg.config)
    if not torch.cuda.is_available (): cg.config['device'] = 'cpu'

    loader_arg, data_arg = cg.get_data_args(cg.config)
    train_set = dataloader (cg.config['root'], "train", **loader_arg)
    eval_set = dataloader (cg.config['root'], "train", transformation=None,
            **loader_arg)
    test_set = dataloader (cg.config['root'], "test", **loader_arg)
    train_loader = DataLoader(dataset=train_set, shuffle=True, **data_arg)
    eval_loader = DataLoader(dataset=eval_set, shuffle=False, **data_arg)
    test_loader = DataLoader(dataset=test_set, shuffle=False, **data_arg)

    if torch.cuda.is_available(): model = model.cuda ()
    train_loop (model, cg.config, [train_loader, eval_loader, test_loader])
