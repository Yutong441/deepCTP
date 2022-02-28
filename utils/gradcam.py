import os
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from medcam import medcam
from data_raw.CTP import load_nib 

def save_gradcam (model, loader_set, cfg, save_dir=None, mode='test'):
    if save_dir is None:
        save_dir = cfg ['save_prefix'] + '_gradcam'
        if not os.path.exists (save_dir): os.mkdir (save_dir)

    model.eval ()
    ytrue = loader_set.annotations [cfg ['outcome_col']].values
    ypred = np.load (cfg['save_prefix']+'_tmp/ypred_{}.npz'.format (
        mode)) ['arr_0']

    model = medcam.inject(model, backend="gcam", layer=cfg['gcam'],
            save_maps=False, label='best', replace=True)
    torch.backends.cudnn.enabled=False
    for i in range (loader_set.__len__ () ):
        img, lab = loader_set.__getitem__ (i)
        act = model (img.unsqueeze (0).to(cfg['device']))
        save_obj = {'map': act.detach().cpu().numpy(),
                'ypred': float (ypred[i]), 'ytrue': float (ytrue[i])}
        filename = loader_set.annotations.index[i]
        np.savez_compressed (save_dir+'/cam_'+filename, save_obj)

def central_slice (img, depths=21):
    start_depth = (img.shape[2] - depths)//2
    return img [:,:, start_depth:(start_depth + depths)]

class IndexTracker:
    def __init__(self, ax, X, Y, cmap_type='jet', title=None):
        self.ax = ax
        if title is not None: ax.set_title (title)
        self.X, self.Y = X, Y
        rows, cols, self.slices = Y.shape
        self.ind = self.slices//2
        self.im1 = ax.imshow(self.X[:, :, self.ind], cmap=cmap_type)
        self.im2 = ax.imshow(self.Y[:, :, self.ind], cmap='viridis', alpha=0.5)
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up': self.ind = (self.ind + 1) % self.slices
        else: self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im1.set_data(self.X[:, :, self.ind])
        self.im2.set_data(self.Y[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im1.axes.figure.canvas.draw()
        self.im2.axes.figure.canvas.draw()

def multi_plot3D (img, overlay, ncols=4, channel_name=None):
    '''
    Args:
        `img`: [H, W, D, C]
        `overlay`: [H, W, D]
    '''
    if channel_name is None:
        maps = ['MIP', 'CBFA', 'CBVA', 'TTP', 'TTDA', 'MTTA', 'PMBA']
    chan = img.shape[-1]
    nrows = int (np.ceil (chan/ncols))
    fig, ax = plt.subplots(nrows, ncols, squeeze=False)
    [one_ax.axes.xaxis.set_ticks ([]) for one_ax in ax.ravel()]
    [one_ax.axes.yaxis.set_ticks ([]) for one_ax in ax.ravel()]

    tracker_list = []
    for i in range (chan):
        cmap_type = 'jet' if maps[i] != 'MIP' else 'gray'
        tracker = IndexTracker(ax[i//ncols,i%ncols], 
                img[...,i], overlay, cmap_type=cmap_type, title=maps[i])
        fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
        #tracker_list.append (tracker)

class MultiIndexTracker:
    def __init__(self, ax, X, Y, maps, ncols=4, title=None):
        [one_ax.axes.xaxis.set_ticks ([]) for one_ax in ax.ravel()]
        [one_ax.axes.yaxis.set_ticks ([]) for one_ax in ax.ravel()]
        [one_ax.spines['top'].set_visible(False) for one_ax in ax.ravel()]
        [one_ax.spines['bottom'].set_visible(False) for one_ax in ax.ravel()]
        [one_ax.spines['right'].set_visible(False) for one_ax in ax.ravel()]
        [one_ax.spines['left'].set_visible(False) for one_ax in ax.ravel()]
        self.ax = ax

        self.X, self.Y = X, Y
        rows, cols, self.slices = Y.shape
        self.ind = self.slices//2
        self.list1, self.list2 = [], []
        self.ncols = ncols
        self.maps = maps

        for i in range (X.shape [-1]):
            cmap_type = 'jet' if maps[i] != 'MIP' else 'gray'
            cmap_type = mpl.cm.get_cmap(cmap_type).copy()
            cmap_type.set_under(color='black') 

            vmax = X [...,i].max()
            self.ax [i//ncols, i%ncols].set_title (maps[i])
            self.list1.append (self.ax [i//ncols, i%ncols].imshow(
                    self.X[:, :, self.ind, i], cmap=cmap_type, vmin=1e-2,
                    vmax=vmax))
            if maps [i] == 'MIP':
                self.list2.append (self.ax [i//ncols, i%ncols].imshow(
                        self.Y[:, :, self.ind], cmap='viridis', alpha=0.5))
            else: self.list2.append (None)
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up': self.ind = (self.ind + 1) % self.slices
        else: self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        ncols = self.ncols
        for i in range (self.X.shape[-1]):
            self.list1[i].set_data(self.X[:, :, self.ind, i])
            self.ax [i//ncols, i%ncols].set_ylabel('slice %s' % self.ind)
            self.list1[i].axes.figure.canvas.draw()
            if self.maps [i] == 'MIP':
                self.list2[i].set_data(self.Y[:, :, self.ind])
                self.list2[i].axes.figure.canvas.draw()

def load_plot_3D (sample_num='001', img_path='data/CTP/CTP_pro/CTP', 
        overlay_path= 'results/CTP_gradcam/cam_CTP', ncols=4,
        channel_name=None):
    obj = np.load (overlay_path+sample_num+'.npz', allow_pickle=True)
    plot_title = 'predicted: {}, truth: {}'.format (
          np.round (obj['arr_0'].item()['ypred'], 1), 
          obj['arr_0'].item()['ytrue'])
    act = obj['arr_0'].item()['map'].squeeze()
    act = np.nansum (act, axis=1)
    act = np.moveaxis (act, 0, 2)
    img = central_slice (load_nib (img_path+sample_num+'.nii.gz'))

    chan = img.shape[-1]
    nrows = int (np.ceil (chan/ncols))
    fig, ax = plt.subplots(nrows, ncols, squeeze=False)
    if channel_name is None:
        maps = ['MIP', 'CBFA', 'CBVA', 'TTP', 'TTDA', 'MTTA', 'PMBA']
    tracker = MultiIndexTracker(ax, img, act, maps)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.suptitle (plot_title)
    plt.show ()

load_plot_3D ('063')
plt.show ()
