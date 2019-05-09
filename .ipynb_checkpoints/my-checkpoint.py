"""
Juyoung's data analysis functions
"""

import numpy as np
import matplotlib.pyplot as plt
import pyret.filtertools as ft
from pyret.nonlinearities import Binterp, RBF, Sigmoid
from scipy.stats import sem, pearsonr


def model_r_init(model_names, dataset=["training","val"], n_cell=1):
    # model_r [cell index] [data set] [model_name]
    # argument: model_names, dataset=["training","test"], n_cell=1
    values = np.zeros(len(model_names))
    model_r = [dict() for i in range(n_cell)]

    for i in range(n_cell):
        # initialization 1 - dict + zip(keys, values[0,0,..0])
        # initialization 2 - {key:value for key[0] in ...}
        model_r[i] = {key:dict(zip(model_names, values)) for key in dataset}

    return model_r

def LN_model_summary(trace, stim, nbins=30, n_testset=2000):
    # trace = (cell id, trace)
    # stim  = (frame, space dim)
    # default setting for test datasets
    # -  after training set: test
    # - before training set: test2
    
    assert trace.shape[1] == stim.shape[0], "Data point number mismatch."
    assert n_testset < 0.5*stim.shape[0], "Testset size is more than half of the training set. Too short training data or too large testset. Lower n_testset." # < half of total sampling number
    nbins = np.ceil(nbins)
    nbins = int(nbins)
    print('nbins = ', nbins)
    
    n_cells = trace.shape[0]
    n_subplots = 7
    
    # model_r definition
    
    model_r = model_r_init(["L","LN"], dataset=["training","test","test2"], n_cell=n_cells)
    
    for i in range(n_cells):
        resp = trace[i]
        resp_training = resp[n_testset:-n_testset:]
        stim_training = stim[n_testset:-n_testset:]

        # Linear filter (or RF) by rev. corr.
        rc, lags   = ft.revcorr(stim_training, resp_training, nbins)
        
        # Linear predictionn
        pred_training = ft.linear_response(rc[::-1], stim_training)
        pred_heldout = ft.linear_response(rc[::-1], stim[-n_testset:])
        pred_heldout2 = ft.linear_response(rc[::-1], stim[:n_testset])
        # Nonlinearity fitting
        binterp2 = Binterp(20) # nbins for nonlinearity estimation. Binterp class object.
        binterp2.fit(pred_training, resp_training)
        # LN model output
        LN_output_training_data = binterp2.predict(pred_training)
        LN_output_heldout_data = binterp2.predict(pred_heldout)
        LN_output_heldout_data2 = binterp2.predict(pred_heldout2)

        # Figure setting
        fig = plt.figure(figsize=(18, 2.5))
        ax1 = plt.subplot2grid((1, n_subplots), (0,0), colspan=2)
        ax2 = plt.subplot2grid((1, n_subplots), (0,2), colspan=1)
        ax3 = plt.subplot2grid((1, n_subplots), (0,3), colspan=n_subplots-3)

        # Plot Linear filter
        # cmap='seismic'
        plt.sca(ax1)
        rf_imshow(rc.T, ax=ax1)
        plt.axis('off')

        # model r
        model_r[i]['training']["L"] = pearsonr(resp_training, pred_training)[0]
        model_r[i]['test']["L"]     = pearsonr(resp[-n_testset:], pred_heldout)[0]
        model_r[i]['test2']["L"]    = pearsonr(resp[:n_testset], pred_heldout2)[0]

        model_r[i]['training']["LN"] = pearsonr(resp_training, LN_output_training_data)[0]
        model_r[i]['test']["LN"]     = pearsonr(resp[-n_testset:], LN_output_heldout_data)[0]
        model_r[i]['test2']["LN"]    = pearsonr(resp[:n_testset], LN_output_heldout_data2)[0]
        #
        print('Cell %d: w/ training data (%.3f and %.3f), w/ test data (%.3f and %.3f)' 
              %  (i+1, model_r[i]['training']["L"], model_r[i]['training']["LN"], model_r[i]['test']["L"], model_r[i]['test']["LN"]))

        # Plot 1 : Nonlinearity fitting plot (scatter & binterp)
        model_out = pred_training
        plt.sca(ax2)
        plt.plot(model_out, resp_training, linestyle='none', marker='+', mew=0.5) #mec='w'
        binterp2.plot((model_out.min(),model_out.max()), linewidth=5, label='Binterp') # axes?
        plt.xlabel('Filtered stimulus')
        #plt.ylabel(data_key)
        ax2.set_ylim([-4,4])
        ax2.set_xticklabels([])

        # Plot trace
        LN_output_heldout_data -= np.mean(LN_output_heldout_data, axis=0)
        LN_output_heldout_data /= np.std(LN_output_heldout_data, axis=0)
        #fig = plt.figure()
        #fig, ax = plt.subplots(figsize=(15,3))
        #ax = plt.subplot2grid((n_cells, n_subplots), (i, 2), colspan=n_subplots-2)
        plt.sca(ax3)
        plt.plot(resp[-n_testset:], color='gray')
        plt.plot(LN_output_heldout_data)
        ax3.set_xlim([0, n_testset])
        #plt.title('Heldout data vs LN model output')
        plt.show()
        
    model_r_bar_plot(model_r, dataset=["training","test","test2"], models=["L","LN"])
    #
    return model_r#, LN_result


#__all__ = []

# Function for rev correlation (rolled_stim, output)
def corr_with_rolled_stim(rolled_stim, output):
    # rolled_stim = [N, d1, d2, ...]
    # output = [N, cells]  e.g. ROI trace. truncated.
    # corr  = [cells, d1, d2, ...]
    assert rolled_stim.shape[0] == output.shape[0]
    assert output.ndim == 2
    #
    # corr dim tuple
    d = rolled_stim.shape[1:]
    d_corr = np.insert(np.asarray(d), 0, output.shape[1])
    corr = np.zeros(d_corr)
    #
    for cell in range(output.shape[1]):
        for t in range(output.shape[0]):
            corr[cell] += rolled_stim[t] * output[t, cell]

    return corr

# Function for RF visualization
def rf_imshow(rf_data, **kwargs):
    # [d1, d2]           - Create a figrue or draw on given axis. arg 'climit' for range. 
    # [num_cell, d1, d2] - Create a figure. Always 1-D raw image array.
    assert rf_data.ndim < 4, "Dim >3. Too large dimention for rf visualization"
            
    # single cell case
    if rf_data.ndim is 2: 
        
        rf = rf_data
        
        if 'ax' in kwargs.keys():
            ax = kwargs.pop('ax')
        else:
            ax = plt.axes() # create an axis
        
        if 'climit' in kwargs.keys():
            c_limit = kwargs.pop('climit')
        else:
            c_limit = max([abs(rf.min()), abs(rf.max())])
            
        #plt.subplot(1, numcell, cell+1)
        img = ax.imshow(rf, aspect='auto', cmap='seismic', vmin = -c_limit, vmax = c_limit)
        cbar = plt.colorbar(img, ticks=[-c_limit, c_limit], ax=ax)
        cbar.ax.set_yticklabels(['-', '+'])
        
    else:
        numcell = rf_data.shape[0]
        # image along a row. 
        fig, axes = plt.subplots(1, numcell, figsize=(18, 2.5))   
        
        for cell in range(numcell):
            rf = rf_data[cell]
            ax = axes[cell]
            c_limit = max([abs(rf.min()), abs(rf.max())])
            #plt.subplot(1, numcell, cell+1)
            img = ax.imshow(rf, aspect='auto', cmap='seismic', vmin = -c_limit, vmax = c_limit)
            cbar = plt.colorbar(img, ticks=[-c_limit, c_limit], ax=ax)
            cbar.ax.set_yticklabels(['-', '+'])
            #plt.title('cell: %d' %(cell))


# 4D tensor visualization for pytorch model (e.g. for 2nd layer)
# (out_ch, in_ch, dim1, dim2)
def plot_kernels_in_ch_cols(tensor): 
    num_rows = tensor.shape[0]
    num_cols = tensor.shape[1]
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*1.5)) 
    
    for i in range(num_rows):     # over out channels
        # set limit
        climit = max([abs(tensor[i].min()), abs(tensor[i].max())])

        for j in range(num_cols): # over  in channels
            
            if num_cols == 1 and num_rows == 1:
                # For (1,1) case, ax is not subscriptable.
                ax = axes
            elif num_cols == 1:
                ax = axes[i]
            elif num_rows == 1:
                ax = axes[j]
            else:
                ax = axes[i,j]    

            rf_imshow(tensor[i,j], ax=ax, clmit=climit) # c_limit...!!!!???
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

    
def plot_kernels_out_ch_cols(tensor):
# (out_ch, in_ch, dim1, dim2)
# plot out channels into cols (e.g. 1st layer)
    num_cols = tensor.shape[0]
    num_rows = tensor.shape[1]
    #fig = plt.figure()
    #fig.patch.set_facecolor((1, 1, 1))
    
    fig, axes = plt.subplots(num_rows, num_cols+1, figsize=((num_cols+1)*3, num_rows*1))
    
    for j in range(num_rows): # over in channels
        
        climit = max([abs(tensor[j].min()), abs(tensor[j].max())])
        
        for i in range(num_cols):
            #ax1 = plt.subplot(num_rows, num_cols, j*num_cols + i + 1)
            
            if num_cols == 1 and num_rows == 1:
                # For (1,1) case, ax is not subscriptable.
                ax = axes
            elif num_cols == 1:
                ax = axes[j]
            elif num_rows == 1:
                ax = axes[i]
            else:
                ax = axes[j,i]    
            
            rf_imshow(tensor[i,j], ax=ax, climit=climit)
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
    # temporal profile plot
    

    
def max_slice(img, axis=0):
    # axis 0? Fix axis 0 (e.g. space) and slice it.
    # Find index of abs max pixel.
    
    assert img.ndim is 2, "input dim should be 2."
    assert axis < 2, "axis should be either 0 or 1."
    
    ind = np.unravel_index(np.argmax(abs(img)), img.shape)
    
    if axis is 0:
        line = img[ind[0],:]
        max_ind = ind[1]
    elif axis is 1:
        line = img[:,ind[1]]
        max_ind = ind[0]
    else:
        line = []
        max_ind = []
        
    return line, max_ind      
            

def model_r_bar_plot(model_r, dataset=None, models=None):
    # model_r [cell index] [data set] [model_name]
    # loop over cell id > data set > model type
    # color depends on only model type

    bar_width = 0.48
    margin_dataset = 0.15
    margin_cell = 0.5
    fsize = 20 # fontsize
    n_cell = len(model_r)
    new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    n_color = len(new_colors)
    #
    fig, ax = plt.subplots(figsize=(n_cell*2, 3))

    for i in range(n_cell):

        if dataset is None:
            dataset = model_r[i].keys()
        if not isinstance(dataset, (list,)):
            if i is 1: # print only once
                print('dataset argument should be ''list'', not ''str''. Try with [].')
        n_dataset = len(dataset)
        j = 0 # dataset

        #for dataset in model_r[i].keys():
        for d in dataset:

            if models is None:
                models = model_r[i][d].keys()
            if not isinstance(models, (list,)):
                print('models argument should be ''list'', not ''str''. Try with [].')

            n_model = len(models)
            k = 0 # model index

            for model in models:

                model_spacing = 0.5
                dataset_spacing = n_model * model_spacing  + margin_dataset
                cell_spacing = n_dataset * dataset_spacing + margin_cell

                plt.bar(i*cell_spacing + j*dataset_spacing + k*model_spacing + margin_cell/2. + margin_dataset/2., model_r[i][d][model], bar_width, color=new_colors[k%n_color])
                k += 1
            #
            j += 1
        #
    #print('%s' % dataset) # of last cell
    #print('%s' % models)  # of last cell & last dataset

    ax.set_ylabel('Correlation', fontsize=fsize)
    ybottom, ytop = plt.ylim()
    ax.set_ylim(top = ytop + 0.03)
    #ax.set_xticks([i*spacing + (n_keys+0.5)*width for i in range(n_cell)])
    ax.set_xticks([(i+0.5)*cell_spacing for i in range(n_cell)])
    ax.set_xticklabels(['cell %d' %(i+1) for i in range(n_cell)], horizontalalignment='center')
    #ax.set_xlabel("va=baseline")
    ax.tick_params(axis='both', labelsize=fsize)
    ax.tick_params(axis='x', length=0)
    ax.yaxis.grid()
    plt.title('%s models on %s dataset' % (models, dataset))
    #
    return ax

    #ybottom, ytop = plt.ylim()
    #ax.set_ylim(top = ytop + 0.03)
    #ax.set_xticks([i*spacing + (n_keys+0.5)*width for i in range(n_cell)])
    #ax.set_xticks([(i+0.5)*spacing for i in range(n_cell)])
    #ax.set_xticklabels(['cell %d' %(i+1) for i in range(n_cell)])
    #ax.tick_params(axis='both', labelsize=fsize)
    #ax.tick_params(axis='x', length=0)
    #ax.yaxis.grid()

def figure_setting():
    # Figure setting
    plt.rcParams['figure.figsize'] = [5, 3]
    # plot style test
    print(plt.style.available)
    #plt.style.use(['classic'])
    plt.style.use(['seaborn-white','dark_background'])
    plt.style.use(['dark_background'])
    # plt.style.use('seaborn-dark') # Working in dark theme jupyter lab. Figures for white presentation
    #
    plt.rcParams['font.size'] = 20
    #plt.rcParams['axes.labelsize'] = 40
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    # etc
    #ax2.xaxis.grid()
    #ax2.set_xlim([0,n_data_res])
    #ax2.set_yticklabels([])
