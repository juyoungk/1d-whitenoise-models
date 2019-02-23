"""
Juyoung's data analysis functions
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def model_r_init(model_names, dataset=["training","test"], n_cell=1):
# argument: model_names, dataset=["training","test"], n_cell=1
    values = np.zeros(len(model_names))
    model_r = [dict() for i in range(n_cell)]

    for i in range(n_cell):
        # initialization 1 - dict + zip(keys, values[0,0,..0])
        # initialization 2 - {key:value for key[0] in ...}
        model_r[i] = {key:dict(zip(model_names, values)) for key in dataset}
        
    return model_r

# models
class SimpleModel(nn.Module):
    def __init__(self, D_stim, D_out):
        # D_stim : [ch, dim1, dim2] e.g. [color, space, time]
        # D_out: # of cells (or ROIs)
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(D_stim[0], D_out,  kernel_size = D_stim[1:])
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)    # x.size(0) = batch number
        #x = x.view(-1, D_out)        # [batch #, output cell#]
        #return torch.tanh(x) # For RELU, self.conv1(x).clamp(min=0) For SELU, nn.functional.selu(x)
        #x = nn.functional.softplus(x)
        x = torch.tanh(x)
        # Additional conv for temporal kinetics of Ca indicator. No linear combination over channels.
        return x

class LN_TemConv(nn.Module):
# 2-layer model: Conv1 (over entire space) + Conv2(= FC)
    def __init__(self, D_stim, H, D_out, temp_filter_size = 15):
        # D_stim : [ch, dim1, dim2] e.g. [color, space, time]
        #     H  : num of layer1 channels
        # D_out  : num of cells (or ROIs)
        
        max_space_filtering    = D_stim[1] # conv over entire space
        k1 = [max_space_filtering, temp_filter_size] # subunit spatiotemporal filter. # [space, time] ~ [40*7 um, (1/15Hz)*6=400 ms]
        #k2 = [D_stim[1]-max_space_filtering+1, temp_filter_size] # filter for integrating subunits.
        k2 = [D_stim[1]-max_space_filtering+1, D_stim[2]-temp_filter_size+1] # filter for integrating subunits.

        super(LN_TemConv, self).__init__()
        self.relu = nn.ReLU(inplace=True) # inplace=True: update the input directly. 
        self.softplus = nn.Softplus()
        self.conv1 = nn.Conv2d(D_stim[0], H, kernel_size = k1)
        self.conv2 = nn.Conv2d(H,     D_out, kernel_size = k2) # equivalent to FC layer.
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.softplus(x)   # rectifying nonlinearity.
        x = self.conv2(x)      # Temporal convolution.
        # x = (batch, ch, dim1, dim2)
        assert x.size(2) == 1 # Final dim1 (space) convolution should integrate all subunits.
        assert x.size(3) == 1 # Final dim1 (space) convolution should integrate all subunits.
        x = x.view(x.size(0), -1)
        x = torch.tanh(x)     # Final nonlinearity
        return x

class CNN_2layer(nn.Module):
# 2-layer model: Conv1 + Conv2(= FC)
    def __init__(self, D_stim, H, D_out, temp_filter_size = 15, space_filter_size = 7, space_stride=1):
        # D_stim : [ch, dim1, dim2] e.g. [color, space, time]
        #     H  : num of channels (types in conv1 layer) 
        # D_out  : num of cells (or ROIs)
        
        max_space_filtering    = space_filter_size;
        max_temporal_filtering = temp_filter_size;
        # filter size as tuple
        k1 = (max_space_filtering, max_temporal_filtering) # subunit spatiotemporal filter. # [space, time] ~ [40*7 um, (1/15Hz)*6=400 ms]
        #k2 = [D_stim[1]-max_space_filtering+1, max_temporal_filtering] # filter for integrating subunits.
        conv1_output_space = int((D_stim[1]-max_space_filtering)/space_stride+1)
        k2 = (conv1_output_space, D_stim[2]-max_temporal_filtering+1) # filter for integrating subunits.
        #
        assert k2[0]%1 == 0, "Non-integer filter size probably due to the stride."

        super(CNN_2layer, self).__init__()
        self.name = 'CNN_2layer'
        self.relu = nn.ReLU(inplace=True) # inplace=True: update the input directly. 
        self.softplus = nn.Softplus()
        self.conv1 = nn.Conv2d(D_stim[0], H, k1, stride = (space_stride, 1))
        self.conv2 = nn.Conv2d(H,     D_out, k2, stride = 1) # equivalent to FC layer.
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.softplus(x)     # rectifying nonlinearity.
        x = self.conv2(x)    # saturating nonlinearity.
        # x = (batch, ch, dim1, dim2)
        assert x.size(2) == 1 # Final dim1 (space) convolution should integrate all subunits.
        assert x.size(3) == 1 # Final dim1 (space) convolution should integrate all subunits.
        x = x.view(x.size(0), -1)
        x = torch.tanh(x)
        return x
    
    
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
def rf_imshow(rf_data):
    # [cell id, dim1, dim2, ...]
    
    if rf_data.ndim is 2:
        # single cell case
        rf_data = rf_data[None, :]
        
    numcell = rf_data.shape[0]
    for cell in range(numcell):
        rf = rf_data[cell]
        c_limit = max([abs(rf.min()), abs(rf.max())])  
        #plt.subplot(1, numcell, cell+1)
        plt.imshow(rf, aspect='auto', cmap='seismic', vmin = -c_limit, vmax = c_limit)
        cbar = plt.colorbar(ticks=[-c_limit, c_limit])
        cbar.ax.set_yticklabels(['-', '+']) 
        #plt.title('cell: %d' %(cell))
        
        
# 4D tensor visualization for pytorch model
# (out_ch, in_ch, dim1, dim2)
def plot_kernels_in_ch_cols(tensor):
    num_rows = tensor.shape[0]
    num_cols = tensor.shape[1]    
    fig = plt.figure()
    #fig.patch.set_facecolor((1, 1, 1))
    for i in range(num_rows):     # over out channels 
        # set limit
        c_limit = max([abs(tensor[i].min()), abs(tensor[i].max())])  
        
        for j in range(num_cols): # over  in channels
            ax1 = plt.subplot(num_rows, num_cols, i*num_cols + j + 1)
            rf_imshow(tensor[i,j])
            #ax1.imshow(tensor[i,j])
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show() 
    
def plot_kernels_out_ch_cols(tensor):
# (out_ch, in_ch, dim1, dim2)
# plot out channels into cols
    num_cols = tensor.shape[0]
    num_rows = tensor.shape[1]
    fig = plt.figure()
    #fig.patch.set_facecolor((1, 1, 1))
    for j in range(num_rows): # over in channels
        for i in range(num_cols):
            ax1 = plt.subplot(num_rows, num_cols, j*num_cols + i + 1)
            rf_imshow(tensor[i,j])
            #ax1.imshow(tensor[i,j])
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])    
                                 
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
            print('dataset argument should be ''list'', not ''str''. Try with [].')
        n_dataset = len(dataset)
        j = 0 # dataset 
        
        #for dataset in model_r[i].keys():
        for d in dataset:

            if models is None:
                models = model_r[i][d].keys()
            if not isinstance(models, (list,)):
                print('models argument should be ''list'', not ''str''. Try with [].')
            n_model   = len(models)
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
    print('%s' % dataset) # of last cell
    print('%s' % models)  # of last cell & last dataset
    
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
    plt.title(tuple(models))
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