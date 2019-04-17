"""
Juyoung's models
"""
import torch
import torch.nn as nn
import time
import my
import matplotlib.pyplot as plt

# models
class SimpleModel(nn.Module):
    def __init__(self, D_stim, D_out):
        # D_stim : [ch, dim1, dim2] e.g. [color, space, time]
        # D_out: # of cells (or ROIs)
        super(SimpleModel, self).__init__()
        self.n_cell = D_out
        self.conv1 = nn.Conv2d(D_stim[0], D_out,  kernel_size = D_stim[1:])

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)    # x.size(0) = batch number
        #x = x.view(-1, D_out)        # [batch #, output cell#]
        #return torch.tanh(x) # For RELU, self.conv1(x).clamp(min=0) For SELU, nn.functional.selu(x)
        #x = nn.functional.softplus(x)
        x = torch.tanh(x)
        # Additional conv for temporal kinetics of Ca indicator. No linear combination over channels.
        return x
    
    def reg_conv1_L1(self):
        # Define regularization term for this model..
        return self.conv1.weight.abs().sum()
    
    def visualize(self):
        fig = plt.figure(figsize=(3*self.n_cell, 2)) 
        w_conv1 = self.conv1.weight.data.cpu().numpy()
        my.plot_kernels_out_ch_cols(w_conv1)


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
        self.name = 'LN_TemConv'
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
# 2-layer model: Conv1 + Conv2 (= FC)
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
        self.n_cell = D_out
        self.num_types = H
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
    
    def reg_conv1_L1(self):
        # Define regularization term for this model..
        return self.conv1.weight.abs().sum()
    
    def reg_conv2_L1(self):
        # Define regularization term for this model..
        return self.conv2.weight.abs().sum()
    
    def visualize(self):
        fig = plt.figure(figsize=(4*self.num_types, 1)) 
        w_conv1 = self.conv1.weight.data.cpu().numpy()
        my.plot_kernels_out_ch_cols(w_conv1)
        
        #plt.title('L1 reg %.1e,   L2 reg %.1e' % (coeff_L1, coeff_L2))
        fig = plt.figure(figsize=(4*self.num_types, 2*self.n_cell)) 
        w_conv2 = self.conv2.weight.data.cpu().numpy()
        my.plot_kernels_in_ch_cols(w_conv2)

