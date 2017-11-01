# Author: Yan Zhang  
# Email: zhangyan.cse (@) gmail.com

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import gzip
import model
from torch.utils import data
import torch
import torch.optim as optim
from torch.autograd import Variable
from time import gmtime, strftime
import sys
import torch.nn as nn

use_gpu = 1

conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5


down_sample_ratio = 16
epochs = 10
HiC_max_value = 100



# This block is the actual training data used in the training. The training data is too large to put on Github, so only toy data is used. 
# cell = "GM12878_replicate"
# chrN_range1 = '1_8'
# chrN_range = '1_8'

# low_resolution_samples = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/'+cell+'down16_chr'+chrN_range+'.npy.gz', "r")).astype(np.float32) * down_sample_ratio
# high_resolution_samples = np.load(gzip.GzipFile('/home/zhangyan/SRHiC_samples/original10k/'+cell+'_original_chr'+chrN_range+'.npy.gz', "r")).astype(np.float32)

# low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
# high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)


low_resolution_samples = np.load(gzip.GzipFile('../../data/GM12878_replicate_down16_chr19_22.npy.gz', "r")).astype(np.float32) * down_sample_ratio

low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)

batch_size = low_resolution_samples.shape[0]

# Reshape the high-quality Hi-C sample as the target value of the training. 
sample_size = low_resolution_samples.shape[-1]
padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
half_padding = padding / 2
output_length = sample_size - padding


print low_resolution_samples.shape

lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=batch_size, shuffle=False)

production = False
try:
    high_resolution_samples = np.load(gzip.GzipFile('../../data/GM12878_replicate_original_chr19_22.npy.gz', "r")).astype(np.float32)
    high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)
    Y = []
    for i in range(high_resolution_samples.shape[0]):
        no_padding_sample = high_resolution_samples[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
        Y.append(no_padding_sample)
    Y = np.array(Y).astype(np.float32)
    hires_set = data.TensorDataset(torch.from_numpy(Y), torch.from_numpy(np.zeros(Y.shape[0])))
    hires_loader = torch.utils.data.DataLoader(hires_set, batch_size=batch_size, shuffle=False)
except:
    production = True
    hires_loader = lowres_loader

Net = model.Net(40, 28)
Net.load_state_dict(torch.load('../model/pytorch_model_12000'))
if use_gpu:
    Net = Net.cuda()

_loss = nn.MSELoss()


running_loss = 0.0
running_loss_validate = 0.0
reg_loss = 0.0


for i, (v1, v2) in enumerate(zip(lowres_loader, hires_loader)):    
    _lowRes, _ = v1
    _highRes, _ = v2
    

    _lowRes = Variable(_lowRes)
    _highRes = Variable(_highRes)

    
    if use_gpu:
        _lowRes = _lowRes.cuda()
        _highRes = _highRes.cuda()
    y_prediction = Net(_lowRes)
    if (not production):
        loss = _loss(y_prediction, _highRes) 


    running_loss += loss.data[0]
    
print '-------', i, running_loss, strftime("%Y-%m-%d %H:%M:%S", gmtime())


y_prediction = y_prediction.data.cpu().numpy()

print y_prediction.shape






