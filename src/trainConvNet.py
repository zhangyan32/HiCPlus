# Author: Yan Zhang  
# Email: zhangyan.cse (@) gmail.com

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import gzip

import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet
from lasagne.updates import sgd

sys.setrecursionlimit(10000)
conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5

down_sample_ratio = 16
learning_rate = 0.00001
epochs = 10
HiC_max_value = 100

# Read the input sample pairs. 
# The shape of the samples should be (N, 1, n, n), where N is the number of the samples, and n is the size of the samples. Both of the samples should have exact the same size. 
low_resolution_samples = np.load(gzip.GzipFile('../data/GM12878_replicate_down16_chr19_22.npy.gz', "r")) * down_sample_ratio
high_resolution_samples = np.load(gzip.GzipFile('../data/GM12878_replicate_original_chr19_22.npy.gz', "r")) 

low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)

# Reshape the high-quality Hi-C sample as the target value of the training. 
sample_size = low_resolution_samples.shape[-1]
padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
half_padding = padding / 2
output_length = sample_size - padding
Y = []
for i in range(high_resolution_samples.shape[0]):
    no_padding_sample = high_resolution_samples[i][0][half_padding:(sample_size-half_padding) , half_padding:(sample_size - half_padding)]
    Y.append(no_padding_sample)
Y = np.array(Y).astype(np.float32)
Y = Y.reshape((Y.shape[0], -1))

X = low_resolution_samples

net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv2d1', layers.Conv2DLayer),
        ('conv2d2', layers.Conv2DLayer),
        ('conv2d3', layers.Conv2DLayer),
        ('output_layer', layers.FlattenLayer),
        ],
    input_shape=(None, 1, sample_size, sample_size),
    conv2d1_num_filters=conv2d1_filters_numbers, 
    conv2d1_filter_size = (conv2d1_filters_size, conv2d1_filters_size),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.GlorotUniform(),  
    conv2d2_num_filters=conv2d2_filters_numbers, 
    conv2d2_filter_size = (conv2d2_filters_size, conv2d2_filters_size), 
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,   
    conv2d3_num_filters=conv2d3_filters_numbers, 
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d3_filter_size = (conv2d3_filters_size, conv2d3_filters_size),
    update=sgd,       
    update_learning_rate = learning_rate,
    regression=True,
    max_epochs= epochs,
    verbose=1,
    )

net1.fit(X, Y)

output_model_name = '../model/test_model'

f = file(output_model_name + '.net', 'wb')

pickle.dump(net1,f,protocol=pickle.HIGHEST_PROTOCOL)
f.close()

