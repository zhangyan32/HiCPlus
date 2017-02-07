# Author: Yan Zhang  
# Email: zhangyan.cse (@) gmail.com

import sys
import os
import urllib
import gzip
import cPickle
import pickle
sys.setrecursionlimit(10000)

import numpy as np
import theano.tensor as T

import lasagne
from lasagne import layers
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.updates import sgd

#hyperparameters in the ConvNet.
learning_rate = 0.00001
conv2d1_filters_numbers = 16
conv2d1_filters_size = 9
conv2d2_filters_numbers = 16
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5

Resume = False
resumed_model = ''
use_ChIA_PET = False

path = '/home/zhangyan/temptesttemptest'
bicubic_HiC = np.load(path + '/bicubic_x4_chr19_20.npy')

output_model_name = path + '/model.net'

y_train_full = np.load(path + '/experimentalHiCRes_chr19_20.npy')
y_train_full = np.minimum(100, y_train_full)

sample_num = bicubic_HiC.shape[0]

if (use_ChIA_PET):
    X_train = np.zeros((sample_num,3,40,40)).astype(np.float32)
    X_train[:,0:1,:,:] = bicubic_HiC
    X_train[:,1:2,:,:] = chip_pet
    X_train[:,2:3,:,:] = chip_pet2
else:
    X_train = bicubic_HiC




y_train = []
padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3
half_padding = padding / 2

output_length = 40 - padding
for i in range(y_train_full.shape[0]):
    no_padding_sample = y_train_full[i][0][half_padding:(40-half_padding) , half_padding:(40 - half_padding)]
    y_train.append(no_padding_sample)
y_train = np.array(y_train).astype(np.float32)
 
# we need our target to be 1 dimensional
Y_out = y_train.reshape((y_train.shape[0], -1))

# set to a large number and Ctrl+C can break the training and save the model
epochs = 300000

if (not Resume):
    net1 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('output_layer', layers.FlattenLayer),
            ],
        input_shape=(None, 1, 40, 40),
        conv2d1_num_filters=conv2d1_filters_numbers, 
        conv2d1_filter_size = (conv2d1_filters_size, conv2d1_filters_size),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,#
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
else:
    f = open(resumed_model)
    net1 = pickle.load(f)

net1.fit(X_train, Y_out)

f = file(output_model_name, 'wb')
pickle.dump(net1,f,protocol=pickle.HIGHEST_PROTOCOL)
f.close()

