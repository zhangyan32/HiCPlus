# Author: Yan Zhang  
# Email: zhangyan.cse (@) gmail.com

import sys
import os
import pickle
import urllib
import gzip
import cPickle
import math
sys.setrecursionlimit(10000)

import numpy as np

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
import theano.tensor as T
import lasagne
from lasagne import layers
from lasagne.updates import sgd

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

path = '/home/zhangyan/temptesttemptest'
input_model_name = path + '/model.net'

f = open(input_model_name)
net1 = pickle.load(f)

X_train = np.load(path + '/bicubic_x4_chr19_20.npy')
X_test = np.load(path + '/bicubic_x4_chr18_18_GM12878_replicate.npy')

Seq_depth_corr = (np.sum(X_test)/X_test.shape[0])/float(np.sum(X_train)/X_train.shape[0])
X_test = X_test * Seq_depth_corr

y_predict = net1.predict(X_test)

length = math.sqrt(y_predict.shape[1])
y_predict = np.reshape(y_predict, (y_predict.shape[0], length, length))

# load the index for recombine the samples to the HiC map
index = np.load(path + '/index18_18_GM12878_replicate.npy')
size = index[0][1]
predictedMatrix = np.zeros((size, size))

# recombine the samples to the entire HiC interaction map
for i in range(0, y_predict.shape[0]):
    x = int(index[i+1][0])
    y = int(index[i+1][1])
    # 6 = half_padding 34 = sample_size - half_padding
    predictedMatrix[x+6:x+34, y+6:y+34] = y_predict[i]
np.save(path + '/predicted_matrix', predictedMatrix.astype(np.float32))


