import matplotlib
matplotlib.use('Agg')

import sys
import matplotlib.pyplot as plt

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import numpy as np
import theano.tensor as T
from nolearn.lasagne import BatchIterator
from theano.sandbox.neighbours import neibs2images
from lasagne.objectives import squared_error
from lasagne.updates import sgd
from scipy.stats import poisson

from lasagne.nonlinearities import tanh
import pickle
import sys
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import precision_score
import os
import urllib
import gzip
import cPickle
import sys
sys.setrecursionlimit(10000)
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr

import math

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

index = np.load(path + '/index18_18_GM12878_replicate.npy')

size = index[0][1]
predictedMatrix = np.zeros((size, size))
print predictedMatrix.shape
print y_predict.shape

for i in range(0, y_predict.shape[0]):
    x = int(index[i+1][0])
    y = int(index[i+1][1])
    # 6 = half_padding 34 = sample_size - half_padding
    predictedMatrix[x+6:x+34, y+6:y+34] = y_predict[i]
np.save(path + '/predicted_matrix', predictedMatrix.astype(np.float32))


