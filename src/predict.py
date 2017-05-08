# Author: Yan Zhang  
# Email: zhangyan.cse (@) gmail.com

import sys
import os
import pickle
import gzip
import numpy as np

sys.setrecursionlimit(10000)

input_model_name = '../model/test_model.net'

f = open(input_model_name)
net1 = pickle.load(f)
down_sample_ratio = 16
low_resolution_samples = np.load(gzip.GzipFile('../data/GM12878_replicate_down16_chr17_17.npy.gz', "r")) * down_sample_ratio

enhanced_low_resolution_samples = net1.predict(low_resolution_samples)
np.save('../data/enhanced_GM12878_replicate_down16_chr17_17.npy', enhanced_low_resolution_samples)

# The output samples shape is (N, 1, n, n), where N is the number of the sample, and n is the sample size. The user should recombine the enhanced Hi-C samples to the entire Hi-C based on the rules which divided the sample. 

