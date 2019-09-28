# HiCPlus
Resolution Enhancement of HiC interaction heatmap. 
## Deprecation warning
Since Theano is no long deveopled, as well as to avoid too many dependencies, we impletemented the Pytorch version 
https://github.com/wangjuan001/hicplus
Currently, this repo is no longer maintained. 


## Dependency

* [Python] (https://www.python.org) (2.7) with Numpy and Scipy. We recommand use the  [Anaconda] (https://www.continuum.io) distribution to install Python. 
* [Theano] (https://github.com/Theano/Theano) (0.8.0). GPU acceleration is not required but strongly recommended. CuDNN is also recommended to maximize the GPU performance. 
* [Lasagne] (https://github.com/Lasagne/Lasagne) (0.2.dev1)
* [Nolearn] (https://github.com/dnouri/nolearn) (0.6a0.dev0)



## Installation
Just clone the repo to your local folder. 

```
$ git clone https://github.com/zhangyan32/HiCPlus.git

```

## Usage

### Training
In the training stage, both high-resolution Hi-C samples and low-resolution Hi-C samples are needed. Two samples should be in the same shape as (N, 1, n, n), where N is the number of the samples, and n is the size of the samples. The sample index of the sample should be from the sample genomic location in two input data sets. 

### Prediction
Only low-resolution Hi-C samples are needed. The shape of the samples should be the same with the training stage. The prediction generates the enhanced Hi-C data, and the user should recombine the output to obtain the entire Hi-C matrix. 

### Suggested way to generate samples
We suggest that generate a file containing the location of each samples when generate the samples with n x n size. Therefore, after obtaining the high-resolution Hi-C, it is easy to recombine all of the samples to obtain high-resolution Hi-C matrix. 

### Normalization and experimental condition
Hi-C experiments have several different types of cutting enzyme as well as different normalization method. Our model can handle all of the conditions as long as the training and testing are under the same condition. For example, if the KR normalized samples are used in the training stage, the trained model only works for the KR normalized low-resolution sample. 

## Citation

http://biorxiv.org/content/early/2017/03/01/112631




