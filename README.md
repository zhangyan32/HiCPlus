# HiCPlus
Resolution Enhancement of HiC interaction heatmap. The repo is still under construction. 

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
Only high-resolution HiC matrix is needed, and you also need to provide the upscaling factor. We will generate the corresponding low-resolution matrix for the training. For example, if you want to enhance the resolution from 40kb to 10kb, you need to provide the 10kb resolution matrix and upscaling factor, which is 4 in this case. The type of the matrix should be in the same format(same normalization method and the cutting enzyme). The training process will generate a model file, which is in binary form and can only be read in the same environment. 

### Prediction
Load the sample and provide the low-resolution sample. Done. 

### Processing samples
We provide the script src/genSample.py to generate samples for both training and testing. In the training sets, we use the high-resolution experimental HiC map to create the low-resolution map and interpolated map. In the testing sets, you may provide your only low-resolution map with interpolation. 

### About the normalization and experimental condition
HiC experiments have several different types of cutting enzyme as well as different normalization method. Our model can handle all of the conditions as long as the training and testing are under the same condition. For example, if the KR normalized samples are used in the training stage, the trained model only works for the KR normalized low-resolution sample. 

## Citation

http://biorxiv.org/content/early/2017/03/01/112631

## License


