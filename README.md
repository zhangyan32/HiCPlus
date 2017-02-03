# HiCPlus
Resolution Enhancement of HiC interaction heatmap.  

## Dependency

* [Python] (https://www.python.org) (2.7) with Numpy and Scipy. We recommand use the  [Anaconda] (https://www.continuum.io) distribution to install Python. 
* [Theano] (https://github.com/Theano/Theano) (0.8.0). GPU acceleration is not required but strongly recommended. CuDNN is also recommended to maximize the GPU performance. 
* [Lasagne] (https://github.com/Lasagne/Lasagne) (0.2.dev1)
* [Nolearn] (https://github.com/dnouri/nolearn) (0.6a0.dev0)



## Installation
No installation is required. Just clone the repo to your local folder. 

```
$ git clone https://github.com/zhangyan32/HiCPlus.git

```

## Usage

### Training
Only high-resolution HiC matrix is neede and you also need to provide the upscaling factor. We will generate the corresponding low-resolution matrix for the training. For example, if you want to enhance the resolution from 40kb to 10kb, you need to provide the 10kb resolution matrix and upscaling factor, which is 4 in this case. The type of the matrix should be in the same format(same normalization method and the cutting enzyme). The training process will generate a model file, which is in bianry form and can only be read in the same environment. 

### Prediction
Load the sample and provide the low-resolution sample. Done. 


## Citation

TODO: Write citation

## License

TODO: Write license
