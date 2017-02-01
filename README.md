# HiCPlus: Resolution Enhancement of HiC interaction heatmap 

## Dependency

* [Python] (https://www.python.org) (2.7.10). The easiest way to install Python and all of the necessary dependencies is to download and install [Anaconda] (https://www.continuum.io) (2.3.0). I listed the versions of Python and Anaconda I used, but the latest versions should be fine. If you're curious as to what packages in Anaconda are used, they are: [numpy] (http://www.numpy.org/) (1.10.1), [scipy] (http://www.scipy.org/) (0.16.0), and [h5py] (http://www.h5py.org) (2.5.0). 
* [Theano] (https://github.com/Theano/Theano) (latest). At the time I wrote this, Theano 0.7.0 is already included in Anaconda. However, it is missing some crucial helper functions. You need to git clone the latest bleeding edge version since there isn't a version number for it:

```
$ git clone git://github.com/Theano/Theano.git
$ cd Theano
$ python setup.py develop
```

## Installatsdfn

TODO: Describe the installation process

## Usage

TODO: Write usage instructions

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History

TODO: Write history

## Credits

TODO: Write credits

## License

TODO: Write license
