
import numpy as np
import scipy.stats as st

# Generate the 2D Gaussian Kernel
# kernlen, size of the kernal
# nsig, sigma in Gaussian distribution. 
# the code is partially from code.google.com/p/iterative-fusion
def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

# Run the Gaussian smoothing on Hi-C matrix
# matrix is a numpy matrix form of the Hi-C interaction heatmap
def Gaussian_filter(matrix, sigma=4, size=13):
    result = np.zeros(matrix.shape)
    padding = size / 2
    kernel = gkern(13, nsig=sigma)
    for i in range(padding, matrix.shape[0] - padding):
        for j in range(padding, matrix.shape[0] - padding):
            result[i][j] = np.sum(matrix[i - padding : i + padding + 1, j - padding : j + padding + 1] * kernel)
    return result
