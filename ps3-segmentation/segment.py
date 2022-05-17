import sys
import numpy as np
import skimage.io
from sklearn.mixture import GaussianMixture
from scipy.io import loadmat

from Rgb2Luv import Rgb2Luv

# Set the random seed
np.random.seed(0)


def prettyPrintArray(array):
    """
        Pretty-prints a numpy array

        Based on:
        https://stackoverflow.com/questions/13214809/pretty-print-2d-python-list

        Args
        ----------
        array :        The numpy array that you want to print
    """

    s = [[str(e) for e in row] for row in array]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:>{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

def generate_Y(luvImage:np.ndarray)-> np.ndarray:
    """
    Return the fitted Gaussian Mixture model.
    """
    gmm = GaussianMixture(n_components=5, covariance_type='diag', random_state=0)
    gmm.fit(luvImage)
    return gmm

def 


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: The function %s is called as follows" % sys.argv[0])
        print("")
        print("    %s originalImage.png superpixelMap.mat "
              " scribbleMask.mat beta" % sys.argv[0])
        sys.exit(0)

    rgbImageFile = sys.argv[1]
    superpixelMapFile = sys.argv[2]
    scribbleMaskFile = sys.argv[3]
    beta = sys.argv[4]

    # Load the image and convert to LUV space
    rgbImage = skimage.io.imread(rgbImageFile)
    rgbImage = rgbImage
    luvImage = Rgb2Luv().convert(rgbImage)

    # Load the scribble mask
    scribbleMask = loadmat(scribbleMaskFile)['scribble_mask']
    foregroundMask = scribbleMask == 1
    backgroundMask = scribbleMask == 2

    foreground = luvImage[foregroundMask]
    background = luvImage[backgroundMask]

    # Fit the Gaussian Mixture
    f_gmm = generate_Y(foreground)
    b_gmm = generate_Y(background)

    print(f_gmm.means_())
