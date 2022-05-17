import sys
from matplotlib import pyplot as plt
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

def generate_Y(luvImage:np.ndarray)-> GaussianMixture:
    """
    Return the fitted Gaussian Mixture model.
    """
    gmm = GaussianMixture(n_components=5, covariance_type='diag', random_state=0)
    gmm.fit(luvImage)
    return gmm

def generate_adjacency_matrix(SPM:np.ndarray, plot=False) -> np.ndarray:
    """
    Input: superPixelMapFile
    Output: adjacency matrix
    Plot: overlayed markov graph
    """
    # SPM =  loadmat(superPixelMapFile)['labels']
    # print(loadmat(superPixelMapFile).keys())
    # print(loadmat(superPixelMapFile)['modes'].shape)
    
    label_num = SPM.max() + 1
    adj_matrix = np.eye(label_num)
    # scan (avoid dup writing by more reading to improve efficiency)
    for r in range(SPM.shape[0]):
        for c in range(SPM.shape[1]):
            if r < SPM.shape[0] - 2:
                if SPM[r, c] != SPM[r + 1, c] and adj_matrix[SPM[r, c], SPM[r + 1, c]] == 0:
                    adj_matrix[SPM[r, c], SPM[r + 1, c]] = 1
                    adj_matrix[SPM[r + 1, c], SPM[r, c]] = 1
            if c < SPM.shape[1] - 2:
                if SPM[r, c] != SPM[r, c+1] and adj_matrix[SPM[r, c], SPM[r, c+1]] == 0:
                    adj_matrix[SPM[r, c], SPM[r, c+1]] = 1
                    adj_matrix[SPM[r, c+1], SPM[r, c]] = 1
    if not plot:
        return adj_matrix
    
    # plot overlayed adj matrix
    centers = np.zeros([label_num, 2])
    _util_x_vec, _util_y_vec = np.arange(SPM.shape[1]), np.arange(SPM.shape[0])
    util_matrix_x, util_matrix_y = np.meshgrid(_util_x_vec, _util_y_vec)
    # print(_util_x_vec.max(), _util_y_vec.max())
    # print(_util_x_vec, util_matrix_x, SPM == 0, util_matrix_x[SPM == 0], SPM)
    for label in range(label_num):
        centers[label, 0] = util_matrix_x[SPM == label].mean()
        centers[label, 1] = util_matrix_y[SPM == label].mean()

    # print(centers.max(axis=0))
    # plt.imshow(SPM)
    for r in range(label_num):
    # for r in range(4):
        for c in range(r):
            if adj_matrix[r,c]:
                # print(f"{r} {[centers[r, 0], centers[c, 0]],[centers[r, 1], centers[c, 1]]}")
                plt.plot([centers[r, 0], centers[c, 0]],[centers[r, 1], centers[c, 1]] )
    # plt.show()
    plt.savefig("q4-b.png")

def node_potential(gmm:GaussianMixture, y: np.ndarray)->np.ndarray:
    """
    Input: Gaussian Mixture Model, multiple Y_i for the super pixel.
    Output: multiple likelihood
    """
    log_likelihood = gmm.score_samples(y)
    likelihood = np.exp(log_likelihood)
    return log_likelihood, likelihood

def edge_potential(beta:int=2, ):
    pass

def potentials():
    pass



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

    # Load the superpixels
    SPM =  loadmat(superpixelMapFile)['labels']

    # Fit the Gaussian Mixture
    f_gmm = generate_Y(foreground)
    b_gmm = generate_Y(background)

    print(f"foreground cov diag {f_gmm.covariances_.diagonal()}")
    prettyPrintArray(f_gmm.means_)
    print(f"background cov diag {b_gmm.covariances_.diagonal()}")
    prettyPrintArray(b_gmm.means_)


    # superpxiel
    plt.imshow(rgbImage)
    adj_m = generate_adjacency_matrix(SPM, plot=True)


    # 