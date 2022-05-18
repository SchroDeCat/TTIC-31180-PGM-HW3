import sys
from tracemalloc import stop
from matplotlib import pyplot as plt
import numpy as np
import skimage.io
from sklearn.covariance import log_likelihood
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

    return adj_matrix

def node_potential(gmm:GaussianMixture, y: np.ndarray)->np.ndarray:
    """
    Input: Gaussian Mixture Model, Y_i for the super pixel.
    Output: (log) likelihood
    """
    log_likelihood = gmm.score_samples(y)
    likelihood = np.exp(log_likelihood)
    return log_likelihood, likelihood

def _edge_potential(x_i:int, x_j:int, beta:int=2,) -> np.float:
    """
    Output: (log) likelihood
    """
    log_likelihood = -beta * (1-int(x_i == x_j))
    likelihood = np.exp(log_likelihood)
    return log_likelihood, likelihood

def edge_potential_constant(beta:int) -> np.ndarray:
    """
    Output: log likelihood
    """
    _tmp = np.zeros([2,2])
    for i in range(2):
        for j in range(2):
            _tmp[i, j] = _edge_potential(i, j, beta)[0]
    return _tmp
    

def potentials():
    pass

def loopy_BP(luvImage:np.ndarray, SPM:np.ndarray, adj_matrix:np.ndarray, beta:int, 
            b_gmm:GaussianMixture, f_gmm:GaussianMixture, stopping_condition:np.float=1e-5, max_iter:int=1000000) -> np.ndarray:
    """
    Input: 
        @luvImage: 3 * w * h pixel values
        @SPM: w * h pixel labels (superpixel)
        @adj_matrix: adj matrix for superpixels
        @beta: hyperparameter
        @b_gmm: background Gaussian Mixture Model
        @f_gmm: foreground Gaussian Mixture Model
    Output:
        calibrated belief --> np.ndarray.
    """
    pixel_num = adj_matrix.shape[0] # number of superpixel
    belief_list = np.zeros([pixel_num, 2])
    message_mtr = np.zeros([pixel_num, pixel_num, 2])
    assert luvImage.shape[:-1] == SPM.shape

    # _ = plt.figure()
    # plt.imshow(luvImage)
    # plt.title("LuvImage")
    # plt.savefig("luvImage.png")

    y_i = np.array([luvImage[SPM==label].mean(axis=0) for label in range(pixel_num)])

    _img = np.zeros([SPM.shape[0], SPM.shape[1], 3])
    for label in range(pixel_num):
        _img[SPM == label] = y_i[label]
    
    # _ = plt.figure()
    # plt.imshow(_img)
    # plt.title("Y_i")
    # plt.colorbar()
    # plt.savefig("y_i.png")

    return 0

    assert y_i.shape[0] ==  pixel_num
    assert y_i.shape[1] == 3
    unary_phi = np.zeros([pixel_num, 2])              # log value [0, 1]
    edge_phi_constant = edge_potential_constant(beta) # log value [[0,0],[0,1]; [1.0], [1,1]]

    # init phi
    print(f"y_i {y_i[:10]} y_i.shape {y_i.shape} y_i range {y_i.max(), y_i.min()} pixel num {pixel_num} luvImage {luvImage[:10]} range {luvImage.max(), luvImage.min()}")
    unary_phi[:, 0] = node_potential(f_gmm, y_i)[0]     # label = 1
    unary_phi[:, 1] = node_potential(b_gmm, y_i)[0]     # label = 2
    
    def _single_update(_belief_list, _message_mtr):
        print(message_mtr.min(), "min")
        for s in range(pixel_num):
            for t in range(pixel_num):
                # send message s->t
                if adj_matrix[s, t] == 1 and s != t:
                    neighbor_mt = adj_matrix[s,:] == 1 # neighbors other than t, s
                    neighbor_mt[s] = False
                    neighbor_mt[t] = False
                    for t_val in range(2):
                        _tmp_s = [0, 0]
                        for s_val in range(2):   
                            _tmp_s[s_val] = unary_phi[s, s_val] + edge_phi_constant[s_val, t_val] + np.sum(_message_mtr[neighbor_mt, s, s_val])
                        _sum_val = np.sum(np.exp(_tmp_s))
                        try:
                            assert _sum_val > 0
                        except:
                            print(_tmp_s, 'unary', unary_phi[s,:], 'edge', edge_phi_constant, np.sum(neighbor_mt), _message_mtr[neighbor_mt, s, s_val], "sum", np.sum(_message_mtr[neighbor_mt, s, s_val]))
                            raise ValueError(f"tmp_s {_tmp_s}")
                        _message_mtr[s, t, t_val] = np.log(_sum_val)
                    
        # update belief
        for s in range(pixel_num):
            for s_val in range(2):
                neighbors = adj_matrix[s,:] == 1
                neighbors[s] = False
                _belief_list[s, s_val] = unary_phi[s, s_val] + np.sum(message_mtr[neighbors, s, s_val])
                # normalization?
        
        return _belief_list, _message_mtr
    
    for _ in range(max_iter):
        print(message_mtr.min(), message_mtr.max())
        new_belief_list, message_mtr = _single_update(_belief_list=belief_list, _message_mtr=message_mtr)
        if np.abs(np.exp(belief_list) - np.exp(new_belief_list)).max() < stopping_condition:
            belief_list = new_belief_list
            print(f"Stable Condition {np.abs(np.exp(belief_list) - np.exp(new_belief_list)).max()} - {_}")
            print(belief_list.max(), new_belief_list.max())
            break
        belief_list = new_belief_list
    
    
    _img = SPM.copy()
    for label in range(pixel_num):
        _img[SPM == label] = np.exp(belief_list[label, 0])
    
    print(belief_list.max(), belief_list.min())
    _ = plt.figure()
    plt.imshow(_img)
    plt.colorbar()
    plt.show()
    return belief_list



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
    beta = int(sys.argv[4])

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
    # print(luvImage.shape)
    f_gmm = generate_Y(foreground)
    b_gmm = generate_Y(background)

    print(f"foreground cov diag {f_gmm.covariances_.diagonal()}")
    prettyPrintArray(f_gmm.means_)
    print(f"background cov diag {b_gmm.covariances_.diagonal()}")
    prettyPrintArray(b_gmm.means_)


    # superpxiel plot
    plt.imshow(rgbImage)
    adj_matrix = generate_adjacency_matrix(SPM, plot=True)


    # loopy BP
    _ = loopy_BP(luvImage, SPM, adj_matrix, beta, b_gmm, f_gmm)