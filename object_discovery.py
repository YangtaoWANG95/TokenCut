import torch
import torch.nn.functional as F
import numpy as np
#from scipy.linalg.decomp import eig
from scipy.linalg import eigh
from scipy import ndimage
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, im_name='', no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    cls_token = feats[0,0:1,:].cpu().numpy() 

    feats = feats[0,1:,:]
    feats = F.normalize(feats, p=2)
    A = (feats @ feats.transpose(1,0)) 
    A = A.cpu().numpy()
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
  
    # Print second and third smallest eigenvector 
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])

    # method1 avg
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    
    # method2 EM algo
    #second_smallest_vec = eigenvectors[:, 0:1]
    #bipartition = GMM(second_smallest_vec)

    # method3 Kmeans 
    #second_smallest_vec = eigenvectors[:, 0:1]
    #bipartition = Kmeans_partition(second_smallest_vec)
    
    # method4 min energie
    #second_smallest_vec = eigenvectors[:, 0]
    #bipartition = min_energie(A, second_smallest_vec)

    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    pred, _, objects,cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]) ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0],cc[1]] = 1

    return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)

def Kmeans_partition(eigvec):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(eigvec)
    return kmeans.labels_

def GMM(eigvec):
    gmm = GaussianMixture(n_components=2, max_iter=300)
    gmm.fit(eigvec)
    partition = gmm.predict(eigvec)
    return partition

def min_energie(M, eigvec):
    """
    Method proposed in Normalized Cut, compute the best Ncut(A,B)
    """
    best_index = 0
    sorted_index = np.argsort(eigvec)
    length = len(eigvec)
    eps = 1e-5

    min_energie = float("inf")
    for i in range(1,length):
        A = np.zeros((length,1))
        B = np.zeros((length,1))
        A[sorted_index[:i]]=1
        B[sorted_index[i:]]=1
        assAV = np.sum(np.transpose(A) * M)
        cutAB = np.sum(np.transpose(A) * M * B)
        assBV = np.sum(np.transpose(B) * M)
        cutBA = np.sum(np.transpose(B) * M * A)
        if assAV == 0:
            assAV = eps
        if assBV == 0:
            assBV = eps
        energie = cutAB / assAV + cutBA / assBV
        if min_energie > energie:
            min_energie = energie
            best_index = i
    partition = eigvec > eigvec[sorted_index[best_index]]
    return partition

def patch_scoring(M, threshold=0.):
    """
    Patch scoring based on the inverse degree.
    """
    # Cloning important
    A = np.copy(M)

    # Zero diagonal
    np.fill_diagonal(A,0)

    # Make sure symmetric and non nul
    A[A < 0] = 0
    C = A + A.T

    # Sort pixels by inverse degree
    cent = -np.sum(A > threshold, axis=1).astype(np.float32)
    sel = np.argsort(cent)[::-1] # desending order

    return sel, cent

def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition) 
    cc = objects[np.unravel_index(seed, dims)]
    

    if principle_object:
        mask = np.where(objects == cc)
       # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]
         
        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])
        
        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError

