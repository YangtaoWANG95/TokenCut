"""
Main functions for applying Normalized Cut.
Code adapted from LOST: https://github.com/valeoai/LOST
"""
import torch
import torch.nn.functional as F
import numpy as np
#from scipy.linalg.decomp import eig
from scipy.linalg import eigh
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def ncut(feats, dims, scales, init_image_size, eps=1e-5, tau=0.05):
    #feats = feats[0,1:,:].detach().cpu().numpy()
    feats = feats[0,1:,:]
    feats = F.normalize(feats, p=2)
    A = (feats @ feats.transpose(1,0))
    A = A.cpu().numpy()

    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
  
    # Print second and third smallest eigenvector 
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    
    # method 2 avg
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg

    # method3 EM algo
    #second_smallest_vec = eigenvectors[:, 0:1]
    #bipartition = GMM(second_smallest_vec)

    # method 4 Kmeans
    #second_smallest_vec = eigenvectors[:, 0:1]
    #bipartition = Kmeans_partition(second_smallest_vec)

    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        bipartition = np.logical_not(bipartition)
        eigenvec = eigenvec * -1
    bipartition = bipartition.reshape(dims).astype(float)

    # predict BBox
    pred, _ = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size) ## We only extract the principal object BBox
   
    return np.asarray(pred), bipartition, seed, eigenvec.reshape(dims)

def GMM(eigvec):
    gmm = GaussianMixture(n_components=2, max_iter=300)
    gmm.fit(eigvec)
    partition = gmm.predict(eigvec)
    return partition

def Kmeans_partition(eigvec):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(eigvec)
    return kmeans.labels_

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

        return pred, pred_feats
    else:
        raise NotImplementedError

def get_feats(feat_out, shape):
    nb_im, nh, nb_tokens = shape[0:3]  # Batch size, Number of heads, Number of tokens
    qkv = (
            feat_out["qkv"]
            .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
            .permute(2, 0, 3, 1, 4)
        )
    q, k, v = qkv[0], qkv[1], qkv[2]
    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    return k
 
