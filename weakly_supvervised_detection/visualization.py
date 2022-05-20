"""
Vis utilities. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import cv2
import numpy as np
from PIL import Image
import cv2
import scipy
import numpy as np 
import torch
import torch.nn as nn
       
from utils import unnormalize_images
import matplotlib.pyplot as plt

def visualize_img(image):
    image = image.clone().detach().cpu().numpy()
    #image = np.uint8(np.transpose(image, (1,2,0)))
    print(f'image shape: {image.shape}')
    image = np.uint8(image*256)
    cv2.imwrite('./test/test_img.png', image)
        
def visualize_fms(image, mask, seed, im_name, dim, scales, folder, save=True):
    w_featmap, h_featmap = dim
    image = image.clone().detach().cpu().numpy()
    image = np.uint8(np.transpose(image, (1,2,0))*256)
    mask = mask.reshape(w_featmap,h_featmap)

    mask = mask.reshape(w_featmap*h_featmap)
    mask[seed] = 3
    mask = mask.reshape(w_featmap, h_featmap)
    mask = scipy.ndimage.zoom(mask, scales, order=0, mode='nearest')
    if save:
        pltname1 = f"{folder}/LOST_{im_name}.png"
        #Image.fromarray(image).save(pltname1)
        cv2.imwrite(pltname1, image)
 #       print(f"Predictions saved at {pltname1}.")
        pltname2 = f"{folder}/Mask_{im_name}.png"
        plt.imsave(fname=pltname2, arr=mask)
 #       print(f"Predictions saved at {pltname2}.")

def visualize_eigvec(eigvec, folder, im_name, dim, scales, save=True):
    print(f'eigvec shape: {eigvec.shape}, scales: {scales}, dim: {dim}')
    eigvec = scipy.ndimage.zoom(eigvec, scales, order=0, mode='nearest')
    if save:
        pltname= f"{folder}/{im_name}_ours_att.jpg"
        plt.imsave(fname=pltname, arr=eigvec, cmap='cividis')

def visualize_predictions_gt(image, pred, gt, im_name, seed, dim, scales, folder, output_name='ours_box', save=True):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    image = unnormalize_images(image)
    image = image[0].clone().detach().cpu().numpy()
    image = np.transpose(image, (1,2,0))
    image = np.uint8(image*256)
    image = np.ascontiguousarray(image, dtype=np.uint8)

    # Plot the box
    cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 2 # RED
    )
    # Plot the ground truth box
    if len(gt>1):
        for i in range(len(gt)):
            cv2.rectangle(
                image,
                (int(gt[i][0]), int(gt[i][1])),
                (int(gt[i][2]), int(gt[i][3])),
                (0, 0, 255), 3, #BLUE
            )
    
    if save:
        pltname = f"{folder}/{im_name}_{output_name}.jpg"
        Image.fromarray(image).save(pltname)
    #    print(f"Predictions saved at {pltname}.")
    return image

def visualize_attn_LOST(A, dims, scales, folder, im_name):
    """
    Visualization of the maps presented in Figure 2 of the paper.
    """
    w_featmap, h_featmap = dims

    # Binarized similarity
    binA = A.copy()
    binA[binA < 0] = 0
    binA[binA > 0] = 1

    # Save inverse degree
    im_deg = (
        nn.functional.interpolate(
            torch.from_numpy(1 / binA.sum(-1)).reshape(1, 1, w_featmap, h_featmap),
            scale_factor=scales,
            mode="nearest",
        )[0][0].cpu().numpy()
    )
    plt.imsave(fname=f"{folder}/{im_name}_lost_attn.jpg", arr=im_deg, cmap='cividis')

