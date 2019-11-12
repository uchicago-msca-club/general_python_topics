import warnings
warnings.filterwarnings('ignore')

import os, glob
import cv2
import pandas as pd
import numpy as np
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist, euclidean
from skimage import feature, io, color


def plot_corr_heatmap(corrmat, annotate=False, annot_size=15, a4_dims=(11.7, 8.27)):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cutsomcmap = sns.diverging_palette(250, 0, as_cmap=True)
    
    fig, ax = plt.subplots(figsize=a4_dims)
    corrheatmap = \
        sns.heatmap(ax=ax, data=corrmat, mask=mask, annot=annotate,
                    linewidths=0.5, cmap=cutsomcmap, annot_kws={"size": annot_size})
    plt.show()
    

def create_dir(path):
    if os.path.exists(path):
#        print(path, "already exists")
        return False
    return os.mkdir(path)

def display_image(img, winname="", destroy=True, waitKey=0):
    cv2.namedWindow(winname, cv2.WINDOW_KEEPRATIO)
    try:
        cv2.imshow(winname, img)
        cv2.waitKey(waitKey)
        if destroy:
            cv2.destroyWindow(winname)
    except Exception as e:
        print("Coult not open image!")
        print(e)
        cv2.destroyWindow(winname)
    
def img_2dto3d(img):
    img_3d = np.zeros((img.shape[0], img.shape[1], 3))
    img_3d[:,:,0] = img
    img_3d[:,:,1] = img
    img_3d[:,:,2] = img
    return img_3d
    
def tile_images(img_arr, haxis=True, resize=1):
    if resize > 1:
        for i in range(len(img_arr)):
            img_arr[i] = cv2.resize(img_arr[i], (int(img_arr[i].shape[1]/resize),
                                                 int(img_arr[i].shape[0]/resize)))
    
    if haxis:
        new_img = np.zeros((img_arr[0].shape[0], 
                            img_arr[0].shape[1]*len(img_arr),
                            img_arr[0].shape[2]), dtype=np.uint8)
        ctr = 0
        for i in img_arr:
            new_img[:, ctr:ctr+i.shape[1],:] = i
            ctr += i.shape[1]
            
    else:
        new_img = np.zeros((img_arr[0].shape[0]*len(img_arr), 
                            img_arr[0].shape[1],
                            img_arr[0].shape[2]), dtype=np.uint8)
        ctr = 0
        for i in img_arr:
            new_img[ctr:ctr+i.shape[0], :, :] = i
            ctr += i.shape[0]
        
    return new_img
        
        
def alignImages(im1, im2, max_features=500, good_match_percent=0.15):
 
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
#     display_image(imMatches, "matches")

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, imMatches, h


def get_centroid(mask, lbl):
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"]==0:
        return -1, -1
    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY

def get_largest_contour(contours):
    l1 = 0
    cc = None
    for c1 in contours:
        c1 = np.asarray(c1)
        if c1.shape[0] > l1:
            cc = c1
    cc = np.reshape(cc, (cc.shape[0], cc.shape[2]))
    return cc

def get_median_dist(arr):
    dist_mat = distance_matrix(arr, arr)             
    dist_vec = np.tril(dist_mat).ravel()
    dist_vec = dist_vec[dist_vec>0]
    dist_median = np.median(dist_vec)
    return dist_median


def get_median_img(img):
   i1 = img.reshape((img.shape[0]*img.shape[1], 3))
   def get_nonzero_px(_i):
       img2 = []
       for i in _i:
               if (i[0] != 0) & (i[1] != 0) & (i[2] != 0):
                   img2.append(i)
       return np.asarray(img2)
   i1 = get_nonzero_px(i1)
   median_i1 = geometric_median(i1)
   return median_i1


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

# --------------------------------------------------------------------------------
# REGION MERGING UTILS

from skimage import data, io, segmentation, color
from skimage.future import graph

def weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])
