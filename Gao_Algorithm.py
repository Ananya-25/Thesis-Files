import sys
import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from datafold.pcfold import PCManifold

########### MODIFIFED ALGORITHM TO INCLUDE PREVIOUS LANDMARKS 
###### numLmkStart and numLmkEnd to keep track of IDs and landmarks
def GetGPLmk_Euclidean(data, numLmk, batch_size, n_prev_landmarks, BNN, eps_safe=1e-5):
    """
    Parameters
    ==========
    numLmk: number of new landmarks to find
    batch_size: used to estimate epsilon
    n_prev_landmarks: the number of previously found landmarks at the end of the given "data" array
    eps_safe: safety to bound numbers away from zero
    """
    
    n_trials = 100
    total_ep = 0
    rng = np.random.default_rng(42)
    for i in range(n_trials):
        n_samples = data.shape[0]
        idx = rng.choice(n_samples, size=batch_size)
        pcm = PCManifold(data[idx,:])
        pcm.optimize_parameters()
        ep = pcm.kernel.epsilon
        total_ep += ep
        
    bandwidth = total_ep / n_trials
    nbrs = NearestNeighbors(n_neighbors=BNN+1).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    indices = indices.reshape(-1)
    repmat = np.tile(range(data.shape[0]), [1, BNN+1]).reshape(-1)
    kernel = np.exp(-distances**2/(bandwidth*2)).reshape(-1)

    fullPhi = csr_matrix((kernel, (repmat, indices)), (data.shape[0], data.shape[0]))
    fullMatProd = (fullPhi+fullPhi.T)/2
    fullMatProd = fullMatProd.toarray()
    KernelTrace = np.diag(fullMatProd)
    
    GPLmkIdx = np.zeros(numLmk + n_prev_landmarks, dtype=int)
    # if we have previous landmarks, set them at the beginning of the array
    if n_prev_landmarks > 0:
        GPLmkIdx[:n_prev_landmarks] = np.arange(0, n_prev_landmarks)
    
    numLmk_total = GPLmkIdx.shape[0]
    
    invKn = np.zeros((numLmk_total, numLmk_total))
    for j in range(n_prev_landmarks, numLmk_total):
        if j == 0:
            ptuq = KernelTrace
        else:
            if j == 1:
                invKn[0, 0] = 1/fullMatProd[GPLmkIdx[0],GPLmkIdx[0]]
                a = invKn[0,0]*fullMatProd[GPLmkIdx[:j],:]
                b = fullMatProd[:,GPLmkIdx[:j]].T
                c = np.multiply(b, a)
                ptuq = KernelTrace - (np.sum(c, 0))
            else: 
                p = fullMatProd[GPLmkIdx[:(j-1)],GPLmkIdx[j-1]]
                mu = np.divide(1, np.clip(fullMatProd[GPLmkIdx[j-1],GPLmkIdx[j-1]]-
                                   np.dot(p.T, np.dot(invKn[:(j-1),:(j-1)], p)), eps_safe, 1/eps_safe))
                p = p.T
                invKn[:(j-1),:(j)] = np.dot(
                    invKn[:(j-1),:(j-1)], 
                    (np.append(np.identity(j-1) + np.dot(mu, np.dot((np.dot(p, p.T)), 
                    invKn[:(j-1),:(j-1)])), -np.dot(mu, p).reshape(1, -1), axis=0)).T)
                invKn[(j-1),:(j)] = np.append(invKn[:(j-1),(j-1)].T, mu)
                productEntity = np.dot(invKn[:(j),:(j)], fullMatProd[GPLmkIdx[:j],:])
                ptuq = KernelTrace - (sum(np.multiply(fullMatProd[:,GPLmkIdx[:j]].T, productEntity),0).T)
        maxUQIdx = np.argmax(ptuq)
        GPLmkIdx[j] = maxUQIdx
    p = fullMatProd[GPLmkIdx[:-1],GPLmkIdx[-1]]
    mu = np.divide(1, np.clip(fullMatProd[GPLmkIdx[-1],GPLmkIdx[-1]]-np.dot(p.T, np.dot(invKn[:-1,:-1], p)), eps_safe, 1/eps_safe))
    p = p.T
    invKn[:-1,:] = invKn[:-1,:-1] @ np.append(np.identity(numLmk_total-1) + np.dot(mu, np.dot((np.dot(p, p.T)), invKn[:-1,:-1])), 
                                              -np.dot(mu, p).reshape(1, -1), axis=0).T 
    invKn[-1,:] = np.append(invKn[:-1,-1].T, mu)
    ptuq = KernelTrace - sum(np.multiply(fullMatProd[:,GPLmkIdx].T, np.dot(invKn, fullMatProd[GPLmkIdx, :])),0).T
    return (GPLmkIdx, ptuq)