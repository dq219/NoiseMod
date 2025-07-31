import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import torch.nn as nn
import copy 

# get similarity matrix from the weight matrix
def getS(mat):
    x = np.abs(mat) + 1e-7
    norm = np.sqrt(np.sum(x ** 2, axis = 0)[:, np.newaxis] @  np.sum(x ** 2, axis = 0)[np.newaxis, :])
    return x.T @ x / norm

# get random walk Laplacian
def getL(S):
    D = np.diag(S.sum(axis=1))
    D_inv = np.diag(1.0 / np.diag(D))
    res = D_inv @ S
    return np.eye(len(res)) - res

# take a weight matrix, detects community using the first k eigenvectors of the random walk Laplacian 
# re-orders the indexes and gives the community grouping based on the new order
def DetectCommunity(mat, k = 4, seed = 42):
    # get Laplacian
    (dim_in, dim_out) = mat.shape
    S = getS(mat)
    L = getL(S)
    eigvals, eigvecs = eigh(L)
    eigidx = np.arange(len(eigvals)) + 1

    # detect community
    if k == 'auto':
        eigdif = eigvals[1:] - eigvals[:-1]
        k = np.argmax(eigdif) + 1
    k = min(k, dim_out)
    X = eigvecs[:, :k]
    kmeans = KMeans(n_clusters = k, random_state = seed)
    labels = kmeans.fit_predict(X)

    # re-order matrix indices
    yProxy = []
    for j in range(4):
        wProxy = np.sum(np.abs(mat)[:, labels == j], axis = 1) + 1e-8
        yProxy.append(np.sum(wProxy * (-np.arange(len(wProxy))))/np.sum(wProxy))
    yProxy = np.argsort(yProxy)[::-1]
    newlabels = np.argsort(np.array(yProxy))[labels]
    sorted_indices = np.argsort(newlabels)
    inxBounds = [0]
    for j in range(4):
        inxBounds.append(dim_out - np.sum(newlabels > j))

    return sorted_indices, inxBounds, eigvals, eigidx

# main analysis function
def analyse_connectivity(model, k = 4):

    ###############################
    ### Collect weight matrices ###
    ###############################

    WMATS = []
    for i in range(len(model.layer)):
        if isinstance(model.layer[i], nn.Linear):
            WMATS.append(model.layer[i].weight.detach().cpu().numpy().T)
    encode_num = len(WMATS[0])

    ###################################
    ### Calculate Laplacian spectra ###
    ###################################

    EIG_RAW = []
    for i, m in enumerate(WMATS):
        mat = WMATS[i]      
        S = getS(mat)
        L = getL(S)
        eigvals, eigvecs = eigh(L)
        eigidx = np.arange(len(eigvals)) + 1
        
        EIG_RAW.append([eigvals, eigidx])

    #######################################################################################################
    ### Calculate spectra while grouping rows according to the detected cluster from the previous layer ###
    #######################################################################################################
    
    # *** note on the grouping: this does not seem to affect the spectra in any qualitative way for all visualisations

    EIG = []
    sorted_indices = np.arange(encode_num)
    inxBounds = np.arange(encode_num + 1)
    IDX = [[sorted_indices, inxBounds]]

    for i in range(len(WMATS)):

        [sorted_indices, inxBounds] = copy.deepcopy(IDX[-1])
        
        mat = WMATS[i][sorted_indices, :]
        (dim_in, dim_out) = mat.shape
        mat_collapse = np.zeros((k, dim_out))
        for j in range(k):
            mat_collapse[j] = np.sum(np.abs(mat)[inxBounds[j] : inxBounds[j + 1]], axis = 0)
        sorted_indices_new, inxBounds_new, eigvals, eigidx = DetectCommunity(mat_collapse, k = k)
        EIG.append([eigvals, eigidx])
        IDX.append([sorted_indices_new, inxBounds_new])

    return WMATS, EIG_RAW, EIG, IDX




