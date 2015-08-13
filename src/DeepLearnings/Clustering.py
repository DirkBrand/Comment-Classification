'''
Created on 28 Jul 2015

@author: Dirk
'''
from _collections import defaultdict

from numpy import Inf, random
from scipy.spatial.distance import cosine

import numpy as np


def create_sum_of_clusters( wordlist, clusters, clusterSums ):
    
    feature_vec = np.zeros(300)
    for i, word in enumerate(wordlist):
        for j,cl in enumerate(clusters.values()):
            if i in cl:
                feature_vec += clusterSums[j]
        
   
    #
    # Return the "bag of centroids"
    return feature_vec

def crp(vectors):
    clusterVec = dict()         # tracks sum of vectors in a cluster
    clusterVec[0] = np.zeros(300)
    clusterIdx = defaultdict(list)         # array of index arrays. e.g. [[1, 3, 5], [2, 4, 6]]
    clusterIdx[0] = []
    ncluster = 1
    # probablity to create a new table if new customer
    # is not strongly "similar" to any existing table
    pnew = 1.0/ (1 + ncluster)  
    N = len(vectors)
    rands = random.rand(N)         # N rand variables sampled from U(0, 1)
 
    for i in range(N):
        maxSim = -Inf
        maxIdx = 0
        v = vectors[i]
        for j in range(ncluster):
            sim = cosine(v, clusterVec[j])
            if sim > maxSim: # Finds most similar cluster
                maxIdx = j
                maxSim = sim
            if maxSim < pnew:
                if rands[i] < pnew: #new cluster
                    clusterVec[ncluster] = v
                    clusterIdx[ncluster] = [i] 
                    ncluster += 1
                    pnew = 1.0 / (1 + ncluster)
                continue
        clusterVec[maxIdx] = clusterVec[maxIdx] + v
        clusterIdx[maxIdx].append(i)
 
    return clusterIdx, clusterVec, ncluster