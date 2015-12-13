'''
Created on 24 Feb 2015

@author: Dirk
'''
import re

from FeatureExtraction.mainExtractor import read_news24_comments
from gensim.models.word2vec import LineSentence
import nltk
from nltk.corpus import stopwords
from scipy.sparse.csgraph import _validation
from scipy.stats import futil

from DeepLearnings.Clustering import crp, create_sum_of_clusters
from DeepLearnings.ModelTraining import train_model, get_model,\
    comment_to_wordlist
from Features import train_clusterer, getCommentCount,\
    create_bag_of_centroids
from config import  comment_data_path
import numpy as np


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stops = set(stopwords.words("english"))         





retrain = False
model_type = 1
if __name__ == '__main__':   
         
    
    model = get_model(model_type)

    vectors = model.syn0
    cluster_ids, cluster_sums, numClusters = crp(vectors)
    
    print numClusters, "Clusters created"
    idx = model.index2word
    # For the first 10 clusters
    for i, cluster in enumerate(cluster_ids.values()):
        #
        # Print the cluster number  
        print "\nCluster %d" % i
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for id in cluster:
            words.append(idx[id])        
        print words[:20]
    
   
    
        

    
    
    
    
    
    
    
    
    
    