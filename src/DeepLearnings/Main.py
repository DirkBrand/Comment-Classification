'''
Created on 24 Feb 2015

@author: Dirk
'''
import re

from gensim.models.word2vec import LineSentence
from mainExtractor import read_comments
import nltk
from nltk.corpus import stopwords

from DeepLearnings.Clustering import crp, create_sum_of_clusters
from DeepLearnings.ModelTraining import train_model, get_model,\
    comment_to_wordlist
from Features import train_clusterer, getCommentCount,\
    create_bag_of_centroids
from config import  comment_data_path
import numpy as np
from scipy.stats import futil
from scipy.sparse.csgraph import _validation

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
stops = set(stopwords.words("english"))         





retrain = False
model_type = 1
if __name__ == '__main__':   
    
    articleList, commentList, parentList, commentCount = read_comments(comment_data_path + 'trainTestDataSet.txt')
     
    
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
    
    '''
    feature_matrix = np.empty([commentCount, numClusters])
    # Transform the training set reviews into bags of centroids
    index = 0
    for commList in commentList.values():
        for comm in commList:
            feature_matrix[index] = create_sum_of_clusters(comment_to_wordlist(comm.body,True), cluster_ids, cluster_sums )
            index += 1
        '''
    
        

    
    
    
    
    
    
    
    
    
    