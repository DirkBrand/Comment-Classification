'''
Created on 02 Mar 2015

@author: Dirk
'''




import math
from gensim.models.word2vec import LineSentence
from sklearn.cluster.k_means_ import KMeans

from Main import get_model
import pandas as pd
import numpy as np
from numpy.matlib import rand
from sklearn.metrics.pairwise import pairwise_distances, cosine_distances
from sklearn.cluster.dbscan_ import DBSCAN
import sklearn


def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    num_centroids = max( word_centroid_map.values() ) + 1
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v1)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def read_y_file(fname):
    y = []
    with open(fname) as f:
        for l in f:
            y.append(int(l))
    return np.array(y)

model_type = 4
if __name__ == '__main__':    
    if model_type == 1:
        fname = 'D:\REPOS\comments_database\CommentCorps\comms_data.txt'
        y = read_y_file('D:\REPOS\comments_database\CommentCorps\comms_labels.txt')
        model_name = "base_model"
    elif model_type == 2:
        fname = 'D:\REPOS\comments_database\CommentCorps\comms_POS_data.txt'
        y = read_y_file('D:\REPOS\comments_database\CommentCorps\comms_POS_labels.txt')
        model_name = "POS_model"
    elif model_type == 3:
        fname = 'D:\REPOS\comments_database\CommentCorps\comms_lemmatized_data.txt'
        y = read_y_file('D:\REPOS\comments_database\CommentCorps\comms_lemmatized_labels.txt')
        model_name = "lemmatized_model"
    elif model_type == 4:
        fname = 'D:\REPOS\comments_database\CommentCorps\comms_sentence_data.txt'
        y = read_y_file('D:\REPOS\comments_database\CommentCorps\comms_sentence_labels.txt')
        model_name = "sentence_model"
    elif model_type == 5:
        fname = 'D:\REPOS\comments_database\CommentCorps\comms_bigram_data.txt'
        y = read_y_file('D:\REPOS\comments_database\CommentCorps\comms_bigram_labels.txt')
        model_name = "bigram_model"
    elif model_type == 6:
        fname = 'D:\REPOS\comments_database\CommentCorps\comms_sentence_data.txt'
        y = read_y_file('D:\REPOS\comments_database\CommentCorps\comms_sentence_labels.txt')
        model_name = "bigram_from_sentence_model"
    
    
    N = file_len(fname)
    model = get_model(model_type)    
    
    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] / 5
    print N, num_clusters
    print y.shape
    
    clusterer = DBSCAN(metric=cosine_distances).fit_predict(word_vectors)
    print clusterer
    # Initalize a k-means object     and use it to extract centroids
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )
    
    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster 
    word_centroid_map = dict(zip( model.index2word, idx ))
    # For the first 10 clusters
    for cluster in xrange(0,10):
        #
        # Print the cluster number  
        print "\nCluster %d" % cluster
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in xrange(0,len(word_centroid_map.values())):
            if( word_centroid_map.values()[i] == cluster ):
                words.append(word_centroid_map.keys()[i])
        print words
    
    # Pre-allocate an array for the training set bags of centroids (for speed)
    centroids = np.zeros( (N, num_clusters), \
        dtype="float32" )
    
    # Transform the training set reviews into bags of centroids
    counter = 0
    for comm in LineSentence(fname):
        centroids[counter] = create_bag_of_centroids( comm, \
            word_centroid_map )
        counter += 1
    