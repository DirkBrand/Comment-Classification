'''
Created on 05 Mar 2015

@author: Dirk
'''
from _collections import defaultdict
from decimal import Decimal
import math
import re
from time import strptime

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.cluster.k_means_ import KMeans

from DeepLearnings.Main import get_model
from Objects import CommentObject, ArticleObject
import numpy as np


WORD_MIN = 25 # At least that many words per comment
ENGAGE_MIN = 50 # At least that many total votes
VOTES_MIN = 0 # At least that many individual votes

MIN_THREAD_LENGTH = 100 # Threads at least that long




def words(text): return re.findall('[a-z\']+', text.lower())


def generate_bigrams(input):
    bigrams = []
    bigrams2 = []
    for sent in nltk.sent_tokenize(input):
        sent = nltk.word_tokenize(sent.strip("."))
        bigrams += (zip(sent, sent[1:]))
                    
    for tup in bigrams:
        if len(tup[0]) > 1 and len(tup[1]) > 1:
            bigrams2.append(tup[0]+"_"+tup[1])
    
    
    return bigrams2

def getCommentCount(model_type, commentList):
    count = 0
    for commList in commentList.values():
        for comm in commList:
            count += 1
    return count

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


def getFromModel(model, word):
    try:
        print 1
        return (model[word]), 1
    except:            
        print word
        return np.random.random(size=(model.syn0.shape[1])), 0
    
def tfidf_weighted_sum_features(model_type, commentList, commentCount): 
    model = get_model(model_type) 
    M = model.syn0.shape[1]
    N = getCommentCount(model_type, commentList)
    feature_matrix = np.empty([N,M])
    index = 0  
    if model_type == 2: #POS        
        global_body=[]
        for global_comm_list in commentList.values():
            for global_comm in global_comm_list:
                global_body += global_comm.pos_body.split(" ")
                
        global_freq_dist = nltk.FreqDist(global_body)
        
        for commList in commentList.values():
            for comm in commList:
                
                sum = np.zeros(M)
                fdist = nltk.FreqDist(comm.pos_body.split(" "))
                for w in comm.pos_body.split(" "):
                    tf = 0.5 + 0.5*float(fdist[w])/np.max(fdist.values())
                    count = global_freq_dist[w]
                
                    idf = math.log(N / float(1+count))
                    sum += tf * idf * getFromModel(model, w)
                feature_matrix[index] = sum
                index += 1
    elif model_type == 4: #SENTENCE
        global_body=[]
        for global_comm_list in commentList.values():
            for global_comm in global_comm_list:
                global_body += nltk.word_tokenize(global_comm.lemma_body)
                
        global_freq_dist = nltk.FreqDist(global_body)
        
        for commList in commentList.values():
            for comm in commList:
                sentences = nltk.sent_tokenize(comm.lemma_body)
                sum = np.zeros(M)
                for sent in sentences:
                    fdist = nltk.FreqDist(nltk.word_tokenize(sent.strip('.')))
                    for w in nltk.word_tokenize(sent.strip('.')):
                        tf = 0.5 + 0.5*float(fdist[w])/np.max(fdist.values())
                        count = global_freq_dist[w]                        
                    
                        idf = math.log(N / float(1+count))
                        sum += tf * idf * getFromModel(model, w)
                feature_matrix[index] = sum
                index += 1
    elif model_type == 5: #BIGRAMS
        global_body=[]
        for global_comm_list in commentList.values():
            for global_comm in global_comm_list:
                global_body += generate_bigrams(global_comm.lemma_body)
                
        global_freq_dist = nltk.FreqDist(global_body)
        
        for commList in commentList.values():
            for comm in commList:
                sum = np.zeros(M)
                bigrams = generate_bigrams(comm.lemma_body) # Make the bigrams
                fdist = nltk.FreqDist(bigrams)
                for w in bigrams:
                    tf = 0.5 + 0.5*float(fdist[w])/np.max(fdist.values())
                    count = global_freq_dist[w]                  
                
                    idf = math.log(len(commentList) / float(1+count))
                    mod, er = getFromModel(model, w)
                    if er == 1:
                        sum += tf * idf * mod
                feature_matrix[index] = sum
                index += 1
    elif model_type == 99: # GOOGLE
        global_body=[]
        for global_comm_list in commentList.values():
            for global_comm in global_comm_list:
                global_body += words(global_comm.body)
                
        global_freq_dist = nltk.FreqDist(global_body)
        for commList in commentList.values():
            for comm in commList:
                sum = np.zeros(M)
                body = words(comm.body)
                fdist = nltk.FreqDist(body)
                for w in body:
                    tf = 0.5 + 0.5*float(fdist[w])/np.max(fdist.values())
                    count = global_freq_dist[w]                  
                
                    idf = math.log(len(commentList) / float(1+count))
                    mod, er = getFromModel(model, w)
                    if er == 1:
                        sum += tf * idf * mod
                feature_matrix[index] = sum
                index += 1
            
    return feature_matrix
     


def min_max_mean_features(model_type, commentList, commentCount):   
    model = get_model(model_type)
    M = model.syn0.shape[1]
    feature_matrix = np.empty([getCommentCount(model_type, commentList),M*3])
    
    index = 0
    if model_type == 2: #POS
        for commList in commentList.values():
            for comm in commList:
                body = comm.pos_body.split(" ")
                comment_matrix = np.zeros([len(body), M])
                for i in range(len(body)):
                    tok = body[i]
                    comment_matrix[i] = getFromModel(model, tok)
                    
                minVec = np.min(comment_matrix, axis=0)
                maxVec = np.max(comment_matrix, axis=0)
                meanVec = np.mean(comment_matrix, axis=0)
                feature_matrix[index] = np.concatenate([minVec,maxVec,meanVec])
                index += 1
    
    elif model_type == 4: # SENTENCE
        for commList in commentList.values():
            for comm in commList:
                sentences = nltk.sent_tokenize(comm.lemma_body)
                size = 0
                for sent in sentences:
                    body = nltk.word_tokenize(sent.strip('.'))
                    size += len(body)
                    
                comment_matrix = np.zeros([size, M])
                ind = 0
                for sent in sentences:
                    body = nltk.word_tokenize(sent.strip('.'))
                    for i in range(len(body)):
                        tok = body[i]
                        comment_matrix[ind] = getFromModel(model, tok)
                        ind += 1
                    if len(body) < 1:
                        continue
                minVec = np.min(comment_matrix, axis=0)
                maxVec = np.max(comment_matrix, axis=0)
                meanVec = np.mean(comment_matrix, axis=0)
                feature_matrix[index] = np.hstack((np.hstack((minVec,maxVec)), meanVec))
                index += 1
    elif model_type == 5: #BIGRAMS
        for commList in commentList.values():
            for comm in commList:
                bigrams = generate_bigrams(comm.lemma_body) # Make the bigrams
                comment_matrix = np.zeros([len(bigrams), M])
                for i in range(len(bigrams)):
                    tok = bigrams[i]
                    mod, er = getFromModel(model, tok)
                    if er == 1:
                        comment_matrix[i] = mod
                if len(bigrams) < 1:
                    continue
                    
                minVec = np.min(comment_matrix, axis=0)
                maxVec = np.max(comment_matrix, axis=0)
                meanVec = np.mean(comment_matrix, axis=0)
                feature_matrix[index] = np.hstack((np.hstack((minVec,maxVec)), meanVec))
                index += 1
                
    elif model_type == 99: # GOOGLE
        for commList in commentList.values():
            for comm in commList:
                body = words(comm.body)
                comment_matrix = np.zeros([len(body), M])
                for i in range(len(body)):
                    tok = body[i]
                    mod, er = getFromModel(model, tok)
                    if er == 1:
                        comment_matrix[i] = mod
                if len(body) < 1:
                    continue
                minVec = np.min(comment_matrix, axis=0)
                maxVec = np.max(comment_matrix, axis=0)
                meanVec = np.mean(comment_matrix, axis=0)
                feature_matrix[index] = np.hstack((np.hstack((minVec,maxVec)), meanVec))
                index += 1
        
    
    return feature_matrix
    
def bag_of_centroids_features(model_type, commentList, commentCount):   
    model = get_model(model_type)   
    word_vectors = model.syn0
    num_clusters = word_vectors.shape[0] / 50
    
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans( n_clusters = num_clusters , n_jobs=-1)
    idx = kmeans_clustering.fit_predict( word_vectors )
    
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
    feature_matrix = np.zeros( (getCommentCount(model_type, commentList), num_clusters), \
        dtype="float32" )
    
    # Transform the training set reviews into bags of centroids
    index = 0
    for commList in commentList.values():
        for comm in commList:
            if model_type == 2: #POS
                feature_matrix[index] = create_bag_of_centroids(comm.pos_body.split(" "), \
                    word_centroid_map )
                index += 1
            elif model_type == 4: #SENTENCES
                sentences = nltk.sent_tokenize(comm.lemma_body)
                body = []
                for sent in sentences:
                    body += nltk.word_tokenize(sent.strip("."))
                    
                feature_matrix[index] = create_bag_of_centroids(body, \
                    word_centroid_map )
                index += 1
            elif model_type == 5: #BIGRAMS            
                feature_matrix[index] = create_bag_of_centroids(generate_bigrams(comm.lemma_body), \
                    word_centroid_map )
                index += 1
            elif model_type == 99: #GOOGLE            
                feature_matrix[index] = create_bag_of_centroids(words(comm.body), \
                    word_centroid_map )
                index += 1
        
    
    return feature_matrix