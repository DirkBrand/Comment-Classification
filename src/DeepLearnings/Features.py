'''
Created on 05 Mar 2015

@author: Dirk
'''
from _collections import defaultdict
from collections import Counter
from decimal import Decimal
import math
import re
from time import strptime

from gensim.models.doc2vec import LabeledSentence
import nltk
from nltk.cluster.util import cosine_distance
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from numpy import dtype
from scipy.cluster.vq import kmeans, vq
from sklearn.cluster import dbscan_
from sklearn.cluster.dbscan_ import DBSCAN
from sklearn.cluster.k_means_ import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from textblob.tokenizers import SentenceTokenizer

from DeepLearnings.ModelTraining import get_model, comment_to_wordlist,\
    LabeledLineSentence
from Objects import CommentObject, ArticleObject
from config import comment_data_path
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

def getCommentCount(commentList):
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


def make_feature_vec(words, model, num_features):
    feature_vec = np.zeros((num_features,), dtype="float32")
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set:
            feature_vec


 
def get_paragraph_features(model, commentList, commentCount, filepath):    
    vects = model.docvecs
    
    M = model.syn0.shape[1]
    N = getCommentCount(commentList)
    feature_matrix = np.empty((N,M), dtype="float32")
    index = 0 
    for commList in commentList.values():
        for comm in commList:
            feature_matrix[index] = vects[comm.id]
            index += 1
            if index % 1000 == 0:
                print index, "document vectors retrieved"
    
    return feature_matrix
    
def tfidf_weighted_sum_features(model, commentList, commentCount): 
    M = model.syn0.shape[1]
    N = getCommentCount(commentList)
    feature_matrix = np.empty((N,M), dtype="float32")
    
    index = 0  
    
    index2word_set = set(model.index2word)
    
    global_body=[]
    
    # Create bag of words for comment
    for commList in commentList.values():
        for comm in commList:
            global_body += comment_to_wordlist(comm.body, True)
            
    global_freq_dist = nltk.FreqDist(global_body)
    
    IN_COUNT = 1
    OUT_COUNT = 1
    for commList in commentList.values():
        for comm in commList:
            sum = np.zeros((M,), dtype="float32")
            clean_comm = comment_to_wordlist(comm.body, True)
            fdist = nltk.FreqDist(clean_comm)
            for word in clean_comm:                
                tf = 0.5 + 0.5*float(fdist[word])/np.max(fdist.values())
                count = global_freq_dist[word]                        
            
                idf = math.log(N / float(1+count))
                if word in index2word_set:
                    #print "IN - ", word
                    IN_COUNT += 1
                    # TFIDF weighted vector
                    sum += tf * idf * model[word]
                else:
                    #print "OUT - ", word
                    OUT_COUNT += 1
            feature_matrix[index] = sum
            index += 1
    
  
    print IN_COUNT, 'words found'
    print OUT_COUNT, 'words not found'
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
                    comment_matrix[i] = model[tok]
                    
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
                        comment_matrix[ind] = model[tok]
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
                    mod, er = model[tok]
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
                    mod, er = model[tok]
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
  
def train_clusterer(model, commentList):  
    
    count = 0
    corpus = []
    words = []
    index2word_set = set(model.index2word)
    for commList in commentList.values():
        for comm in commList:      
            tokens = []
            for word in comment_to_wordlist(comm.body, True):
                if word in index2word_set:  
                    tokens.append(word)  
                    words.append(word)                
                    count += 1
            corpus.append(' '.join(tokens))
    
    fdist1 = FreqDist(words)   
    most_common = [item[0] for item in fdist1.most_common(10)]
    uniques = Counter(words).keys()
    print fdist1
    print len(most_common)
    print len(uniques)
    wordlist = [item for item in uniques if item not in most_common]
    N = len(wordlist)
    print N
                  
    word_vectors = np.zeros((N, 200))
    i = 0
    for word in wordlist:
        if word not in most_common:
            word_vectors[i] = model[word]
            i += 1
    
    
    
    num_clusters = word_vectors.shape[0] / 10
    print word_vectors.shape
    print num_clusters
    
    # Initalize a k-means object and use it to extract centroids
    #idx = KMeans( n_clusters = num_clusters).fit_predict( word_vectors )
    
    centroids,_  = kmeans(word_vectors, num_clusters,)
    
    idx,_ = vq(word_vectors,centroids)
    
    print "TRAINED"
    
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
        
    return word_centroid_map, num_clusters
    
def bag_of_centroids_features(commentList, commentCount, num_clusters, centroid_map):       
    # Pre-allocate an array for the training set bags of centroids (for speed)
    feature_matrix = np.zeros( (getCommentCount(commentList), num_clusters), \
        dtype="float32" )
    
    # Transform the training set reviews into bags of centroids
    index = 0
    for commList in commentList.values():
        for comm in commList:               
            feature_matrix[index] = create_bag_of_centroids(comment_to_wordlist(comm.body, True), \
                centroid_map )
            index += 1
        
    
    return feature_matrix