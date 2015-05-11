'''
Created on 03 Mar 2014

@author: Dirk
'''

import pickle

import nltk
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model.stochastic_gradient import SGDRegressor

from DeepLearnings.FeatureExtraction import min_max_mean_features,\
    tfidf_weighted_sum_features, bag_of_centroids_features, getCommentCount
from DeepLearnings.Main import get_model
from FeatureExtraction.SentimentUtil import create_classifier, make_full_dict,\
    create_word_scores, find_best_words, evaluate_features, best_word_features,\
    extract_features
from FeatureExtraction.mainExtractor import read_comments, extract_values, extract_feature_matrix,\
    extract_Time_Data, extract_words, extract_topics, read_user_comments,\
    extract_user_values, extract_user_topics, read_user_data,\
    extract_social_features, extract_sentence_values
from config import sentiment_path, feature_set_path, comment_data_path
import numpy as np


def scaleMatrix(mat):
    return (mat - np.min(mat, axis=0)) / (np.max(mat, axis=0) - np.min(mat, axis=0))


def percentage(count, total):
    return 100 * count / total

def unusual_words(text):    
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab.difference(english_vocab)
    return sorted(unusual)

def setup():    
    create_classifier(make_full_dict, sentiment_path + 'sentiment_classifier.pickle')
    print 'Trained Classifier'
    print 'evaluating best %d word features' % (10000)
    
    word_scores = create_word_scores()
    best_words = find_best_words(word_scores, 10000)
    evaluate_features(best_word_features, best_words)


def extractSaveFeatures(articleList, commentList, parentList,commentCount):
    featureMatrix = extract_feature_matrix(articleList, commentList, parentList,commentCount)
    print "Extracted Features"
    save_numpy_matrix(feature_set_path + "featureArray",featureMatrix) 
    
def extractSaveValues(articleList, commentList, parentList,commentCount):
    valueVector = extract_values(articleList,commentList, parentList, commentCount)
    print "Extracted values"
    save_numpy_matrix(feature_set_path + "valueVector",valueVector) 
    
    
def extractSaveSentenceValues(articleList, commentList, parentList,commentCount):
    valueVector = extract_sentence_values(articleList,commentList, parentList, getCommentCount(4, commentList))
    print "Extracted values"
    save_numpy_matrix(feature_set_path + "sentenceValueVector",valueVector) 
    
def extractWordData(articleList, commentList, commentCount):
    bwd, fwd, twd, bbwd, btwd, tbwd, ttwd, qbwd, qtwd = extract_words(commentList, commentCount)
    save_sparse_csr(feature_set_path + "binaryWordData",bwd) 
    save_sparse_csr(feature_set_path + "freqWordData",fwd) 
    save_sparse_csr(feature_set_path + "tfidfWordData",twd) 
    save_sparse_csr(feature_set_path + "bigramBinaryWordData",bbwd) 
    save_sparse_csr(feature_set_path + "bigramTfidfWordData",btwd) 
    save_sparse_csr(feature_set_path + "trigramBinaryWordData",tbwd) 
    save_sparse_csr(feature_set_path + "trigramTfidfWordData",ttwd) 
    save_sparse_csr(feature_set_path + "quadgramBinaryWordData",qbwd) 
    save_sparse_csr(feature_set_path + "quadgramTfidfWordData",qtwd) 
    
def extractSocialData(articleList, commentList, commentCount):
    userList, userCount = read_user_data(comment_data_path + 'userdata.txt');
    socialVector = extract_social_features(userList, commentList, commentCount)            
    print "Extracted social Features"
    save_numpy_matrix(feature_set_path + "socialVector",socialVector) 
    
model_type = 5
def extractSaveWordEmbeddingFeatures(commentList, commentCount): 
    if model_type == 2:
        model_name = "POS_model"
    elif model_type == 3:
        model_name = "lemmatized_model"
    elif model_type == 4:
        model_name = "sentence_model"
    elif model_type == 5:
        model_name = "bigram_model"  
    elif model_type == 99:
        model_name = "google_model"  
    
    print "ONE"
    fs2 = tfidf_weighted_sum_features(model_type, commentList, commentCount)
    save_numpy_matrix(feature_set_path + model_name + "_TfidfWeightedSumFeatures",fs2) 
    print "TWO"
    fs1 = min_max_mean_features(model_type, commentList, commentCount)    
    save_numpy_matrix(feature_set_path + model_name + "_MinMaxMeanFeatures",fs1)
    print "THREE"
    fs3 = bag_of_centroids_features(model_type, commentList, commentCount)
    save_numpy_matrix(feature_set_path + model_name + "_BagOfCentroidsFeatures",fs3)  
    print "Extracted Deep Learning Features"
 
def extractTimeData(articleList, commentList, commentCount):
    td = extract_Time_Data(articleList, commentCount)
    save_sparse_csr("timeData",td) 

def extractTopicData(articleList, commentList, commentCount, numWords):
    w = extract_topics(articleList, commentCount, numWords)
    save_sparse_csr(feature_set_path + "topicData",w) 
    
def extractUserData():
    userList, userCount = read_user_comments(comment_data_path + 'comments.txt')
    print "Processed", userCount, "Users"
    
    valueVector = extract_user_values(userList, userCount) 
    print "Extracted values"
    featureMatrix = extract_user_topics(userList,userCount,100)            
    print "Extracted Features"
    save_sparse_csr("valueVectorTest",valueVector) 
    save_sparse_csr("featureArrayTest",featureMatrix) 
    
    print "Saved User Data"
    
    
def save_sparse_csr(filename,array):
    np.savez(filename, data = array.data, indices=array.indices, indptr =array.indptr, shape=array.shape)

def save_numpy_matrix(filename, array):
    np.save(filename, array)
    
def load_numpy_matrix(filename):
    return np.load(filename)
    
def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix(( loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])

def savePickledList(articleList,commentList,parentList):
    f = open('articleList.pkl', 'wb')
    pickle.dump(articleList, f)
    f.close()
    f = open('commentList.pkl', 'wb')
    pickle.dump(commentList, f)
    f.close()
    f = open('parentList.pkl', 'wb')
    pickle.dump(parentList, f)
    f.close()
    
def loadPickledList():
    artList = []
    commentList = []
    parentList = []

    f =  open("articleList" + '.pkl', 'rb')
    artList = pickle.load(f)
    f.close()
    f =  open("commentList" + '.pkl', 'rb')
    artList = pickle.load(f)
    f.close()
    f =  open("parentList" + '.pkl', 'rb')
    parentList = pickle.load(f)
    f.close()
    
    commCount = 0
    for commList in artList.values(): 
        commCount += len(commList)

    return artList, commentList, parentList, commCount


reLoad = True
if __name__ == '__main__':
    print 'START'
    #setup()
         
    # To process all the comments
    if reLoad:
        articleList, commentList, parentList, commentCount = read_comments(comment_data_path + 'trainTestDataSet.txt')
        #savePickledList(articleList, commentList, parentList)
    else:
        articleList, commentList, parentList, commentCount = loadPickledList()
        
    print "Processed", commentCount, "Comments"
      
    # Deep Learning Extraction
    #extractSaveWordEmbeddingFeatures(commentList, commentCount)
    #extractSaveSentenceValues(articleList,commentList, parentList,commentCount)
    
    # Classic
    #extractSaveValues(articleList,commentList, parentList,commentCount)
    extractSaveFeatures(articleList,commentList, parentList,commentCount)
    #extractSocialData(articleList, commentList, commentCount)
    
    # Vector Space
    #extractWordData(articleList, commentList, commentCount)
    
    
    #extractTopicData(articleList, commentCount,100)    
    #extractTimeData(articleList, commentCount)    
    #extractUserData()    
    #extractAllExperiment(articleList,parentList,500)
    
    
    
    
    
    
    

    
    

