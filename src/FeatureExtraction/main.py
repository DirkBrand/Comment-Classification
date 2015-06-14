'''
Created on 03 Mar 2014

@author: Dirk
'''

import pickle

import nltk
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse.csr import csr_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, \
    CountVectorizer
from sklearn.linear_model.stochastic_gradient import SGDRegressor

from DeepLearnings.FeatureExtraction import min_max_mean_features, \
    tfidf_weighted_sum_features, bag_of_centroids_features, getCommentCount
from DeepLearnings.Main import get_model
from FeatureExtraction.SentimentUtil import create_classifier, make_full_dict, \
    create_word_scores, find_best_words, evaluate_features, best_word_features, \
    extract_features
from FeatureExtraction.mainExtractor import read_comments, extract_values, extract_feature_matrix, \
    extract_Time_Data, read_user_comments, \
    extract_user_values, read_user_data, \
    extract_social_features, extract_sentence_values, extract_word_clusters, \
    extract_global_bag_of_words, extract_words, UnigramAnalyzer, BigramAnalyzer, \
    UnigramBigramAnalyzer, UnigramBigramTrigramAnalyzer, \
    UnigramBigramTrigramQuadgramAnalyzer, TrigramAnalyzer, QuadgramAnalyzer,\
    CharacterAnalyzer, LexicalBigramAnalyzer
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


def extractSaveFeatures(articleList, commentList, parentList, commentCount):
    featureMatrix = extract_feature_matrix(articleList, commentList, parentList, commentCount)
    print "Extracted Features"
    
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train = []
    test = []
    train, test = featureMatrix[train_v], featureMatrix[test_v]
    print train.shape
    print test.shape
    save_numpy_matrix(feature_set_path + "featureArray_train", train) 
    save_numpy_matrix(feature_set_path + "featureArray_test", test) 
    
def extractSaveValues(articleList, commentList, parentList, commentCount):
    valueVector = extract_values(articleList, commentList, parentList, commentCount)
    print "Extracted values"
    save_numpy_matrix(feature_set_path + "valueVector", valueVector) 
    
    
def extractSaveSentenceValues(articleList, commentList, parentList, commentCount):
    valueVector = extract_sentence_values(articleList, commentList, parentList, getCommentCount(4, commentList))
    print "Extracted values"
    save_numpy_matrix(feature_set_path + "sentenceValueVector", valueVector) 

def extractSynsetData(articleList, commentList, commentCount):
    wd = extract_word_clusters(commentList, commentCount)
    save_sparse_csr(feature_set_path + "clusteredWordData", wd) 

def extractLexicalBigramData(articleList, commentList, commentCount):
    processed_comment_list = extract_global_bag_of_words(commentList)   
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train_list = []
    test_list = []
    for v in train_v:
        train_list.append(processed_comment_list[v])
    for v in test_v:
        test_list.append(processed_comment_list[v])
        
    print len(train_list)
    print len(test_list)
    
    print 'Lexical Ngrams Binary'
    cb_train, cb_test = extract_words(CountVectorizer(analyzer=LexicalBigramAnalyzer(), binary=True, dtype=float), train_list, test_list)
    save_sparse_csr(feature_set_path + "binaryLexicalBigramsData_train", cb_train) 
    save_sparse_csr(feature_set_path + "binaryLexicalBigramsData_test", cb_test) 
    
    print 'Lexical Ngrams tfidf'
    ct_train, ct_test = extract_words(TfidfVectorizer(analyzer=LexicalBigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)    
    save_sparse_csr(feature_set_path + "tfidfLexicalBigramsData_train", ct_train) 
    save_sparse_csr(feature_set_path + "tfidfLexicalBigramsData_test", ct_test)
    
def extractCharacterData(articleList, commentList, commentCount):
    processed_comment_list = extract_global_bag_of_words(commentList)   
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train_list = []
    test_list = []
    for v in train_v:
        train_list.append(processed_comment_list[v])
    for v in test_v:
        test_list.append(processed_comment_list[v])
        
    print len(train_list)
    print len(test_list)
    
    print 'Character Ngrams Binary'
    cb_train, cb_test = extract_words(CountVectorizer(analyzer=CharacterAnalyzer(), binary=True, dtype=float), train_list, test_list)
    save_sparse_csr(feature_set_path + "binaryCharacterData_train", cb_train) 
    save_sparse_csr(feature_set_path + "binaryCharacterData_test", cb_test) 
    
    print 'Character Ngrams tfidf'
    ct_train, ct_test = extract_words(TfidfVectorizer(analyzer=CharacterAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)    
    save_sparse_csr(feature_set_path + "tfidfCharacterData_train", ct_train) 
    save_sparse_csr(feature_set_path + "tfidfCharacterData_test", ct_test) 
    
def extractWordData(articleList, commentList, commentCount):
    processed_comment_list = extract_global_bag_of_words(commentList)    
    
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train_list = []
    test_list = []
    for v in train_v:
        train_list.append(processed_comment_list[v])
    for v in test_v:
        test_list.append(processed_comment_list[v])        
        
    
    #train_list = [' '.join(sent) for sent in train_list]  
    #test_list = [' '.join(sent) for sent in test_list]    

    print len(train_list)
    print len(test_list)
    
    print 'Unigram Binary'
    bwd_train, bwd_test = extract_words(CountVectorizer(analyzer=UnigramAnalyzer(), binary=True, dtype=float), train_list, test_list)
    print 'Unigram Frequency'
    fwd_train, fwd_test = extract_words(CountVectorizer(analyzer=UnigramAnalyzer(), dtype=float), train_list, test_list)
    print 'Unigram TFIDF'
    twd_train, twd_test = extract_words(TfidfVectorizer(analyzer=UnigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)
    print 'Bigram Binary'
    bbwd_train, bbwd_test = extract_words(CountVectorizer(analyzer=UnigramBigramAnalyzer(), binary=True, dtype=float),train_list, test_list)
    print 'Bigram TFIDF'
    btwd_train, btwd_test = extract_words(TfidfVectorizer(analyzer=UnigramBigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float),train_list, test_list)
    print 'Trigram Binary'
    tbwd_train, tbwd_test = extract_words(CountVectorizer(analyzer=UnigramBigramTrigramAnalyzer(), binary=True, dtype=float),train_list, test_list)
    print 'Trigram TFIDF'
    ttwd_train, ttwd_test = extract_words(TfidfVectorizer(analyzer=UnigramBigramTrigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float),train_list, test_list)
    print 'Bigram Only Binary'
    bowd_train, bowd_test = extract_words(CountVectorizer(analyzer=BigramAnalyzer(), binary=True, dtype=float),train_list, test_list)
    print 'Bigram Only TFIDF'
    bowd2_train, bowd2_test = extract_words(TfidfVectorizer(analyzer=BigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float),train_list, test_list)
    print 'Trigram Only Binary'
    towd_train, towd_test = extract_words(CountVectorizer(analyzer=TrigramAnalyzer(), binary=True, dtype=float),train_list, test_list)
    print 'Trigram Only TFIDF'
    towd2_train, towd2_test = extract_words(TfidfVectorizer(analyzer=TrigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float),train_list, test_list)
   

    '''
    print 'Quadgram Binary'
    qbwd_train, qbwd_test = extract_words(CountVectorizer(analyzer=UnigramBigramTrigramQuadgramAnalyzer(), binary=True, dtype=float),train_list, test_list)
    print 'Quadgram TFIDF'
    qtwd_train, qtwd_test = extract_words(TfidfVectorizer(analyzer=UnigramBigramTrigramQuadgramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float),train_list, test_list)
    print 'Quadgram Only Binary'
    qowd_train, qowd_test = extract_words(CountVectorizer(analyzer=QuadgramAnalyzer(), binary=True, dtype=float),train_list, test_list)
    print 'Quadgram Only TFIDF'
    qowd2_train, qowd2_test = extract_words(TfidfVectorizer(analyzer=QuadgramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float),train_list, test_list)
    '''
    
    print(feature_set_path + "binaryWordData_train", bwd_train[123,:]) 
    print(feature_set_path + "freqWordData_train", fwd_train[123,:]) 
    print(feature_set_path + "tfidfWordData_train", twd_train[123,:]) 
    print(feature_set_path + "bigramBinaryWordData_train", bbwd_train[123,:]) 
    print(feature_set_path + "bigramTfidfWordData_train", btwd_train[123,:]) 
    print(feature_set_path + "trigramBinaryWordData_train", tbwd_train[123,:]) 
    print(feature_set_path + "trigramTfidfWordData_train", ttwd_train[123,:]) 
    
    print(feature_set_path + "bigramOnlyBinaryWordData_train", bowd_train[123,:])
    print(feature_set_path + "bigramOnlyTfidfWordData_train", bowd2_train[123,:])
    print(feature_set_path + "trigramOnlyBinaryWordData_train", towd_train[123,:])
    print(feature_set_path + "trigramOnlyTfidfWordData_train", towd2_train[123,:])
    
    '''
    print(feature_set_path + "quadgramBinaryWordData_train", qbwd_train[123,:]) 
    print(feature_set_path + "quadgramTfidfWordData_train", qtwd_train[123,:])  
    print(feature_set_path + "quadgramOnlyBinaryWordData_train", qowd_train[123,:]) 
    print(feature_set_path + "quadgramOnlyTfidfWordData_train", qowd2_train[123,:]) 
    '''
    
    
    save_sparse_csr(feature_set_path + "binaryWordData_train", bwd_train) 
    save_sparse_csr(feature_set_path + "freqWordData_train", fwd_train) 
    save_sparse_csr(feature_set_path + "tfidfWordData_train", twd_train) 
    save_sparse_csr(feature_set_path + "bigramBinaryWordData_train", bbwd_train) 
    save_sparse_csr(feature_set_path + "bigramTfidfWordData_train", btwd_train) 
    save_sparse_csr(feature_set_path + "trigramBinaryWordData_train", tbwd_train) 
    save_sparse_csr(feature_set_path + "trigramTfidfWordData_train", ttwd_train)  
    
    save_sparse_csr(feature_set_path + "bigramOnlyBinaryWordData_train", bowd_train)
    save_sparse_csr(feature_set_path + "bigramOnlyTfidfWordData_train", bowd2_train)
    save_sparse_csr(feature_set_path + "trigramOnlyBinaryWordData_train", towd_train)
    save_sparse_csr(feature_set_path + "trigramOnlyTfidfWordData_train", towd2_train)
    
    '''
    save_sparse_csr(feature_set_path + "quadgramBinaryWordData_train", qbwd_train) 
    save_sparse_csr(feature_set_path + "quadgramTfidfWordData_train", qtwd_train) 
    save_sparse_csr(feature_set_path + "quadgramOnlyBinaryWordData_train", qowd_train) 
    save_sparse_csr(feature_set_path + "quadgramOnlyTfidfWordData_train", qowd2_train)     
    '''
    
    save_sparse_csr(feature_set_path + "binaryWordData_test", bwd_test) 
    save_sparse_csr(feature_set_path + "freqWordData_test", fwd_test) 
    save_sparse_csr(feature_set_path + "tfidfWordData_test", twd_test) 
    save_sparse_csr(feature_set_path + "bigramBinaryWordData_test", bbwd_test) 
    save_sparse_csr(feature_set_path + "bigramTfidfWordData_test", btwd_test) 
    save_sparse_csr(feature_set_path + "trigramBinaryWordData_test", tbwd_test) 
    save_sparse_csr(feature_set_path + "trigramTfidfWordData_test", ttwd_test) 
    
    save_sparse_csr(feature_set_path + "bigramOnlyBinaryWordData_test", bowd_test)
    save_sparse_csr(feature_set_path + "bigramOnlyTfidfWordData_test", bowd2_test)
    save_sparse_csr(feature_set_path + "trigramOnlyBinaryWordData_test", towd_test)
    save_sparse_csr(feature_set_path + "trigramOnlyTfidfWordData_test", towd2_test)
    
    '''
    save_sparse_csr(feature_set_path + "quadgramBinaryWordData_test", qbwd_test) 
    save_sparse_csr(feature_set_path + "quadgramTfidfWordData_test", qtwd_test)  
    save_sparse_csr(feature_set_path + "quadgramOnlyBinaryWordData_test", qowd_test) 
    save_sparse_csr(feature_set_path + "quadgramOnlyTfidfWordData_test", qowd2_test) 
    '''
    
def extractSocialData(articleList, commentList, commentCount):
    userList, userCount = read_user_data(comment_data_path + 'userdata.txt');
    socialVector = extract_social_features(userList, commentList, commentCount) 
    
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train = []
    test = []
    train, test = socialVector[train_v], socialVector[test_v]
    print train.shape
    print test.shape
               
    print "Extracted social Features"
    save_numpy_matrix(feature_set_path + "socialVector_train", train) 
    save_numpy_matrix(feature_set_path + "socialVector_test", test) 
    
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
    save_numpy_matrix(feature_set_path + model_name + "_TfidfWeightedSumFeatures", fs2) 
    print "TWO"
    fs1 = min_max_mean_features(model_type, commentList, commentCount)    
    save_numpy_matrix(feature_set_path + model_name + "_MinMaxMeanFeatures", fs1)
    print "THREE"
    fs3 = bag_of_centroids_features(model_type, commentList, commentCount)
    save_numpy_matrix(feature_set_path + model_name + "_BagOfCentroidsFeatures", fs3)  
    print "Extracted Deep Learning Features"
 
def extractTimeData(articleList, commentList, commentCount):
    td = extract_Time_Data(articleList, commentCount)
    save_sparse_csr("timeData", td) 

    
    
    
def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)

def save_numpy_matrix(filename, array):
    np.save(filename, array)
    
def load_numpy_matrix(filename):
    return np.load(filename)
    
def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

def savePickledList(articleList, commentList, parentList):
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

    f = open("articleList" + '.pkl', 'rb')
    artList = pickle.load(f)
    f.close()
    f = open("commentList" + '.pkl', 'rb')
    artList = pickle.load(f)
    f.close()
    f = open("parentList" + '.pkl', 'rb')
    parentList = pickle.load(f)
    f.close()
    
    commCount = 0
    for commList in artList.values(): 
        commCount += len(commList)

    return artList, commentList, parentList, commCount


reLoad = True
if __name__ == '__main__':
    print 'START'
    # setup()
         
    # To process all the comments
    if reLoad:
        articleList, commentList, parentList, commentCount = read_comments(comment_data_path + 'trainTestDataSet.txt')
        # savePickledList(articleList, commentList, parentList)
    else:
        articleList, commentList, parentList, commentCount = loadPickledList()
        
    print "Processed", commentCount, "Comments"
    
    
    '''
    extractSaveValues(articleList, commentList, parentList, commentCount)
    y = load_numpy_matrix(feature_set_path + r'valueVector.npy')[:, 3]   
    sss = StratifiedShuffleSplit(y, 1, test_size=0.40, random_state=42)
    for train, test in sss:
        print train
        np.save('train_vect', train)
        np.save('test_vect', test)
        y_train = y[train]
        y_test = y[test]
    save_numpy_matrix(feature_set_path + "valueVector_train", y_train) 
    save_numpy_matrix(feature_set_path + "valueVector_test", y_test) 
    
    '''
    train = np.load('train_vect.npy')
    print train
    
    # Deep Learning Extraction
    # extractSaveWordEmbeddingFeatures(commentList, commentCount)
    # extractSaveSentenceValues(articleList,commentList, parentList,commentCount)
    
    # Classic
    #extractSaveFeatures(articleList,commentList, parentList,commentCount)
    #extractSocialData(articleList, commentList, commentCount)
    
    # Vector Space
    #extractWordData(articleList, commentList, commentCount)
    #extractCharacterData(articleList, commentList, commentCount)
    extractLexicalBigramData(articleList, commentList, commentCount)
    
    # extractSynsetData(articleList, commentList, commentCount)
    
    # extractTopicData(articleList, commentCount,100)    
    # extractTimeData(articleList, commentCount)    
    # extractUserData()    
    # extractAllExperiment(articleList,parentList,500)
    
    
    
    
    
    
    

    
    

