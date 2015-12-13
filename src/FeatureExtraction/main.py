'''
Created on 03 Mar 2014

@author: Dirk
'''

import datetime
import pickle

from FeatureExtraction.mainExtractor import extract_feature_matrix,\
    extract_values, extract_word_clusters, extract_global_bag_of_words,\
    extract_words, LexicalBigramUnigramAnalyzer, CharacterAnalyzer,\
    CharacterSkipGramAnalyzer, extract_global_bag_of_words_processed,\
    UnigramAnalyzer, UnigramBigramAnalyzer, UnigramBigramTrigramAnalyzer,\
    BigramAnalyzer, TrigramAnalyzer, read_user_data, extract_social_features,\
    extract_Time_Data, read_news24_comments, read_toy_comments,\
    read_slashdot_comments
from gensim import matutils, models, corpora
import nltk
from nltk.stem.snowball import SnowballStemmer
from scipy import sparse
from scipy.sparse.csr import csr_matrix
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, \
    CountVectorizer
from sklearn.linear_model.stochastic_gradient import SGDRegressor

from DeepLearnings.Features import getCommentCount, tfidf_weighted_sum_features, \
    train_clusterer, bag_of_centroids_features, get_paragraph_features
from DeepLearnings.Main import get_model
from DeepLearnings.ModelTraining import comment_to_wordlist,\
    comment_to_words_for_topics
from config import sentiment_path, feature_set_path, comment_data_path,\
    model_path
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




def extractSaveFeatures(df_comments, df_thread_groupby, tag):
    featureMatrix = extract_feature_matrix(df_comments, df_thread_groupby)
    print "Extracted Features"
    
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train = []
    test = []
    train, test = featureMatrix[train_v], featureMatrix[test_v]
    print train.shape
    print test.shape
    save_numpy_matrix(feature_set_path + "featureArray" + tag + "_train", train) 
    save_numpy_matrix(feature_set_path + "featureArray" + tag + "_test", test) 
    
def extractSaveValues(df_comments, filename, datatype):
    valueVector = extract_values(df_comments, datatype)
    print "Extracted values"
    save_numpy_matrix(filename, valueVector) 
    


def extractSynsetData(articleList, commentList, commentCount):
    wd = extract_word_clusters(commentList, commentCount)
    save_sparse_csr(feature_set_path + "clusteredWordData", wd) 

def extractLexicalBigramData(articleList, commentList, commentCount):
    comment_list = extract_global_bag_of_words(commentList)   
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train_list = []
    test_list = []
    for v in train_v:
        train_list.append(comment_list[v])
    for v in test_v:
        test_list.append(comment_list[v])
        
    print len(train_list)
    print len(test_list)
    
    print 'Lexical Ngrams Binary'
    cb_train, cb_test = extract_words(CountVectorizer(analyzer=LexicalBigramUnigramAnalyzer(), binary=True, dtype=float), train_list, test_list)
    save_sparse_csr(feature_set_path + "binaryLexicalBigramsData_train", cb_train) 
    save_sparse_csr(feature_set_path + "binaryLexicalBigramsData_test", cb_test) 
    
    print 'Lexical Ngrams tfidf'
    ct_train, ct_test = extract_words(TfidfVectorizer(analyzer=LexicalBigramUnigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)    
    save_sparse_csr(feature_set_path + "tfidfLexicalBigramsData_train", ct_train) 
    save_sparse_csr(feature_set_path + "tfidfLexicalBigramsData_test", ct_test)
    
def extractCharacterData(df_comments, tag):
    comment_list = extract_global_bag_of_words(df_comments)   
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train_list = []
    test_list = []
    for v in train_v:
        train_list.append(comment_list[v])
    for v in test_v:
        test_list.append(comment_list[v])
        
    print len(train_list)
    print len(test_list)
    
    print 'Character Ngrams Binary'
    cb_train, cb_test = extract_words(CountVectorizer(analyzer=CharacterAnalyzer(), binary=True, dtype=float), train_list, test_list)
    save_sparse_csr(feature_set_path + "binaryCharacterData" + tag + "_train", cb_train) 
    save_sparse_csr(feature_set_path + "binaryCharacterData" + tag + "_test", cb_test) 
    
    print 'Character Ngrams tfidf'
    ct_train, ct_test = extract_words(TfidfVectorizer(analyzer=CharacterAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)    
    save_sparse_csr(feature_set_path + "tfidfCharacterData" + tag + "_train", ct_train) 
    save_sparse_csr(feature_set_path + "tfidfCharacterData" + tag + "_test", ct_test) 
     
    print 'Character skipgrams Binary'
    sb_train, sb_test = extract_words(CountVectorizer(analyzer=CharacterSkipGramAnalyzer(), binary=True, dtype=float), train_list, test_list)
    save_sparse_csr(feature_set_path + "binaryCharacterSkipgramData" + tag + "_train", sb_train) 
    save_sparse_csr(feature_set_path + "binaryCharacterSkipgramData" + tag + "_test", sb_test) 
    
    print 'Character skipgrams TFIDF'
    sb_train, sb_test = extract_words(TfidfVectorizer(analyzer=CharacterSkipGramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)
    save_sparse_csr(feature_set_path + "tfidfCharacterSkipgramData" + tag + "_train", sb_train) 
    save_sparse_csr(feature_set_path + "tfidfCharacterSkipgramData" + tag + "_test", sb_test)     

    
    
def extractTopicModelData(df_comments, set_tag, tag):
    corpus = []
    i = 0
    for _, row in df_comments.iterrows():   
        comm = row['comment_content']
        corpus.append(comment_to_words_for_topics(comm))
        i += 1
        if i % 100 == 0:
            print i,datetime.datetime.now().time()
    
         
    print len(corpus)
    
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train_list = []
    test_list = []
    for v in train_v:
        train_list.append(corpus[v])
    for v in test_v:
        test_list.append(corpus[v])    
        
    lda = models.LdaModel.load(model_path + set_tag.replace("_","") + "_lda_model")
    
    dictionary = corpora.Dictionary.load(model_path + set_tag.replace("_","") + "_dictionary")
    train = [dictionary.doc2bow(text) for text in train_list]
    test = [dictionary.doc2bow(text) for text in test_list]
    
    lda.print_topics(20, 5)
    
    docTopicProbMat_train = lda[train]
    docTopicProbMat_test = lda[test]
    
    #print lda.top_topics(docTopicProbMat_train, 10)
    
    train_lda=matutils.corpus2dense(docTopicProbMat_train, 100, num_docs=len(train)).transpose()
    test_lda=matutils.corpus2dense(docTopicProbMat_test, 100, num_docs=len(test)).transpose()
      
    print train_lda.shape
    print test_lda.shape
    
    save_sparse_csr(feature_set_path + set_tag + "lda" + tag + "_train", sparse.csr_matrix(train_lda)) 
    save_sparse_csr(feature_set_path + set_tag + "lda" + tag + "_test",  sparse.csr_matrix(test_lda)) 
    
    print "DONE LDA"
    
    
def extractWordData(df_comments, tag):
    processed_comment_list = extract_global_bag_of_words_processed(df_comments)    
    print len(processed_comment_list)
    
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train_list = []
    test_list = []
    for v in train_v:
        train_list.append(processed_comment_list[v])
    for v in test_v:
        test_list.append(processed_comment_list[v])        
        
    
    # train_list = [' '.join(sent) for sent in train_list]  
    # test_list = [' '.join(sent) for sent in test_list]    

    print len(train_list)
    print len(test_list)
    
    print 'Unigram Binary'
    bwd_train, bwd_test = extract_words(CountVectorizer(analyzer=UnigramAnalyzer(), binary=True, dtype=float), train_list, test_list)
    print 'Unigram Frequency'
    fwd_train, fwd_test = extract_words(CountVectorizer(analyzer=UnigramAnalyzer(), dtype=float), train_list, test_list)
    print 'Unigram TFIDF'
    twd_train, twd_test = extract_words(TfidfVectorizer(analyzer=UnigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)
    print 'Bigram Binary'
    bbwd_train, bbwd_test = extract_words(CountVectorizer(analyzer=UnigramBigramAnalyzer(), binary=True, dtype=float), train_list, test_list)
    print 'Bigram TFIDF'
    btwd_train, btwd_test = extract_words(TfidfVectorizer(analyzer=UnigramBigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)
    print 'Trigram Binary'
    tbwd_train, tbwd_test = extract_words(CountVectorizer(analyzer=UnigramBigramTrigramAnalyzer(), binary=True, dtype=float), train_list, test_list)
    print 'Trigram TFIDF'
    ttwd_train, ttwd_test = extract_words(TfidfVectorizer(analyzer=UnigramBigramTrigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)
    print 'Bigram Only Binary'
    bowd_train, bowd_test = extract_words(CountVectorizer(analyzer=BigramAnalyzer(), binary=True, dtype=float), train_list, test_list)
    print 'Bigram Only TFIDF'
    bowd2_train, bowd2_test = extract_words(TfidfVectorizer(analyzer=BigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)
    print 'Trigram Only Binary'
    towd_train, towd_test = extract_words(CountVectorizer(analyzer=TrigramAnalyzer(), binary=True, dtype=float), train_list, test_list)
    print 'Trigram Only TFIDF'
    towd2_train, towd2_test = extract_words(TfidfVectorizer(analyzer=TrigramAnalyzer(), use_idf=True, smooth_idf=True, dtype=float), train_list, test_list)
  
    print(feature_set_path + "binaryWordData_train", bwd_train[123, :]) 
    print(feature_set_path + "freqWordData_train", fwd_train[123, :]) 
    print(feature_set_path + "tfidfWordData_train", twd_train[123, :]) 
    print(feature_set_path + "bigramBinaryWordData_train", bbwd_train[123, :]) 
    print(feature_set_path + "bigramTfidfWordData_train", btwd_train[123, :]) 
    print(feature_set_path + "trigramBinaryWordData_train", tbwd_train[123, :]) 
    print(feature_set_path + "trigramTfidfWordData_train", ttwd_train[123, :]) 
    
    print(feature_set_path + "bigramOnlyBinaryWordData_train", bowd_train[123, :])
    print(feature_set_path + "bigramOnlyTfidfWordData_train", bowd2_train[123, :])
    print(feature_set_path + "trigramOnlyBinaryWordData_train", towd_train[123, :])
    print(feature_set_path + "trigramOnlyTfidfWordData_train", towd2_train[123, :])
    
   
    
    
    save_sparse_csr(feature_set_path + "binaryWordData" + tag + "_train", bwd_train) 
    save_sparse_csr(feature_set_path + "freqWordData" + tag + "_train", fwd_train) 
    save_sparse_csr(feature_set_path + "tfidfWordData" + tag + "_train", twd_train) 
    save_sparse_csr(feature_set_path + "bigramBinaryWordData" + tag + "_train", bbwd_train) 
    save_sparse_csr(feature_set_path + "bigramTfidfWordData" + tag + "_train", btwd_train) 
    save_sparse_csr(feature_set_path + "trigramBinaryWordData" + tag + "_train", tbwd_train) 
    save_sparse_csr(feature_set_path + "trigramTfidfWordData" + tag + "_train", ttwd_train)  
    
    save_sparse_csr(feature_set_path + "bigramOnlyBinaryWordData" + tag + "_train", bowd_train)
    save_sparse_csr(feature_set_path + "bigramOnlyTfidfWordData" + tag + "_train", bowd2_train)
    save_sparse_csr(feature_set_path + "trigramOnlyBinaryWordData" + tag + "_train", towd_train)
    save_sparse_csr(feature_set_path + "trigramOnlyTfidfWordData" + tag + "_train", towd2_train)
    
   
    save_sparse_csr(feature_set_path + "binaryWordData" + tag + "_test", bwd_test) 
    save_sparse_csr(feature_set_path + "freqWordData" + tag + "_test", fwd_test) 
    save_sparse_csr(feature_set_path + "tfidfWordData" + tag + "_test", twd_test) 
    save_sparse_csr(feature_set_path + "bigramBinaryWordData" + tag + "_test", bbwd_test) 
    save_sparse_csr(feature_set_path + "bigramTfidfWordData" + tag + "_test", btwd_test) 
    save_sparse_csr(feature_set_path + "trigramBinaryWordData" + tag + "_test", tbwd_test) 
    save_sparse_csr(feature_set_path + "trigramTfidfWordData" + tag + "_test", ttwd_test) 
    
    save_sparse_csr(feature_set_path + "bigramOnlyBinaryWordData" + tag + "_test", bowd_test)
    save_sparse_csr(feature_set_path + "bigramOnlyTfidfWordData" + tag + "_test", bowd2_test)
    save_sparse_csr(feature_set_path + "trigramOnlyBinaryWordData" + tag + "_test", towd_test)
    save_sparse_csr(feature_set_path + "trigramOnlyTfidfWordData" + tag + "_test", towd2_test)
   
    
def extractSocialData(df_comments, tag):
    socialVector = extract_social_features(df_comments)
    
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train = []
    test = []
    train, test = socialVector[train_v], socialVector[test_v]
    print train.shape
    print test.shape
               
    print "Extracted social Features"
    save_numpy_matrix(feature_set_path + "socialVector" + tag + "_train", train) 
    save_numpy_matrix(feature_set_path + "socialVector" + tag + "_test", test) 
    

def extractSaveWordEmbeddingFeatures(df_comments, modeltag, datatag):       
    
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train = []
    test = []
    
    print len(train_v), len(test_v)
   
    print "ONE"
    fs1 = tfidf_weighted_sum_features(get_model(1,"Basic300" + modeltag), df_comments)
    train, test = fs1[train_v], fs1[test_v]
    save_numpy_matrix(feature_set_path + "Basic300" + modeltag + "_TfidfFeatures" + datatag + "_train", train) 
    save_numpy_matrix(feature_set_path + "Basic300" + modeltag + "_TfidfFeatures" + datatag + "_test", test) 
      
    
    '''
    print "TWO"    
    centroid_map, num_clusters = train_clusterer(get_model(1,"Basic300" + modeltag), commentList)
    fs2 = bag_of_centroids_features(commentList, commentCount, num_clusters, centroid_map)   
    train, test = fs2[train_v], fs2[test_v]
    save_numpy_matrix(feature_set_path + "Basic300" + modeltag + "_BOCFeatures" + datatag + "_train", train) 
    save_numpy_matrix(feature_set_path + "Basic300" + modeltag + "_BOCFeatures" + datatag + "_test", test)  
    '''
    print "THREE"
    fs1 = get_paragraph_features(get_model(10,"DocBasic300" + modeltag), df_comments)
    train, test = fs1[train_v], fs1[test_v]
    save_numpy_matrix(feature_set_path + "DocBasic300" + modeltag + "_ParagraphFeatures" + datatag + "_train", train) 
    save_numpy_matrix(feature_set_path + "DocBasic300" + modeltag + "_ParagraphFeatures" + datatag + "_test", test)   
    
         
    print "FOUR"
    fs1 = tfidf_weighted_sum_features(get_model(99,"GoogleNews-vectors-negative300.bin"), df_comments)
    train, test = fs1[train_v], fs1[test_v]
    save_numpy_matrix(feature_set_path + "Google_TfidfFeatures" + datatag + "_train", train) 
    save_numpy_matrix(feature_set_path + "Google_TfidfFeatures" + datatag + "_test", test)      
         
    
    
    print train.shape
    print test.shape
    
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


set = 3
if __name__ == '__main__':
    print 'START'
         
    # To process all the comments
    if set == 1:
        df_comments = read_news24_comments(comment_data_path + 'trainTestDataSet.txt')
        df_comments.sort('date', inplace=True)
        df_comments.reset_index(inplace=True, drop=True)
        df_thread_groupby = df_comments.groupby('thread_root_id')
        set_tag = "_news24"
        tag = '_main'
    elif set == 2:
        df_comments = read_toy_comments(comment_data_path + 'trainTestDataSet.txt', comment_data_path + 'toyComments.csv')
        df_comments.sort('date', inplace=True)
        df_comments.reset_index(inplace=True, drop=True)
        df_thread_groupby = df_comments.groupby('thread_root_id')
        set_tag = "_news24"
        tag = '_toy'
    elif set == 3:
        df_comments = read_slashdot_comments(comment_data_path + 'slashdotDataSet_latest.txt')
        df_comments.sort('date', inplace=True)
        df_comments.reset_index(inplace=True, drop=True)
        df_thread_groupby = df_comments.groupby('thread_root_id')
        set_tag = "_slashdot"
        tag = '_slashdot'
        
   
    print df_comments.shape
    print df_comments.head()         
    
    
    # Get values and split train-test
    extractSaveValues(df_comments, feature_set_path + "valueVector" + tag, set)
    y = load_numpy_matrix(feature_set_path + r'valueVector' + tag + '.npy') 
    sss = StratifiedShuffleSplit(y , 1, test_size=0.40, random_state=42)
    for train, test in sss:
        np.save('train_vect', train)
        np.save('test_vect', test)
        y_train = y[train]
        y_test = y[test]
        save_numpy_matrix(feature_set_path + "valueVector" + tag + "_train", y_train) 
        save_numpy_matrix(feature_set_path + "valueVector" + tag + "_test", y_test) 
        print y_train.shape
        print y_test.shape

    
    # Features  
    extractSaveFeatures(df_comments, df_thread_groupby, tag)
    extractSocialData(df_comments, tag)
    
    
    # Ngrams
    extractWordData(df_comments, tag) 
    extractCharacterData(df_comments, tag)  
    
    # LDA
    #extractTopicModelData(df_comments, set_tag, tag)
    
    # Deep Learning
    #extractSaveWordEmbeddingFeatures(df_comments, set_tag, tag)    
    
    
    
   
    
    
    
    

    
    

