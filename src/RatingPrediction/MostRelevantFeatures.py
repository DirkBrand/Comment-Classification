'''
Created on 18 Jun 2015

@author: Dirk
'''

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2

from FeatureExtraction.main import load_numpy_matrix, load_sparse_csr
from FeatureExtraction.mainExtractor import extract_global_bag_of_words_processed,\
    read_comments, extract_values, UnigramAnalyzer, BigramAnalyzer
from config import feature_set_path, comment_data_path

import numpy as np


def extract_words(vectorizor, train_list, test_list):
    count_vect = vectorizor.fit(train_list)
    train = count_vect.transform(train_list)
    test = count_vect.transform(test_list)
    
    #print count_vect.get_feature_names()[1000:1010]
    
    #print count_vect.get_feature_names()
    print "Train:", train.shape    
    print "Test:", test.shape  
    print  count_vect.vocabulary_
    
    return train, test, count_vect.get_feature_names()

if __name__ == '__main__':
    articleList, commentList, parentList, commentCount = read_comments(comment_data_path + 'trainTestDataSet.txt')
    
    # Values
    y = extract_values(articleList, commentList, parentList, commentCount)[:, 3]   
    sss = StratifiedShuffleSplit(y, 1, test_size=0.40, random_state=42)
    y_train = []
    y_test = []
    for train, test in sss:
        print train
        np.save('train_vect', train)
        np.save('test_vect', test)
        y_train = y[train]
        y_test = y[test]
    
    processed_comment_list = extract_global_bag_of_words_processed(commentList)  
    train_v, test_v = np.load('train_vect.npy'), np.load('test_vect.npy')
    train_list = []
    test_list = []
    for v in train_v:
        train_list.append(processed_comment_list[v])
    for v in test_v:
        test_list.append(processed_comment_list[v])
        
    #train, test, terms = extract_words(CountVectorizer(analyzer=UnigramAnalyzer(), dtype=float), train_list, test_list)
    train, test, terms = extract_words(CountVectorizer(analyzer=BigramAnalyzer(), dtype=float), train_list, test_list)
    
    
    
    
    
    selector2 = SelectKBest(score_func=chi2, k=min(50, train.shape[1])).fit(train,y_train)
    ind = [zero_based_index for zero_based_index in list(selector2.get_support(indices=True))]
    print np.asarray(terms)[selector2.get_support()]
    
    