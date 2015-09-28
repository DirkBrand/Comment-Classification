'''
Created on 18 Jun 2015

@author: Dirk
'''

import operator

from FeatureExtraction.main import load_numpy_matrix, load_sparse_csr
from FeatureExtraction.mainExtractor import extract_global_bag_of_words_processed,\
    read_comments, extract_values, UnigramAnalyzer, BigramAnalyzer,\
    TrigramAnalyzer, read_toy_comments, read_slashdot_comments,\
    CharacterSkipGramAnalyzer, CharacterAnalyzer, extract_global_bag_of_words
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.lda import LDA

from config import feature_set_path, comment_data_path
import numpy as np


def extract_words(vectorizor, train_list, test_list):
    count_vect = vectorizor.fit(train_list)
    train = count_vect.transform(train_list)
    test = count_vect.transform(test_list)
        
    return train, test, count_vect.get_feature_names()

set = 1
if __name__ == '__main__':
    if set == 1:
        articleList, commentList, parentList, commentCount = read_comments(comment_data_path + 'trainTestDataSet.txt', skip_mtn=False)
    elif set == 2:
        articleList, commentList, parentList, commentCount = read_toy_comments(comment_data_path + 'trainTestDataSet.txt', comment_data_path + 'toyComments.csv')
    elif set == 3:
        articleList, commentList, commentCount = read_slashdot_comments(comment_data_path + 'slashdotDataSet.txt', limit=100000)
    
    # Values
    y = extract_values(articleList, commentList, commentCount, set)
    sss = StratifiedShuffleSplit(y, 1, test_size=0.95, random_state=42)
    y_train = []
    y_test = []
    for train, test in sss:
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
        
        
        
        
        
    train, test, terms = extract_words(CountVectorizer(analyzer=UnigramAnalyzer(), dtype=float), train_list, test_list)  
    print train.shape 
    model = LDA()
    model.fit(train.toarray(), y_train)
    values = []
    for i, v in enumerate(model.coef_[0]):
        values.append(tuple([i,v]))
        
    values.sort(key=operator.itemgetter(1))
    values = values[::-1]
    values = values[:10]
    print ["%3d : %0.5f" % (i[0],i[1]) for i in values]
    print [terms[i[0]] for i in values]

    print "---------------------------"
    
    train, test, terms = extract_words(CountVectorizer(analyzer=BigramAnalyzer(), dtype=float), train_list, test_list)
    print train.shape 
    model = LDA()
    model.fit(train.toarray(), y_train)
    values = []
    for i, v in enumerate(model.coef_[0]):
        values.append(tuple([i,v]))
        
    values.sort(key=operator.itemgetter(1))
    values = values[::-1]
    values = values[:10]
    print ["%3d : %0.5f" % (i[0],i[1]) for i in values]
    print [terms[i[0]] for i in values]

    print "---------------------------"
    
    train, test, terms = extract_words(CountVectorizer(analyzer=TrigramAnalyzer(), dtype=float), train_list, test_list)
    print train.shape   
    model = LDA()
    model.fit(train.toarray(), y_train)
    values = []
    for i, v in enumerate(model.coef_[0]):
        values.append(tuple([i,v]))
        
    values.sort(key=operator.itemgetter(1))
    values = values[::-1]
    values = values[:10]
    print ["%3d : %0.5f" % (i[0],i[1]) for i in values]
    print [terms[i[0]] for i in values]

    print "---------------------------"
    
    
    
    
    
    
    
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
    
    
    
    train, test, terms = extract_words(CountVectorizer(analyzer=CharacterAnalyzer(), binary=True, dtype=float), train_list, test_list)
    print train.shape 
    model = LDA()
    model.fit(train.toarray(), y_train)
    values = []
    for i, v in enumerate(model.coef_[0]):
        values.append(tuple([i,v]))
        
    values.sort(key=operator.itemgetter(1))
    values = values[::-1]
    values = values[:10]
    print ["%3d : %0.5f" % (i[0],i[1]) for i in values]
    print [terms[i[0]] for i in values]

    print "---------------------------"
    
    
    train, test, terms = extract_words(CountVectorizer(analyzer=CharacterSkipGramAnalyzer(), binary=True, dtype=float), train_list, test_list)
    print train.shape 
    model = LDA()
    model.fit(train.toarray(), y_train)
    values = []
    for i, v in enumerate(model.coef_[0]):
        values.append(tuple([i,v]))
        
    values.sort(key=operator.itemgetter(1))
    values = values[::-1]
    values = values[:10]
    print ["%3d : %0.5f" % (i[0],i[1]) for i in values]
    print [terms[i[0]] for i in values]

    print "---------------------------"