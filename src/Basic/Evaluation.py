'''
Created on 12 Aug 2015

@author: Dirk
'''
from FeatureExtraction.SentimentUtil import load_classifier
from FeatureExtraction.main import load_sparse_csr, load_numpy_matrix
from FeatureExtraction.mainExtractor import read_comments, read_toy_comments,\
    read_slashdot_comments, extract_global_bag_of_words_processed, extract_words,\
    UnigramBigramAnalyzer, extract_values
import scipy
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.metrics import classification_report, precision_score,\
    recall_score, f1_score, accuracy_score

from RatingPrediction.Classification import draw_confusion_matrix
from config import comment_data_path, spam_set_path, feature_set_path
import numpy as np


vals = ['ham', 'spam']

set = 2
if __name__ == '__main__':
    clf = load_classifier(spam_set_path + 'spam_classifier.pickle')
    vectorizer = load_classifier(spam_set_path + 'spam_vectorizer.pickle')
    
    # To process all the comments
    if set == 1:
        articleList, commentList, parentList, commentCount = read_comments(comment_data_path + 'trainTestDataSet.txt')
        tag = '_main'
    elif set == 2:
        articleList, commentList, parentList, commentCount = read_toy_comments(comment_data_path + 'trainTestDataSet.txt', comment_data_path + 'toyComments.csv')
        tag = '_toy'
    elif set == 3:
        articleList, commentList, commentCount = read_slashdot_comments(comment_data_path + 'slashdotDataSet.txt', limit=100000)
        tag = '_slashdot'
    
    processed_comment_list = []
    for art in commentList.items():        
        for comm in art[1]:  
            processed_comment_list.append(comm.body.decode('ascii','ignore'))
    features = vectorizer.transform(processed_comment_list)

    y_train = load_numpy_matrix(feature_set_path +  r'valueVector'+tag+'_train.npy')
    y_test = load_numpy_matrix(feature_set_path +  r'valueVector'+tag+'_test.npy')
    
    
    print features.shape
    print y_train.shape
    print y_test.shape
      
    valueVector = np.concatenate([y_train,y_test]) 
    print 
    print valueVector.shape
    
       
    
    # train_list = [' '.join(sent) for sent in train_list]  
    # test_list = [' '.join(sent) for sent in test_list]    
    predicted = [float(v) for v in clf.predict(features)]
    
                
    print "Accuracy: %0.3f " % (accuracy_score(valueVector, predicted))
                
    
    print classification_report(valueVector, predicted, target_names=['0','1'])
    print draw_confusion_matrix(valueVector, predicted, ['ham','spam'])
    
    
    
    