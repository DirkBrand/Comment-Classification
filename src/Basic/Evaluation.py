'''
Created on 12 Aug 2015

@author: Dirk
'''
from FeatureExtraction.SentimentUtil import load_classifier
from FeatureExtraction.main import load_sparse_csr, load_numpy_matrix
from FeatureExtraction.mainExtractor import read_news24_comments,read_toy_comments, read_slashdot_comments
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
        df_comments = read_slashdot_comments(comment_data_path + 'slashdotDataSet_latest.txt')[:100000]
        df_comments.sort('date', inplace=True)
        df_comments.reset_index(inplace=True, drop=True)
        df_thread_groupby = df_comments.groupby('thread_root_id')
        set_tag = "_slashdot"
        tag = '_slashdot'
        
    
    processed_comment_list = []
    for _, row in df_comments.iterrows():  
        comm = row['comment_content'] 
        processed_comment_list.append(comm.decode('ascii','ignore'))
        
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
                
    
    print classification_report(valueVector, predicted, target_names=['0','1'],digits=3 )
    print draw_confusion_matrix(valueVector, predicted, ['ham','spam'])
    
    
    
    