'''
Created on 24 Mar 2014

@author: Dirk
'''

from collections import Counter
import os
import pickle

from FeatureExtraction.main import load_sparse_csr, load_numpy_matrix
import nltk
from nltk.metrics.scores import precision, recall
from scipy.sparse import sparsetools
from sklearn import cross_validation
from sklearn import preprocessing, decomposition as deco, svm
from sklearn.datasets import make_classification
from sklearn.externals import joblib
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.feature_selection.univariate_selection import f_regression, chi2
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score,\
    metrics
from sklearn.metrics.metrics import precision_score, recall_score,\
    classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from RatingPrediction.Classification import *
from RatingPrediction.Regression import *
from config import feature_set_path
import numpy as np


VALUES = [ 'totalVotes', 'percentage of total','ratio','Status']
FEATURES = [ 'CF', 'BTF','FTF','TTF', 'BBTF', 'BTTF', 'TBTF', 'TTTF', 'QBTF', 'QTTF', 'BiOnlyBi','BiOnlyTFIDF', 'TriOnlyBi', 'TriOnlyTFIDF','QuadOnlyBi','QuadOnlyTFIDF', 'binaryChar', 'tfidfChar', 'binary skipgram', 'tfidf skipgram',  'LDA', 'word2vec tfidf','doc2vec','google tfidf']
            



def load_training_data(x_filename, y_filename):    
    tempCWD = os.getcwd()
    
    os.chdir('D:\Workspace\NLTK comments\src\FeatureExtraction')
    
    fs = np.load(x_filename)
    vv = np.load(y_filename)
    
    os.chdir(tempCWD)
    return fs, vv
   

def binScaling(values, nrBins):
    bins = []
    jump = 100 / nrBins
    for i in range(nrBins - 1):
        bins.append(np.percentile(values, jump * (i + 1)))
    
    bins.append(np.max(values))
    #print bins
    
    temp = np.zeros(len(values))
    for i in range(len(values)):
        val = 1
        for b in bins:
            if values[i] <= b:
                temp[i] = val
                break
            val += 1
        
    return temp

def getDateTime(filename):
       
    tempCWD = os.getcwd()
    
    os.chdir('D:\Workspace\NLTK comments\src\FeatureExtraction')
    
    dt = np.load(filename)
    
    os.chdir(tempCWD)
    
    return dt
    
    

def runRegressionModelTest(featureSet, valueVector, model):    
    output = ''
    clf = 0
    if model == 1:
        print "\nLINEAR REGRESSION\n"
        clf = linear_regression_fit(featureSet, valueVector)
    elif model == 2:
        print "\nSVR\n"
        clf = SVR_fit(featureSet, valueVector)
    elif model == 4:
        print "\nSTOCHASTIC\n"
        clf = SGD_r_fit(featureSet, valueVector)
        joblib.dump(clf, 'sgd.pkl')
    elif model == 5:        
        print "\nNEIGHBOURS\n"
        clf = neighbours_fit(featureSet, valueVector)
    elif model == 6:        
        print "\nLOGISTIC\n"
        clf = log_regression_fit(featureSet, valueVector)
    elif model == 7:        
        print "\nBAYESIANRIDGE\n"
        clf = bayesian_ridge_fit(featureSet, valueVector)
    elif model == 8:        
        print "\nRIDGE\n"
        clf = ridge_fit(featureSet, valueVector)
    elif model == 9:        
        print "\nELASTIC NET\n"
        clf = elastic_fit(featureSet, valueVector)
    elif model == 10:        
        print "\nLASSO\n"
        clf = lasso_fit(featureSet, valueVector)
    else :
        print 'Invalid choice\n'
    
    
    return clf 

def runClassificationTest(X, y, model, featureset, datatype):
    clf = 0
    kernel='rbf'
    C=0
    Lc = 0
    gamma=0
    if datatype==1:
        if featureset == 0: #Custom Features
            C = 10000
            Lc = 10000
            gamma = 1.0
        if featureset == 1: #Binary word Features
            C = 10
            Lc = 1
            gamma = 0.1
        if featureset == 2: #Freq Word Features
            C = 10
            Lc = 1
            gamma = 0.01 
        if featureset == 3: #TFIDF word Features
            C =10
            Lc = 1
            gamma = 0.1
        if featureset == 4: #Bigram binary word Features
            C = 10
            Lc = 0.01
            gamma = 0.01
        if featureset == 5: #Bigram tfidf word Features
            C = 1000
            Lc = 10
            gamma = 0.01
        if featureset == 6: #Trigram binary word Features
            C = 100
            Lc = 0.1
            gamma = 0.001
        if featureset == 7: #Trigram tfidf word Features
            C = 1000
            Lc = 10
            gamma = 0.01
        if featureset == 8: #Quadgram binary word Features
            C = 10
            Lc = 0.1
            gamma = 0.01
        if featureset == 9: #Quadgram tfidf word Features
            C = 10
            Lc = 10
            gamma = 0.37275937203149417
        if featureset == 10: #Bigram ONLY binary word Features
            C = 100000
            Lc = 10
            gamma = 0.1
        if featureset == 11: #Bigram ONLY tfidf word Features
            C = 100
            Lc = 10
            gamma = 0.01
        if featureset == 12: #Triigram ONLY binary word Features
            C = 100
            Lc = 0.1
            gamma = 0.001
        if featureset == 13: #Triigram ONLY tfidf word Features
            C = 10
            Lc = 1
            gamma = 0.1
        if featureset == 14: #Quadgram ONLY binary word Features
            C = 1
            Lc = 10
            gamma = 0.071968567300115208
        if featureset == 15: #Quadgram ONLY tfidf word Features
            C = 1000000
            Lc = 1000
            gamma = 0.013894954943731374
        if featureset == 16: #Binary Character Features
            C = 100 
            Lc =  0.001
            gamma = 0.0001
        if featureset == 17: #TFDIF Character Features
            C = 100000 
            Lc = 1000 
            gamma = 0.001
        if featureset == 18: #Binary character skipgram Features
            C = 10 
            Lc = 0.01
            gamma =10
        if featureset == 19: #TFIDF character skipgram Features
            C = 100 
            Lc = 1 
            gamma =1
        if featureset == 20: #LDA Features
            C = 1 
            Lc = 1 
            gamma = 10
        if featureset == 21: # Basic300 tfidf
            C = 1
            gamma = 0.0001
            Lc = 10
        if featureset == 22: # DocBasic300 
            C = 1
            gamma = 10
            Lc = 0.1
        if featureset == 23: # Google tfidf
            C = 1
            gamma = 0.001
            Lc = 10
    elif datatype==2: # TOY
        if featureset == 0: 
            C = 10000000
            Lc = 10000
            gamma = 0.0001
        if featureset == 1:
            C = 100000
            gamma = 0.000001
            Lc = 1
        if featureset == 2:
            C = 100000
            gamma = 0.1
            Lc = 1
        if featureset == 3:
            C = 1000
            gamma = 0.0001
            Lc = 1
        if featureset == 4:
            C = 1000000
            gamma = 0.000001
            Lc = 1
        if featureset == 5:
            C = 1000000
            gamma = 0.000001
            Lc = 1
        if featureset == 6:
            C = 10
            gamma = 0.001
            Lc = 1
        if featureset == 7:
            C = 10
            gamma = 0.001
            Lc = 1
        if featureset == 10:
            C = 1000000
            gamma = 0.000001
            Lc = 1
        if featureset == 11:
            C = 1000000
            gamma = 0.000001
            Lc = 1
        if featureset == 12:
            C = 100000
            gamma = 0.00001
            Lc = 1
        if featureset == 13:
            C = 100000
            gamma = 0.00001
            Lc = 1
        if featureset == 16:
            C = 100
            gamma = 0.0001
            Lc = 0.010
        if featureset == 17:
            C = 1000
            gamma = 0.00001
            Lc = 0.01
        if featureset == 18:
            C = 100
            gamma = 0.001
            Lc = 0.1
        if featureset == 19:
            C = 10000
            gamma = 0.00001
            Lc = 0.1
        if featureset == 20:
            C = 10000
            gamma = 0.1
            Lc = 1
        if featureset == 21:
            C = 1
            gamma = 10
            Lc = 0.1
        if featureset == 22:
            C = 1
            gamma = 10
            Lc = 1
        if featureset == 23:
            C = 1
            gamma = 10
            Lc = 10
    elif datatype==3: # SLASHDOT
        if featureset == 0:
            C = 1000
            gamma = 0.01
            Lc = 10
        if featureset == 1:
            C = 100
            gamma = 0.001
            Lc = 0.1
        if featureset == 2:
            C = 10000
            gamma = 0.00001
            Lc = 0.1
        if featureset == 3:
            C = 100
            gamma = 0.001
            Lc = 0.1
        if featureset == 4:
            C = 1000000
            gamma = 0.000001
            Lc = 1
        if featureset == 5:
            C = 100000
            gamma = 0.000001
            Lc = 0.1
        if featureset == 6:
            C = 10
            gamma = 0.01
            Lc = 1
        if featureset == 7:
            C = 10
            gamma = 0.01
            Lc = 0.1
        if featureset == 10:
            C = 10
            gamma = 0.1
            Lc = 1
        if featureset == 11:
            C = 10
            gamma = 0.1
            Lc = 1
        if featureset == 12:
            C = 10
            gamma = 0.1
            Lc = 1
        if featureset == 13:
            C = 10
            gamma = 0.1
            Lc = 10
        if featureset == 16:
            C = 1000
            gamma = 0.01
            Lc = 1
        if featureset == 17:
            C = 10
            gamma = 0.01
            Lc = 0.1
        if featureset == 18:
            C = 1000
            gamma = 0.00001
            Lc = 0.01
        if featureset == 19:
            C = 1
            gamma = 0.001
            Lc = 0.1
        if featureset == 20:
            C = 10
            gamma = 1000
            Lc = 10
        if featureset == 21:
            C = 1
            gamma = 10
            Lc = 0.1
        if featureset == 22:
            C = 1
            gamma = 10
            Lc = 1
        if featureset == 23:
            C = 1
            gamma = 10
            Lc = 10
        
           
    if model == 1:
        print "\nSVC\n"
        clf = svc_fit(X, y, kernel=kernel, C=C, gamma=gamma)
    elif model == 2:
        print '\nLinearSVC\n'
        clf = linear_svc_fit(X, y, C=Lc)
    elif model == 3:
        print '\nStochasticGradientDescent\n'
        clf = SGD_c_fit(X, y)  
    elif model == 4:
        print '\nKNearestNeighbours\n'
        clf = nearest_fit(X, y)
    elif model == 5:
        print '\nRandomForest\n'
        clf = random_forest_fit(X, y)
    elif model == 6:
        print '\nLogistic\n'
        clf = log_regression_fit(X, y)
    elif model == 7:
        print '\nDecisionTrees\n'
        clf = decision_tree_fit(X, y)   

    return clf;


def normalize_sets_sparse (train, test):
    norm = train.copy()
    norm.data **= 2 # Square every value
    norm = norm.sum(axis=0) # Sum every column
    n_nonzeros = np.where(norm > 0)
    norm[n_nonzeros] = 1.0 / np.sqrt(norm[n_nonzeros])
    norm = np.array(norm).T[0]
    
    sparsetools.csr_scale_columns(train.shape[0], train.shape[1], train.indptr, train.indices, train.data, norm)
    sparsetools.csr_scale_columns(test.shape[0], test.shape[1], test.indptr, test.indices, test.data, norm)
    
    return train, test


def normalized(a, norm):
    l2 = np.atleast_1d(norm)
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis=0)

def normalize_sets_dense (train, test):    
    norm1 =  np.linalg.norm(train, axis=0)       
    return normalized(train, norm1), normalized(test, norm1)



reg = False
scale = True
datatype = 3

  
if __name__ == '__main__':
    if datatype == 1:
        tag = '_main'
    elif datatype == 2:
        tag = "_toy"
    elif datatype == 3:
        tag = '_slashdot'
    
    for featureV in [0,1,2,3,4,5,6,7,10,11,12,13,16,17,18,19,20]: 
    #for featureV in [21,22,23]:          
        y_train = load_numpy_matrix(feature_set_path +  r'valueVector'+tag+'_train.npy')
        y_test = load_numpy_matrix(feature_set_path +  r'valueVector'+tag+'_test.npy')
        
        if featureV == 0:
            X_train = load_numpy_matrix(feature_set_path +  r'featureArray'+tag+'_train.npy')
            sd = load_numpy_matrix(feature_set_path +  r'socialVector'+tag+'_train.npy')
            X_train =  np.hstack((X_train,sd))
            X_test = load_numpy_matrix(feature_set_path +  r'featureArray'+tag+'_test.npy')
            sd2 = load_numpy_matrix(feature_set_path +  r'socialVector'+tag+'_test.npy')
            X_test =  np.hstack((X_test,sd2))
            perc = 80
        elif featureV == 1:
            X_train = load_sparse_csr(feature_set_path +  r'binaryWordData'+tag+'_train.npz')  
            X_test = load_sparse_csr(feature_set_path +  r'binaryWordData'+tag+'_test.npz')  
        elif featureV == 2:
            X_train = load_sparse_csr(feature_set_path +  r'freqWordData'+tag+'_train.npz')  
            X_test = load_sparse_csr(feature_set_path +  r'freqWordData'+tag+'_test.npz') 
        elif featureV == 3:
            X_train = load_sparse_csr(feature_set_path +  r'tfidfWordData'+tag+'_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'tfidfWordData'+tag+'_test.npz') 
        elif featureV == 4:
            X_train = load_sparse_csr(feature_set_path +  r'bigramBinaryWordData'+tag+'_train.npz')  
            X_test = load_sparse_csr(feature_set_path +  r'bigramBinaryWordData'+tag+'_test.npz')  
        elif featureV == 5:
            X_train = load_sparse_csr(feature_set_path +  r'bigramTfidfWordData'+tag+'_train.npz')  
            X_test = load_sparse_csr(feature_set_path +  r'bigramTfidfWordData'+tag+'_test.npz')  
        elif featureV == 6:
            X_train = load_sparse_csr(feature_set_path +  r'trigramBinaryWordData'+tag+'_train.npz')  
            X_test = load_sparse_csr(feature_set_path +  r'trigramBinaryWordData'+tag+'_test.npz') 
        elif featureV == 7:
            X_train = load_sparse_csr(feature_set_path +  r'trigramTfidfWordData'+tag+'_train.npz')  
            X_test = load_sparse_csr(feature_set_path +  r'trigramTfidfWordData'+tag+'_test.npz')  
        elif featureV == 8:
            X_train = load_sparse_csr(feature_set_path +  r'quadgramBinaryWordData'+tag+'_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'quadgramBinaryWordData'+tag+'_test.npz')  
        elif featureV == 9:
            X_train = load_sparse_csr(feature_set_path +  r'quadgramTfidfWordData'+tag+'_train.npz')  
            X_test = load_sparse_csr(feature_set_path +  r'quadgramTfidfWordData'+tag+'_test.npz')
        elif featureV == 10:
            X_train = load_sparse_csr(feature_set_path +  r'bigramOnlyBinaryWordData'+tag+'_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'bigramOnlyBinaryWordData'+tag+'_test.npz') 
        elif featureV == 11:
            X_train = load_sparse_csr(feature_set_path +  r'bigramOnlyTfidfWordData'+tag+'_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'bigramOnlyTfidfWordData'+tag+'_test.npz') 
        elif featureV == 12:
            X_train = load_sparse_csr(feature_set_path +  r'trigramOnlyBinaryWordData'+tag+'_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'trigramOnlyBinaryWordData'+tag+'_test.npz') 
        elif featureV == 13:
            X_train = load_sparse_csr(feature_set_path +  r'trigramOnlyTfidfWordData'+tag+'_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'trigramOnlyTfidfWordData'+tag+'_test.npz')
        elif featureV == 14:
            X_train = load_sparse_csr(feature_set_path +  r'quadgramOnlyBinaryWordData'+tag+'_train.npz')  
            X_test = load_sparse_csr(feature_set_path +  r'quadgramOnlyBinaryWordData'+tag+'_test.npz')  
        elif featureV == 15:
            X_train = load_sparse_csr(feature_set_path +  r'quadgramOnlyTfidfWordData'+tag+'_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'quadgramOnlyTfidfWordData'+tag+'_test.npz') 
            
        elif featureV == 16:
            X_train = load_sparse_csr(feature_set_path +  r'binaryCharacterData' + tag + '_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'binaryCharacterData' + tag + '_test.npz') 
        elif featureV == 17:
            X_train = load_sparse_csr(feature_set_path +  r'tfidfCharacterData' + tag + '_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'tfidfCharacterData' + tag + '_test.npz') 
        elif featureV == 18:
            X_train = load_sparse_csr(feature_set_path +  r'binaryCharacterSkipgramData' + tag + '_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'binaryCharacterSkipgramData' + tag + '_test.npz') 
        elif featureV == 19:
            X_train = load_sparse_csr(feature_set_path +  r'tfidfCharacterSkipgramData' + tag + '_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'tfidfCharacterSkipgramData' + tag + '_test.npz') 
        elif featureV == 20:
            #X_train = load_sparse_csr(feature_set_path +  r'_news24lda' + tag + '_train.npz') 
            #X_test = load_sparse_csr(feature_set_path +  r'_news24lda' + tag + '_test.npz') 
            X_train = load_sparse_csr(feature_set_path +  r'_slashdotlda' + tag + '_train.npz') 
            X_test = load_sparse_csr(feature_set_path +  r'_slashdotlda' + tag + '_test.npz') 
        elif featureV == 21:
            X_train = load_numpy_matrix(feature_set_path + "Basic300_news24_TfidfFeatures" + tag + "_train.npy")
            X_test = load_numpy_matrix(feature_set_path + "Basic300_news24_TfidfFeatures" + tag + "_test.npy")
            #X_train = load_numpy_matrix(feature_set_path + "Basic300_slashdot_TfidfFeatures" + tag + "_train.npy")
            #X_test = load_numpy_matrix(feature_set_path + "Basic300_slashdot_TfidfFeatures" + tag + "_test.npy")
        elif featureV == 22:
            X_train = load_numpy_matrix(feature_set_path + "DocBasic300_news24_ParagraphFeatures" + tag + "_train.npy")
            X_test = load_numpy_matrix(feature_set_path + "DocBasic300_news24_ParagraphFeatures" + tag + "_test.npy")
            #X_train = load_numpy_matrix(feature_set_path + "DocBasic300_slashdot_ParagraphFeatures" + tag + "_train.npy")
            #X_test = load_numpy_matrix(feature_set_path + "DocBasic300_slashdot_ParagraphFeatures" + tag + "_test.npy")
        elif featureV == 23:
            X_train = load_numpy_matrix(feature_set_path + "Google_TfidfFeatures" + tag + "_train.npy")
            X_test = load_numpy_matrix(feature_set_path + "Google_TfidfFeatures" + tag + "_test.npy")
            
            
             
          
        
        # Reduce Dataset
        '''
        factor1 = 1
        X_orig = X_orig[:X_orig.shape[0]/factor1,:]
        '''
       # test(Xn, yn)   
            
             
        print "\nFeatures",FEATURES[featureV]
        print '\nTotal:', X_train.shape[0] + X_test.shape[0] 
        print 'Features:', X_train.shape[1]   
        print "\nClass distribution", Counter(y_train)
        
            
        # FEATURE SELECT
        if featureV == 0:
            selector =  SelectPercentile(score_func=f_classif, percentile=perc).fit(X_train, y_train)
        else:
            if featureV == 21 or featureV == 22 or featureV == 23:
                selector = SelectKBest(score_func=f_classif, k=min(200000, X_train.shape[1])).fit(X_train,y_train)
            else:                
                selector = SelectKBest(score_func=chi2, k=min(200000, X_train.shape[1])).fit(X_train,y_train)
        
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        
        
        print X_train.shape
        print X_test.shape
        
    
        # FEATURE SCALING   
         
        if featureV == 0 or featureV == 21 or featureV == 22 or featureV == 23:
            X_train, X_test = normalize_sets_dense(X_train, X_test)
        elif featureV != 20:
            X_train, X_test = normalize_sets_sparse(X_train, X_test)
           
        
        #Xn = preprocessing.normalize(Xn, axis=0, copy=False)
        
        
          
        
        if reg:      
            print "\nREGRESSION\n"
            for m in [2]:
                
                clf =  runRegressionModelTest(X_train, y_test, m) 
                
                cv = cross_validation.StratifiedShuffleSplit(X_train.shape[0], n_iter=5,test_size=0.33,random_state=42)
                a = cross_validation.cross_val_score(clf, X_test, y_test, cv=cv)
                a = a[a > 0]
                print 'Cross V score: :' +  ' '.join("%10.3f" % x for x in a) 
                print ('Mean Score: %.3f' % np.mean(a))
        else:
            print "\nCLASSIFICATION\n"
            print "Nr Of Features", X_train.shape[1]
            print "Nr Of train Rows", X_train.shape[0]
            print "Nr Of test Rows", X_test.shape[0]
            for m in [1, 2]:
                print "STARTING CLASSIFICATION"
                clf = runClassificationTest(X_train, y_train, m, featureV, datatype)
                
                predicted= clf.predict(X_test)
                print "Accuracy: %0.3f " % (accuracy_score(y_test,predicted ))
                
                '''
                print "precision ", (precision_score(y_test, clf.predict(X_test), average=None))
                print "recall  ", (recall_score(y_test, clf.predict(X_test), average=None))
                print "F1 Score ", (f1_score(y_test, clf.predict(X_test), average=None))
                '''
                
        
                if datatype == 3:
                    print classification_report(y_test, predicted, target_names=['0','1', '2'], digits=3)
                    print draw_confusion_matrix(y_test, predicted, [0,1,2])
                else:
                    print classification_report(y_test, predicted, target_names=['0','1'], digits=3)
                    print draw_confusion_matrix(y_test, predicted, [0,1])
                    
        
    
    

    
    
    
   
            
    
    
