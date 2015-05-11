'''
Created on 24 Mar 2014

@author: Dirk
'''

from collections import Counter
import os
import pickle

import nltk
from nltk.metrics.scores import precision, recall
from sklearn import cross_validation
from sklearn import preprocessing, decomposition as deco, svm
from sklearn.datasets import make_classification
from sklearn.externals import joblib
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.feature_selection.univariate_selection import f_regression, chi2
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score,\
    metrics
from sklearn.metrics.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from FeatureExtraction.main import load_sparse_csr, load_numpy_matrix
from RatingPrediction import elm
from RatingPrediction.Classification import *
from RatingPrediction.Regression import *
from config import feature_set_path
import numpy as np


VALUES = [ 'totalVotes', 'percentage of total','ratio','Status']
FEATURES = [ 'CF', 'BWF','FWF','TWF', 'BTF', 'TTF', 'SL-MMM', 'SL-TWS', 'SL-BOC', 'POS-MMM', 'POS-TWS', 'POS-BOC', 'BG-MMM', 'BG-TWS', 'BG-BOC', 'Google-MMM', 'Google-TWS', 'Google-BOC']
            



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
    
    

def getDataSets(normalize, cross_multiply, selected, data_representation=0):
    output = ''
    
      
    tempCWD = os.getcwd()
    
    os.chdir('D:\Workspace\NLTK comments\src\FeatureExtraction')
    
         
    featureSet = load_numpy_matrix("featureArray.npy")
    valueVector = load_numpy_matrix("valueVector.npy")
    binaryWordData = load_sparse_csr('binaryWordData.npz')  
    freqWordData = load_sparse_csr('freqWordData.npz')  
    tfidfWordData = load_sparse_csr('tfidfWordData.npz') 
    rawTfidfWordData = load_sparse_csr('rawTfidfWordData.npz')
    bigramTfidfWordData = load_sparse_csr('bigramTfidfWordData.npz')  
    topicData = load_numpy_matrix('topicData.npy')   
    socialVector = load_numpy_matrix('socialVector.npy')   
    
    if data_representation == 0: # BINARY
        wordDataVector = binaryWordData
    elif data_representation == 1: # FREQUENCY
        wordDataVector = freqWordData
    elif data_representation == 2: # TFIDF
        wordDataVector = tfidfWordData
    elif data_representation == 3: # RAW TFIDF
        wordDataVector = rawTfidfWordData
    elif data_representation == 4: # BIGRAM TFIDF
        wordDataVector = bigramTfidfWordData
    
    
    output += 'Loaded training and testing data\n'
    
    os.chdir(tempCWD)
    
    
    # Adding Columns 2*6*12*16*17*18
    if cross_multiply:
        bestFeat = [5, 13, 19, 20, 23, 25]

        for i in range(0,len(bestFeat)-1):
            for j in range(i+1,len(bestFeat)) :
                featureSet = np.c_[featureSet, featureSet[:, bestFeat[i]] * featureSet[:, bestFeat[j]]]

        output += "Added new columns\n"

    
    # NORMALIZATION
    if normalize:
        featureSet = preprocessing.scale(featureSet)
        #valueVector = preprocessing.scale(valueVector)
        #binaryWordData = preprocessing.scale(binaryWordData)
        #freqWordData = preprocessing.scale(freqWordData)
        #tfidfWordData = preprocessing.scale(tfidfWordData)
        topicData = preprocessing.scale(topicData)
        socialVector = preprocessing.scale(socialVector)
        output += "Scaled training & testing Features\n"
    
    # FEATURE SELECTION 
    if selected:
        featureSet = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(featureSet, valueVector[:,0])
        wordDataVector = SelectPercentile(score_func=chi2, percentile=20).fit(wordDataVector, valueVector).transform(wordDataVector)
        
        
    
    # Binning
    '''
    featureSet[:,0] = binScaling(featureSet[:,0], 10)
    featureSet[:,14] = binScaling(featureSet[:,14], 10)
    
    X_test[:,0] = binScaling(X_test[:,0], 10)
    X_test[:,14] = binScaling(X_test[:,14], 10)
    
    '''
    
    
    
    
    return featureSet, valueVector, wordDataVector ,topicData, socialVector, output

def runRegressionModelTest(featureSet, valueVector, model):    
    output = ''
    clf = 0
    if model == 1:
        print "\nLINEAR REGRESSION\n"
        clf = linear_regression_fit(featureSet, valueVector)
    elif model == 2:
        print "\nSVR\n"
        clf = SVR_fit(featureSet, valueVector)
    elif model == 3:
        print "\nEXTREME LEARNING MACHINE\n"
        clf = elm.ELMRegressor()
        clf.fit(featureSet, valueVector)
        joblib.dump(clf, 'elm.pkl')
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

def runClassificationTest(X, y, model, featureset):
    clf = 0
    kernel='rbf'
    C=0
    Lc = 0
    gamma=0
    if featureset == 0: #Custom Features
        C = 222962
        Lc = 100
        gamma = 0.3644  
    if featureset == 1: #Binary word Features
        C = 278
        Lc = 10
        gamma = 0.0175    
    if featureset == 2: #Freq Word Features
        C = 3618
        Lc = 10
        gamma = 0.0034            
    if featureset == 3: #TFIDF word Features
        C = 75361
        Lc = 10
        gamma = 1.2526   
    if featureset == 4: #Bigram word Features
        C = 390255
        Lc = 5
        gamma = 0.3361
    if featureset == 5: #Trigram word Features
        C = 139602
        Lc = 10
        gamma = 0.1312
    if featureset == 6: #sentence MMM Features
        C = 7524
        Lc = 100000
        gamma = 0.0769
    if featureset == 7: #sentence TWS Features
        C = 12336
        Lc = 1000
        gamma = 0.3406
    if featureset == 8: #sentence BOC Features
        C = 71830
        Lc = 100
        gamma = 0.3963
    if featureset == 9: #POS mmm Features
        C = 36738
        Lc = 1000
        gamma = 0.3638
    if featureset == 10: #POS tfidf Features
        C = 164211
        Lc = 10000
        gamma = 0.0028
    if featureset == 11: #POS BOC Features
        C = 68903
        Lc = 100
        gamma = 0.264
    if featureset == 12: #Bigram mmm Features
        C = 1000
        Lc = 10000
        gamma = 0.0
    if featureset == 13: #Bigram tfidf Features
        C = 100000
        Lc = 1000
        gamma = 0.5    
    if featureset == 14: #Bigram BOC Features
        C = 120
        Lc = 10000
        gamma = 0.1539
    if featureset == 15: #Google mmm Features
        C = 117795
        Lc = 1000000
        gamma = 0.0209
    if featureset == 16: #Google tfidf Features
        C = 32598
        Lc = 1000
        gamma = 0.664
        
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

def shuffle_in_unison(features, values):
    rng_state = np.random.get_state()
    
    # shuffle features
    np.random.set_state(rng_state)
    indexList = np.arange(np.shape(features)[0])
    np.random.shuffle(indexList)
    features = features[indexList,:]
    
    # shuffle values
    np.random.set_state(rng_state)
    np.random.shuffle(values)

def test(X, y):
       
    ###############################################################################
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function: the 10% most significant features
    
    selector2 = SelectKBest(score_func=f_regression, k=6)
    selector2.fit(X, y)
    print [zero_based_index for zero_based_index in list(selector2.get_support(indices=True))]


reg = False
scale = True
valueV = 3
featureV = 1
perc = 20
  
if __name__ == '__main__':
    
    
    
    
    y = load_numpy_matrix(feature_set_path +  r'valueVector.npy')[:,valueV]
    
    if featureV == 0:
        X = load_numpy_matrix(feature_set_path +  r'featureArray.npy')
        sd = load_numpy_matrix(feature_set_path +  r'socialVector.npy')
        X =  np.hstack((X,sd))
        perc = 50
    elif featureV == 1:
        X = load_sparse_csr(feature_set_path +  r'binaryWordData.npz')  
    elif featureV == 2:
        X = load_sparse_csr(feature_set_path +  r'freqWordData.npz')  
    elif featureV == 3:
        X = load_sparse_csr(feature_set_path +  r'tfidfWordData.npz') 
    elif featureV == 4:
        X = load_sparse_csr(feature_set_path +  r'bigramTfidfWordData.npz')  
    elif featureV == 5:
        X = load_sparse_csr(feature_set_path +  r'trigramTfidfWordData.npz')  
    elif featureV == 6:
        X = load_numpy_matrix(feature_set_path +  r'sentence_model_MinMaxMeanFeatures.npy')  
    elif featureV == 7:
        X = load_numpy_matrix(feature_set_path +  r'sentence_model_TfidfWeightedSumFeatures.npy')  
    elif featureV == 8:
        X = load_numpy_matrix(feature_set_path +  r'sentence_model_BagOfCentroidsFeatures.npy')   
    elif featureV == 9:
        X = load_numpy_matrix(feature_set_path +  r'POS_model_MinMaxMeanFeatures.npy')       
    elif featureV == 10:
        X = load_numpy_matrix(feature_set_path +  r'POS_model_TfidfWeightedSumFeatures.npy')       
    elif featureV == 11:
        X = load_numpy_matrix(feature_set_path +  r'POS_model_BagOfCentroidsFeatures.npy')    
    elif featureV == 12:
        X = load_numpy_matrix(feature_set_path +  r'bigram_model_MinMaxMeanFeatures.npy')       
    elif featureV == 13:
        X = load_numpy_matrix(feature_set_path +  r'bigram_model_TfidfWeightedSumFeatures.npy')       
    elif featureV == 14:
        X = load_numpy_matrix(feature_set_path +  r'bigram_model_BagOfCentroidsFeatures.npy')      
    elif featureV == 15:
        X = load_numpy_matrix(feature_set_path +  r'google_model_MinMaxMeanFeatures.npy')      
    elif featureV == 16:
        X = load_numpy_matrix(feature_set_path +  r'google_model_TfidfWeightedSumFeatures.npy')  
    
    
    
   # test(Xn, yn)   
        
    
    # Reduce Dataset
    '''
    factor1 = 1
    X_orig = X_orig[:X_orig.shape[0]/factor1,:]
    '''
        
    # FEATURE SELECT
    X = SelectPercentile(score_func=chi2, percentile=perc).fit_transform(X,y) 
    
    # GET TEST SET
    sss = cross_validation.StratifiedShuffleSplit(y, n_iter=1, test_size=0.75, random_state=42)
    for train, test in sss:
        Xt , yt = X[test], y[test]
        Xn , yn = X[train], y[train]
    
    
    yn = [int(i) for i in yn]
    
    print 'Number of training rows:', Xn.shape[0]/3*2, "| number of test rows:", Xn.shape[0]/3
    print 'Total:', Xn.shape[0]
    print 'Number of features:', Xn.shape[1], '\n'
    
    print "\n","Values",VALUES[valueV],'\n'
    print "\n","Features",FEATURES[featureV],'\n'
    print "Class distribution %.3f" %(np.sum(yn)/Xn.shape[0])
    
    
    # FEATURE SCALING
    normalizer = preprocessing.Normalizer().fit(Xn)
    if scale:       
        Xn = normalizer.transform(Xn)
    
    
    
    
    
      
    
    if reg:      
        print "\nREGRESSION\n"
        for m in [2]:
            
            clf =  runRegressionModelTest(Xn, yn, m) 
            
            cv = cross_validation.StratifiedShuffleSplit(Xn.shape[0], n_iter=5,test_size=0.33,random_state=42)
            a = cross_validation.cross_val_score(clf, Xn, yn, cv=cv)
            a = a[a > 0]
            print 'Cross V score: :' +  ' '.join("%10.3f" % x for x in a) 
            print ('Mean Score: %.3f' % np.mean(a))
    else:
        print "\nCLASSIFICATION\n"
        print "Nr Of Features", Xn.shape[1]
        print "Nr Of test Rows", Xn.shape[0]/3
        for m in [1]:
            print "STARTING CLASSIFICATION"
            clf = runClassificationTest(Xn, yn, m, featureV)
            
            cv = cross_validation.StratifiedShuffleSplit(yn, n_iter=10,test_size=0.33,random_state=0)
            a = cross_validation.cross_val_score(clf, Xn, yn, cv=cv)
            a = a[a > 0]
            print "Accuracy: %0.3f (+/- %0.4f)" % (a.mean(), a.std() / 2)
            c = cross_validation.cross_val_score(clf, Xn, yn, cv=cv, scoring='precision')
            c = c[c > 0]
            print "precision %0.3f (+/- %0.4f)" % (c.mean(), c.std() / 2)
            d = cross_validation.cross_val_score(clf, Xn, yn, cv=cv, scoring='recall')
            d = d[d > 0]
            print "recall %0.3f (+/- %0.4f)" % (d.mean(), d.std() / 2)
            b = cross_validation.cross_val_score(clf, Xn, yn, cv=cv, scoring='f1')
            b = b[b > 0]
            print "F1 Score %0.3f (+/- %0.4f)" % (b.mean(), b.std() / 2)
            e = cross_validation.cross_val_score(clf, Xn, yn, cv=cv, scoring='roc_auc')
            e = e[e > 0]
            print "AUC Score %0.3f (+/- %0.4f)" % (e.mean(), e.std() / 2)
    
            print draw_confusion_matrix(yt, clf.predict(Xt), [0,1])
        
    
    

    
    
    
   
            
    
    
