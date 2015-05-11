'''
Created on 24 Mar 2014

@author: Dirk
'''

from collections import Counter
import os
import pickle

import nltk
from sklearn import cross_validation
from sklearn import preprocessing, decomposition as deco, svm
from sklearn.datasets import make_classification
from sklearn.externals import joblib
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.feature_selection.univariate_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score,\
    metrics
from sklearn.preprocessing import StandardScaler

from FeatureExtraction.LexicalFeatures import words, known, swears, entropy, \
    withoutStopWords, pos_freq
from FeatureExtraction.SentimentUtil import load_classifier, make_full_dict, \
    get_subjectivity
from FeatureExtraction.SurfaceFeatures import lengthiness, question_frequency, \
    exclamation_frequency, capital_frequency, F_K_score, avgTermFreq, \
    onSubForumTopic
from FeatureExtraction.mainExtractor import pattern
from RatingPrediction import elm
from RatingPrediction.Classification import svc_fit, \
    draw_confusion_matrix, linear_svc_fit, nearest_fit,\
    random_forest_fit, SGD_c_fit, log_regression_fit
from RatingPrediction.Regression import linear_regression_fit, SVR_fit, \
    decision_tree_fit, \
    bayesian_ridge_fit, neighbours_fit, SGD_r_fit
import matplotlib.pyplot as plt
import numpy as np


VALUES = ['ratio', 'totalVotes', 'percentage of total','Percentage * Ratio']
    

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
    
    temp = np.copy(values)
    for i in range(len(values)):
        val = 1
        for b in bins:
            if values[i] <= b:
                temp[i] = val
                break
            val += 1
        
    return temp

def getDataSets(normalize, selected):
    output = ''
    
      
    tempCWD = os.getcwd()
    
    os.chdir('D:\Workspace\NLTK comments\src\FeatureExtraction')
    
         
    featureSet, valueVector = load_training_data("featureArrayTest.npy", "valueVectorTest.npy")
    
    
    
    output += 'Loaded training and testing data\n'
    
    os.chdir(tempCWD)
   
    # NORMALIZATION
    if normalize:
        valueVector = preprocessing.scale(valueVector)
        featureSet = preprocessing.scale(featureSet)
        output += "Scaled training & testing Features\n"
    
    # FEATURE SELECTION 
    if selected:
        featureSet = np.c_[featureSet[:,12],featureSet[:,14],featureSet[:,15],featureSet[:,18],featureSet[:,20],featureSet[:,23]]
    
    # Binning
    '''
    featureSet[:,0] = binScaling(featureSet[:,0], 10)
    featureSet[:,14] = binScaling(featureSet[:,14], 10)
    
    X_test[:,0] = binScaling(X_test[:,0], 10)
    X_test[:,14] = binScaling(X_test[:,14], 10)
    
    '''
    
    
    
    
    return featureSet, valueVector, output

def runRegressionModelTest(featureSet, valueVector, X_test, y_test, model):    
    output = ''
    score = 0
    clf = 0
    if model == 1:
        output += "\nLINEAR REGRESSION\n"
        clf = linear_regression_fit(featureSet, valueVector)
    elif model == 2:
        output += "\nSVR\n"
        clf = SVR_fit(featureSet, valueVector)
    elif model == 3:
        output += "\nEXTREME LEARNING MACHINE\n"
        clf = elm.ELMRegressor()
        clf.fit(featureSet, valueVector)
        joblib.dump(clf, 'elm.pkl')
    elif model == 4:
        output += "\nSTOCHASTIC\n"
        clf = SGD_r_fit(featureSet, valueVector)
        joblib.dump(clf, 'sgd.pkl')
    elif model == 5:        
        output += "\nNEIGHBOURS\n"
        clf = neighbours_fit(featureSet, valueVector)
    elif model == 6:        
        output += "\nLOGISTIC\n"
        clf = log_regression_fit(featureSet, valueVector)
    elif model == 7:        
        output += "\nBAYESIANRIDGE\n"
        clf = bayesian_ridge_fit(featureSet, valueVector)
    else :
        output += 'Invalid choice\n'
    
    score = mean_squared_error(y_test, clf.predict(X_test))
    score2 = r2_score(y_test, clf.predict(X_test))
    cv = cross_validation.ShuffleSplit(featureSet.shape[0], n_iter=50,test_size=0.25,random_state=0)
    a = cross_validation.cross_val_score(clf, featureSet, valueVector, cv=cv)
    a = a[a > 0]
    output += 'Cross V score: :' +  ' '.join("%10.3f" % x for x in a) + '\n'
    output += ('Mean Score: %.3f\n' % np.mean(a))
    output += ('Mean Squared Error: %.3f\n' % score)
    output += ('R^2: %.3f\n' % score2)
    
    return output 

def runClassificationTest(X, y, Xt, yt, model, labs):
    output = ''
    clf = 0
    
    
    if model == 1:
        output += "\nSVC\n"
        clf = svc_fit(X, y)
    elif model == 2:
        output += '\nLinearSVC\n'
        clf = linear_svc_fit(X, y)
    elif model == 3:
        output += '\nStochasticGradientDescent\n'
        clf = SGD_c_fit(X, y)  
    elif model == 4:
        output += '\nKNearestNeighbours\n'
        clf = nearest_fit(X, y)
    elif model == 5:
        output += '\nRandomForest\n'
        clf = random_forest_fit(X, y)
    elif model == 6:
        output += '\nLogistic\n'
        clf = log_regression_fit(X, y)
        
        
    
    accuracy = accuracy_score(yt, clf.predict(Xt))
    f1 = f1_score(yt, clf.predict(Xt), labels=labs)
    
    
    cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=50,test_size=0.3,random_state=0)
    a = cross_validation.cross_val_score(clf, X, y, cv=cv)
    a = a[a > 0]
    output += 'Cross V score: :' +  ' '.join("%10.3f" % x for x in a) + '\n'
    output += "\n\nAccuracy " + str(accuracy)
    output += "\nF1 Score " + str(f1)
    

    return clf, output;
    
def printPredictions(clf, X, y): 
    for i in range(len(y)):
        print "%.4f - %.4f" % (y[i], clf.predict(X[i, :])) 
    plt.scatter(clf.predict(X), y)
    plt.show()

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def test(X, y):
       
    ###############################################################################
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function: the 10% most significant features
    
    selector2 = SelectKBest(score_func=f_regression, k=6)
    selector2.fit(X, y)
    print [zero_based_index for zero_based_index in list(selector2.get_support(indices=True))]


reg = True
valueV = 2
  
if __name__ == '__main__':
    
    X_orig, y_orig, out = getDataSets(normalize=True, selected=False)
    print out
    
      
    #test(X,y)
    
        
        
    #shuffle_in_unison(X_orig,y_orig)
    
    Xn = X_orig
    
    factor1 = 1
    # Reduce Dataset
    X_orig = X_orig[:X_orig.shape[0]/factor1,:]
    y_orig = y_orig[:len(y_orig)/factor1]
    
    print 'Number of training rows:', len(Xn)/3*2, "| number of test rows:", len(Xn)/3
    print 'Total:',len(Xn)
    print 'Number of features:', Xn.shape[1], '\n'
    
    
    for v in [1,0]:
        
        yn = y_orig[:,v]
        
        X,Xt,y,yt = cross_validation.train_test_split(Xn,yn, test_size=0.33, random_state=0)      
          
        
        
        print "\n","Values",VALUES[v],'\n'
        
        
        if reg:      
            print "\nREGRESSION\n"
            for m in [1]:
                print runRegressionModelTest(X, y, Xt, yt, m) 
        else:
            print "\nCLASSIFICATION\n"
            for fi in [2,3,5]:
                labels = np.arange(1,fi+1)
                yb = binScaling(y, len(labels)).astype(np.int)
                ytb = binScaling(yt, len(labels)).astype(np.int)       
                
                
                print "Unique", np.unique(yb)
                print "Nr Of Features", Xt.shape[1]
                print "Nr Of test Rows", Xt.shape[0]
                for m in [1,2,5,6]:
                    clf, out = runClassificationTest(X, yb, Xt, ytb, m, labels)
                    print out
                    print draw_confusion_matrix(ytb, clf.predict(Xt), labels)
        
        
    
    

    
    
    
   
            
    
    
