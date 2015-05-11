'''
Created on 24 Mar 2014

@author: Dirk
'''

import math
import os
import random
from scipy import stats

from sklearn import cross_validation
from sklearn.feature_selection import SelectPercentile,  SelectKBest
from sklearn.feature_selection.univariate_selection import f_regression
from sklearn.metrics import r2_score

from RatingPrediction.Regression import SVR_fit
from RatingPrediction.main import getDataSets, shuffle_in_unison, getDateTime
import numpy as np


VALUES = [ 'totalVotes', 'percentage of total','ratio','Percentage * Ratio','Ttest']
            



def load_training_data(x_filename, y_filename):    
    tempCWD = os.getcwd()
    
    os.chdir('D:\Workspace\NLTK comments\src\FeatureExtraction')
    
    fs = np.load(x_filename)
    vv = np.load(y_filename)
    
    os.chdir(tempCWD)
    return fs, vv
   

def binning_tuple(opList, nrBins):
    bins = []
    jump = 100 / nrBins
    for i in range(nrBins - 1):
        bins.append(np.percentile(opList, jump * (i + 1)))
    
    bins.append(np.max(opList))
    #print bins
    
    temp = np.copy(opList)
    for i in range(len(opList)):
        val = 0
        for b in bins:
            if opList[i] <= b:
                temp[i] = val
                break
            val += 1
        
    return temp


    
    
def test(X, y):
       
    ###############################################################################
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function: the 10% most significant features
    selector = SelectPercentile(f_regression, percentile=20)
    selector.fit(X, y)
    print [zero_based_index for zero_based_index in list(selector.get_support(indices=True))]
    
    selector2 = SelectKBest(score_func=f_regression, k=6)
    selector2.fit(X, y)
    print [zero_based_index for zero_based_index in list(selector2.get_support(indices=True))]



def determine_NDGC(X,y,K,cv,clf_min,typev):
    dt = getDateTime('timeData.npy')
    
    NDGC = []
    tau = []
    print "K=",K
    print "CV=",cv
    for j in np.arange(0,cv):
        # Train Classifier
        score = 0
        while score < clf_min:
            shuffle_in_unison(X0,y0)
            X,Xt,y,yt = cross_validation.train_test_split(X0,y0, test_size=0.3, random_state=0)
            clf = SVR_fit(X,y)
            #clf = linear_regression_fit(X, y)
            score =  r2_score(yt, clf.predict(Xt))
        
        
        commList = []
        
        N = len(Xt)
        
        predicted = []
        recorded = []
        for i,row in enumerate(Xt):
            predicted.append(clf.predict(row))
            recorded.append(yt[i])
        
        tau.append(stats.kendalltau(predicted, recorded)[0])
        
        # bin the recorded values
        recorded = binning_tuple(recorded,5)
        for i, t in enumerate(recorded):
            commList.append((predicted[i],  t, dt[i]))
            
        
        
        
        DCG = 0
        iDCG = 0
        ind = 1
        
        sorted_by_ratio = sorted(commList, key=lambda tup: tup[1])[::-1]
        
        rankedList = []
        # Build Ranked List
        for i,tup in enumerate(sorted_by_ratio):
            rankedList.append((i+1, tup[0], tup[1], tup[2]))
            
        
        # Sort by predictions
        if typev == 1:        
            print "Classifier"
            rankedList = sorted(rankedList, key=lambda tup: tup[1])[::-1]
        if typev == 2: 
            print "TimeStamp"
            rankedList = sorted(rankedList, key=lambda tup: tup[3])[::-1]  
        if typev == 3:      
            print "Random"      
            random.shuffle(rankedList)
        
        
        ind = 1
        for tup in rankedList:
            fav = N - tup[0] + 1
            rank = tup[2]
            pow = 2**rank - 1
            CG = rank/math.log(ind+1,2)
            DCG += CG
            if ind == K:
                break;
            ind += 1
            
            
        # Sort by community
        rankedList = sorted(rankedList, key=lambda tup: tup[2])[::-1]
        #print "Ranked ratio List",rankedList  
        ind = 1
        for tup in rankedList:
            fav = N - tup[0] + 1
            rank = tup[2]
            pow = 2**rank - 1
            CG = rank/math.log(ind+1,2)
            iDCG += CG
            if ind == K:
                break;
            ind += 1
        
        
        
        print 'Test',j," - ", DCG/float(iDCG)
        NDGC.append(DCG/float(iDCG))
        
    return np.mean(NDGC), np.mean(tau)
     
reg = True
value = 1
feature = 1
if __name__ == '__main__':
    
    X0, y0, wordData,topicData,socialData, out = getDataSets(normalize=True, selected=False)
    y0 = y0[:,value]
    print out
        
    #test(X,y)
        
    if feature == 0:
        X0 = X0
    elif feature == 1:
        X0 = wordData
    elif feature == 2:
        X0 = topicData
    elif feature == 3:
        X0 = np.hstack((X0,wordData))
    elif feature == 4:
        X0 = np.hstack((X0,topicData))
    elif feature == 5:
        X0 = np.hstack((np.hstack((X0,topicData)), wordData))
    
    
    print X0.shape, y0.shape
    print "\n","Values",VALUES[value],'\n'
    
    
    
    print 'Number of training rows:', len(X0)
    print 'Number of features:', X0.shape[1], '\n'
       
    for t in [3,2,1]:
        score, tau = determine_NDGC(X0, y0, 20, cv=10, clf_min=0.00001,type=t) 
        print "Mean Score:",score
        print "Mean Tau:", tau
    
    

    
    
    
   
            
    
    
