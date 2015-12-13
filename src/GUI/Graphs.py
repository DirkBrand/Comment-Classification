'''
Created on 08 Apr 2014

@author: Dirk
'''
from numpy.random import normal
import os

from FeatureExtraction.main import load_numpy_matrix
from sklearn import preprocessing

from RatingPrediction.main import load_training_data, binScaling,\
    getDateTime
from config import feature_set_path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


# [commLen, qf, ef, cf, qf*ef, qf*cf, spelled, spelledPerc, badWords, badWordsPerc, spelledPerc*badWordsPerc, complexity, readibility, informativeness, sentiment, subj_obj]
headers = ['timely' , 'timePassing' , 'commLengthiness' , 'numberCharacters' ,'vf', 'nf', 'pronouns' , 
            'cf' , 'qf', 'ef' , 'scf', 'complexity', 'diversity' , 'spelled' , 'spelled freq'
             'badWords', 'badWordsPerc', 'meantermFreq', 'informativeness', 'readibility', 'threadRelevance', 'articleRelevance', 'sentiment', 'subj_obj', 'polarity_overlap']

datatype = 1

if datatype == 1:
    tag = '_main'
elif datatype == 2:
    tag = "_toy"
elif datatype == 3:
    tag = '_slashdot'
    
def getDataSets(normalize=False, selected=False):
    
    y_train = load_numpy_matrix(feature_set_path +  r'valueVector'+tag+'_train.npy')
    y_test = load_numpy_matrix(feature_set_path +  r'valueVector'+tag+'_test.npy')
    
    
    X_train = load_numpy_matrix(feature_set_path +  r'featureArray'+tag+'_train.npy')
    sd = load_numpy_matrix(feature_set_path +  r'socialVector'+tag+'_train.npy')
    X_train =  np.hstack((X_train,sd))
    X_test = load_numpy_matrix(feature_set_path +  r'featureArray'+tag+'_test.npy')
    sd2 = load_numpy_matrix(feature_set_path +  r'socialVector'+tag+'_test.npy')
    X_test =  np.hstack((X_test,sd2))
    
    return X_train, X_test, y_train, y_test
            
def is_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh                                             
                              
bincount = 500


if __name__ == '__main__': 
    
    X, Xt, y, yt = getDataSets(False, False)  
    
    print np.min(y)
    print np.max(y)
    print np.mean(y)
    print  'Loaded testing data\n'
    
    
    print 'Number of training rows:', len(X)
        
    for feat in range(33):
        print feat
        Xn = X[:,feat]
        x_range = max(Xn) - min(Xn)
        plt.figure()
        plt.hist(Xn, bins=bincount)
        if feat != 7:
            plt.xlim([min(Xn)-x_range*0.05,max(Xn)+x_range*0.05])
        #plt.title(headers[feat])
        #plt.ylabel("Frequency")
        #plt.xlabel("Value")
        plt.savefig(r'D:\REPOS\meesters-documentation\Final Report\Pictures\Graphs\g' + str(feat+1) + tag +  '.png', bbox_inches='tight')

       
    
    
    
    
