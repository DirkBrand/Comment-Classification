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

def getDataSets(normalize=False, selected=False):
    if datatype == 1:
        tag = '_main'
    elif datatype == 2:
        tag = "_toy"
    elif datatype == 3:
        tag = '_slashdot'
        
        
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
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh                                             
                                             
type = ['(TEST)', '(TRAIN)']
scaled = ['(PLAIN)', '(NORMALIZED)']

bincount = 500


values = False
features = True

timeData = False



if __name__ == '__main__': 
    
    os.chdir('D:\Workspace\NLTK comments\src\RatingPrediction')
    X, Xt, y, yt = getDataSets(False, False)  
    
    print np.min(y)
    print np.max(y)
    print np.mean(y)
    print  'Loaded testing data\n'
    
    
    print 'Number of training rows:', len(X)
    
    feat = 33
    
    
    print feat
    Xn = X[:,feat]
    Xn = Xn[~is_outlier(Xn)]
    plt.hist(Xn, bins=bincount)
    #plt.title(headers[feat])
    plt.ylabel("Frequency")
    plt.xlabel("Value")
    plt.savefig(r'D:\REPOS\meesters-documentation\Final Report\Pictures\Graphs\g' + str(feat+1) +  '.png', bbox_inches='tight')

       
    
    
    
    
