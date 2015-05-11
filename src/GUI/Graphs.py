'''
Created on 08 Apr 2014

@author: Dirk
'''
import os

from numpy.random import normal
from sklearn import preprocessing

from RatingPrediction.main import load_training_data, binScaling, getDataSets,\
    getDateTime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


# [commLen, qf, ef, cf, qf*ef, qf*cf, spelled, spelledPerc, badWords, badWordsPerc, spelledPerc*badWordsPerc, complexity, readibility, informativeness, sentiment, subj_obj]
headers = ['Length' , 'Question frequency' , 'Exclamation frequency' , 'Capital frequency' ,'qf*ef', 'qf*cf', 'Spelled Correctly' ,
            'Spelled Freq' , 'BadWords' , 'BadWords Freq', 'spelledPerc*badWordsPerc', 'Complexity' , 'Readibility' ,
             'AvgTermFreq', 'Sentiment', 'SubjObj', 'Verb Count', "Noun Count", 'relevance thread', 'relevance article', 'timeliness', 'timeTotal', 'diversity','engagement','2*6' , '2*12' , 
             '2*13' , '2*16' , '2*17' , '6*12' , '6*13' , '6*16' , '6*17' , '12*13' , '12*16' , '12*17' , '13*16' , '13*17' , '16*17']
type = ['(TEST)', '(TRAIN)']
scaled = ['(PLAIN)', '(NORMALIZED)']
col = 8
valueCol = 4
bincount = 500
values = True
features = False

timeData = False

if __name__ == '__main__':    
    tempCWD = os.getcwd()
    
    os.chdir('D:\Workspace\NLTK comments\src\RatingPrediction')
    X, y, wd,td, out = getDataSets(normalize=False, selected=False)  
    
    y = y[:,valueCol]
    print np.min(y)
    print np.max(y)
    print np.mean(y)
    print  'Loaded testing data\n'
    X_scaled = preprocessing.scale(X)
    print  "Scaled testing Features\n"
    
    X_binned = binScaling(X[:,col], 10)
    print 'Binned Features'
    
    
    
    
    
    print 'Number of training rows:', len(X)
    os.chdir(tempCWD)
    
    if features:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(8,8))
        ax1.set_title(headers[col] + ' ' + scaled[0])
        ax1.hist(X[:, col], bins=bincount)
        ax2.hist(X_scaled[:, col], bins=bincount)
        X[:,col] = X_binned
        ax3.hist(X[:,col], bins=bincount)
        ax3.set_title(headers[col] + ' ' + scaled[1])
        f.show()
    
    
    
    if values:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(8,8))
        ax1.hist(y, bins=bincount)
        y_Binned = binScaling(y,3)
        print "Binned the values"
        ax2.hist(preprocessing.scale(y),bins=bincount)
        ax3.hist(y_Binned, bins=bincount)
        f.suptitle('Y Values')
        f.show()
            
    if timeData:
        for i in range(len(headers)):
            dt = getDateTime('timeData.npy')
            plt.scatter(dt, preprocessing.scale(y),label='Values')
            plt.gcf().autofmt_xdate()
            plt.xlabel('Time')
            plt.grid()
            plt.show()
            i += 1
    
    
    plt.show()
    
    
    
    
