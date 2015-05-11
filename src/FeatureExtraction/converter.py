'''
Created on 09 May 2014

@author: Dirk
'''
import os

from scipy.io.matlab.mio5_utils import scipy

from FeatureExtraction.main import load_numpy_matrix, load_sparse_csr
from RatingPrediction.main import getDataSets
import numpy as np


def saveIt(filename, matrix, header, delimiter=","):
    with open(filename, 'w') as fh:        
        fh.write(delimiter.join(header) +"\n")
        for row in matrix:
            line = delimiter.join(str(int(value)) if value.is_integer() else "%.6f" % value for value in row)
            fh.write(line+"\n")
            
if __name__ == '__main__':
    #get_new_test_X()
    #get_new_test_Y()
    tempCWD = os.getcwd()
    
    os.chdir('D:\Workspace\NLTK comments\src\FeatureExtraction')    
         
    X_orig = load_numpy_matrix("featureArray.npy")
    y_orig = load_numpy_matrix("valueVector.npy")
    #wd = load_sparse_csr("freqWordData.npz").todense()
    
    print X_orig.shape
    print y_orig[:,3].shape
    #print wd.shape
    Xn = np.c_[X_orig, y_orig[:,3]]
    #wd = np.c_[wd, y_orig[:,3]]
    
    
    head = ["Lengthiness",
            "Questionfrequency",
            "Exclamationfrequency",
            "CapitalFrequency",
            "SentenceCapitalFrequency",
            "SpelledCorrectly",
            "SpelledFreq",
            "BadWords",
            "BadWordsFreq",
            "spelledPerc*badWordsPerc",
            "Complexity",
            "Readibility",
            "AvgTermFreq",
            "Sentiment",
            "SubjObj",
            "VerbCount",
            "NounCount",
            "ThreadRelevance",
            "ArticleRelevence",
            "Timeliness",
            "Totaltime",
            "Diversity",
            "PolarityOverlap",
            "Informativeness",
            "NumberOfCharacters",
            "NumberOfPronouns",
            "VALUE"]
    #head = "Exclamation frequency,Spelled Correctly,AvgTermFreq,Verb Count,Noun Count,timeliness,Ratio"
    
    saveIt("features.tab", Xn ,header=head, delimiter='\t')
    
    print "SAVED TO CSV"