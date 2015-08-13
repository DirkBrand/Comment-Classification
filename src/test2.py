'''
Created on 22 Jul 2015

@author: Dirk
'''
from collections import Counter
import re

from FeatureExtraction.mainExtractor import read_slashdot_comments,\
    read_comments, read_toy_comments
from nltk.tokenize import word_tokenize

from config import comment_data_path


if __name__ == '__main__':
    #articleList, commentList, commentCount = read_slashdot_comments(comment_data_path + 'slashdotDataSet.txt', skip=False)
    #articleList, commentList, parList, commentCount = read_comments(comment_data_path + 'trainTestDataSet.txt', skip=False)
    articleList, commentList, parList, commentCount = read_toy_comments(comment_data_path + 'trainTestDataSet.txt', comment_data_path + 'toyComments.csv')
    
    
    totalComms = 0
    totalWords = 0
    totalArt = 0
    numberAnon = 0
    for art in commentList.items():        
        totalArt += 1
        for comm in art[1]:
            totalComms += 1
            if comm.author.lower() == 'anonymous coward':
                numberAnon += 1
            totalWords += len(word_tokenize(comm.body))
            print totalComms
            
    
    print "Total comms", totalComms
    print "Average per art", float(totalComms)/totalArt
    print "Number anon commes", numberAnon
    print "Total words", totalWords
    print "Ave words per comm", float(totalWords)/totalComms
    print 
    print "Number of parent comments", len(parList.keys())
    print "Number of child comments", totalComms - len(parList.keys())
    print "Ave Number of child comments per parent", sum(parList.values())/float(len(parList.values()))