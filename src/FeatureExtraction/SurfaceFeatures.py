'''
Created on 05 Mar 2014

@author: Dirk
'''

from curses.ascii import isdigit 
import math

from nltk import data
from nltk import word_tokenize
import nltk 
from nltk.corpus import cmudict

from Profiling.timer import timefunc


r = cmudict
d = r.dict()

fakeSentenceCount = 1

def nsyl(word):
    if word.lower() not in d:
        return 1
    
    return max([len([y for y in x if isdigit(y[-1])]) for x in d[word.lower()]])


    
@timefunc
def timeliness(commDate, previous, aveTime):
    if aveTime == 0:
        return 0
    return (float(commDate)-previous) / aveTime
    
    
@timefunc
def lengthiness(avgLen, tokens):
    return len(tokens) / avgLen

@timefunc       
def question_frequency(sentences):
    count = 0
    total = 0
    for s in sentences:
        if len(s) > 1:
            total += 1
        if s.endswith('?'):
            count += 1
            
    if total == 0:
        return 0
    
    return float(count + fakeSentenceCount) / float(total + fakeSentenceCount)

@timefunc   
def exclamation_frequency(sentences):
    count = 0
    total = 0
    for s in sentences:
        if len(s) > 1:
            total += 1
        if s.endswith('!'):
            count += 1
            
    if total == 0:
        return 0
    
    return float(count + fakeSentenceCount) / float(total + fakeSentenceCount)

@timefunc   
def capital_frequency(tokens):
    if len(tokens) == 0:
        return 0
    
    count = 0
    for word in tokens:
        if word.isupper() and len(word) > 1:
            count += 1
            
    return count / float(len(tokens)) # PERCENTAGE

@timefunc
def sentence_capital_frequency(sentences):
    if len(sentences) == 0:
        return 0

    count = 0
    total = 0
    for sent in sentences:
        if len(sent) > 1:
            total += 1
        if sent[0].isupper():
            count += 1

    return  float(count + fakeSentenceCount) / float(total + fakeSentenceCount)

@timefunc       
def F_K_score(text, sentences):
    myWords = word_tokenize(text)
    syll_count = 0
    for w in myWords:
        syll_count += nsyl(w)
        
    return 206.835 - 1.015*(float(len(myWords)) / float(len(sentences))) - 84.6*(float(syll_count) / float(len(myWords)))

@timefunc
def avgTermFreq(text, theWords):
    sumtf = 0.0
    for w in theWords:
        tf = termf(text.count(w), theWords)
        sumtf += tf
    
    return sumtf / float(len(theWords))
    

def termf(wordFreq, theWords):
    fdist = nltk.FreqDist(theWords)
    tf = (float(wordFreq))/max(fdist.values())
    return tf

@timefunc           
def tf_idf(text, theWords, article):
    sum_inform = 0
    for w in theWords:
        tf = termf(text.count(w), theWords)
        count = 0
        for comm in article:
            if w in comm.body.lower():
                count += 1
        
        idf = max(0,math.log(len(article) / (count+1.0)))
        sum_inform += tf * idf
    
    return sum_inform 
        
@timefunc
def onSubForumTopic(tokens, BoW):
    count = 0
    for t in tokens:
        if t in BoW:
            count += 1
    
    return float(count) / len(tokens)

    



    




