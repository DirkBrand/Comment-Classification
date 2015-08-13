'''
Created on 13 Mar 2014

@author: Dirk
'''
import collections
import csv
import itertools
import math
import os
import pickle
import random
import re

import nltk
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.metrics.association import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from config import sentiment_path


POLARITY_DATA_DIR = os.path.join(sentiment_path +  'polarityData', 'rt-polaritydata')
twitter_data = os.path.join(POLARITY_DATA_DIR, 'training.1600000.processed.noemoticon.csv')
RT_POLARITY_POS_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
RT_POLARITY_NEG_FILE = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt')

def load_classifier(filename):
    
    
    f = open(filename)
    classifier = pickle.load(f)
    f.close()
    return classifier

def make_full_dict(words):
    return dict([(word, True) for word in words])

def get_subjectivity(probDist):
    p = probDist.prob('pos')
    n = probDist.prob('neg')
    
    # Subjective = 1, Objective = 0
    if abs(p - n) < 0.1:
        return 0
    else:
        return 1


clf = load_classifier(sentiment_path + 'sentiment_classifier.pickle')
#clf = 0
def get_polarity_overlap(articleWords, commentWords):
    articleSet = dict()
    refWords = make_full_dict(articleWords)
    articleSet.update(refWords)

    commentSet = dict()
    refWords = make_full_dict(commentWords)
    commentSet.update(refWords)

    articleProbDist = clf.prob_classify(articleSet)
    commentProbDist = clf.prob_classify(commentSet)

    if (articleProbDist.prob('pos')  < 0.5 and commentProbDist.prob('pos')  < 0.5) or (articleProbDist.prob('pos')  > 0.5 and commentProbDist.prob('pos')  > 0.5):
        return 1
    else:
        return 0

def getTweets():
    tweets = []
    for (words, sentiment) in pos_tweets + neg_tweets:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        tweets.append((words_filtered, sentiment))
    return tweets
    
    
    
pos_tweets = [('I love this car', 'positive'),
             ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]

twets = getTweets()

 # FUNCTIONS    
       
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words 

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(twets))
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def create_sent_classifier():   
    training_set = nltk.classify.apply_features(extract_features, twets)
        
    classifier = nltk.NaiveBayesClassifier.train(training_set)  
    f = open("sentiment_classifier.pickle", 'wb')
    pickle.dump(classifier, f)
    f.close()  
    print "Saved Classifier"
        
    
def read_in_tweets(filename):
    sentences = []
    with open(filename, "rb") as f_obj:
        reader = csv.reader(f_obj)
        count = 1
        for row in reader:
            sentences.append((row[0],row[5]))
            
    return sentences

        
def create_classifier(feature_select, filename):    
    posFeatures = []
    negFeatures = []
    # http://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation
    # breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
    
      
    sentences = read_in_tweets(twitter_data)
    random.shuffle(sentences)
    sentences = sentences[:100000]
    
    posSentences = []
    negSentences = []
    for tup in sentences:
        if tup[0]=='0':
            negSentences.append(tup[1])
        elif tup[0]=='4':
            posSentences.append(tup[1])
    
   
    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        posWords = [feature_select(posWords), 'pos']
        posFeatures.append(posWords)

    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        negWords = [feature_select(negWords), 'neg']
        negFeatures.append(negWords)

    
    # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    trainFeatures = negFeatures[:] + posFeatures[:]

    # trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)
    f = open(filename, 'wb')
    pickle.dump(classifier, f)
    f.close()



# this function takes a feature selection mechanism and returns its performance in a variety of metrics

def evaluate_features(feature_select, best_words):
    posFeatures = []
    negFeatures = []
   
   
      
    sentences = read_in_tweets(twitter_data)
    random.shuffle(sentences)
    sentences = sentences[:100000]
    
    posSentences = []
    negSentences = []
    for tup in sentences:
        if tup[0]=='0':
            negSentences.append(tup[1])
        elif tup[0]=='4':
            posSentences.append(tup[1])
    
   
    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        posWords = [feature_select(posWords,best_words), 'pos']
        posFeatures.append(posWords)

    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        negWords = [feature_select(negWords,best_words), 'neg']
        negFeatures.append(negWords)


    
    # selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures) * 3 / 4))
    negCutoff = int(math.floor(len(negFeatures) * 3 / 4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    # trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)    

    # initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)    

    # puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)    

    # prints metrics to show how well the feature selection did
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
    classifier.show_most_informative_features(10)


# scores words based on chi-squared test to show information gain (http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/)

def create_word_scores():
    # creates lists of all positive and negative words
    posWords = []
    negWords = []
      
    sentences = read_in_tweets(twitter_data)
    random.shuffle(sentences)
    sentences = sentences[:100000]
    
    posSentences = []
    negSentences = []
    for tup in sentences:
        if tup[0]=='0':
            negSentences.append(tup[1])
        if tup[0]=='4':
            posSentences.append(tup[1])
    
   
    for i in posSentences:
        posWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        posWords.append(posWord)

    for i in negSentences:
        negWord = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        negWords.append(negWord)
        
    
    
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    # build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd.inc(word.lower())
        cond_word_fd['pos'].inc(word.lower())
    for word in negWords:
        word_fd.inc(word.lower())
        cond_word_fd['neg'].inc(word.lower())

    # finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    # builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores



# finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

# creates feature selection mechanism that only uses best words
def best_word_features(words, best_words):
    return dict([(word, True) for word in words if word in best_words])
