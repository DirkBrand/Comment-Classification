'''
Created on 08 Nov 2015

@author: Dirk
'''

from _collections import defaultdict
from math import log
import math
import pickle
import re
import time

import nltk
from nltk.corpus import names, wordnet as wn, cmudict 

from config import comment_data_path, sentiment_path


def words(text): return re.findall('[a-z]+', text.lower())

def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        #print f.__name__, 'took', end - start, 'time'
        return result
    return f_timer

@timefunc
def timeliness(comment_date, previous_comment_date, ave_time_thread):
    return (float(comment_date)-previous_comment_date)/ave_time_thread

@timefunc
def lengthiness(tokens, avgLen):
    return len(tokens) / avgLen


@timefunc
def pos_freq(tokens):
    tags = nltk.tag.pos_tag(tokens)
        
    verb_fr = 0
    noun_fr = 0
    pronoun_fr = 0
    
    for word, tag in tags:
        if tag in ['VB', 'VBD','VBG','VBN','VBP','VBZ']:
            verb_fr += 1
        if tag in ['NN', 'NNP','NNS','NNPS']:
            noun_fr += 1        
        if tag in ['PRP', 'PRP$','WP','WP$','WDT','WRB']:
            pronoun_fr += 1
    
    
    return verb_fr, noun_fr, pronoun_fr


@timefunc
def sent_frequency(sentences, mark):
    count, total = 0, 0
    for s in sentences:
        if len(s) > 1:
            total += 1
        if s.rstrip().endswith(mark):
            count += 1            
    if total == 0:
        return 0
    
    return float(count) / float(total) # PERCENTAGE

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

    count, total = 0, 0
    for sent in sentences:
        if len(sent) > 1:
            total += 1
        if sent[0].isupper():
            count += 1

    if total == 0:
        return 0
            
    return  float(count) / float(total)

@timefunc
def entropy(comment):
    tokens = words(comment)
    if len(tokens) == 0:
        return 0
    
    freq = nltk.FreqDist(tokens)
    number_of_uniques = len(set(tokens))
    sum_entropy = 0
    for f in freq.keys():
        p = freq[f]
        sum_entropy += p * (log(number_of_uniques, 10) - log(p, 10))
        
    return 1 / float(number_of_uniques) * sum_entropy


@timefunc
def lexical_diversity(tokens):
    if len(tokens) == 0:
        return 0
    return len(set(tokens)) / float(len(tokens))

r = cmudict
pronounce_dict = r.dict()

def nsyl(word):
    if word.lower() not in pronounce_dict:
        return 1
    
    return max([len(list(y for y in x if y[-1].isdigit())) for x in pronounce_dict[word.lower()]])

@timefunc
def F_K_score(sentences, tokens):
    syll_count = 0
    for w in tokens:
        syll_count += nsyl(w)
        
    return 206.835 - 1.015*(float(len(tokens)) / float(len(sentences))) - 84.6*(float(syll_count) / float(len(tokens)))


@timefunc
def termf(wordFreq, theWords):
    fdist = nltk.FreqDist(theWords)
    tf = (float(wordFreq))/max(fdist.values())
    return tf

@timefunc
def tf_idf(comment, comment_thread):
    tokens = words(comment)
    
    sum_inform = 0
    for w in tokens:
        tf = termf(comment.count(w), tokens)
        count = 0
        for comm in comment_thread:
            if w in comm.decode('ascii', 'ignore').lower():
                count += 1
        
        idf = max(0,math.log(len(comment_thread) / (count+1.0)))
        sum_inform += tf * idf
    
    return sum_inform 


@timefunc
def onSubForumTopic(tokens, forum_tokens):
    if len(tokens) == 0:
        return 0
    count = len(set(tokens) & set(forum_tokens))
    
    return float(count) / len(tokens)


# SPELLING
@timefunc
def words_from_list(filepath):
    f = open(filepath, 'r') 
    word_ret = []
    for line in f:
        word_ret.append(line.strip())
    return word_ret

BADWORDS = words_from_list(comment_data_path + 'badwords.txt')
FNAMES = set(w.lower() for w in names.words('female.txt'))
MNAMES = names.words('male.txt')

def train(features):
    model = defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model
NWORDS = train(words(file(comment_data_path + 'big.txt').read()))


@timefunc
def missing_words(words): 
    return [w for w in words if not w in NWORDS and not w in FNAMES and not w in MNAMES]

@timefunc
def known_words(words): return [w for w in words if w in NWORDS or w in FNAMES or w in MNAMES]


@timefunc
def swears(words): 
    return [w for w in words if w in BADWORDS]


# SENTIMENT
@timefunc
def load_classifier(filename):   
    f = open(filename)
    classifier = pickle.load(f)
    f.close()
    return classifier


@timefunc
def make_full_dict(words):
    return dict([(word, True) for word in words])

@timefunc
def get_subjectivity(probDist):
    p = probDist.prob('pos')
    n = probDist.prob('neg')
    
    # Subjective = 1, Objective = 0
    if abs(p - n) < 0.2:
        return 0
    else:
        return 1
    

@timefunc
def get_polarity_overlap(articleWords, commentWords, clf):
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
    
    
@timefunc
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


@timefunc
def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


@timefunc
def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


@timefunc
def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

@timefunc
def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN