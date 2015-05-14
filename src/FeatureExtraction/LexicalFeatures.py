'''
Created on 06 Mar 2014

@author: Dirk
'''
from collections import Counter
from math import log
import re, collections

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import names, stopwords

from Profiling.timer import timefunc
from config import comment_data_path


def words(text): return re.findall('[a-z]+', text.lower())

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

def words_from_list(filepath):
    f = open(filepath, 'r') 
    words = ['-']
    for line in f:
        words.append(line.strip())
        
    return words
     
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

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
 
def getVF(pf):  
    vf = 0  
    if 'VB' in pf.keys():
        vf += pf['VB']
    if 'VBD' in pf.keys():
        vf += pf['VBD']
    if 'VBG' in pf.keys():
        vf += pf['VBG']
    if 'VBN' in pf.keys():
        vf += pf['VBN']
    if 'VBP' in pf.keys():
        vf += pf['VBP']
    if 'VBZ' in pf.keys():
        vf += pf['VBZ']
        
    return vf  

def getNF(pf):
    nf = 0
    if 'NN' in pf.keys():
        nf += pf['NN']
    if 'NNP' in pf.keys():
        nf += pf['NNP']
    if 'NNS' in pf.keys():
        nf += pf['NNS']
    if 'NNPS' in pf.keys():
        nf += pf['NNPS']
    if 'WDT' in pf.keys():
        nf += pf['WDT']
    if 'WP' in pf.keys():
        nf += pf['WP']
    if 'WP$' in pf.keys():
        nf += pf['WP$']
    if 'WRB' in pf.keys():
        nf += pf['WRB']
    if 'PRP' in pf.keys():
        nf += pf['PRP']
    if 'PRP$' in pf.keys():
        nf += pf['PRP$']
        
    return nf

def getPN(pf):
    count = 0
    if 'PRP' in pf.keys():
        count += pf['PRP']
    if 'PRP$' in pf.keys():
        count += pf['PRP$']
    if 'WP' in pf.keys():
        count += pf['WP']
    if 'WP$ ' in pf.keys():
        count += pf['WP$ ']

    return count
NWORDS = train(words(file(comment_data_path + 'big.txt').read()))
BADWORDS = words_from_list(comment_data_path + 'badwords.txt')
FNAMES = set(w.lower() for w in names.words('female.txt'))
MNAMES = names.words('female.txt')

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
   replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return [w for w in words if w in NWORDS or w in FNAMES or w in MNAMES]


def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)


# COMPLEXITY
@timefunc   
def entropy(body, mywords):
    freq = nltk.FreqDist(body)
    length = len(set(mywords))
    sum = 0
    for f in freq.keys():
        p = freq[f]
        sum += p * (log(length, 10) - log(p, 10))
        
    return 1 / float(length) * sum

# SWEAR WORDs
@timefunc   
def swears(words): return set(w for w in words if w in BADWORDS)

# TAGGED POS COUNTER
def pos_freq(tokens):
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    
    counts = Counter(tag for word, tag in tags)
    return counts

@timefunc
def withoutStopWords(tokens):
    stop = stopwords.words('english')
    return [i for i in tokens if i not in stop]
    
    
