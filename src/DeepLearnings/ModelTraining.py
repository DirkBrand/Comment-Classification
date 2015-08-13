'''
Created on 25 Jun 2015

@author: Dirk
'''
import logging
import os
import re
import string

from gensim.models.doc2vec import Doc2Vec, LabeledSentence, TaggedDocument
from gensim.models.phrases import Phrases
from gensim.models.word2vec import Word2Vec
from mainExtractor import read_comments
import nltk
from nltk.corpus import stopwords
from sklearn.manifold.t_sne import TSNE
from textblob.tokenizers import SentenceTokenizer

from config import model_path, comment_data_path
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
    level=logging.INFO)

stops = set(stopwords.words("english"))

def comment_to_chunked_wordlist(line):    
    text = re.sub("[^a-zA-Z]", " ", line )
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
    
    entity_names = []
    for tree in chunked_sentences:  
        for chunk in tree:
            if type(chunk) == nltk.Tree:
                entity_names.append(' '.join(c[0] for c in chunk.leaves()))
            else:
                entity_names.append(chunk[0])
        entity_names = [word.strip(string.punctuation).lower() for word in entity_names if len(word.strip(string.punctuation)) > 1]
    
    words = [w for w in entity_names if not w in stops]        
    return words



def comment_to_wordlist(line, remove_stops=False):
    text = re.sub("[^a-zA-Z]", " ", line )
    words = text.split(" ")
    if remove_stops:
        words = [w for w in words if not w in stops and len(w) >0]
        
    words = [w.lower() for w in words if len(w) > 0]
    return words
        
def comment_to_sentences(comment, remove_stops=False):
    sentencer = SentenceTokenizer();
    
    corpus = []
    for sent in sentencer.tokenize(comment):  
        if len(sent) > 0 :  
            corpus.append(comment_to_wordlist(sent, remove_stops))
    
    return corpus

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
    
    def __iter__(self):
        f = open(self.filename, 'r')
        for line in f:
            body = line.split("|")[1]
            for sent in comment_to_sentences(body):
                yield sent
                
class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename):
            body = line.split("|")[1]
            uid = line.split("|")[0]
            yield LabeledSentence(comment_to_wordlist(body), [uid])
    def to_array(self):
        self.sentences = []
        for line in open(self.filename):
            body = line.split("|")[1]
            uid = line.split("|")[0]
            self.sentences.append(LabeledSentence(comment_to_wordlist(body), [uid]))
        return self.sentences
    
    def sentences_perm(self):
        return np.random.permutation(self.sentences)


# Set values for various parameters
num_features = 200  # Word vector dimensionality                      
min_word_count = 10  # Minimum word count                        
num_workers = 4 # Number of threads to run in parallel
context = 10  # Context window size                                                                                    
downsampling = 1e-3  # Downsample setting for frequent words

def train_model(model_type, model_name, filename):  
    print "Training model..."
    print model_name
    
    if model_type < 10:
        model = Word2Vec(MySentences(filename), workers=num_workers, sg=2, \
                    size=num_features, min_count=min_word_count, \
                    window=context, sample=downsampling)
    else:
        
        model = Doc2Vec(alpha=0.025, min_alpha=0.025, workers=num_workers, size=num_features, min_count=1, sample=downsampling)
        sentences = LabeledLineSentence(filename)
        model.build_vocab(sentences.to_array())
        for epoch in range(5):
            model.train(sentences.sentences_perm())
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        
            
          
    model.init_sims(replace=True)
           
    model.save(model_path + model_name)

        

 
model_type = 1
tag = "_news24"
#tag = "_slashdot"
if __name__ == '__main__':   
    
    if model_type == 1:
        model_name = "Basic300" + tag
    if model_type == 10:
        model_name = "DocBasic300" + tag
        
    train_model(model_type, model_name, comment_data_path + "comms_data.txt")
    #train_model(model_type, model_name, comment_data_path + "slashdotCommData.txt") 

    
    

def get_model(model_num, model_names):
    
    
    if model_num < 10:
        model = Word2Vec.load(model_path + model_names)
    elif model_num < 99:
        model = Doc2Vec.load(model_path + model_names)
    else:
        model = Word2Vec.load_word2vec_format(model_path + model_names, binary=True)  # C text format
    return model
