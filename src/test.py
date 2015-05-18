'''
Created on 13 Apr 2015

@author: Dirk
'''
from nltk.tokenize import word_tokenize
'''
x_train = [[1.0, 5.0, 1.0],
                    [2.0, 6.0, 2.0]]
x_test = [[0.0, 5.0,  1.0],
                   [2.0, 10.0, 1.0]]

train = csr_matrix(x_train)
test = csr_matrix(x_test)

norm = train.copy()
norm.data **= 2 # Square every value
norm = norm.sum(axis=0) # Sum every column
n_nonzeros = np.where(norm > 0)
norm[n_nonzeros] = 1.0 / np.sqrt(norm[n_nonzeros])
norm = np.array(norm).T[0]

print norm1
print
print normalized(x_train,norm1)
print
print normalized(x_test,norm1)
print

sparsetools.csr_scale_columns(train.shape[0], train.shape[1], train.indptr, train.indices, train.data, norm)
print train.todense()
sparsetools.csr_scale_columns(test.shape[0], test.shape[1], test.indptr, test.indices, test.data, norm)
print test.todense()
'''
'''
puncts = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

fh = open('FeatureExtraction\\WordNetResources\\wn-domains-2.0-20050210', 'r')
dbdomains = anydbm.open('dbdomains', 'c')
for line in fh:
    offset, domain = line.split('\t')
    dbdomains[offset[:-2]] = domain
fh.close()

sentences = "Bringing the party into disrepute? Ha ha ha. Like he could've made it look any worse than it already was"

print nltk.version_info
for sentence in sent_detector.tokenize(sentences.strip()):
    print sentence       
    dis = disambiguate(sentence, algorithm=maxsim, similarity_option='wup', keepLemmas=True)
    for w in dis:
        if w[2] is None:
            continue
        
          
        print w[1] ," - ", w[2], " - ", w[2].definition()
        print dbdomains.get('0' + str(w[2].offset))
'''

import anydbm
import collections
from math import sqrt
import pprint
import re
import string

from nltk import cluster
import nltk
from nltk.corpus import webtext, treebank, reuters, wordnet_ic, wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from pywsd import disambiguate
from pywsd.lesk import simple_lesk
from pywsd.similarity import max_similarity as maxsim, max_similarity, sim
from scipy.sparse import sparsetools
from scipy.sparse.csr import csr_matrix
from sklearn.cluster.k_means_ import KMeans

from FeatureExtraction.LexicalFeatures import known
from FeatureExtraction.mainExtractor import onlyWords, stops
import numpy as np

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
 
synsets, hypernyms, common_hypernyms, part_meronyms = {}, {}, {}, {}
similarity_dict = {}                      
 
cats = ['doorknob', 'chandelier', 'radio', 'motorcycle', 'keyboard', 'chair', 'blender', 'flashlight', 'dice', 'cup', 'fireplace',
        'piano', 'hat', 'plate', 'handle', 'cannon', 'screwdriver', 'fan', 'desk', 'handgun', 'stapler', 'gun', 'locomotive', 'bicycle',
        'globe', 'sign', 'turntable', 'computer', 'trombone', 'candle', 'hammer', 'hourglass', 'clock', 'harmonica', 'microwave', 'umbrella',
         'mailbox', 'guitar', 'sink', 'key', 'anchor', 'pliers', 'headphones', 'skateboard', 'blimp', 'grenade', 'stoplight', 'drums', 'scissors',
          'hydrant', 'submarine', 'synthesizer', 'camera', 'helicopter', 'fish', 'door', 'television', 'tank', 'skull', 'car', 'blade', 'tree',
          'bed', 'banana', 'donut', 'cake', 'spotlight', 'toilet', 'lock', 'telephone', 'airplane', 'trumpet', 'tyrannosaurus', 'sailboat', 'chessboard',
           'toaster', 'cross', 'couch', 'speaker', 'elephant', 'dresser', 'windmill', 'plant', 'ladder', 'remote', 'yacht', 'person', 'wrench', 'bottle', 'balloon']
           
verbs = ['camp', 'cure', 'relax', 'double', 'challenge', 'greet', 'cry', 'imagine', 'breathe', 'spoil', 'clear', 'bat', 'offend', 'scold', 'count', 'suspend',
           'attend', 'slap', 'risk', 'prevent', 'settle', 'soak', 'deceive', 'hand', 'scorch', 'eat', 'hit']                                                                                                                         
 
adjectives = [    'dazzling', 'spurious', 'efficient', 'polite', 'cruel', 'chunky', 'sophisticated', 'obtainable', 'mighty', 'succinct', 'unarmed',
              'crazy', 'small', 'unknown', 'parsimonious', 'tender', 'deranged', 'rambunctious', 'pointless', 'puzzling', 'wonderful', 'pathetic', 'closed'] 

test_sentence = 'The sophisticated man camped under the small chandelier wearing efficient headphones and using a camera with a pathetic handgun.  He offended and scolded everyone.' 

cats.sort()
 
for cat in cats:
    if cat == 'wrench':
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[2]
    elif cat == 'plant':                              
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'toaster':                            
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'toilet':                             
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'cake':                               
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[2]
    elif cat == 'banana':                             
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'blade':                              
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[3]
    elif cat == 'television':                         
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[2]
    elif cat == 'synthesizer':                        
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'hydrant':                            
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'stoplight':                          
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'spotlight':                          
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'microwave':                          
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'hammer':                             
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'globe':                              
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    elif cat == 'plate':                              
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[3]
    elif cat == 'radio':                              
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[1]
    else:                                             
        synsets[cat] = wn.synsets(cat, pos=wn.NOUN)[0]
 
    hypernyms[cat] = synsets[cat].hypernym_paths()
    part_meronyms[cat] = synsets[cat].part_meronyms()

for verb in verbs:
    synsets[verb] = wn.synsets(verb, pos=wn.VERB)[0]
    
    
for adj in adjectives:
    syns =  wn.synsets(adj, pos=wn.ADJ)
    synsets[adj] = syns[0]

for outer_cat in cats + verbs + adjectives:
    common_hypernyms[outer_cat] = {}
 
    similarity_dict[outer_cat] = {}         
 
    for inner_cat in cats + verbs + adjectives:
        common_hypernyms[outer_cat][inner_cat] = synsets[outer_cat].lowest_common_hypernyms(synsets[inner_cat])
 
        # similarity_dict[outer_cat][inner_cat] = max(wn.path_similarity(synsets[outer_cat],synsets[inner_cat]), wn.path_similarity(synsets[inner_cat],synsets[outer_cat]))
        print synsets[inner_cat].pos(),synsets[outer_cat].pos()
        if synsets[inner_cat].pos() == synsets[outer_cat].pos():
            similarity_dict[outer_cat][inner_cat] = synsets[outer_cat].lin_similarity(synsets[inner_cat], brown_ic)
        else:
            similarity_dict[outer_cat][inner_cat] = max(wn.path_similarity(synsets[outer_cat], synsets[inner_cat]), wn.path_similarity(synsets[inner_cat], synsets[outer_cat]))
            

tuples = [(i[0], i[1].values()) for i in similarity_dict.items()] 
vectors = [np.array(tup[1]) for tup in tuples]

    
# Rule of thumb
n = sqrt()
print "Number of clusters", n
km_model = KMeans(n_clusters=n)
km_model.fit(vectors)

clustering = collections.defaultdict(list)
for idx, label in enumerate(km_model.labels_):
    clustering[label].append(tuples[idx][0])
    
pprint.pprint(dict(clustering), width=1)

vect = np.zeros([1, 20])

for sentence in sent_detector.tokenize(test_sentence.strip()):
    # print sentence           
    for w in word_tokenize(sentence):                  
        # print w[1] ," - ", w[2], " - ", w[2].definition()
        for key, clust in clustering.items():
            if w[1] in clust:
                vect[0][key] += 1

print vect

def write_csv(name, cats, dict):
    csv = ''
    csv = 'Name' + ',' + ','.join(cats) + '\n'
    for outer_cat in cats:
        csv += outer_cat + ','
        for inner_cat in cats:
            csv += str(dict[outer_cat][inner_cat])
            if inner_cat != cats[-1]:
                csv += ','
        csv += '\n'
 
    f = open(name + '.csv', 'w')
    f.write(csv)
    f.close()
 

#write_csv("similarities", cats + verbs, similarity_dict)
