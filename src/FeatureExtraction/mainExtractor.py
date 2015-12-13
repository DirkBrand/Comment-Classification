'''
Created on 13 May 2014

@author: Dirk
'''

from __builtin__ import dict
from _collections import defaultdict
import codecs
from collections import Counter
import collections
from datetime import datetime
from math import exp, sqrt
import pickle
import pprint
import re
import string
import sys
from time import  mktime, strptime

from FeatureExtraction.feature_utils import sent_frequency, words,\
    load_classifier, pos_freq, capital_frequency, sentence_capital_frequency,\
    entropy, F_K_score, missing_words, swears, make_full_dict, get_subjectivity,\
    get_polarity_overlap, timeliness, tf_idf, termf, onSubForumTopic,\
    lengthiness, penn_to_wn, known_words, lexical_diversity
import networkx
import nltk
from nltk.chunk import ne_chunk
from nltk.corpus import wordnet as wn, stopwords
from nltk.corpus import wordnet_ic
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.regexp import RegexpTokenizer, wordpunct_tokenize
from nltk.util import ngrams
from numpy import Inf
from pywsd import disambiguate
from pywsd.lesk import simple_lesk
from pywsd.similarity import max_similarity as maxsim
from scipy import stats
from sklearn.cluster.k_means_ import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob.blob import Blobber
from textblob.tokenizers import SentenceTokenizer, WordTokenizer
from textblob_aptagger.taggers import PerceptronTagger

from Profiling.timer import timefunc
from config import sentiment_path
import numpy as np
import pandas as pd 


pattern = r'''(?x) # set flag to allow verbose regexps
 ([A-Z]\.)+ # abbreviations, e.g. U.S.A.
 | \w+(-\w+)* # words with optional internal hyphens
 | \$?\d+(\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
 | \.\.\. # ellipsis
 | [][.,;"'?():-_`] # these are separate tokens'''


stops = set(nltk.corpus.stopwords.words("english"))

            
correction = 0

@timefunc 
def extract_entities(words):
    entities = []
    for chunk in ne_chunk(pos_tag(words)):
        if hasattr(chunk, 'node'):            
            performer = ' '.join(c[0] for c in chunk.leaves())
            entities.append(performer.lower())
    return entities


def printValues(author, likeCount, dislikeCount, ratio, commLen, qf, ef, cf, NF, VF, spelled, badWords, complexity, readibility, informativeness, sent):
    print "Author:", author, " - Likes:", likeCount, " - Dislikes", dislikeCount, " - Ratio:%.4f" % (ratio)        
    print 'length: %f - Question Frequency: %.2f - Exclamation Frequency: %.2f - Caps Frequency: %.2f' % (commLen , qf * 100, ef * 100 , cf * 100)
 
               
    print 'Noun Freq: %.2f' % (NF) 
    print 'Verb Freq: %.2f' % (VF)
    
    print 'Correctly Spelled: %.2f' % (spelled * 100)
    print 'Bad Words: %.2f' % (badWords * 100)
    print 'entropy:', complexity
    print 'F-K:', readibility
    print 'TF-IDF:', informativeness
    print 'sentiment', sent
    print "\n"
 
 
c = 0
def extract_values(df_comments, type):
    valueVector = np.empty([df_comments.shape[0]])
    index = 0
                 
    for _,row in df_comments.iterrows():  
        if type == 1: # MAIN
            if int(row.status) == 1:
                valueVector[index] = 0
            else:
                valueVector[index] = 1  
        elif type == 2: # TOY           
            if int(row.score) == 1:
                valueVector[index] = 1
            elif int(row.score) == 2:
                valueVector[index] = 0
            elif int(row.score) == 3:
                valueVector[index] = 0 
            else:
                valueVector[index] = 0 
        elif type == 3: # SLASHDOT
            if int(row.score) <= 2:
                valueVector[index] = 1
            else:
                valueVector[index] = 0             
                                              
            
        index += 1
        if index % 1000 == 0:
            print "extracted", index, "values"
        
           
    print "\nClass distribution %.3f" %(np.sum(valueVector)/valueVector.shape[0])
                
    return valueVector

def extract_sentence_values(articleList, commentList, parentList, commentCount):
    valueVector = np.empty([commentCount,4])
    index = 0
                 

       
    for commList in commentList.values():
        sumVotes = 0
        for comm in commList:
            sumVotes += comm.likeCount + comm.dislikeCount
            
        for comm in commList:
            sentences = nltk.sent_tokenize(comm.lemma_body)
            for sent in sentences:        
            
                tokens = nltk.regexp_tokenize(sent, pattern)
                theWords = words(comm.body)
                uniqueWords = set(theWords)
                
                if len(tokens) == 0 or len(uniqueWords) == 0:
                    continue
                
                
                ratio = (comm.likeCount) / (float(max(1,comm.likeCount+comm.dislikeCount)))
                
                
                #print ttest
                
                totalVotes = comm.likeCount + comm.dislikeCount
                    
                                           
                valueVector[index,0] = totalVotes
                valueVector[index,1] = ratio
                if comm.reported > 0:
                    valueVector[index,2] = 1
                else:
                    valueVector[index,2] = 0
                
                if comm.status == 1:
                    valueVector[index,3] = 0
                else:
                    valueVector[index,3] = 1                
                
                index += 1
                if index % 1000 == 0:
                    print "extracted", index, "values"
            
                if index >= commentCount:
                    break
            if index >= commentCount:
                break
                
    return valueVector



def extract_feature_matrix(df_comments, df_thread_groupby):
    print "START"
    # Sentence Tokenizer
    sentencer = SentenceTokenizer()
    
    clf = load_classifier(sentiment_path + 'sentiment_classifier.pickle')
        
    featureMatrix = np.empty([df_comments.shape[0],25])
    
    feature_dict = dict()
    for ix, row in df_comments.iterrows():
        feature_dict[row['comment_id']] = ix
    
    feature_count = 0
    
    for _,row in df_comments.iterrows():
        index = feature_dict[row['comment_id']]
        
        comm = row['comment_content'].decode('ASCII', 'ignore')
        tokens = words(comm)
        unique_tokens = set(tokens)
        sentences = sentencer.tokenize(comm)
        
        featureMatrix[index][3] =  len(comm)
        
        verb_fr, noun_fr, pronoun_fr = pos_freq(tokens)
        featureMatrix[index][4] = verb_fr
        featureMatrix[index][5] = noun_fr
        featureMatrix[index][6] = pronoun_fr
        
        featureMatrix[index][7] = capital_frequency(tokens)
        featureMatrix[index][8] = sent_frequency(sentences, '?')
        featureMatrix[index][9] = sent_frequency(sentences, '!')
        featureMatrix[index][10] = sentence_capital_frequency(sentences)
        
        featureMatrix[index][11] = entropy(comm)
        featureMatrix[index][12] = lexical_diversity(tokens)
        
        
        if len(tokens) == 0:
            featureMatrix[index][13] =  0
            featureMatrix[index][14] =  0
            featureMatrix[index][15] =  0
            featureMatrix[index][16] =  0
        else:
            spelt_wrong = missing_words(unique_tokens)
            bad_words_list = swears(unique_tokens)
            
            featureMatrix[index][13] =  len(spelt_wrong)
            featureMatrix[index][14] =  len(spelt_wrong)/float(len(unique_tokens))
            featureMatrix[index][15] =  len(bad_words_list)
            featureMatrix[index][16] =  len(bad_words_list)/float(len(unique_tokens))
            
            
        featureMatrix[index][19] =  F_K_score(sentences, tokens)
        
        testSet = dict()
        refWords = make_full_dict(tokens)
        testSet.update(refWords)
    
        probDist = clf.prob_classify(testSet)                
        sentiment = probDist.prob('pos')            
        subj_obj = get_subjectivity(probDist)
    
        polarity_overlap = get_polarity_overlap(words(row['article_body']), tokens, clf)
        featureMatrix[index][22] =  sentiment
        featureMatrix[index][23] =  subj_obj
        featureMatrix[index][24] =  polarity_overlap
        
        feature_count += 1
        if feature_count % 1000 == 0:
            print feature_count
    
    print "DONE"
    
    feature_count = 0
    # Grouped
    for _,group in df_thread_groupby:
        thread_comments = [row['comment_content'] for _,row in group.iterrows()]
        
        # Get average time
        sumTime = 0 
        count = 0                
        previous = mktime(group.iloc[0]['date'])
        first = mktime(group.iloc[0]['date'])
        
        # Average length
        sumLen = 0 
        
        
        thread_tokens = []    
        
        # Within Thread
        for _, row in group.iterrows():
            index = feature_dict[row['comment_id']]
            comm = row['comment_content'].decode('ascii','ignore')
            tokens = words(comm)
            sentences = sentencer.tokenize(comm)
            
            # Ongoing average time
            sumTime += mktime(row['date']) - previous
            count += 1            
            avgTime = sumTime/float(count)
            
            # Ongoing average length
            sumLen += len(words(row['comment_content']))
            avgLen = sumLen/float(count)
            
            ######################################################################
            # Get chunked sentences
            for sent in sentences:
                sent_tokens = words(sent)
                sent_tokens_tagged = nltk.pos_tag(sent_tokens)
                chunks = nltk.ne_chunk(sent_tokens_tagged, binary=True)
                doc = [] 
                for chunk in chunks:
                    if type(chunk) == nltk.Tree:
                        doc.append(' '.join(c[0] for c in chunk.leaves()))
                    else:
                        doc.append(chunk[0])
                doc = [word.strip(string.punctuation) for word in doc if len(word.strip(string.punctuation)) > 1]
                
                # The cumulative tokens up to this point
                thread_tokens += doc
            
            ######################################################################
            article_tokens = []
            article_sentences = sentencer.tokenize(row['article_body'])
            for sent in article_sentences:
                sent_tokens = words(sent)
                sent_tokens_tagged = nltk.pos_tag(sent_tokens)
                chunks = nltk.ne_chunk(sent_tokens_tagged, binary=True)
                doc = []
                for chunk in chunks:
                    if type(chunk) == nltk.Tree:
                        doc.append(' '.join(c[0] for c in chunk.leaves()))
                    else:
                        doc.append(chunk[0])
                article_tokens = [word.strip(string.punctuation) for word in doc if len(word.strip(string.punctuation)) > 1]
            
            ######################################################################
            
            
            featureMatrix[index][0] = timeliness(mktime(row['date']), previous, max(avgTime, 1))
            previous = mktime(row['date'])        
            
            featureMatrix[index][1] =  mktime(row['date']) - first  
            
            featureMatrix[index][2] = lengthiness(words(row['comment_content']), max(avgLen, 1))  
            
            featureMatrix[index][17] =  np.mean([termf(comm.count(w), tokens) for w in set(tokens)])  
            featureMatrix[index][18] =  tf_idf(comm, thread_comments)     
            
            featureMatrix[index][20] =  onSubForumTopic(tokens, thread_tokens)
            featureMatrix[index][21] =  onSubForumTopic(tokens, article_tokens)
    
    
            feature_count += 1
            if feature_count % 1000 == 0:
                print feature_count
    
    return featureMatrix





def extract_social_features(df_comments):
    socialVector = np.empty([df_comments.shape[0],8])
    index = 0
        
    graph = networkx.DiGraph()   
    
    userdict = dict()
    for _, row in df_comments.iterrows():
        userdict[row['comment_id']] = row['author']
        
    for user in set(userdict.values()):
        graph.add_node(user)
        
         
    for _, row in df_comments.iterrows():
        if not userdict.has_key(row['thread_root_id']):
            continue
        
        source = userdict[row['comment_id']]
        dest = userdict[row['thread_root_id']]
        if source == dest:
            continue
        graph.add_edge(source, dest)
    
    pageranker = networkx.pagerank(graph, alpha=0.85)
    hubs, auths = networkx.hits(graph)
    
    author_groupby = df_comments.groupby('author')
    user_age_dict = {}
    user_nr_posts_dict = {}
    for _,group in author_groupby:
        first_date = datetime.fromtimestamp(mktime(group.date.values[0]))
        last_date = datetime.fromtimestamp(mktime(group.date.values[-1]))
        diff = last_date - first_date
        days = diff.days
        user_age_dict[group.author.values[0]] = days + 1
        user_nr_posts_dict[group.author.values[0]] = len(group)
        
    for ix, row in df_comments.iterrows():            
        user = userdict[row['comment_id']]
        socialVector[ix][0] = graph.in_degree(user) #In Degree
        socialVector[ix][1] = graph.out_degree(user) #Out Degree
        socialVector[ix][2] = user_age_dict[user] #User Age
        socialVector[ix][3] = user_nr_posts_dict[user] #Nr of Posts
        socialVector[ix][4] = user_nr_posts_dict[user]/float(user_age_dict[user]) # Postrate
        socialVector[ix][5] = pageranker[user] # Pagerank
        socialVector[ix][6] = hubs[user] # Pagerank
        socialVector[ix][7] = auths[user] # Pagerank
    
        index += 1
        if index % 1000 == 0:
            print "extracted", index, "values"
        
                
    return socialVector


class CharacterSkipGramAnalyzer(object):   
    def __init__(self):
        self.sentencer = SentenceTokenizer()
        self.worder = WordTokenizer();
    def __call__(self, doc):  
        tokens = []      
        for sent in self.sentencer.tokenize(doc.lower()):
            words = ''.join([ch for ch in sent if ch not in string.punctuation])
            words = self.worder.tokenize(words)
            
            for word in words:
                tokens.append(word.strip())
                if len(word) > 2:
                    for j in range(0,len(word)):    
                        term = word[:j] + word[j+1:] 
                        tokens.append(term.strip())
        return tokens
    
class CharacterAnalyzer(object):   
    def __init__(self):
        self.sentencer = SentenceTokenizer()
        self.max = 8
        self.min = 2
    def __call__(self, doc):  
        tokens = []      
        for sent in self.sentencer.tokenize(doc.lower()):
            words = ''.join([ch for ch in sent if ch not in string.punctuation])
            for n in range(self.min,self.max+1):
                ngr = [words[i:i+n] for i in range(len(words)-n+1)]
                if len(ngr) > 0:
                    tokens += ngr
        return tokens

class UnigramAnalyzer(object):   
    def __call__(self, doc):  
        
        filtered_words = doc.split(" ")
        tokens = []
            
        for word in filtered_words:
            tokens.append(word)
        return tokens
    

                
class BigramAnalyzer(object):   
    def __call__(self, doc):   
        
        filtered_words = doc.split(" ")
        tokens = []
            
        for bigram in ngrams(filtered_words,2):
            tokens.append('%s %s' %bigram)
        return tokens
    
class TrigramAnalyzer(object):   
    def __call__(self, doc):   
        
        filtered_words = doc.split(" ")
        tokens = []
            
        for trigram in ngrams(filtered_words,3):
            tokens.append('%s %s %s' %trigram)
        return tokens
    
class QuadgramAnalyzer(object):  
    def __call__(self, doc):   
        
        filtered_words = doc.split(" ")
        tokens = []
            
        for qgram in ngrams(filtered_words,4):
            tokens.append('%s %s %s %s' %qgram)
        return tokens
    
class UnigramBigramAnalyzer(object):   
    def __call__(self, doc):
        
        filtered_words = doc.split(" ")
        tokens = []
            
            
        for word in filtered_words:
            tokens.append(word)
        for bigram in ngrams(filtered_words,2):
            tokens.append('%s %s' %bigram)
        return tokens

    
class UnigramBigramTrigramAnalyzer(object):  
    def __call__(self, doc): 
        filtered_words = doc.split(" ")
        tokens = []
            
            
        for word in filtered_words:
            tokens.append(word)
        for bigram in ngrams(filtered_words,2):
            tokens.append('%s %s' %bigram)
        for trigram in ngrams(filtered_words,3):
            tokens.append('%s %s %s' %trigram)
        return tokens

    
class UnigramBigramTrigramQuadgramAnalyzer(object): 
    def __call__(self, doc): 
        filtered_words = doc.split(" ")
        tokens = []
            
        for word in filtered_words:
            tokens.append(word)
        for bigram in ngrams(filtered_words,2):
            tokens.append('%s %s' %bigram)
        for trigram in ngrams(filtered_words,3):
            tokens.append('%s %s %s' %trigram)
        for qgram in ngrams(filtered_words,4):
            tokens.append('%s %s %s %s' %qgram)
        return tokens
    
    
class LexicalBigramUnigramAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer()
    def __call__(self, doc):   
        tokens = []     
        for sent in self.sentencer.tokenize(doc.decode('ascii','ignore')):
            tagged = self.tb(sent.lower()).tags    
            
            tagged = [(t[0], penn_to_wn(t[1])) for t in tagged]
            tagged = [(t[0], t[1]) for t in tagged if t[0] not in stopwords.words('english')]
            ng = zip(tagged, tagged[1:])
            rule1 = [(t[0],t[1]) for t in ng if t[0][1]== wn.ADJ and t[1][1]== wn.NOUN]
            rule2 = [(t[0],t[1]) for t in ng if (t[0][1]== wn.ADV and t[1][1]== wn.VERB) or (t[0][1]== wn.VERB and t[1][1]== wn.ADV)]
            rule3 = [(t[0],t[1]) for t in ng if t[0][1]== wn.VERB and t[1][1]== wn.VERB]
            rule4 = [(t[0],t[1]) for t in ng if t[0][1]== wn.NOUN and t[1][1]== wn.NOUN]
            
            filtered_list = rule1 + rule2 + rule3 + rule4
                             
                    
            # Lemmatize
            filtered_bigrams = [self.lemmatizer.lemmatize(t[0][0], t[0][1]) + ' ' + self.lemmatizer.lemmatize(t[1][0], t[1][1]) for t in filtered_list]
            filtered_unigrams = [self.lemmatizer.lemmatize(w[0], w[1]) for w in tagged]
            for bigram in filtered_bigrams:
                tokens.append(bigram)
            for unigram in filtered_unigrams:
                tokens.append(unigram)
        return tokens

def extract_global_bag_of_words_processed(df_comments):
    corpus = []   
    i = 0
    lemmatizer = WordNetLemmatizer()    
    tb = Blobber(pos_tagger=PerceptronTagger())
    sentencer = SentenceTokenizer()
    for _,row in df_comments.iterrows():  
        comm = row['comment_content']
        tokens = []   
        for sent in sentencer.tokenize(comm.decode('ascii','ignore')):
            tagged = tb(sent.lower()).tags    
            # Remove stops
            filtered_words = [w for w in tagged if not w[0] in stopwords.words('english')]
                   
            # Remove punctuation
            filtered_words = [(re.findall('[a-z]+', w[0].lower())[0], w[1]) for w in filtered_words if len(re.findall('[a-z]+', w[0].lower())) > 0]             
                    
            # Lemmatize
            filtered_words = [lemmatizer.lemmatize(w[0], penn_to_wn(w[1])) for w in filtered_words]  
            
            filtered_words = [w for w in filtered_words if len(w) > 1]
            
            for word in filtered_words:
                tokens.append(word)  
        corpus.append(' '.join(tokens))
        i += 1
        if i % 1000 == 0:
            print i, "words processed for Ngrams"
                
            
    return corpus


def extract_global_bag_of_words(df_comments):
    corpus = []   
    i = 0
    for _,row in df_comments.iterrows():   
        comm = row['comment_content'].decode('ascii','ignore')
        corpus.append(comm)
        i += 1
        if i % 1000 == 0:
            print i, "extracted"
                
            
    return corpus

def process_text(commentList):
    """ Tokenize text and stem words removing punctuation """
    tokens = []
    i = 0
    for art in commentList.items():        
        for comm in art[1]:
            text = re.findall('[a-z]+',comm.body.lower())
            tokens.append(' '.join(text))
            if i % 1000 == 0:
                print i, "processed"
            i += 1
            
    return tokens
    
def extract_global_bag_of_synsets(commentList):
    corpus = []
    global_synset_set = set()
    
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    
    # ISSUE throws away named entities
    i = 0
    for art in commentList.items():        
        for comm in art[1]:
            filtered_words = []
            for sentence in sent_detector.tokenize(comm.body.strip()):
                #print sentence       
                dis = disambiguate(sentence, algorithm=maxsim, similarity_option='wup')
                for w in dis:
                    # Only found words and nouns+verbs
                    if w[1] is None:
                        continue  
                    
                    if not w[1].pos() == wn.NOUN and not w[1].pos() == wn.VERB:
                        continue
                               
                    #print w[0] ," - ", w[1], " - ", w[1].definition()
                      
                    filtered_words.append(w[1])
                    global_synset_set.add(w[1])
                 
            corpus.append(filtered_words)
            i += 1
            print i
            if i % 1000 == 0:
                print i, "processed"
                break
        if i % 1000 == 0:
            print i, "processed"
            break
            
    return global_synset_set, corpus
    
       
def extract_words(vectorizer, train_list, test_list): 
    count_vect = vectorizer.fit(train_list)
    train = count_vect.transform(train_list)
    test = count_vect.transform(test_list)
    
    #print count_vect.get_feature_names()[1000:1010]
    
    #print count_vect.get_feature_names()
    print "Train:", train.shape    
    print "Test:", test.shape  
    print  
    
    return train, test



def extract_word_clusters(commentList, commentCount):
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    a, corpus, global_synsets = extract_global_bag_of_words(commentList, True)
    similarity_dict = {}
    i = 0
    t = len(global_synsets)**2
    
    for syn_out in global_synsets:
        similarity_dict[syn_out] = {} 
        for syn_in in global_synsets:
            if syn_in.pos() == syn_out.pos():
                similarity_dict[syn_out][syn_in] = syn_out.lin_similarity(syn_in, brown_ic)
            else:
                similarity_dict[syn_out][syn_in] = max(wn.path_similarity(syn_out,syn_in), wn.path_similarity(syn_in,syn_out))
        
            if i % 10000 == 0:
                print i, 'synsets processed out of',len(global_synsets)**2, '(',float(i)/(t),'%)'
            i += 1

    tuples = [(i[0], i[1].values()) for i in similarity_dict.items()] 
    vectors = [np.array(tup[1]) for tup in tuples]

    
    # Rule of thumb
    n = sqrt(len(global_synsets)/2)
    print "Number of clusters", n
    km_model = KMeans(n_clusters=n)
    km_model.fit(vectors)
    
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(tuples[idx][0])
        
    pprint.pprint(dict(clustering), width=1)
    
    feature_vector = np.zeros([len(corpus),n])
    
    for i,comment in enumerate(corpus):
        for w in comment:
            for key, clust in clustering.items():
                if w in clust:
                    feature_vector[i][key] += 1
        if i % 1000 == 0:
            print i, 'comments processed'
        
    print feature_vector
    '''
    #corpus = extract_global_bag_of_words(commentList)
    corpus = process_text(commentList)
    vectorizer = TfidfVectorizer(analyzer='word', use_idf=True, smooth_idf=True, max_df=0.5,
                                 min_df=0.1)
 
    tfidf_model = vectorizer.fit_transform(corpus)
    km_model = KMeans(n_clusters=1000)
    km_model.fit(tfidf_model)
    
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)
        
    pprint(dict(clustering))
    '''

def create_global_topic_list(articleList):
    e = re.compile(r"\s(de)\s")
    u = re.compile(r"\s(du)\s")
    globalTopicList = []
    
    i = 0
    for commList in articleList.values():
        # Article body + all comments 
        art = commList[0].artBody        
        for comm in commList:
            art += comm.body
            
        # Global list of named entities
        art = u.sub(" Du ", art)            
        art = e.sub(" De ", art)
        entities = extract_entities(wordpunct_tokenize(art))
        globalTopicList += entities 
        i += 1
        if i % 100 == 0:
            print i,"comments processed for global vector" 

    globalTopicList = nltk.FreqDist(globalTopicList)

    tempVector = dict()
    for k in globalTopicList.items()[:100]:
        tempVector[k[0]] = 0
    
    f = open("globalTopics" + '.pkl', 'wb')
    pickle.dump(tempVector, f, pickle.HIGHEST_PROTOCOL)
    f.close()

            

def extract_bigrams(articleList, commentCount):
    featureMatrix = np.zeros([commentCount,100])

    index = 0
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    bagOfWords = []
    for art in articleList.items():        
        for comm in art[1]:
            mywords = words(comm.body)
            mywords = known_words(mywords)
            # Remove Stops
            filtered_words = [w for w in mywords if not w in stopwords.words('english')]
            # Stemming
            stemmed_words = [stemmer.stem(w) for w in filtered_words]
            bagOfWords += stemmed_words
            bagOfWords.append("\n")
            
    tempVector = dict()
        
    #Create your bigrams
    bgs = nltk.bigrams(bagOfWords)

    fdist = nltk.FreqDist(bgs)   
    
    for k in fdist.keys()[:100]:
        tempVector[k] = 0
    
    
    theKeys = tempVector.keys()
    
    for art in articleList.items():        
        for comm in art[1]:
            mywords = words(comm.body)
            mywords = known_words(mywords)
            # Remove Stops
            filtered_words = [w for w in mywords if not w in stopwords.words('english')]
            # Stemming
            stemmed_words = [stemmer.stem(w) for w in filtered_words]
            bgs = nltk.bigrams(stemmed_words)
            for word in (w for w in bgs if tempVector.has_key(w)):
                keyInd = theKeys.index(word)      
                featureMatrix[index][keyInd] += 1
                           
            index += 1
            if index % 100 == 0:
                print "extracted", index, "features"
        
            if index >= commentCount:
                break            
            
            
    
    
    print "non-zero",np.count_nonzero(featureMatrix)
    print "Percentage filled:%.2f" %(float(np.count_nonzero(featureMatrix))/(featureMatrix.shape[0]*featureMatrix.shape[1]))
    return featureMatrix

def extract_Time_Data(articleList, commentCount):
    
    timeData = np.empty(commentCount,dtype=datetime)
    index = 0
    for art in articleList.items():        
        for comm in art[1]:
            dt = datetime(*comm.date[:6])
            timeData[index] = dt
            index += 1
            if index % 1000 == 0:
                print "extracted", index, "dates"
        
            if index >= commentCount:
                break
            
        if index >= commentCount:
            break
        
                
    return timeData;


def read_slashdot_comments(filename, skip=True):
    values = defaultdict(list)
    headers = ['article_id', 'comment_id', 'thread_root_id', 'parent_id', 'author', 'score', 'flag', 'date', 'wtf',
               'article_title', 'article_body', 'comment_title', 'has_link', 'comment_content', 'quoted_text']

    skippedCount = 0
    commentCount = 0
    f1 = open(filename, 'r')

    for line in f1:
        temp = line.split('\t')
        if len(temp) < 14:
            continue
        if len(words(temp[13])) == 0:
            continue

        for i, v in enumerate(temp):
            values[headers[i]].append(v)

    # Create Dataframe
    df_slashdot = pd.DataFrame(values)
    df_slashdot.drop('wtf', axis=1, inplace=True)
    # Decode Strings
    for col in df_slashdot.columns:
        df_slashdot[col] = df_slashdot[col].str.decode('iso-8859-1').str.encode('utf-8')

    # Add root for null roots
    cond = df_slashdot.thread_root_id == 'NULL'
    df_slashdot.loc[cond, 'thread_root_id'] = df_slashdot['comment_id']
    print df_slashdot[df_slashdot.thread_root_id == 'NULL'].shape


    # Replace date with datetime
    def map_date(date):
        date_ret = None
        try:
            date_ret = strptime(date, "<> on %A %B %d, %Y @%H:%M%p ()")
        except:
            date_ret = strptime(date, "on %A %B %d, %Y @%H:%M%p ()")
        return date_ret

    df_slashdot.date = df_slashdot.date.map(map_date)

    if (skip):
        df_slashdot = df_slashdot[df_slashdot['author'].str.lower() != 'anonymous coward']
        df_slashdot = df_slashdot[df_slashdot['score'] != '2']

    print "Done with comments"

    return df_slashdot

def read_toy_comments(mainfilename, toyfilename):
    f1 = open(mainfilename, 'r')        
    f2 = open(toyfilename, 'r')        
        
    ## Short list of comments
    values = defaultdict(list)
    headers_news24_large = ['article_id', 'comment_id','thread_root_id', 'user_id', 'likes','dislikes','reported','status','rating','date','author','article_title','article_body','comment_content','lemma_body','pos_body']

    
    commentCount = 0
    
    for line in f1:
        temp = line.split('&')            
        commentCount += 1
        if commentCount % 10000 == 0:
            print "Read", commentCount, "comments"
            
        for i,v in enumerate(temp):
            values[headers_news24_large[i]].append(v)
    
    
    df_news24_large = pd.DataFrame(values)
    
    print df_news24_large.shape
    print Counter(df_news24_large.status)
    
    print "Done with main comments"
    
    values = defaultdict(list)
    
    commentCount = 0
    # Toy comment set
    for line in f2:        
        temp = line.split('|')
        if len(temp) < 5:
            continue        
        
        comment = temp[0].replace('&','and').replace('..', '.').replace('.', '. ').replace('@','') 
        rating = temp[1]
        
        if not (rating == '1' or rating == '2' or rating == '3'):
            continue
        
        if len(comment) == 0:
            continue
        
        #print comment
        
        s2 = comment.translate(string.maketrans("",""), string.punctuation).replace(" ","")
        
        for ix, row in df_news24_large.iterrows():
            comm = row['comment_content']
            s1 = comm.translate(string.maketrans("",""), string.punctuation).replace(" ","")
            if s2 in s1:
                df_news24_large.set_value(ix, 'score', int(rating))
                
                commentCount += 1
                if commentCount % 100 == 0:
                    print "Read", commentCount, "comments"                    
                break
            
    # Replace date with datetime
    def map_date(date):
        date_ret = strptime(date, "%Y-%m-%d %H:%M:%S.%f")      
        return date_ret
         
                                
    df_news24_large.date = df_news24_large.date.map(map_date) 
    cond = df_news24_large.thread_root_id == 'null'    
    df_news24_large.loc[cond, 'thread_root_id'] = df_news24_large['comment_id']
    
    df_news24_small = df_news24_large[df_news24_large['score'].notnull()]

    print Counter(df_news24_small.score.values)
    print "Saved",commentCount
    return df_news24_small

def levenshteinDistance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]
 


WORD_MIN = 20 # At least that many words per comment (TWEET)
MIN_THREAD_LENGTH = 20 # Threads at least that long

def read_news24_comments(filename, skip=True, skip_mtn=False, limit = -1):
    # Short list of comments
    values = defaultdict(list)
    headers_news24 = ['article_id', 'comment_id','thread_root_id', 'user_id', 'likes','dislikes','reported','status','rating','date','author','article_title','article_body','comment_content','lemma_body','pos_body']
    f1 = open(filename, 'r')
    
    commentCount = 0
    totalCount = 0
    lessThanCount = 0
    mtnCount = 0
    
    for line in f1:
        temp = line.split('&')
        
        body = temp[13].lower()
        if skip:
            if len(words(body)) < WORD_MIN:
                lessThanCount += 1
                continue
            
        if skip_mtn:
            if "mtn" in body or "honda" in body or "toyota" in body or "form" in body or "camry" in body or "service" in body :
                mtnCount += 1
                continue
            
        commentCount += 1
        if commentCount % 10000 == 0:
            print "Read", commentCount, "comments"
            
        for i,v in enumerate(temp):
            values[headers_news24[i]].append(v)
    
   
    df_news24_large = pd.DataFrame(values)
    
    
    # Replace date with datetime
    def map_date(date):
        date_ret = strptime(date, "%Y-%m-%d %H:%M:%S.%f")      
        return date_ret
         
                                
    df_news24_large.date = df_news24_large.date.map(map_date) 
    
    
    # Add root for null roots
    cond = df_news24_large.thread_root_id == 'null'    
    df_news24_large.loc[cond, 'thread_root_id'] = df_news24_large['comment_id']
    print df_news24_large[cond].shape
    
   
    print "Saved",commentCount,"comments out of", totalCount
    print lessThanCount, "comments less than", WORD_MIN
    print mtnCount, "mtn comments"
    
    return df_news24_large



def read_user_data(filename):
    f = open(filename, 'r')        
        
    # To process all the comments
    userList = dict()
    commentCount = 0
    for line in f:        
        temp = line.split('&')
        if len(temp) < 9:
            continue
        
        userid = temp[0]
        inDeg = temp[1]
        outDeg = temp[2]
        age = temp[3]
        postCount = temp[4]
        postRate = temp[5]
        pageRank = temp[6]
        hub = temp[7]
        auth = temp[8]
        
        
        comm = [inDeg, outDeg, age, postCount, postRate, pageRank, hub, auth]
        
        userList[userid] = comm
        
        commentCount += 1
        if commentCount % 10000 == 0:
            print "Read", commentCount, "user comments"
        

    print "done reading"
            
        
    return userList, len(userList)

