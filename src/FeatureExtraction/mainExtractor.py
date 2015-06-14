'''
Created on 13 May 2014

@author: Dirk
'''

from __builtin__ import dict
from _collections import defaultdict
from collections import Counter
import collections
from datetime import datetime
from math import exp, sqrt
import pickle
import pprint
import re
import string
from time import  mktime, strptime

import nltk
from nltk.chunk import ne_chunk
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.regexp import RegexpTokenizer, wordpunct_tokenize
from pywsd import disambiguate
from pywsd.lesk import simple_lesk
from pywsd.similarity import max_similarity as maxsim
from scipy import stats
from sklearn.cluster.k_means_ import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob.blob import Blobber
from textblob.tokenizers import SentenceTokenizer
from textblob_aptagger.taggers import PerceptronTagger

from DeepLearnings.FeatureExtraction import ENGAGE_MIN
from FeatureExtraction.LexicalFeatures import words, pos_freq, getVF, getNF, \
    getPN, known, swears, entropy, penn_to_wn
from FeatureExtraction.SentimentUtil import load_classifier, make_full_dict, \
    get_subjectivity, get_polarity_overlap
from FeatureExtraction.SurfaceFeatures import lengthiness, question_frequency, \
    exclamation_frequency, capital_frequency, sentence_capital_frequency, \
    onSubForumTopic, F_K_score, tf_idf, avgTermFreq, timeliness
from Objects import CommentObject, ArticleObject, UserCommentObject, UserObject, \
    ArticleCommentObject
from Profiling.timer import timefunc
from config import sentiment_path
import numpy as np
from nltk.util import ngrams


pattern = r'''(?x) # set flag to allow verbose regexps
 ([A-Z]\.)+ # abbreviations, e.g. U.S.A.
 | \w+(-\w+)* # words with optional internal hyphens
 | \$?\d+(\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
 | \.\.\. # ellipsis
 | [][.,;"'?():-_`] # these are separate tokens'''


stops = set(nltk.corpus.stopwords.words("english"))

def onlyWords(text): return re.findall('[a-z]+', text.lower()) 
            
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
 
 
def lexical_diversity(text):
    return len(set(text)) / len(text)

c = 0
def extract_values(articleList, commentList, parentList, commentCount):
    valueVector = np.empty([commentCount,4])
    index = 0
                 
    sumLikes = 0         
    sumDislikes = 0
    sumRatio = 0
    for art in commentList.values():
        for comm in art:
            sumLikes += comm.likeCount 
            sumDislikes += comm.dislikeCount
            sumRatio += comm.likeCount/float(max(1,comm.likeCount+comm.dislikeCount))
    
    globalMean = np.mean(np.append(np.ones(sumLikes),(np.zeros(sumDislikes))))
    print "Global vote mean:", globalMean
    
    for art in commentList.values():
        sumVotes = 0
        for comm in art:
            sumVotes += comm.likeCount + comm.dislikeCount
        
        for comm in art:
            
            tokens = nltk.regexp_tokenize(comm.body, pattern)
            theWords = words(comm.body)
            uniqueWords = set(theWords)
            
            if len(tokens) == 0 or len(uniqueWords) == 0:
                continue
            
            arr = np.append(np.ones(comm.likeCount),-(np.ones(comm.dislikeCount)))
            ttest, p =  stats.ttest_1samp(arr, globalMean)
            
            
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

def extract_feature_matrix(articleList, commentList,  parentList, commentCount):
    # Sentence Tokenizer
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    
    # Sentiment Classifier
    clf = load_classifier(sentiment_path + 'sentiment_classifier.pickle')
    
    featureMatrix = np.empty([commentCount,23])
    
    
    index = 0
    for commList in commentList.values():
        sumLen = 0
        for comm in commList:
            sumLen += len(words(comm.body))
            
        avgLen = float(sumLen) / len(commList)
        
        # Thread BOW: with CHUNKING
        cnt = Counter()
        for comm in commList:
            sentences = nltk.sent_tokenize(comm.body)
            sentences = [nltk.word_tokenize(sent) for sent in sentences]
            sentences = [nltk.pos_tag(sent) for sent in sentences]
            
            for sent in sentences:
                chunks = nltk.ne_chunk(sent, binary=True)
                doc = [] 
                for chunk in chunks:
                    if type(chunk) == nltk.Tree:
                        doc.append(' '.join(c[0] for c in chunk.leaves()))
                    else:
                        doc.append(chunk[0])
                doc = [word.strip(string.punctuation) for word in doc if len(word.strip(string.punctuation)) > 1]
                for w in doc:
                    cnt[w] += 1



        # Article BOW: with CHUNKING
        articleCnt = Counter()
        sentences = nltk.sent_tokenize(articleList[commList[0].article_id].synopsis)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [nltk.pos_tag(sent) for sent in sentences]
        for sent in sentences:
            chunks = nltk.ne_chunk(sent, binary=True)
            doc = []
            for chunk in chunks:
                if type(chunk) == nltk.Tree:
                    doc.append(' '.join(c[0] for c in chunk.leaves()))
                else:
                    doc.append(chunk[0])
            doc = [word.strip(string.punctuation) for word in doc if len(word.strip(string.punctuation)) > 1]
    
            for w in doc:
                articleCnt[w] += 1
            
                     
        # Average Timeliness
        startTime = mktime(commList[0].date)
        sum = 0
        for comm in commList:
            diff = mktime(comm.date) - startTime
            startTime = mktime(comm.date)
            sum += diff
        
        aveTime = float(sum) / float(len(commList))

        for ind in range(len(commList)):
            comm = commList[ind]

            tokens = nltk.regexp_tokenize(comm.body, pattern)
            text = nltk.Text([w.lower() for w in tokens])
            theWords = words(comm.body)
            uniqueWords = set(theWords)
            sentences = sent_detector.tokenize(comm.body.strip())
            
            if len(tokens) == 0 or len(uniqueWords) == 0:
                continue
            
                    
            commLengthiness = lengthiness(avgLen, theWords)
            numberCharacters = len(comm.body)
            diversity = lexical_diversity(text)
                
            qf = question_frequency(sentences)
            ef = exclamation_frequency(sentences)
            cf = capital_frequency(tokens)
            scf = sentence_capital_frequency(sentences)
            
            pf = pos_freq(theWords)
            vf = getVF(pf)
            nf = getNF(pf)

            pronouns = getPN(pf)
                       
                
            #threadRelevance = onSubForumTopic(tokens, cnt.keys())
            articleRelevance = onSubForumTopic(tokens, articleCnt.keys())
            
            
            spelled = len(known(uniqueWords))
            spelledPerc = float(spelled) / len(uniqueWords)
            badWords = len(swears(uniqueWords))
            badWordsPerc = float(badWords) / len(uniqueWords)
            complexity = entropy(comm.body, theWords)
            readibility = F_K_score(comm.body, sentences)
            #informativeness = tf_idf(text, theWords, commList)
            meantermFreq = avgTermFreq(text, theWords)
            #termFreq = 0
            
            testSet = dict()
            refWords = make_full_dict(theWords)
            testSet.update(refWords)
                
                
            probDist = clf.prob_classify(testSet)                
            sentiment = probDist.prob('pos')            
            subj_obj = get_subjectivity(probDist)

            polarity_overlap = get_polarity_overlap(nltk.regexp_tokenize(articleList[commList[0].article_id].body, pattern), theWords)


            if ind > 0:
                timely = timeliness(mktime(comm.date), mktime(commList[ind-1].date), aveTime)
            else:
                timely = timeliness(mktime(comm.date), mktime(comm.date), aveTime)


            timePassing = mktime(comm.date) - mktime(commList[0].date)
            
            if timePassing < 0 or timely < 0:
                timePassing = 0
                timely = 0
                '''
            # Community features
            likes = comm.likeCount 
            dislikes = comm.dislikeCount
            reports = 0
            if comm.reported > 0:
                reports = 1 
            
            engagement = 0
            if parentList.has_key(comm.id):
                engagement = parentList[comm.id]
                
                
            ratio = (comm.likeCount + 1) / (float(comm.likeCount+comm.dislikeCount + 2))
            '''
            #printValues(comm.author, comm.likeCount, comm.dislikeCount, (comm.likeCount+correction) / (comm.dislikeCount+comm.likeCount+2*correction), commLen, qf, ef, cf, nf, vf, spelled, badWords, complexity, readibility, termFreq, sentiment)
            
            featureMatrix[index] = np.array([timely, #19
                                             timePassing, #20
                                             commLengthiness, #0
                                             numberCharacters, #24
                                             vf, #15
                                             nf, #16
                                             pronouns,
                                             cf, #3
                                             qf, #1
                                             ef, #2
                                             scf, #4
                                             complexity, #10
                                             diversity, #21
                                             spelled, #5
                                             spelledPerc, #6
                                             badWords, #7
                                             badWordsPerc, #8
                                             meantermFreq, #12
                                             #informativeness, #23
                                             readibility, #11
                                             #threadRelevance, #17
                                             articleRelevance, #18
                                             sentiment, #13
                                             subj_obj, #14
                                             polarity_overlap, #22
                                             #likes,
                                             #dislikes,
                                             #ratio,
                                             #reports,
                                             #engagement
                                             ], float) #25
            index += 1
            if index % 100 == 0:
                print "extracted", index, "features"

            if index >= commentCount:
                break
            
        if index >= commentCount:
            break
        
                
    return featureMatrix



def extract_user_values(userList, userCount):
    valueVector = np.empty([userCount,4])
    index = 0
    corr = 10
    for user in userList:   
        allText = ''            
        sumVotes = 0
        for comm in user.comments:
            sumVotes += comm.likes + comm.dislikes
            allText += comm.body
        
            
        tokens = nltk.regexp_tokenize(allText, pattern)
        theWords = words(allText)
        uniqueWords = set(theWords)
        
        if len(tokens) == 0 or len(uniqueWords) == 0:
            continue
        
        ratio = (user.likeSum + corr) / (float(user.totalVotes + 2*corr) )
            
        valueVector[index,0] = ratio
        valueVector[index,1] = user.totalVotes
        
        index += 1
        if index % 1000 == 0:
            print "extracted", index, "values"           
                     
        
        if index >= userCount:
            break
                
    return valueVector


def extract_social_features(userList, articleList, commentCount):
    socialVector = np.empty([commentCount,8])
    index = 0
                
    
    for commList in articleList.values():             
        for comm in commList:
            if not userList.has_key(comm.userId):
                continue
            
            user = userList[comm.userId] 
            
        
            socialVector[index,0] = float(user[0])                      #In Degree
            socialVector[index,1] = float(user[1])                      #Out Degree
            socialVector[index,2] = float(user[2])                      #User Age
            socialVector[index,3] = float(user[3])                      #Nr of Posts
            socialVector[index,4] = float(user[4])                      #Post Rate
            socialVector[index,5] = float(user[5].strip())              #PageRank
            socialVector[index,6] = float(user[6].strip())              #Hub
            socialVector[index,7] = float(user[7].strip().strip('.'))   #Auth
            
            index += 1
            if index % 1000 == 0:
                print "extracted", index, "values"
        
            if index >= commentCount:
                break
        if index >= commentCount:
            break
                
    return socialVector


class CharacterAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer();
        self.num = 6;
    def __call__(self, doc):  
        tokens = []      
        for sent in self.sentencer.tokenize(doc.decode('ascii','ignore')):
            sent = sent.lower()
            for word in word_tokenize(sent):
                word= ''.join([ch for ch in word if ch not in string.punctuation])
                for n in range(3,self.num+1):
                    ngr = [word[i:i+n] for i in range(len(word)-n+1)]
                    if len(ngr) > 0:
                        tokens += ngr
       
        return tokens

class UnigramAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer()
    def __call__(self, doc):  
        tokens = []      
        for sent in self.sentencer.tokenize(doc.decode('ascii','ignore')):
            sent = sent.lower()
            tagged = self.tb(sent).tags    
            # Remove stops
            filtered_words = [w for w in tagged if not w[0] in stops]
                   
            # Remove punctuation
            filtered_words = [(re.findall('[a-z]+', w[0].lower())[0], w[1]) for w in filtered_words if len(re.findall('[a-z]+', w[0].lower())) > 0]             
                    
            # Lemmatize
            filtered_words = [self.lemmatizer.lemmatize(w[0], penn_to_wn(w[1])) for w in filtered_words]
            for word in filtered_words:
                tokens.append(word)
        return tokens
    

                
class BigramAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer()
    def __call__(self, doc):   
        tokens = []     
        for sent in self.sentencer.tokenize(doc.decode('ascii','ignore')):
            tagged = self.tb(sent.lower()).tags    
            # Remove stops
            filtered_words = [w for w in tagged if not w[0] in stops]
                   
            # Remove punctuation
            filtered_words = [(re.findall('[a-z]+', w[0].lower())[0], w[1]) for w in filtered_words if len(re.findall('[a-z]+', w[0].lower())) > 0]             
                    
            # Lemmatize
            filtered_words = [self.lemmatizer.lemmatize(w[0], penn_to_wn(w[1])) for w in filtered_words]
            for bigram in ngrams(filtered_words,2):
                tokens.append('%s %s' %bigram)
        return tokens
    
class TrigramAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer()
    def __call__(self, doc):   
        tokens = []     
        for sent in self.sentencer.tokenize(doc.decode('ascii','ignore')):
            tagged = self.tb(sent.lower()).tags    
            # Remove stops
            filtered_words = [w for w in tagged if not w[0] in stops]
                   
            # Remove punctuation
            filtered_words = [(re.findall('[a-z]+', w[0].lower())[0], w[1]) for w in filtered_words if len(re.findall('[a-z]+', w[0].lower())) > 0]             
                    
            # Lemmatize
            filtered_words = [self.lemmatizer.lemmatize(w[0], penn_to_wn(w[1])) for w in filtered_words]
            for trigram in ngrams(filtered_words,3):
                tokens.append('%s %s %s' %trigram)
        return tokens
    
class QuadgramAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer()
    def __call__(self, doc):   
        tokens = []     
        for sent in self.sentencer.tokenize(doc.decode('ascii','ignore')):
            tagged = self.tb(sent.lower()).tags    
            # Remove stops
            filtered_words = [w for w in tagged if not w[0] in stops]
                   
            # Remove punctuation
            filtered_words = [(re.findall('[a-z]+', w[0].lower())[0], w[1]) for w in filtered_words if len(re.findall('[a-z]+', w[0].lower())) > 0]             
                    
            # Lemmatize
            filtered_words = [self.lemmatizer.lemmatize(w[0], penn_to_wn(w[1])) for w in filtered_words]
            for qgram in ngrams(filtered_words,4):
                tokens.append('%s %s %s %s' %qgram)
        return tokens
    
class UnigramBigramAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer()
    def __call__(self, doc):   
        tokens = []     
        for sent in self.sentencer.tokenize(doc.decode('ascii','ignore')):
            tagged = self.tb(sent.lower()).tags    
            # Remove stops
            filtered_words = [w for w in tagged if not w[0] in stops]
                   
            # Remove punctuation
            filtered_words = [(re.findall('[a-z]+', w[0].lower())[0], w[1]) for w in filtered_words if len(re.findall('[a-z]+', w[0].lower())) > 0]             
                    
            # Lemmatize
            filtered_words = [self.lemmatizer.lemmatize(w[0], penn_to_wn(w[1])) for w in filtered_words]
            
            for word in filtered_words:
                tokens.append(word)
            for bigram in ngrams(filtered_words,2):
                tokens.append('%s %s' %bigram)
        return tokens

class LexicalBigramAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer()
    def __call__(self, doc):   
        tokens = []     
        for sent in self.sentencer.tokenize(doc.decode('ascii','ignore')):
            tagged = self.tb(sent.lower()).tags    
            
            tagged = [(t[0], penn_to_wn(t[1])) for t in tagged]
            
            ng = zip(tagged, tagged[1:])
            rule1 = [(t[0],t[1]) for t in ng if t[0][1]== wn.ADJ and t[1][1]== wn.NOUN]
            rule2 = [(t[0],t[1]) for t in ng if (t[0][1]== wn.ADV and t[1][1]== wn.VERB) or (t[0][1]== wn.VERB and t[1][1]== wn.ADV)]
            
            filtered_list = rule1 + rule2
                             
                    
            # Lemmatize
            filtered_list = [self.lemmatizer.lemmatize(t[0][0], t[0][1]) + ' ' + self.lemmatizer.lemmatize(t[1][0], t[1][1]) for t in filtered_list]
            for bigram in filtered_list:
                tokens.append(bigram)
        return tokens
    
class UnigramBigramTrigramAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer()
    def __call__(self, doc):   
        tokens = []     
        for sent in self.sentencer.tokenize(doc.decode('ascii','ignore')):
            tagged = self.tb(sent.lower()).tags    
            # Remove stops
            filtered_words = [w for w in tagged if not w[0] in stops]
                   
            # Remove punctuation
            filtered_words = [(re.findall('[a-z]+', w[0].lower())[0], w[1]) for w in filtered_words if len(re.findall('[a-z]+', w[0].lower())) > 0]             
                    
            # Lemmatize
            filtered_words = [self.lemmatizer.lemmatize(w[0], penn_to_wn(w[1])) for w in filtered_words]
            
            for word in filtered_words:
                tokens.append(word)
            for bigram in ngrams(filtered_words,2):
                tokens.append('%s %s' %bigram)
            for trigram in ngrams(filtered_words,3):
                tokens.append('%s %s %s' %trigram)
        return tokens

    
class UnigramBigramTrigramQuadgramAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer()
    def __call__(self, doc):   
        tokens = []     
        for sent in self.sentencer.tokenize(doc.decode('ascii','ignore')):
            tagged = self.tb(sent.lower()).tags    
            # Remove stops
            filtered_words = [w for w in tagged if not w[0] in stops]
                   
            # Remove punctuation
            filtered_words = [(re.findall('[a-z]+', w[0].lower())[0], w[1]) for w in filtered_words if len(re.findall('[a-z]+', w[0].lower())) > 0]             
                    
            # Lemmatize
            filtered_words = [self.lemmatizer.lemmatize(w[0], penn_to_wn(w[1])) for w in filtered_words]
            
            for word in filtered_words:
                tokens.append(word)
            for bigram in ngrams(filtered_words,2):
                tokens.append('%s %s' %bigram)
            for trigram in ngrams(filtered_words,3):
                tokens.append('%s %s %s' %trigram)
            for qgram in ngrams(filtered_words,4):
                tokens.append('%s %s %s %s' %qgram)
        return tokens

def extract_global_bag_of_words(commentList):
    corpus = []   
    i = 0
    for art in commentList.items():        
        for comm in art[1]:           
            corpus.append(comm.body)
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
            mywords = onlyWords(comm.body)
            mywords = known(mywords)
            # Remove Stops
            filtered_words = [w for w in mywords if not w in stops]
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
            mywords = onlyWords(comm.body)
            mywords = known(mywords)
            # Remove Stops
            filtered_words = [w for w in mywords if not w in stops]
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



WORD_MIN = 25 # At least that many words per comment (TWEET)
ENGAGE_MIN = 20 # At least that many total votes
VOTES_MIN = 0 # At least that many individual votes
MIN_THREAD_LENGTH = 20 # Threads at least that long

def read_comments(filename):
    f = open(filename, 'r')        
        
    # To process all the comments
    articleList = dict()
    CommentsList = defaultdict(list)
    parentList = dict()
    
    unique_comments = set()
    
    commentCount = 0
    totalCount = 0
    for line in f:
        temp = line.split('&')
        if len(temp) < 16:
            continue
        
        CO_id = temp[0]
        C_id = temp[1]
        P_id = temp[2]
        U_id = temp[3]
        likes = temp[4]
        dislikes = temp[5]
        reported = temp[6]
        status = temp[7]
        rating = temp[8] 
        date = strptime(temp[9].split('.')[0], "%Y-%m-%d %H:%M:%S")
        author = temp[10]
        articleTitle = temp[11].replace('..', '.').replace('.', '. ')       
        articleSynopsis = temp[12].replace('..', '.').replace('.', '. ')    
        body = temp[13].replace('..', '.').replace('.', '. ')          
        lemma_body = temp[14].replace('..', '.').replace('.', '. ') 
        pos_body = temp[15].replace('..', '.').replace('.', '. ')             
    
        totalCount += 1
        '''
        if body in unique_comments:
            continue
        else:
            unique_comments.add(body)
        '''
        
        comm = CommentObject.CommentObject(C_id, CO_id, P_id, U_id, likes, dislikes, reported,status, date, author, body,lemma_body, pos_body)
        
        '''
        if likes + dislikes == 0:
            continue
        
        if likes + dislikes < ENGAGE_MIN:
            continue
        if likes < VOTES_MIN:
            continue
        
        if dislikes < VOTES_MIN:
            continue
        '''
        comm.setWords(words(comm.body))
        if len(comm.words) < WORD_MIN:
            continue
        
        
        if rating == 0 or rating == 4:
            continue
        
        article = ArticleObject.ArticleObject(CO_id, articleTitle, articleSynopsis, "")


        if P_id == 'null' and not parentList.has_key(C_id):
            parentList[C_id] = 0
        else:
            if parentList.has_key(P_id):           
                parentList[P_id] += 1
            else:
                parentList[P_id] = 1


        articleList[CO_id] = article
        CommentsList[CO_id].append(comm)

        commentCount += 1
        if commentCount % 10000 == 0:
            print "Read", commentCount, "comments"
    
    '''
    commentCount = 0
    # Too few comments
    for a in CommentsList.items():
        if len(a[1]) < MIN_THREAD_LENGTH:
            CommentsList.pop(a[0])
            articleList.pop(a[0])
        else:
            commentCount += len(a[1])    
    '''
    print "Saved",commentCount,"comments out of", totalCount
    
    return articleList, CommentsList, parentList, commentCount

def comments_stats(filename):
    f = open(filename, 'r')        
        
    # To process all the comments
    commentCount = 0
    sum_of_words = 0
    parent_comment_count = 0
    articleList = dict()
    parentList = dict()
    for line in f:
        temp = line.split('&')
        if len(temp) < 16:
            continue
        
        CO_id = temp[0]
        C_id = temp[1]
        P_id = temp[2]
        U_id = temp[3]
        likes = temp[4]
        dislikes = temp[5]
        reported = temp[6]
        status = temp[7]
        rating = temp[8] 
        date = strptime(temp[9].split('.')[0], "%Y-%m-%d %H:%M:%S")
        author = temp[10]
        articleTitle = temp[11].replace('..', '.').replace('.', '. ')       
        articleSynopsis = temp[12].replace('..', '.').replace('.', '. ')    
        body = temp[13].replace('..', '.').replace('.', '. ')          
        lemma_body = temp[14].replace('..', '.').replace('.', '. ') 
        pos_body = temp[15].replace('..', '.').replace('.', '. ')             
                     
          
        if rating == 0 or rating == 4:
            continue
        

        sum_of_words += len(words(body))

        if P_id == 'null'  and not parentList.has_key(C_id):
            parent_comment_count += 1
            parentList[C_id] = 0
        else:
            if parentList.has_key(P_id):           
                parentList[P_id] += 1
            else:
                parentList[P_id] = 1
            
            

        if articleList.has_key(CO_id):
            articleList[CO_id] += 1
        else:
            articleList[CO_id] = 1
            

        commentCount += 1
        if commentCount % 10000 == 0:
            print "Read", commentCount, "comments"
            
    print commentCount
    print parent_comment_count
    print commentCount - parent_comment_count
    print np.mean(articleList.values())
    print np.mean(parentList.values())
    print sum_of_words
    print sum_of_words / float(commentCount)


def read_user_comments(filename):
    f = open(filename, 'r')        
        
    # To process all the comments
    userCommList = defaultdict(list)
    commentCount = 0
    for line in f:        
        temp = line.split('&')
        if len(temp) < 16:
            continue
        
        C_id = temp[1]
        P_id = temp[2]
        U_id = temp[3]
        likes = temp[6]
        dislikes = temp[7]
        date = strptime(temp[9].split('.')[0], "%Y-%m-%d %H:%M:%S")
        author = temp[10] 
        body = temp[15].replace('..', '.').replace('.', '. ')      
        
        
        
        if likes == 'null':
            likes = 0
        else:
            likes = int(likes)
            
            
        if dislikes == 'null':
            dislikes = 0
        else:
            dislikes = int(dislikes)
        
        if likes + dislikes < 50:
            continue
        
        
        if len(words(body)) < 50:
            continue
        
        
        comm = UserCommentObject(U_id, C_id, P_id, author, likes, dislikes, body)
        
        userCommList[U_id].append(comm)
        
        commentCount += 1
        if commentCount % 10000 == 0:
            print "Read", commentCount, "user comments"
        

    print "done reading"
          
    userList = [] 
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    for uco in userCommList.values():    
        u = uco[0]
        user = UserObject(u.userid, u.author, uco)
        for comm in user.comments:
            mywords = onlyWords(comm.body)
            mywords = known(mywords)
            # Remove Stops
            filtered_words = [w for w in mywords if not w in stops]
            # Stemming
            stemmed_words = [stemmer.stem(w) for w in filtered_words]
            user.bagOfWords += stemmed_words
            
            
        userList.append(user)
            
        
    return userList, len(userList)


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



def read_article_comments(filename):
    f = open(filename, 'r')        
        
    # To process all the comments
    articleList = defaultdict(list)
    commentCount = 0
    for line in f:
        temp = line.split('&')
        if len(temp) < 16:
            continue
        
        CO_id = temp[0]
        author = temp[10]
        articleBody = temp[14].replace('..', '.').replace('.', '. ')       
        body = temp[15].replace('..', '.').replace('.', '. ')  
        
        
        
        # Clean HTML
        articleBody = nltk.clean_html(articleBody).replace('and#39;', '\'').replace('\x93', '').replace('\x94', '')
        
        comm = ArticleCommentObject(CO_id,author, body, articleBody)
                
        articleList[CO_id].append(comm)
        
        commentCount += 1
        if commentCount % 1000 == 0:
            print "Read", commentCount, "comments"
            break
    
    
    return articleList, commentCount

def read_articles(filename):
    f = open(filename, 'r')        
        
    # To process all the comments
    articleList = defaultdict(list)
    commentCount = 0
    for line in f:
        temp = line.split('&')
        if len(temp) < 5:
            continue
        
        art_id = temp[0];
        date = temp[1];
        title = temp[2];
        synopsis = temp[3];
        body = temp[4];
        body = nltk.clean_html(body)
        
        
        body = nltk.clean_html(body).replace('and#39;', '\'')
        articleList[art_id] = ArticleObject(id, date, title, synopsis, body)
        
        commentCount += 1
        if commentCount % 10000 == 0:
            print "Read", commentCount, "articles"
            break
            
        #print textwrap.fill(body, width=80),'\n'
        return articleList
        