import re
import string

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from textblob.blob import Blobber
from textblob.tokenizers import SentenceTokenizer
from textblob_aptagger.taggers import PerceptronTagger

from FeatureExtraction.LexicalFeatures import penn_to_wn
from FeatureExtraction.mainExtractor import stops

from nltk.corpus import wordnet as wn

class BigramAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer()
    def __call__(self, doc):   
        tokens = []     
        for sent in self.sentencer.tokenize(doc):
            sent
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
    
class CharacterAnalyzer(object):   
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()    
        self.tb = Blobber(pos_tagger=PerceptronTagger())
        self.sentencer = SentenceTokenizer();
        self.num = 8;
    def __call__(self, doc):  
        tokens = []      
        for sent in self.sentencer.tokenize(doc.lower()):
            for word in word_tokenize(sent):
                word= ''.join([ch for ch in word if ch not in string.punctuation])
                for n in range(2,self.num+1):
                    ngr = [word[i:i+n] for i in range(len(word)-n+1)]
                    if len(ngr) > 0:
                        tokens += ngr
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
    
text = [r'He is my best friend.  I like him as a brother', r'The black ball quickly rolls extremely fast']
vect = CountVectorizer(analyzer=LexicalBigramAnalyzer()).fit(text)
print vect
print('Vocabulary: %s' %vect.get_feature_names())
