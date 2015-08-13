import string

from FeatureExtraction.mainExtractor import CharacterAnalyzer
from textblob.tokenizers import SentenceTokenizer, WordTokenizer


sentencer = SentenceTokenizer()
worder = WordTokenizer();

sentences = ['How are you? I am fine!']

tokens = []      
for sent in sentencer.tokenize(sentences[0].lower()):
    words = ''.join([ch for ch in sent if ch not in string.punctuation])
    words = worder.tokenize(words)
    
    for word in words:
        tokens.append(word.strip())
        if len(word) > 2:
            for j in range(0,len(word)):    
                term = word[:j] + word[j+1:] 
                tokens.append(term.strip())

print tokens