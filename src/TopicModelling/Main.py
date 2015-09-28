'''
Created on 06 Aug 2015

@author: Dirk
'''
from gensim import  models, corpora

from DeepLearnings.ModelTraining import comment_to_sentences,\
    comment_to_wordlist, comment_to_words_for_topics
from config import comment_data_path, model_path


class MyCorpus(object):
    def __init__(self, filename):
        self.file = filename
        
    def __iter__(self):
        for line in open(self.file):
            body = line.split("|")[1]
            yield comment_to_words_for_topics(body)


'''
filename = comment_data_path + "comms_data.txt"
tag = 'news24'

'''
filename = comment_data_path + "slashdotCommData.txt"
tag = 'slashdot'


load_dictionary = False
if __name__ == '__main__':
    texts = MyCorpus(filename)
      
      
    if load_dictionary:
        dictionary = corpora.Dictionary.load(model_path + tag+ "_dictionary")
    else:          
        dictionary = corpora.Dictionary(texts)
        dictionary.filter_extremes(no_below=10)
        dictionary.save(model_path + tag+ "_dictionary")
        
    
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    
    lda = models.LdaModel(corpus=corpus, 
                  id2word = dictionary, 
                  num_topics = 100, chunksize=10000, alpha='auto')
    print "DONE"
    
    lda.save(model_path + tag+ "_lda_model")
    print "DONE"
    lda.print_topics(10, 20)
    
    
    

    