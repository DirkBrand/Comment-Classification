'''
Created on 13 May 2014

@author: Dirk
'''

from __builtin__ import dict
from _collections import defaultdict
from collections import Counter
from datetime import datetime
import pickle
import re
import string
from time import  mktime,strptime

import nltk
from nltk.chunk import ne_chunk
from nltk.stem.snowball import SnowballStemmer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize
from nltk.tokenize.regexp import RegexpTokenizer, wordpunct_tokenize
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from FeatureExtraction.LexicalFeatures import words, pos_freq, getVF, getNF,\
    getPN, known, swears, entropy
from FeatureExtraction.SentimentUtil import load_classifier, make_full_dict,\
    get_subjectivity, get_polarity_overlap
from FeatureExtraction.SurfaceFeatures import lengthiness, question_frequency,\
    exclamation_frequency, capital_frequency, sentence_capital_frequency,\
    onSubForumTopic, F_K_score, tf_idf, avgTermFreq, timeliness
from Objects import CommentObject, ArticleObject, UserCommentObject, UserObject,\
    ArticleCommentObject
from Profiling.timer import timefunc
from TopicDiscovery import vocabulary
from TopicDiscovery.lda import LDA, lda_learning
from config import sentiment_path
import numpy as np
from DeepLearnings.FeatureExtraction import ENGAGE_MIN
from nltk.stem.wordnet import WordNetLemmatizer


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
    
    featureMatrix = np.empty([commentCount,30])
    
    
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
                       
                
            threadRelevance = onSubForumTopic(tokens, cnt.keys())
            articleRelevance = onSubForumTopic(tokens, articleCnt.keys())
            
            
            spelled = len(known(uniqueWords))
            spelledPerc = float(spelled) / len(uniqueWords)
            badWords = len(swears(uniqueWords))
            badWordsPerc = float(badWords) / len(uniqueWords)
            complexity = entropy(comm.body, theWords)
            readibility = F_K_score(comm.body, sentences)
            informativeness = tf_idf(text, theWords, commList)
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
                                             informativeness, #23
                                             readibility, #11
                                             threadRelevance, #17
                                             articleRelevance, #18
                                             sentiment, #13
                                             subj_obj, #14
                                             polarity_overlap, #22
                                             likes,
                                             dislikes,
                                             ratio,
                                             reports,
                                             engagement
                                             ], float) #25
            index += 1
            if index % 100 == 0:
                print "extracted", index, "features"

            if index >= commentCount:
                break
            
        if index >= commentCount:
            break
        
                
    return featureMatrix


def extract_user_topics(userList, userCount, numTopics):
    featureMatrix = np.zeros([userCount,numTopics])

    
    tokenizer = RegexpTokenizer(r'[\w\']+')
    
    clf = load_classifier('sentiment_classifier.pickle')
    
    #Create global topicList
    #create_global_topic_list(articleList)
    tempVector = dict()
    f = open('globalTopics' + '.pkl', 'rb')
    tempVector =  pickle.load(f)
    f.close()
    theKeys = tempVector.keys()
    
    
    
    #Get topics of each article    
    K = 1
    c = 20
    i = 10
    df = 1
    
    index = 0
    for user in userList: 
        # Article body + all comments 
        art = ''        
        for comm in user.comments:
            art += comm.body
            
        texts = [] 
        
        # Build document for LDA
        for line in sent_tokenize(art):
            doc = []       
            tokens = wordpunct_tokenize(line)
            words =  [word for word in tokens if word not in vocabulary.stopwords_list]              
            
            chunks = ne_chunk(pos_tag(words))
            for chunk in chunks:
                if hasattr(chunk, 'node'):            
                    doc.append(' '.join(c[0] for c in chunk.leaves()))
                else:
                    doc.append(chunk[0])
            
            doc = [word.strip(string.punctuation) for word in doc if len(word.strip(string.punctuation)) > 1]
            if len(doc) > 0:
                texts.append(doc)
                
        voca = vocabulary.Vocabulary(True)
        docs = [voca.doc_to_ids(doc) for doc in texts]
        if df > 0: docs = voca.cut_low_freq(docs, df)        
        lda = LDA(K, c,  0.5, 0.5, docs, voca.size(), False)  
        if np.sum([w for w in docs]) == 0:
            continue
        topics = lda_learning(lda, 10 , voca)
       
        
        # Sentiment
        testSet = dict()
        tokens = tokenizer.tokenize(art)
        refWords = make_full_dict(tokens)
        testSet.update(refWords)
        probDist = clf.prob_classify(testSet)         
        subj_obj = get_subjectivity(probDist)
        
        for top in (t for t in topics if tempVector.has_key(t)):
            keyInd = theKeys.index(top)      
            featureMatrix[index][keyInd] += subj_obj
        index += 1
        if index % 1000 == 0:
            print "extracted", index, "features"
                 
    
    
    print "non-zero",np.count_nonzero(featureMatrix)
    print "Percentage filled:%.2f" %(float(np.count_nonzero(featureMatrix))/(featureMatrix.shape[0]*featureMatrix.shape[1]))
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

def extract_global_bag_of_words(commentList):
    corpus = []
    
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    lemmatizer = WordNetLemmatizer()
    
    i = 0
    for art in commentList.items():        
        for comm in art[1]:
            only = onlyWords(comm.lemma_body)
            knownWords = known(only)
            
            # Remove Stops
            filtered_words = [w for w in knownWords if not w in stops]
            # Stemming
            #filtered_words = [lemmatizer.lemmatize(w) for w in filtered_words]
            
            corpus.append(' '.join(filtered_words))
            if i % 1000 == 0:
                print i, "lemmatized"
            i += 1
            
            
    return corpus
    
    
    
def extract_words(commentList, commentCount):    
    processed_comment_list = extract_global_bag_of_words(commentList)
    
    binary_count_vect = CountVectorizer(analyzer='word', max_df=0.5, binary=True, dtype=float)
    freq_count_vect = CountVectorizer(analyzer='word', max_df=0.5, dtype=float)
    tfidf_count_vect = TfidfVectorizer(analyzer='word', max_df=0.5, use_idf=True, smooth_idf=True, dtype=float)
    bigram_binary_count_vect = CountVectorizer(analyzer='word', ngram_range=(1,2), max_df=0.5, binary=True, dtype=float)
    bigram_tfidf_count_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_df=0.5, use_idf=True, smooth_idf=True, dtype=float)
    trigram_binary_count_vect = CountVectorizer(analyzer='word', ngram_range=(1,3), max_df=0.5, binary=True, dtype=float)
    trigram_tfidf_count_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_df=0.5, use_idf=True, smooth_idf=True, dtype=float)
    quadgram_binary_count_vect = CountVectorizer(analyzer='word', ngram_range=(1,4), max_df=0.5, binary=True, dtype=float)
    quadgram_tfidf_count_vect = TfidfVectorizer(analyzer='word', ngram_range=(1,4), max_df=0.5, use_idf=True, smooth_idf=True, dtype=float)
    
    
    
    
    binary_word_features = binary_count_vect.fit_transform(processed_comment_list)
    frequency_word_features = freq_count_vect.fit_transform(processed_comment_list)
    tfidf_word_features = tfidf_count_vect.fit_transform(processed_comment_list)
    
    bigram_binary_count_vect = bigram_binary_count_vect.fit_transform(processed_comment_list)
    bigram_tfidf_word_features = bigram_tfidf_count_vect.fit_transform(processed_comment_list)
    
    trigram_binary_count_vect = trigram_binary_count_vect.fit_transform(processed_comment_list)
    trigram_tfidf_word_features = trigram_tfidf_count_vect.fit_transform(processed_comment_list)
    
    quadgram_binary_count_vect = quadgram_binary_count_vect.fit_transform(processed_comment_list)
    quadgram_tfidf_count_vect = quadgram_tfidf_count_vect.fit_transform(processed_comment_list)
                
    
    print binary_count_vect.vocabulary_
    print binary_word_features.shape
    print frequency_word_features.shape
    print tfidf_word_features.shape
    print bigram_binary_count_vect.shape
    print bigram_tfidf_word_features.shape
    print trigram_binary_count_vect.shape
    print trigram_tfidf_word_features.shape
    print quadgram_binary_count_vect.shape
    print quadgram_tfidf_count_vect.shape
    
    return binary_word_features, frequency_word_features, tfidf_word_features, bigram_binary_count_vect,  bigram_tfidf_word_features, trigram_binary_count_vect, trigram_tfidf_word_features, quadgram_binary_count_vect,quadgram_tfidf_count_vect 


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


def extract_topics(articleList, commentCount, numWords):
    featureMatrix = np.zeros([commentCount,numWords])

    
    tokenizer = RegexpTokenizer(r'[\w\']+')
    
    clf = load_classifier('sentiment_classifier.pickle')
    
    #Create global topicList
    #create_global_topic_list(articleList)
    tempVector = dict()
    f = open('globalTopics' + '.pkl', 'rb')
    tempVector =  pickle.load(f)
    f.close()
    theKeys = tempVector.keys()
    
    
    
    #Get topics of each article    
    K = 1
    c = 20
    i = 10
    df = 1
    
    index = 0
    for commList in articleList.values(): 
        # Article body + all comments 
        art = commList[0].artBody        
        for comm in commList:
            art += comm.body
            
        texts = [] 
        
        # Build document for LDA
        for line in sent_tokenize(art):
            doc = []       
            tokens = wordpunct_tokenize(line)
            words =  [word for word in tokens if word not in vocabulary.stopwords_list]              
            
            chunks = ne_chunk(pos_tag(words))
            for chunk in chunks:
                if hasattr(chunk, 'node'):            
                    doc.append(' '.join(c[0] for c in chunk.leaves()))
                else:
                    doc.append(chunk[0])
            
            doc = [word.strip(string.punctuation) for word in doc if len(word.strip(string.punctuation)) > 1]
            if len(doc) > 0:
                texts.append(doc)
                
        voca = vocabulary.Vocabulary(True)
        docs = [voca.doc_to_ids(doc) for doc in texts]
        if df > 0: docs = voca.cut_low_freq(docs, df)        
        lda = LDA(K, c,  0.5, 0.5, docs, voca.size(), False)        
        topics = lda_learning(lda, 10 , voca)
        
        # Sentiment
        for comm in commList:
            testSet = dict()
            tokens = tokenizer.tokenize(comm.body)
            refWords = make_full_dict(tokens)
            testSet.update(refWords)
            probDist = clf.prob_classify(testSet)         
            subj_obj = get_subjectivity(probDist)
            
            for top in (t for t in topics if tempVector.has_key(t)):
                keyInd = theKeys.index(top)      
                featureMatrix[index][keyInd] += probDist.prob('pos')
            index += 1
            if index % 1000 == 0:
                print "extracted", index, "features"
                 
    
    
    print "non-zero",np.count_nonzero(featureMatrix)
    print "Percentage filled:%.2f" %(float(np.count_nonzero(featureMatrix))/(featureMatrix.shape[0]*featureMatrix.shape[1]))
    return featureMatrix
            

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
        if likes + dislikes == 0:
            continue
        
        if likes + dislikes < ENGAGE_MIN:
            continue
        if likes < VOTES_MIN:
            continue
        
        if dislikes < VOTES_MIN:
            continue
        
        comm.setWords(words(comm.body))
        if len(comm.words) < WORD_MIN:
            continue
        '''
        
        if rating == 0 or rating == 4:
            continue
        
        comm = CommentObject.CommentObject(C_id, CO_id, P_id, U_id, likes, dislikes, reported,status, date, author, body,lemma_body, pos_body)
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
        