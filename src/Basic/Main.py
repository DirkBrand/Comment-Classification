import codecs
import io
import os
import pickle

from FeatureExtraction.mainExtractor import UnigramBigramAnalyzer
import numpy
from pandas import DataFrame
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,\
    TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import spam_set_path




NEWLINE = '\n'

HAM = 0.0
SPAM = 1.0

SOURCES = [
    (spam_set_path + 'spam',        SPAM),
    (spam_set_path + 'easy_ham',    HAM),
    (spam_set_path + 'hard_ham',    HAM),
    (spam_set_path + 'beck-s',      HAM),
    (spam_set_path + 'farmer-d',    HAM),
    (spam_set_path + 'kaminski-v',  HAM),
    (spam_set_path + 'kitchen-l',   HAM),
    (spam_set_path + 'lokay-m',     HAM),
    (spam_set_path + 'williams-w3', HAM),
    (spam_set_path + 'BG',          SPAM),
    (spam_set_path + 'GP',          SPAM),
    (spam_set_path + 'SH',          SPAM)
]

SKIP_FILES = {'cmds'}


def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = io.open(file_path, encoding="latin-1", errors='ignore')
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


def build_data_frame(path, classification):
    rows = []
    index = []
    for file_name, text in read_files(path):
        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame

data = DataFrame({'text': [], 'class': []})
for path, classification in SOURCES:
    data = data.append(build_data_frame(path, classification))

data = data.reindex(numpy.random.permutation(data.index))

vectorizer = TfidfVectorizer(ngram_range =(1,2)) 
classifier = MultinomialNB()

k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)
    
    vectorizer = vectorizer.fit(train_text)
    train_vec = vectorizer.transform(train_text)
    test_vec = vectorizer.transform(test_text)
    
    classifier.fit(train_vec, train_y)
    
    predictions = classifier.predict(test_vec)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=str(SPAM))
    scores.append(score)

f = open(spam_set_path + 'spam_vectorizer.pickle', 'wb')
pickle.dump(vectorizer, f)
f.close()


f = open(spam_set_path + 'spam_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

print('Total emails classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)