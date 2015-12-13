
import operator

from FeatureExtraction.main import load_sparse_csr, load_numpy_matrix
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.lda import LDA
from sklearn.linear_model import LinearRegression
from sklearn.svm.classes import SVR

from RatingPrediction.main import normalize_sets_sparse
from config import feature_set_path
import numpy as np

names=[
'timely', #0
'timePassing', #1
'commLengthiness', #2
'numberCharacters', #3
'vf', #4
'nf', #5
'pronouns', #6
'cf', #7
'qf', #8
'ef', #9
'scf', #10
'complexity', #11
'diversity', #12
'spelled', #13
'spelledPerc', #14
'badWords', #15
'badWordsPerc', #16
'meantermFreq', #17
'informativeness', #18
'readibility', #19
'threadRelevance', #20
'articleRelevance', #21
'sentiment', #22
'subj_obj', #23
'polarity_overlap', #24
'in degree', #25
'Out Degree', #26
'User Age', #27
'Nr of Posts', #28
'Post Rate', #29
'PageRank', #30
'Hub', #31
'Auth']






tag='_toy'


X_train = load_numpy_matrix(feature_set_path +  r'featureArray'+tag+'_train.npy')
sd = load_numpy_matrix(feature_set_path +  r'socialVector'+tag+'_train.npy')
print X_train.shape
print sd.shape

X_train =  np.hstack((X_train,sd))
X_test = load_numpy_matrix(feature_set_path +  r'featureArray'+tag+'_test.npy')
sd2 = load_numpy_matrix(feature_set_path +  r'socialVector'+tag+'_test.npy')
X_test =  np.hstack((X_test,sd2))
'''
X_train = load_sparse_csr(feature_set_path +  r'binaryWordData'+tag+'_train.npz')  
X_test = load_sparse_csr(feature_set_path +  r'binaryWordData'+tag+'_test.npz')  
'''    
y_train = load_numpy_matrix(feature_set_path +  r'valueVector'+tag+'_train.npy')
y_test = load_numpy_matrix(feature_set_path +  r'valueVector'+tag+'_test.npy')

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print np.min(y_train)
print np.max(y_train)
print np.mean(y_train)
print  'Loaded testing data\n'


#model = SVR(C=1.0, gamma=1.0)
model = LDA()
model.fit(X_train, y_train)
values = []
for i, v in enumerate(model.coef_[0]):
    values.append(tuple([i,abs(v)]))
    
values.sort(key=operator.itemgetter(1))
values = values[::-1]
values_chosen = values[:10]
print ["%3d : %0.5f" % (i[0],i[1]) for i in values_chosen]
print [names[i[0]] for i in values_chosen]

values_chosen = values[-10:]
print ["%3d : %0.5f" % (i[0],i[1]) for i in values_chosen]
print [names[i[0]] for i in values_chosen]

