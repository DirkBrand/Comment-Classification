import os

import scipy
from scipy.sparse.csr import csr_matrix
from sklearn import cross_validation, preprocessing
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.feature_selection.univariate_selection import SelectPercentile,\
    chi2, f_classif, SelectKBest
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics.metrics import classification_report
from sklearn.svm.classes import SVC, LinearSVC

from FeatureExtraction.main import load_numpy_matrix, load_sparse_csr
from RatingPrediction.main import valueV, runClassificationTest
from config import feature_set_path
import numpy as np


perc = 50
scale = 1

yn = load_numpy_matrix(feature_set_path+ 'valueVector_train.npy')
#yn = load_numpy_matrix(feature_set_path+ 'sentenceValueVector.npy')[:,valueV]

#filepath = 'MANUAL'
#print load_numpy_matrix(feature_set_path+ 'featureArray_train.npy').shape
#print load_numpy_matrix(feature_set_path+ 'socialVector_train.npy').shape
#Xn = np.hstack((load_numpy_matrix(feature_set_path+ 'featureArray_train.npy'),load_numpy_matrix(feature_set_path+ 'socialVector_train.npy') ))

#filepath = feature_set_path+ 'binaryWordData_train.npz'
#filepath = feature_set_path+ 'freqWordData_train.npz'
#filepath = feature_set_path+ 'tfidfWordData_train.npz'
#filepath = feature_set_path+ 'bigramBinaryWordData_train.npz'
#filepath = feature_set_path+ 'bigramTfidfWordData_train.npz'
#filepath = feature_set_path+ 'trigramBinaryWordData_train.npz'
#filepath = feature_set_path+ 'trigramTfidfWordData_train.npz'
#filepath = feature_set_path+ 'quadgramBinaryWordData_train.npz'
#filepath = feature_set_path+ 'quadgramTfidfWordData_train.npz'

#filepath = feature_set_path+ 'bigramOnlyBinaryWordData_train.npz'
#filepath = feature_set_path+ 'bigramOnlyTfidfWordData_train.npz'
#filepath = feature_set_path+ 'trigramOnlyBinaryWordData_train.npz'
#filepath = feature_set_path+ 'trigramOnlyTfidfWordData_train.npz'
#filepath = feature_set_path+ 'quadgramOnlyBinaryWordData_train.npz'
#filepath = feature_set_path+ 'quadgramOnlyTfidfWordData_train.npz'

filepath = feature_set_path+ 'binaryCharacterData_train.npz'
#filepath = feature_set_path+ 'quadgramOnlyTfidfWordData_train.npz'

Xn = load_sparse_csr(filepath)

#Xn = load_numpy_matrix(feature_set_path+ 'POS_model_MinMaxMeanFeatures.npy')
#Xn = load_numpy_matrix(feature_set_path+ 'POS_model_TfidfWeightedSumFeatures.npy')
#Xn = load_numpy_matrix(feature_set_path+ 'POS_model_BagOfCentroidsFeatures.npy')

#Xn = load_numpy_matrix(feature_set_path+ 'sentence_model_MinMaxMeanFeatures.npy')
#Xn = load_numpy_matrix(feature_set_path+ 'sentence_model_TfidfWeightedSumFeatures.npy')
#Xn = load_numpy_matrix(feature_set_path+ 'sentence_model_BagOfCentroidsFeatures.npy')

#Xn = load_numpy_matrix(feature_set_path+ 'bigram_model_MinMaxMeanFeatures.npy')
#Xn = load_numpy_matrix(feature_set_path+ 'bigram_model_TfidfWeightedSumFeatures.npy')
#Xn = load_numpy_matrix(feature_set_path+ 'bigram_model_BagOfCentroidsFeatures.npy')


#Xn = load_numpy_matrix(feature_set_path+ 'google_model_MinMaxMeanFeatures.npy')
#Xn = load_numpy_matrix(feature_set_path+ 'google_model_TfidfWeightedSumFeatures.npy')
#Xn = load_numpy_matrix(feature_set_path+ 'google_model_BagOfCentroidsFeatures.npy')


print filepath

sss = StratifiedShuffleSplit(yn, 1, test_size=0.75, random_state=0)
for train, test in sss:
    Xn , yn = Xn[train], yn[train]
    
print Xn.shape

# FEATURE SELECTION
#Xn = SelectPercentile(score_func=f_classif, percentile=perc).fit_transform(Xn,yn) 
#Xn = SelectPercentile(score_func=chi2, percentile=perc).fit_transform(Xn,yn) 
Xn = SelectKBest(score_func=chi2, k=min(100000, Xn.shape[1])).fit_transform(Xn,yn) 



print Xn.shape
print yn.shape


# FEATURE SCALING

if scale == 1:
    Xn = preprocessing.normalize(Xn, axis=0, copy=False)
elif scale == 2:
    Xn = preprocessing.normalize(Xn, copy=False)
   
   




tuned_parameters = [{'kernel': ['rbf'], 'C': np.logspace(-1, 7, 9),
                     'gamma': np.logspace(-4, 1, 8)}]

linear_parameters = [{'kernel': ['linear'], 'C': np.logspace(-3, 4, 8)}]

params = {'kernel': ['rbf'], 'C': scipy.stats.expon(scale=100000),
                     'gamma': scipy.stats.expon(scale=0.1)}


print("# Tuning hyper-parameters for %s" % 'F1-score')
print

cv = cross_validation.StratifiedKFold(yn,shuffle=True, n_folds=3, random_state=42)
clf = GridSearchCV(estimator=SVC(C=1, cache_size=1000, class_weight='auto',verbose=True),param_grid=tuned_parameters, cv=cv, scoring='f1').fit(Xn, yn)
#clf = RandomizedSearchCV(estimator=SVC(C=1, cache_size=1000, class_weight='auto',verbose=True), param_distributions=params, n_iter=50, cv=cv, scoring='f1').fit(Xn, yn)

print("Best parameters set found on development set:")
print
print(clf.best_estimator_)
print(clf.best_score_)
print()
print("Grid scores on development set:")
print
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() / 2, params))
print

clf = GridSearchCV(estimator=SVC(C=1, cache_size=1000, class_weight='auto'),param_grid=linear_parameters, cv=cv, scoring='f1').fit(Xn, yn)

print("Best parameters set found on development set:")
print
print(clf.best_estimator_)
print(clf.best_score_)
print()
print("Grid scores on development set:")
print
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() / 2, params))
print
