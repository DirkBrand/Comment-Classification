import os

import scipy
from scipy.sparse.csr import csr_matrix
from sklearn import cross_validation, preprocessing
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.feature_selection.univariate_selection import SelectPercentile,\
    chi2, f_classif
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics.metrics import classification_report
from sklearn.svm.classes import SVC

from FeatureExtraction.main import load_numpy_matrix, load_sparse_csr
from RatingPrediction.main import valueV, runClassificationTest
from config import feature_set_path
import numpy as np


perc = 20
scale = 1

yn = load_numpy_matrix(feature_set_path+ 'valueVector.npy')[:,valueV]
#yn = load_numpy_matrix(feature_set_path+ 'sentenceValueVector.npy')[:,valueV]

#Xn = np.hstack((load_numpy_matrix(feature_set_path+ 'featureArray.npy'),load_numpy_matrix(feature_set_path+ 'socialVector.npy') ))
#Xn = load_sparse_csr(feature_set_path+ 'binaryWordData.npz') 
#Xn = load_sparse_csr(feature_set_path+ 'freqWordData.npz')  
#Xn = load_sparse_csr(feature_set_path+ 'tfidfWordData.npz') 
#Xn = load_sparse_csr(feature_set_path+ 'bigramBinaryWordData.npz')
#Xn = load_sparse_csr(feature_set_path+ 'bigramTfidfWordData.npz')
#Xn = load_sparse_csr(feature_set_path+ 'trigramBinaryWordData.npz')
#Xn = load_sparse_csr(feature_set_path+ 'trigramTfidfWordData.npz')
#Xn = load_sparse_csr(feature_set_path+ 'quadgramBinaryWordData.npz')
#Xn = load_sparse_csr(feature_set_path+ 'quadgramTfidfWordData.npz')
Xn = load_sparse_csr(feature_set_path+ 'quadgramOnlyTfidfWordData.npz')

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



print Xn.shape

# FEATURE SELECTION
Xn = SelectPercentile(score_func=chi2, percentile=perc).fit_transform(Xn,yn) 




# SUBSET SELECTION
sss = StratifiedShuffleSplit(yn, 1, test_size=0.90, random_state=0)
for train, test in sss:
    Xn , yn = Xn[train], yn[train]

print Xn.shape
print yn.shape


# FEATURE SCALING
if scale == 1:
    Xn = preprocessing.normalize(Xn, axis=0, copy=False)
elif scale == 2:
    Xn = preprocessing.normalize(Xn, copy=False)
    




tuned_parameters = [{'kernel': ['rbf'], 'C': [0.1, 10, 100, 1000, 10000, 100000, 1000000],
                     'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]},
                    {'kernel': ['linear'], 'C': [0.001, 0.1, 10, 1000, 10000]}]


params = {'kernel': ['rbf'], 'C': scipy.stats.expon(scale=100000),
                     'gamma': scipy.stats.expon(scale=0.1)}

linear_params = {'kernel': ['linear'], 'C': [0.1, 0.5, 1, 5, 10, 100, 1000, 10000, 1000000]}

print("# Tuning hyper-parameters for %s" % 'F1-score')
print

cv = cross_validation.StratifiedKFold(yn,shuffle=True, n_folds=5, random_state=42)
#clf = GridSearchCV(estimator=SVC(C=1, cache_size=1000, class_weight='auto'),param_grid=tuned_parameters, cv=cv, scoring='f1').fit(Xn, yn)
clf = RandomizedSearchCV(estimator=SVC(C=1, cache_size=1000, class_weight='auto'), param_distributions=params, n_iter=50, cv=cv, scoring='f1').fit(Xn, yn)

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

clf = GridSearchCV(estimator=SVC(C=1, cache_size=1000, class_weight='auto'),param_grid=linear_params, cv=cv, scoring='f1').fit(Xn, yn)

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

