from collections import Counter
import os

from FeatureExtraction.main import load_numpy_matrix, load_sparse_csr
import scipy
from scipy.sparse.csr import csr_matrix
from sklearn import cross_validation, preprocessing
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.feature_selection.univariate_selection import SelectPercentile,\
    chi2, f_classif, SelectKBest
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics.metrics import classification_report, f1_score
from sklearn.svm.classes import SVC, LinearSVC

from RatingPrediction.main import  runClassificationTest
from config import feature_set_path
import numpy as np
from sklearn.multiclass import OneVsRestClassifier


perc = 50
scale = 1
#tag = "_toy"
#tag = '_main'
tag = '_slashdot'

Xn = csr_matrix(np.array((0,0)))
yn = load_numpy_matrix(feature_set_path+ 'valueVector' + tag + '_train.npy')
#yn = load_numpy_matrix(feature_set_path+ 'sentenceValueVector.npy')[:,valueV]

'''
filepath = 'MANUAL'
print load_numpy_matrix(feature_set_path+ 'featureArray' + tag + '_train.npy').shape
print load_numpy_matrix(feature_set_path+ 'socialVector' + tag + '_train.npy').shape
Xn = np.hstack((load_numpy_matrix(feature_set_path+ 'featureArray' + tag + '_train.npy'),load_numpy_matrix(feature_set_path+ 'socialVector' + tag + '_train.npy') ))
'''
#filepath = feature_set_path+ 'binaryWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'freqWordData' + tag + '_train.npz'
filepath = feature_set_path+ 'tfidfWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'bigramBinaryWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'bigramTfidfWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'trigramBinaryWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'trigramTfidfWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'quadgramBinaryWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'quadgramTfidfWordData' + tag + '_train.npz'

#filepath = feature_set_path+ 'bigramOnlyBinaryWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'bigramOnlyTfidfWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'trigramOnlyBinaryWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'trigramOnlyTfidfWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'quadgramOnlyBinaryWordData' + tag + '_train.npz'
#filepath = feature_set_path+ 'quadgramOnlyTfidfWordData' + tag + '_train.npz'

#filepath = feature_set_path+ 'binaryCharacterData_train.npz'
#filepath = feature_set_path+ 'tfidfCharacterData_train.npz'

#filepath = feature_set_path+ 'binaryCharacterSkipgramData_train.npz'
#filepath = feature_set_path+ 'tfidfCharacterSkipgramData_train.npz'

#filepath = feature_set_path+ 'binaryLexicalBigramsData_train.npz'
#filepath = feature_set_path+ 'tfidfLexicalBigramsData_train.npz'




if filepath != 'MANUAL':
    Xn = load_sparse_csr(filepath)


#filepath = feature_set_path+ 'Basic300_TfidfFeatures_train.npy'
#filepath = feature_set_path+ 'Basic300_BOCFeatures_train.npy'

#filepath = feature_set_path+ 'google_model_TfidfFeatures_train.npy'
#filepath = feature_set_path+ 'google_model_BOCFeatures_train.npy'

'''
if filepath != 'MANUAL':
    Xn = load_numpy_matrix(filepath)
'''
print filepath

sss = StratifiedShuffleSplit(yn, 1, test_size=0.90, random_state=0)
for train, test in sss:
    Xn , yn = Xn[train], yn[train]
 
print Xn.shape

# FEATURE SELECTION
#Xn = SelectPercentile(score_func=f_classif, percentile=perc).fit_transform(Xn,yn) 
#Xn = SelectPercentile(score_func=chi2, percentile=perc).fit_transform(Xn,yn) 
#Xn = SelectKBest(score_func=chi2, k=min(200000, Xn.shape[1])).fit_transform(Xn,yn) 


print "Class distribution:", Counter(yn)

print Xn.shape
print yn.shape


# FEATURE SCALING

if scale == 1:
    Xn = preprocessing.normalize(Xn, axis=0, copy=False)
elif scale == 2:
    Xn = preprocessing.normalize(Xn, copy=False)
   
   




tuned_parameters = [{'kernel': ['rbf'], 'C': np.logspace(-3, 9, 13),
                     'gamma': np.logspace(-6, 1, 8)}]

tuned_parameters2 = [{'estimator__kernel': ['rbf'], 'estimator__C': np.logspace(-3, 9, 13),
                     'estimator__gamma': np.logspace(-6, 1, 8)}]

linear_parameters = [{'kernel': ['linear'], 'C': np.logspace(-3, 4, 8)}]

params = {'kernel': ['rbf'], 'C': scipy.stats.expon(scale=100000),
                     'gamma': scipy.stats.expon(scale=0.1)}


print("# Tuning hyper-parameters for %s" % 'F1-score')
print

cv = cross_validation.StratifiedKFold(yn,shuffle=True, n_folds=3, random_state=42)

                                   
clf = GridSearchCV(estimator=SVC(C=1, cache_size=1000, class_weight='auto'),param_grid=tuned_parameters, cv=cv, score_func=f1_score).fit(Xn, yn)
#clf = GridSearchCV(estimator=OneVsRestClassifier(SVC(C=1, cache_size=1000, class_weight='auto', probability=True)),param_grid=tuned_parameters2, cv=cv, scoring='f1').fit(Xn, yn)
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

clf = GridSearchCV(estimator=SVC(C=1, cache_size=1000, class_weight='auto'),param_grid=linear_parameters, cv=cv, score_func=f1_score).fit(Xn, yn)

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
