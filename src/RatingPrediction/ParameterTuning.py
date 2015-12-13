from collections import Counter

from FeatureExtraction.main import load_numpy_matrix, load_sparse_csr
from scipy.sparse.csr import csr_matrix
from sklearn import cross_validation, preprocessing
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection.univariate_selection import SelectPercentile,\
    chi2, f_classif, SelectKBest
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.metrics.scorer import make_scorer
from sklearn.svm.classes import SVC

import numpy as np
from config import feature_set_path

perc = 50

split = True

manual_tests = False
sparse_tests = True
sparse_2_tests = False
dense_tests = False

RBF = True
LINEAR = True

#tag = '_main'
#tag = '_toy'
tag = '_slashdot'


def parameter_tuning(Xn, yn, scale=1):
    
    # FEATURE SELECTION  
    print Xn.shape
    print yn.shape    
    
    # FEATURE SCALING    
    if scale == 1:
        Xn = preprocessing.scale(Xn, with_mean=True)
        print 'NORMALIZING'
    elif scale == 2:
        Xn = preprocessing.scale(Xn, with_mean=False)
        print 'NORMALIZING'
        
    tuned_parameters = [{'kernel': ['rbf'], 'C': np.logspace(-2, 7, 10),
                         'gamma': np.logspace(-4, 2, 7)}]
    
    tuned_parameters2 = {'kernel': ['rbf'], 'C': np.logspace(-2, 7, 10),
                         'gamma': np.logspace(-4, 2, 7)}
    
    linear_parameters = [{'kernel': ['linear'], 'C': np.logspace(-3, 4, 8)}]
    
    linear_parameters2 = {'kernel': ['linear'], 'C': np.logspace(-3, 4, 8)}
    

    cv = cross_validation.StratifiedKFold(yn,shuffle=True, n_folds=3, random_state=42)

    if RBF:
        clf = RandomizedSearchCV(estimator=SVC(C=1, cache_size=1000), param_distributions=tuned_parameters2, cv=cv, scoring='accuracy', n_iter=30, verbose=1, n_jobs=2).fit(Xn, yn)

        print("Best parameters set found on development set:")
        print
        print(clf.best_estimator_)
        print(clf.best_score_)
        print()
        print confusion_matrix(yn, clf.predict(Xn))

    if LINEAR:
        clf = GridSearchCV(estimator=SVC(C=1, cache_size=1000), param_grid=linear_parameters, cv=cv, scoring='accuracy', verbose=1, n_jobs=2).fit(Xn, yn)


        print("Best parameters set found on development set:")
        print
        print(clf.best_estimator_)
        print(clf.best_score_)
        print()
        print confusion_matrix(yn, clf.predict(Xn))

if __name__ == '__main__':

    if manual_tests:
        Xn = csr_matrix(np.array((0,0)))
        yn = load_numpy_matrix(feature_set_path+ 'valueVector' + tag + '_train.npy')
        print Counter(yn)

        filepath = 'MANUAL'
        print load_numpy_matrix(feature_set_path+ 'featureArray' + tag + '_train.npy').shape
        print load_numpy_matrix(feature_set_path+ 'socialVector' + tag + '_train.npy').shape
        Xn = np.hstack((load_numpy_matrix(feature_set_path+ 'featureArray' + tag + '_train.npy'),load_numpy_matrix(feature_set_path+ 'socialVector' + tag + '_train.npy') ))

        Xn = SelectPercentile(score_func=f_classif, percentile=perc).fit_transform(Xn,yn)


        if split:
            sss = StratifiedShuffleSplit(yn, 1, test_size=0.85, random_state=42)
            for train, test in sss:
                Xn , yn = Xn[train], yn[train]


        parameter_tuning(Xn, yn, scale=1)
        print "DONE WITH MANUAL"


    if sparse_tests:
        filepaths = list()
#         filepaths.append(feature_set_path + 'binaryWordData' + tag + '_train.npz')
#         filepaths.append(feature_set_path + 'freqWordData' + tag + '_train.npz')
#         filepaths.append(feature_set_path + 'tfidfWordData' + tag + '_train.npz')
#         filepaths.append(feature_set_path + 'bigramBinaryWordData' + tag + '_train.npz')
#         filepaths.append(feature_set_path + 'bigramTfidfWordData' + tag + '_train.npz')
#         filepaths.append(feature_set_path + 'trigramBinaryWordData' + tag + '_train.npz')
#         filepaths.append(feature_set_path + 'trigramTfidfWordData' + tag + '_train.npz')

        filepaths.append(feature_set_path + 'bigramOnlyBinaryWordData' + tag + '_train.npz')
#         filepaths.append(feature_set_path + 'bigramOnlyTfidfWordData' + tag + '_train.npz')
#         filepaths.append(feature_set_path + 'trigramOnlyBinaryWordData' + tag + '_train.npz')
#         filepaths.append(feature_set_path + 'trigramOnlyTfidfWordData' + tag + '_train.npz')

        for file in filepaths:
            print file
            print tag

            Xn = csr_matrix(np.array((0,0)))
            yn = load_numpy_matrix(feature_set_path + r'valueVector' + tag + '_train.npy')
            print Counter(yn)
            Xn = load_sparse_csr(file)
            Xn = SelectKBest(score_func=chi2, k=min(200000, int(Xn.shape[1]*(perc/100.0)))).fit_transform(Xn,yn)

            if split:
                sss = StratifiedShuffleSplit(yn, 1, test_size=0.75)
                for train, test in sss:
                    Xn , yn = Xn[train], yn[train]

            parameter_tuning(Xn, yn, scale=-1)

    if sparse_2_tests:
        filepaths = list()
#         filepaths.append(feature_set_path+ 'binaryCharacterData' + tag + '_train.npz')
#         filepaths.append(feature_set_path+ 'tfidfCharacterData' + tag + '_train.npz')
# 
#         filepaths.append(feature_set_path+ 'binaryCharacterSkipgramData' + tag + '_train.npz')
#         filepaths.append(feature_set_path+ 'tfidfCharacterSkipgramData' + tag + '_train.npz')

        if tag == '_slashdot':
            filepaths.append(feature_set_path + "_slashdotlda" + tag + "_train.npz")
        else:
            filepaths.append(feature_set_path + "_news24lda" + tag + "_train.npz")

        for file in filepaths:
            print file
            print tag

            Xn = csr_matrix(np.array((0,0)))
            yn = load_numpy_matrix(feature_set_path+ 'valueVector' + tag + '_train.npy')
            print Counter(yn)
            Xn = load_sparse_csr(file)
            Xn = SelectKBest(score_func=chi2, k=min(200000, int(Xn.shape[1]*(perc/100.0)))).fit_transform(Xn,yn)

            if split:
                sss = StratifiedShuffleSplit(yn, 1, test_size=0.85, random_state=42)
                for train, test in sss:
                    Xn , yn = Xn[train], yn[train]

            parameter_tuning(Xn, yn, scale=-1)

        print "DONE WITH SPARSE"

    if dense_tests:
        filepaths = []
        if tag == '_slashdot':
            filepaths.append(feature_set_path + "Basic300_slashdot_TfidfFeatures" + tag + "_train.npy")
            filepaths.append(feature_set_path + "Basic200_slashdot_BOCFeatures" + tag + "_train.npy")
            filepaths.append(feature_set_path + "DocBasic300_slashdot_ParagraphFeatures" + tag + "_train.npy")
        else:
            filepaths.append(feature_set_path + "Basic300_news24_TfidfFeatures" + tag + "_train.npy")
            filepaths.append(feature_set_path + "Basic200_news24_BOCFeatures" + tag + "_train.npy")
            filepaths.append(feature_set_path + "DocBasic300_news24_ParagraphFeatures" + tag + "_train.npy")

        filepaths.append(feature_set_path + "Google_TfidfFeatures" + tag + "_train.npy")

        for file in filepaths:
            print file
            print tag

            yn = load_numpy_matrix(feature_set_path+ 'valueVector' + tag + '_train.npy')
            print Counter(yn)
            Xn = load_numpy_matrix(file)

            if split:
                sss = StratifiedShuffleSplit(yn, 1, test_size=0.85, random_state=42)
                for train, test in sss:
                    Xn , yn = Xn[train], yn[train]

            parameter_tuning(Xn, yn, scale=1)

        print "DONE WITH DENSE"