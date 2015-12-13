'''
Created on 24 Mar 2014

@author: Dirk
'''

from collections import Counter
import os

from scipy.sparse import sparsetools
from sklearn import cross_validation, preprocessing
from sklearn.externals import joblib
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.feature_selection.univariate_selection import chi2
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
import numpy as np

from FeatureExtraction.main import load_sparse_csr, load_numpy_matrix
from config import feature_set_path
from RatingPrediction.Classification import log_regression_fit, svc_fit, linear_svc_fit, SGD_c_fit, nearest_fit, \
    random_forest_fit, draw_confusion_matrix
from RatingPrediction.Regression import linear_regression_fit, SVR_fit, SGD_r_fit, neighbours_fit, \
    bayesian_ridge_fit, ridge_fit, elastic_fit, lasso_fit

FEATURES = ['CF', 'BTF', 'FTF', 'TTF', 'BBTF', 'BTTF', 'TBTF', 'TTTF', 'BiOnlyBi', 'BiOnlyTFIDF',
            'TriOnlyBi', 'TriOnlyTFIDF', 'binaryChar', 'tfidfChar', 'binary skipgram',
            'tfidf skipgram', 'LDA', 'word2vec tfidf', 'word2vec BOC', 'doc2vec', 'google tfidf']


def load_training_data(x_filename, y_filename):
    tempCWD = os.getcwd()

    os.chdir('D:\Workspace\NLTK comments\src\FeatureExtraction')

    fs = np.load(x_filename)
    vv = np.load(y_filename)

    os.chdir(tempCWD)
    return fs, vv


def runRegressionModelTest(featureSet, valueVector, model):
    output = ''
    clf = 0
    if model == 1:
        print "\nLINEAR REGRESSION\n"
        clf = linear_regression_fit(featureSet, valueVector)
    elif model == 2:
        print "\nSVR\n"
        clf = SVR_fit(featureSet, valueVector)
    elif model == 4:
        print "\nSTOCHASTIC\n"
        clf = SGD_r_fit(featureSet, valueVector)
        joblib.dump(clf, 'sgd.pkl')
    elif model == 5:
        print "\nNEIGHBOURS\n"
        clf = neighbours_fit(featureSet, valueVector)
    elif model == 6:
        print "\nLOGISTIC\n"
        clf = log_regression_fit(featureSet, valueVector)
    elif model == 7:
        print "\nBAYESIANRIDGE\n"
        clf = bayesian_ridge_fit(featureSet, valueVector)
    elif model == 8:
        print "\nRIDGE\n"
        clf = ridge_fit(featureSet, valueVector)
    elif model == 9:
        print "\nELASTIC NET\n"
        clf = elastic_fit(featureSet, valueVector)
    elif model == 10:
        print "\nLASSO\n"
        clf = lasso_fit(featureSet, valueVector)
    else:
        print 'Invalid choice\n'

    return clf


def train_classifier(X, y, model, featureset, data_source):
    kernel = 'rbf'

    parameters = np.zeros([21, 3])
    if data_source == 1:
        parameters[0] = [10000, 0.001, 10000] # Manual
        parameters[1] = [10, 0.01, 0.1] # Unigram
        parameters[2] = [10, 0.001, 0.1]
        parameters[3] = [10, 0.1, 1]
        parameters[4] = [100000, 0.001, 0.1] # Bigram
        parameters[5] = [100, 0.1, 10]
        parameters[6] = [100, 0.001, 0.1] # Trigram
        parameters[7] = [10, 0.1, 10000]
        parameters[8] = [1000, 0.001, 1]  # Bigram only
        parameters[9] = [10, 0.1, 100]
        parameters[10] = [0.01, 0.1, 1]  # Trigram only
        parameters[11] = [0.01, 10, 100]
        parameters[12] = [10, 0.0001, 0.001] # Character Ngram
        parameters[13] = [10, 1, 10]
        parameters[14] = [10, 0.001, 0.01] # Character Skipgram
        parameters[15] = [10000, 0.0001, 1]
        parameters[16] = [1000000, 10.0, 0.001] # LDA
        parameters[17] = [10000, 0.0001, 1] # Word2Vec TFIDF
        parameters[18] = [10000000, 0.0001, 10000] # Word2Vec BOC
        parameters[19] = [1, 0.01, 10] # Doc2Vec
        parameters[20] = [10, 0.001, 0.01] # Google Word2vec TFIDF
    elif data_source == 2:  # TOY
        parameters[0] = [0.1, 0.1, 10]  # Manual
        parameters[1] = [100, 0.01, 1]  # Unigram
        parameters[2] = [100, 0.01, 1]
        parameters[3] = [10, 1, 10]
        parameters[4] = [10, 0.01, 0.1]  # Bigram
        parameters[5] = [100, 0.1, 1000]
        parameters[6] = [10, 0.01, 10]  # Trigram
        parameters[7] = [10, 0.1, 10000]
        parameters[8] = [1000, 0.1, 1]  # Bigram only
        parameters[9] = [10000, 0.1, 10000]
        parameters[10] = [10, 0.1, 1]  # Trigram only
        parameters[11] = [10, 10, 1000]
        parameters[12] = [10, 0.001, 0.1]  # Character Ngram
        parameters[13] = [100, 0.1, 100]
        parameters[14] = [100, 0.001, 0.1]  # Character Skipgram
        parameters[15] = [10, 1, 10]
        parameters[16] = [10000000, 1, 1] # LDA
        parameters[17] = [10, 0.01, 0.01]  # Word2Vec TFIDF
        parameters[18] = [1000, 0.0001, 0.1]  # Word2Vec BOC
        parameters[19] = [100, 0.0001, 0.01]  # Doc2Vec
        parameters[20] = [10, 0.001, 1]  # Google Word2vec TFIDF
    elif data_source == 3:  # SLASHDOT
        parameters[0] = [1000000,0.0001, 1000] # Manual
        parameters[1] = [1000, 0.1, 1000] # Unigram
        parameters[2] = [1000, 0.1, 1]
        parameters[3] = [1000, 1, 100]
        parameters[4] = [100, 0.001, 1000] # Bigram
        parameters[5] = [1, 1, 10000]
        parameters[6] = [100, 0.001, 10000] # Trigram
        parameters[7] = [1, 0.1, 10000]
        parameters[8] = [1000, 0.001, 1]  # Bigram only
        parameters[9] = [0.1, 1, 100]
        parameters[10] = [10, 0.1, 1]  # Trigram only
        parameters[11] = [100, 1, 1000]
        parameters[12] = [1000, 0.001, 0.01] # Character Ngram
        parameters[13] = [10, 1, 100]
        parameters[14] = [10, 0.001, 0.1] # Character Skipgram
        parameters[15] = [10, 1, 10]
        parameters[16] = [1000000, 0.01, 10] # LDA
        parameters[17] = [10, 0.01, 100] # Word2Vec TFIDF
        parameters[18] = [1, 0.001, 0.001] # Word2Vec BOC
        parameters[19] = [1000000, 0.0001, 1000] # Doc2Vec
        parameters[20] = [10000000, 0.0001, 10000] # Google Word2vec TFIDF

    C = parameters[featureset][0]
    gamma = parameters[featureset][1]
    Lc = parameters[featureset][2]

    if model == 1:
        print "\nSVC\n"
        clf = svc_fit(X, y, kernel=kernel, C=C, gamma=gamma)
    elif model == 2:
        print '\nLinearSVC\n'
        clf = linear_svc_fit(X, y, C=Lc)
    elif model == 3:
        print '\nStochasticGradientDescent\n'
        clf = SGD_c_fit(X, y)
    elif model == 4:
        print '\nKNearestNeighbours\n'
        clf = nearest_fit(X, y)
    elif model == 5:
        print '\nRandomForest\n'
        clf = random_forest_fit(X, y)
    elif model == 6:
        print '\nLogistic\n'
        clf = log_regression_fit(X, y)

    return clf


def normalize_sets_sparse(train, test):
    norm = train.copy()
    norm.data **= 2  # Square every value
    norm = norm.sum(axis=0)  # Sum every column
    n_nonzeros = np.where(norm > 0)
    norm[n_nonzeros] = 1.0 / np.sqrt(norm[n_nonzeros])
    norm = np.array(norm).T[0]

    sparsetools.csr_scale_columns(train.shape[0], train.shape[1], train.indptr, train.indices, train.data, norm)
    sparsetools.csr_scale_columns(test.shape[0], test.shape[1], test.indptr, test.indices, test.data, norm)

    return train, test


def normalized(a, norm):
    l2 = np.atleast_1d(norm)
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis=0)


def normalize_sets_dense(train, test):
    norm1 = np.linalg.norm(train, axis=0)
    return normalized(train, norm1), normalized(test, norm1)


reg = False
datatype = 3
perc = 50

if __name__ == '__main__':
    if datatype == 1:
        tag = '_main'
        model_tag = '_news24'
    elif datatype == 2:
        tag = "_toy"
        model_tag = '_news24'
    elif datatype == 3:
        tag = '_slashdot'
        model_tag = '_slashdot'

    for featureV in [8]:
        y_train = load_numpy_matrix(feature_set_path + r'valueVector' + tag + '_train.npy')
        y_test = load_numpy_matrix(feature_set_path + r'valueVector' + tag + '_test.npy')

        if featureV == 0:
            X_train = load_numpy_matrix(feature_set_path + r'featureArray' + tag + '_train.npy')
            sd = load_numpy_matrix(feature_set_path + r'socialVector' + tag + '_train.npy')
            X_train = np.hstack((X_train, sd))
            X_test = load_numpy_matrix(feature_set_path + r'featureArray' + tag + '_test.npy')
            sd2 = load_numpy_matrix(feature_set_path + r'socialVector' + tag + '_test.npy')
            X_test = np.hstack((X_test, sd2))
            perc = 80
        elif featureV == 1:
            X_train = load_sparse_csr(feature_set_path + r'binaryWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'binaryWordData' + tag + '_test.npz')
        elif featureV == 2:
            X_train = load_sparse_csr(feature_set_path + r'freqWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'freqWordData' + tag + '_test.npz')
        elif featureV == 3:
            X_train = load_sparse_csr(feature_set_path + r'tfidfWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'tfidfWordData' + tag + '_test.npz')
        elif featureV == 4:
            X_train = load_sparse_csr(feature_set_path + r'bigramBinaryWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'bigramBinaryWordData' + tag + '_test.npz')
        elif featureV == 5:
            X_train = load_sparse_csr(feature_set_path + r'bigramTfidfWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'bigramTfidfWordData' + tag + '_test.npz')
        elif featureV == 6:
            X_train = load_sparse_csr(feature_set_path + r'trigramBinaryWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'trigramBinaryWordData' + tag + '_test.npz')
        elif featureV == 7:
            X_train = load_sparse_csr(feature_set_path + r'trigramTfidfWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'trigramTfidfWordData' + tag + '_test.npz')
        elif featureV == 8:
            X_train = load_sparse_csr(feature_set_path + r'bigramOnlyBinaryWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'bigramOnlyBinaryWordData' + tag + '_test.npz')
        elif featureV == 9:
            X_train = load_sparse_csr(feature_set_path + r'bigramOnlyTfidfWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'bigramOnlyTfidfWordData' + tag + '_test.npz')
        elif featureV == 10:
            X_train = load_sparse_csr(feature_set_path + r'trigramOnlyBinaryWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'trigramOnlyBinaryWordData' + tag + '_test.npz')
        elif featureV == 11:
            X_train = load_sparse_csr(feature_set_path + r'trigramOnlyTfidfWordData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'trigramOnlyTfidfWordData' + tag + '_test.npz')

        elif featureV == 12:
            X_train = load_sparse_csr(feature_set_path + r'binaryCharacterData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'binaryCharacterData' + tag + '_test.npz')
        elif featureV == 13:
            X_train = load_sparse_csr(feature_set_path + r'tfidfCharacterData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'tfidfCharacterData' + tag + '_test.npz')
        elif featureV == 14:
            X_train = load_sparse_csr(feature_set_path + r'binaryCharacterSkipgramData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'binaryCharacterSkipgramData' + tag + '_test.npz')
        elif featureV == 15:
            X_train = load_sparse_csr(feature_set_path + r'tfidfCharacterSkipgramData' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + r'tfidfCharacterSkipgramData' + tag + '_test.npz')
        elif featureV == 16:
            X_train = load_sparse_csr(feature_set_path + model_tag + r'lda' + tag + '_train.npz')
            X_test = load_sparse_csr(feature_set_path + model_tag + r'lda' + tag + '_test.npz')
        elif featureV == 17:
            X_train = load_numpy_matrix(feature_set_path + 'Basic300' + model_tag + '_TfidfFeatures' + tag + '_train.npy')
            X_test = load_numpy_matrix(feature_set_path + 'Basic300' + model_tag + '_TfidfFeatures' + tag + '_test.npy')
        elif featureV == 18:
            X_train = load_numpy_matrix(feature_set_path + 'Basic200' + model_tag + '_BOCFeatures' + tag + '_train.npy')
            X_test = load_numpy_matrix(feature_set_path + 'Basic200' + model_tag + '_BOCFeatures' + tag + '_test.npy')
        elif featureV == 19:
            X_train = load_numpy_matrix(
                feature_set_path + 'DocBasic300' + model_tag + '_ParagraphFeatures' + tag + '_train.npy')
            X_test = load_numpy_matrix(
                feature_set_path + 'DocBasic300' + model_tag + '_ParagraphFeatures' + tag + '_test.npy')
        elif featureV == 20:
            X_train = load_numpy_matrix(feature_set_path + "Google_TfidfFeatures" + tag + "_train.npy")
            X_test = load_numpy_matrix(feature_set_path + "Google_TfidfFeatures" + tag + "_test.npy")

        print "\nFeatures", FEATURES[featureV]
        print '\nTotal:', X_train.shape[0] + X_test.shape[0]
        print 'Features:', X_train.shape[1]
        print "\nClass distribution", Counter(y_train)
        print "\nClass distribution", Counter(y_test)

        # FEATURE SELECT
        if featureV == 0:
            selector = SelectPercentile(score_func=f_classif, percentile=perc).fit(X_train, y_train)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)
        elif featureV <2:
            selector = SelectKBest(score_func=chi2, k=min(200000, int(X_train.shape[1]*(perc/100.0)))).fit(X_train,y_train)
            X_train = selector.transform(X_train)
            X_test = selector.transform(X_test)



        print X_train.shape
        print X_test.shape

        # FEATURE SCALING
        if featureV == 0:
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            X_train, X_test = normalize_sets_sparse(X_train, X_test)
            

        if reg:
            print "\nREGRESSION\n"
            for m in [2]:
                clf = runRegressionModelTest(X_train, y_test, m)

                cv = cross_validation.StratifiedShuffleSplit(X_train.shape[0], n_iter=5, test_size=0.33,
                                                             random_state=42)
                a = cross_validation.cross_val_score(clf, X_test, y_test, cv=cv)
                a = a[a > 0]
                print 'Cross V score: :' + ' '.join("%10.3f" % x for x in a)
                print ('Mean Score: %.3f' % np.mean(a))
        else:
            print "\nCLASSIFICATION\n"
            print "Nr Of Features", X_train.shape[1]
            print "Nr Of train Rows", X_train.shape[0]
            print "Nr Of test Rows", X_test.shape[0]
            for m in [1, 2]:
                print "STARTING CLASSIFICATION"
                clf = train_classifier(X_train, y_train, m, featureV, datatype)

                predicted = clf.predict(X_test)
                print "Accuracy: %0.3f " % (accuracy_score(y_test, predicted))

                print classification_report(y_test, predicted, target_names=['0', '1'], digits=3)
                print classification_report(y_train, clf.predict(X_train), target_names=['0', '1'], digits=3)
                print draw_confusion_matrix(y_test, predicted, [0, 1])
