'''
Created on 25 Apr 2014

@author: Dirk
'''

from sklearn import svm, cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics.metrics import f1_score
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.tree.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd


def svc_fit(X,y,kernel, C, gamma ):
    
    clf = svm.SVC(C=C, gamma=gamma, kernel=kernel, class_weight='auto', cache_size=1000, verbose=True, max_iter=250000)
    print clf
    return clf.fit(X, y)

def linear_svc_fit(X,y,C):
    clf = svm.SVC(kernel='linear', C=C, verbose=True, class_weight='auto', cache_size=1000, max_iter=250000)   
    print clf          
    return clf.fit(X, y)


def log_regression_fit(X,y):
    lso = LogisticRegression(tol=1e-8, penalty='l2')
    return lso.fit(X, y)

def SGD_c_fit(X,y):
    clf = SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, shuffle=True)
    return clf.fit(X, y)

def nearest_fit(X,y):
    clf = KNeighborsClassifier(7, 'distance')
    return clf.fit(X, y)

def random_forest_fit(X,y):
    clf = RandomForestClassifier(n_estimators=50, criterion="entropy", n_jobs=-1, min_samples_leaf=5)
    return clf.fit(X, y)

def decision_tree_fit(X,y):
    clf = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
    return clf.fit(X, y)


def draw_confusion_matrix(y_test, y_pred, labels):
    cm = sk_confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(data=cm, columns=labels, index=labels)
    cm.columns.name = 'Predicted label'
    cm.index.name = 'True label'
    error_rate = (y_pred != y_test).mean()
    print('mean error rate: %.2f' % error_rate)
    return cm