'''
Created on 24 Mar 2014

@author: Dirk
'''

from sklearn import svm, datasets, linear_model, tree, neighbors
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing.data import StandardScaler

import numpy as np


def SVR_fit(X, y):
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'epsilon':[0.1, 0.3, 0.5]}
    scores = ['precision', 'recall']
    for score in scores:
        clf = GridSearchCV(svm.SVR(), parameters, cv=5, scoring=score)
        clf.fit(X, y)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()
    return clf
    
def linear_regression_fit(X, y):
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    return regr

def SGD_r_fit(X,y):
    regr = linear_model.SGDRegressor(loss="squared_loss",penalty='l1')
    return regr.fit(X, y)

def decision_tree_fit(X,y):
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X,y)
    return clf

def neighbours_fit(X,y):
    rdg = neighbors.KNeighborsRegressor(7, weights='distance')
    return rdg.fit(X, y)
    
    
def bayesian_ridge_fit(X,y):
    br = linear_model.BayesianRidge()
    return br.fit(X,y)


def ridge_fit(X,y):
    br = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    return br.fit(X,y)


def lasso_fit(X,y):
    br =  linear_model.LassoCV(alphas=[0.1, 1.0, 10])
    return br.fit(X,y)


def elastic_fit(X,y):
    br =  linear_model.ElasticNet(alpha=0.1)
    return br.fit(X,y)
    
    