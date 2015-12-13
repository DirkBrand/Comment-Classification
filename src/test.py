from FeatureExtraction.main import load_sparse_csr, load_numpy_matrix
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.coordinate_descent import Lasso, ElasticNet
from sklearn.metrics.regression import mean_squared_error
from sklearn.preprocessing.data import normalize, Normalizer
from sklearn.svm.classes import SVR

from RatingPrediction.main import normalize_sets_sparse
from config import feature_set_path
import numpy as np
from sklearn.metrics.metrics import r2_score

import matplotlib.pyplot as plt

tag='_slashdot'


X_train = load_numpy_matrix(feature_set_path +  r'featureArray'+tag+'_train.npy')
sd = load_numpy_matrix(feature_set_path +  r'socialVector'+tag+'_train.npy')
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


print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape

print X_train[123,:]

'''
norm1 =  np.linalg.norm(y_train)    
if norm1 != 0:   
    y_train, y_test =  y_train/norm1, y_test/norm1
print norm1
'''

print y_train.shape

model = SVR(C=1.0, gamma=1.0)
model = LinearRegression()

lasso = Lasso(alpha=0.1).fit(X_train, y_train)
enet = ElasticNet(alpha=0.1, l1_ratio=0.7).fit(X_train, y_train)

y_pred = lasso.predict(X_test)

print "MSE", mean_squared_error(y_test, y_pred)
m = np.mean(y_test)
print "MSE (Mean)",mean_squared_error(y_test, m*np.ones(len(y_test)))


print "r^2 on test data", r2_score(y_test, y_pred)

plt.plot(enet.coef_, label='Elastic net coefficients')
plt.plot(lasso.coef_, label='Lasso coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score(y_test, lasso.predict(X_test)), r2_score(y_test, enet.predict(X_test))))
plt.show()
