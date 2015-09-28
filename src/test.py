from FeatureExtraction.main import load_sparse_csr, load_numpy_matrix
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics.regression import mean_squared_error
from sklearn.preprocessing.data import normalize, Normalizer
from sklearn.svm.classes import SVR

from RatingPrediction.main import normalize_sets_sparse
from config import feature_set_path
import numpy as np


tag='_slashdot'

'''
X_train = load_numpy_matrix(feature_set_path +  r'featureArray'+tag+'_train.npy')
sd = load_numpy_matrix(feature_set_path +  r'socialVector'+tag+'_train.npy')
X_train =  np.hstack((X_train,sd))
X_test = load_numpy_matrix(feature_set_path +  r'featureArray'+tag+'_test.npy')
sd2 = load_numpy_matrix(feature_set_path +  r'socialVector'+tag+'_test.npy')
X_test =  np.hstack((X_test,sd2))
'''
X_train = load_sparse_csr(feature_set_path +  r'binaryWordData'+tag+'_train.npz')  
X_test = load_sparse_csr(feature_set_path +  r'binaryWordData'+tag+'_test.npz')  
            
y_train = load_numpy_matrix(feature_set_path +  r'valueVector'+tag+'_train.npy')
y_test = load_numpy_matrix(feature_set_path +  r'valueVector'+tag+'_test.npy')

normal = Normalizer()
y_train = normal.fit_transform(y_train)
y_test = normal.fit_transform(y_test)
'''
norm1 =  np.linalg.norm(y_train)    
if norm1 != 0:   
    y_train, y_test =  y_train/norm1, y_test/norm1
'''

#model = SVR(C=1.0, gamma=1.0)
model = LinearRegression()
model.fit(X_train, y_train)

print mean_squared_error(y_test, model.predict(X_test))
m = np.mean(y_test)
print mean_squared_error(y_test, m*np.ones(len(y_test)))


print model.score(X_train, y_train)
print np.corrcoef(model.predict(X_train), y_train)[0, 1]**2

print model.score(X_test,  y_test)
print np.corrcoef(model.predict(X_test), y_test)[0, 1]**2

