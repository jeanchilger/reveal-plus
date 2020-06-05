from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file

import sys


print ("file eh ==", sys.argv[1])
X, y_train = load_svmlight_file(sys.argv[1])


print(X.shape)
SVD = TruncatedSVD(n_components=10) 
U = SVD.fit_transform(X)
print(U.shape)


dump_svmlight_file(U,y_train,sys.argv[2])
