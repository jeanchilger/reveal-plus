#from .datasets import load_svmlight_file
from sklearn.datasets import load_svmlight_file
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import genfromtxt







def get_data():
    print("file is ", sys.argv[1])
    data = load_svmlight_file(sys.argv[1])
    return data[0], data[1]



def silhuete(X,y): 


    #clusterer = KMeans(n_clusters=2, random_state=10)
    #cluster_labels = clusterer.fit_predict(X)
    silhouette_avg = silhouette_score(X, y)
    print("For n_clusters =", 2,"The average silhouette_score is :", silhouette_avg)
    print(X)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, y)



X, y = get_data()
print(X.shape)
silhuete(X,y)