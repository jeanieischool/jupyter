#matplotlib.use('TkAgg')
import matplotlib
import pylab
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import matplotlib

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
import random

print(__doc__)

# import data
yelp_reviews = pd.read_csv('yelp_reviewers.txt',sep='|',header=0)
yelp_reviews = yelp_reviews.fillna(0)

# sample data
rows = yelp_reviews.sample(100)
X = rows[['q4','q5','q6']]
y = rows['user_id']
from sklearn.metrics import silhouette_score

def plot_silhouette_scores(data_set):
    s=[]
    for n_clusters in range(2,9):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data_set)
    
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
    
        s.append(silhouette_score(data_set,labels,metric='euclidean'))
#         print(s)
    plt.plot(s)
plot_silhouette_scores(X)