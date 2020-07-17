# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:57:19 2020

@author: Hamza
"""
# K-MEANS Clustering

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset 
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# using Elbow methode to determine the Optimale Number of clusters
from sklearn.cluster import KMeans
wssc = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++',random_state = 42)
    kmeans.fit(X)
    wssc.append(kmeans.inertia_)
    
# plot wssc of each number of clusters
plt.plot(range(1,11), wssc, color='blue')
plt.title('Elbow Methode')
plt.xlabel('Number of Clusters')
plt.ylabel('WSSC')

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init='k-means++', random_state= 42)
y_clusters = kmeans.fit_predict(X)

# # Visualising the clusters
plt.scatter(X[y_clusters == 0, 0], X[y_clusters == 0, 1], color='red', label = 'Careless')
plt.scatter(X[y_clusters == 1, 0], X[y_clusters == 1, 1], color='blue', label = 'Standard')
plt.scatter(X[y_clusters == 2, 0], X[y_clusters == 2, 1], color='green', label = '>>> Target <<<')
plt.scatter(X[y_clusters == 3, 0], X[y_clusters == 3, 1], color='orange', label = 'average')
plt.scatter(X[y_clusters == 4, 0], X[y_clusters == 4, 1], color='yellow', label = 'Careful')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()



