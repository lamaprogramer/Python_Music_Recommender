from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def display_groups(groups):
  for key, item in groups:
    print(groups.get_group(key))
    
def kmeans_clustering(data: pd.DataFrame, n_clusters: int=10, random_state=42):
  k_m = KMeans(n_clusters=n_clusters, init="random", n_init=20, max_iter=1000, tol=1e-04, random_state=random_state)
  return (k_m.fit_predict(data), k_m.cluster_centers_)

def plot_clusters(data: pd.DataFrame, clusters: np.ndarray):
  plt.scatter(data[:,0], data[:,1], c=clusters)
  plt.show()
  
def closest_clusters_euclidean(data_points, cluster_centroids):
  distances = euclidean_distances(data_points, cluster_centroids)
  return [point_distances.argmin() for point_distances in distances]

def closest_clusters_cosine_similarity(data_points, cluster_centroids):
  distances = cosine_similarity(data_points, cluster_centroids)
  return [point_distances.argmax() for point_distances in distances]

def closest_points_euclidean(data_points, data_points2):
  distances = euclidean_distances(data_points, data_points2)
  return distances

def closest_points_cosine_similarity(data_points, data_points2):
  distances = euclidean_distances(data_points, data_points2)
  return distances