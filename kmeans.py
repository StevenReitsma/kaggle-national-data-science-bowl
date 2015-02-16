import numpy as np
import scipy, sklearn, math, os, time
from sklearn.cluster import KMeans
from sklearn.covariance import OAS
from scipy import misc
from numpy import array
import shutil
import tempfile

class kMeansTrainer():
    
    def __init__(self, patches, nr_centroids, nr_it):
        self.patches = patches
        self.nrCentroids = nr_centroids
        self.nr_it = nr_it
        
    def fit(self):
        kmeans = KMeans(self.nr_centroids, init='k-means++', compute_labels = False)
        kmeans.fit(self.patches)
        return kmeans.cluster_centers_
        
    def doeshitofzo(self):
        centroids = self.fit()
        np.savetxt("/DataOutput/centroids.csv", centroids, delimiter = ",")

if __name__ == '__main__':
    km = kMeansTrainer(0, 3000, 10)
    km.doeshitofzo()