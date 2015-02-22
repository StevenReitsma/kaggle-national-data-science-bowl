from __future__ import division
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import util
import batchreader as br
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


class kMeansTrainer():
    
    def __init__(self, nr_centroids, nr_it):
        self.nr_centroids = nr_centroids
        self.nr_it = nr_it
        
    def fit(self, batches):
        kmeans = MiniBatchKMeans(self.nr_centroids, n_init = self.nr_it, init='k-means++')
        maxIterations = batches.nbatches
        print maxIterations
        for i, batch in enumerate(batches):
            util.update_progress(i/maxIterations)
            kmeans.partial_fit(batch)

        util.update_progress(1.0)
        return kmeans.cluster_centers_
        
    
    def saveCentroids(self, centroids, filepath = "../data/centroidskmeans.h5"):
        f = h5py.File(filepath, "w")
        dimensions = (len(centroids), len(centroids[0]))
        dataSet = f.create_dataset("centroids", dimensions, dtype = np.uint8)
        dataSet[...]= centroids
        f.close()

        
    def pipeline(self):
        batches = br.BatchReader(batchsize = 50000)
        centroids = self.fit(batches)
        self.saveCentroids(centroids)         
     
    #   Under construction
    def plotCentroids(self): 
        array = np.zeros(6*6)
        for i in range(6):
            array[i*6+i] = 255
        matrix = np.reshape(array, (6,6))
        plt.gray()
        plt.imsave('testimage.png', matrix)
        print matrix
    
    
if __name__ == '__main__':  
    km = kMeansTrainer(100, 10)
    km.pipeline()