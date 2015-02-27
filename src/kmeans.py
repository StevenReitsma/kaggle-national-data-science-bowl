from __future__ import division
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import util
import batchreader as br
import h5py

import matplotlib.cm as cm
from PIL import Image


class kMeansTrainer():
    
    def __init__(self, nr_centroids, nr_it):
        self.nr_centroids = nr_centroids
        self.nr_it = nr_it
        
    def fit(self, iterations = 1):
        kmeans = MiniBatchKMeans(self.nr_centroids, n_init = self.nr_it, init='k-means++')
        batches = br.BatchReader(batchsize = 50000) # so that we can determine the max iterations
        maxIterations = batches.nbatches
        for it in range(iterations):
            for i, batch in enumerate(batches):
                util.update_progress((i+(it*maxIterations))/(maxIterations*iterations))
                kmeans.partial_fit(batch)
            batches = br.BatchReader(batchsize = 50000) #create new batch reader for the next iteration
        util.update_progress(1.0)
        print "fitting done"
        return kmeans.cluster_centers_
        
    
    def saveCentroids(self, centroids, file_path = "../data/centroidskmeans/"):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        f = h5py.File(file_path + "centroids.h5", "w")
        dataSet = f.create_dataset("centroids", centroids.shape, dtype = np.uint8)
        dataSet[...]= centroids
        f.close()

        
    def pipeline(self):
        centroids = self.fit()
        util.plotCentroids(centroids = centroids, file_path = "../data/centroidskmeans/")
        self.saveCentroids(centroids)         
     


        
    
    
if __name__ == '__main__':  
    km = kMeansTrainer(100, 10)
    km.pipeline()