from __future__ import division
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import util
import batchreader as br
from numpy import array
import randombatchreader as randbr


class kMeansTrainer():
    
    def __init__(self, nr_centroids = 100, nr_it = 10):
        self.nr_centroids = nr_centroids 
        self.nr_it = nr_it
        
        
    def fit(self):
        kmeans = MiniBatchKMeans(self.nr_centroids, init='k-means++')
        
        for it in range(self.nr_it):
            batches = randbr.RandomBatchReader()
            maxIterations = batches.nbatches
            for i, batch in enumerate(batches):
                util.update_progress((i+(it*maxIterations))/(maxIterations*self.nr_it))
                kmeans.partial_fit(batch)
                
        util.update_progress(1.0)
        print "fitting done"
        return kmeans.cluster_centers_
        
    
    def save_centroids(self, centroids, file_path = "../data/centroidskmeans/"):
        if not os.path.exists(file_path):
            os.makedirs(file_path)       
        np.savetxt(file_path + str(len(centroids)) +  "centroids.csv", centroids, delimiter=",")
        

    
    def get_saved_centroids(self, nr_centroids, file_path = "../data/centroidskmeans/"):
        f = open(file_path + str(nr_centroids) + "centroids.csv")
        samples = []
        
        for line in f:
            line = line.strip().split(",")
            sample = [float(x) for x in line]
            samples.append(sample)
            
        return array(samples)

        
    def pipeline(self):
        centroids = self.fit()
        self.save_centroids(centroids)
        util.plot_centroids(centroids = centroids, file_path = "../data/centroidskmeans/")
        

       
    
    
if __name__ == '__main__':    
    km = kMeansTrainer(100, 1)
    km.pipeline()