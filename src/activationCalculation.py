# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 21:49:01 2015

@author: Luc and Tom
"""
from __future__ import division
from pooling import pool
import numpy as np
import kmeans
import batchreader
import os
import h5py
import util


class ActivationCalculation():
            
        
    def distance_to_centroids(self, patches, centroids):
        # Triangle (soft) activation function
        pp = np.sum(np.square(patches), axis=1) # Dot product patchesxpatches
        cc = np.sum(np.square(centroids), axis=1) # Dot product centroidsxcentroids
        pc = 2*np.dot(patches, centroids.T) # 2* Dot product patchesxcentroids
    
        z = np.sqrt(cc + (pp - pc.T).T) # Distance measure
        mu = z.mean(axis=0)
        activation = np.maximum(0, mu-z) # Similarity measure

        return activation
        
    
    def pipeline(self, centroids, file_path = "../data/", batch_size = 729, n_pool_regions = 4):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        

        batches = batchreader.BatchReader(batchsize = batch_size)#   
        f = h5py.File(file_path + "activationkmeans.h5", "w")

        dimensions = (batches.nbatches , len(centroids)*n_pool_regions)
        print dimensions
        dataSet = f.create_dataset("activations", dimensions, dtype = np.uint8)
        
        
        for i, batch in enumerate(batches):
            activation = self.distance_to_centroids(batch, centroids)
            dataSet[i] = pool(activation, n_pool_regions = n_pool_regions)
            util.update_progress(i/batches.nbatches)
        
        print dataSet.shape
        util.update_progress(1)
        f.close()
        
    
if __name__ == '__main__':
    a = np.zeros((10,5))
    print a
#    
#    km = kmeans.kMeansTrainer()
#    centroids = km.get_centroids(new = False)
#    sup_km = ActivationCalculation()
#    sup_km.pipeline(centroids = centroids)


