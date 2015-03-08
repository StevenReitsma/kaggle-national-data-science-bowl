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
        
    
    def normalize(self, activations):
        activations = activations - np.mean(activations, axis = 0)
        activations = activations/np.std(activations, axis = 0)
        
        
        return activations
  
    
    
    def pipeline(self, centroids, file_path = "../data/activations/", batch_size = 729, n_pool_regions = 4):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        

        batches = batchreader.BatchReader(batchsize = batch_size)#   
        

        dimensions = (batches.nbatches , len(centroids)*n_pool_regions) # Set dimensions to #imagesx4*#centroids
        activations = np.zeros(dimensions)
        
        
        
        for i, batch in enumerate(batches):
            activation = self.distance_to_centroids(batch, centroids) # Calculate activations for each patch to each centroid
            activations[i] = pool(activation, n_pool_regions = n_pool_regions) # Returns a vector with length 4x#centroids
            util.update_progress(i/batches.nbatches)
            
        util.update_progress(1)
        print "Normalizing activations..."
        activations = self.normalize(activations)
        print "Normalizing done"
        print "Writing activations to file:"
        f = h5py.File(file_path + str(len(centroids)) + "activationkmeans.h5", "w")
        dataSet = f.create_dataset("activations", dimensions, dtype = np.float64)
        dataSet[...] = activations
        f.close()
        print "Writing done"
        
        return activations

        
    
if __name__ == '__main__':
    km = kmeans.kMeansTrainer()
    centroids = km.get_saved_centroids()
    sup_km = ActivationCalculation()
    sup_km.pipeline(centroids = centroids)


