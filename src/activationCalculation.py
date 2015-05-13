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
import matplotlib.pyplot as plt


class ActivationCalculation():
            
        
    def distance_to_centroids(self, patches, centroids):
        # Triangle (soft) activation function
        pp = np.sum(np.square(patches), axis=1) # Dot product patchesxpatches\
        cc = np.sum(np.square(centroids), axis=1) # Dot product centroidsxcentroids
        pc = 2*np.dot(patches, centroids.T) # 2* Dot product patchesxcentroids

        z = np.sqrt(cc + (pp - pc)) # Distance measure
        mu = z.mean(axis=1)
        activation = np.maximum(0, mu-z) # Similarity measure

        return activation
    
 
    def _distance_to_centroids(self, patches, centroids):
        #self.visualize_activation(centroids)
        
        activations = np.zeros((patches.shape[0],centroids.shape[0]) )
        
        for i, patch in enumerate(patches):
            for j, centroid in enumerate(centroids):

                #print "Centroid/patch"
                #plt.imshow(centroid.reshape(6, 6), interpolation='nearest')
                #plt.show()
                #plt.imshow(patch.reshape(6,6), interpolation='nearest')
                #plt.show()
                
                act = np.square(centroid - patch)
                
                #print "Activation"
                #plt.imshow(act.reshape(6,6), interpolation='nearest')
                #plt.show()
                
                #print i, j
                #print np.max(centroid), np.max(patch)
               # print np.mean(centroid), np.mean(patch)

                
                activations[i, j] = np.sum(act)
            
        return activations
        
        
    
    def normalize(self, activations):
        std = np.std(activations, axis = 0)
        mean = np.mean(activations, axis = 0)
        
        for i, act in enumerate(activations):
            activations[i] = (act-mean)/std
        #activations = activations - np.mean(activations, axis = 0)
        #activations = activations/std
        
        
        return activations
  
    
    
    def pipeline(self, centroids, data_file = "../data/preprocessed.h5", file_path = "../data/activations/", batch_size = -1, n_pool_regions = 4):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        

        if batch_size == -1:
            meta = util.load_metadata()
            batch_size = meta['patches_per_image']
        
        batches = batchreader.BatchReader(batchsize = batch_size, filepath=data_file)#   
        

        dimensions = (batches.nbatches , len(centroids)*n_pool_regions) # Set dimensions to #imagesx4*#centroids
        activations = np.zeros(dimensions)
        
        
        
        for i, batch in enumerate(batches):
            activation = self.distance_to_centroids(batch, centroids) # Calculate activations for each patch to each centroid
            
            #self.visualize_activation(activation.T)            
            
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

        
        
        
    def visualize_activation(self, activations):
        print activations.shape
        patch_size = np.sqrt(activations.shape[1])
        n_features = activations.shape[0]
    
        # Reshape to 2D slabs
        reshaped = np.reshape(activations, (n_features, patch_size, patch_size))
        

        length = int(np.sqrt(reshaped.shape[0]))
        
        f, ax = plt.subplots(length, length)
        
        for i in range(0, length):
            for j in range(0, length):
             ax[i, j].imshow(reshaped[i*length+j], cmap = 'Greys', interpolation = 'nearest')
             ax[i, j].axis('off')
        
        plt.show()
        
        
    def visualize_activation_alt(self, activations):
        patch_size = np.sqrt(activations.shape[0])
        
        one = activations[:,0]
        
        im = np.reshape(one, (patch_size,patch_size))

        plt.imshow(im, cmap='Greys', interpolation= 'nearest')        
        
        plt.show()

    
    
if __name__ == '__main__':
    km = kmeans.kMeansTrainer()
    centroids = km.get_saved_centroids(100)
    #util.plot_centroids(centroids, "../data/centroidskmeans")
    sup_km = ActivationCalculation()
    sup_km.pipeline(centroids = centroids, data_file="../data/preprocessed_test.h5")
    
    


