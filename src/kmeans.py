from __future__ import division
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import util
from numpy import array
import randombatchreader as randbr


class kMeansTrainer():
    
    def __init__(self, nr_centroids = 100, nr_it = 10, rotational_invariant_training = False):
        self.nr_centroids = nr_centroids 
        self.nr_it = nr_it
        self.rotational_invariant_training = rotational_invariant_training
        
        meta = util.load_metadata()
        self.patch_width = meta['patch_size']
        
    def rotate_patches_90_degrees(self, batch, times):
        
        dup = np.copy(batch)
        
        for i, patch in enumerate(dup):
			temp = np.rot90(patch.reshape((self.patch_width, self.patch_width), order="F"), times)
			dup[i,:] = np.reshape(temp, self.patch_width**2, order="F")

	return dup
        
    
    def fit(self):
        kmeans = MiniBatchKMeans(self.nr_centroids, init='k-means++')
        
        for it in range(self.nr_it):
            print "Iteration {0} out of {1}".format(it, self.nr_it)            
            
            batches = randbr.RandomBatchReader()
            maxIterations = batches.nbatches
            for i, batch in enumerate(batches):
                
                if self.rotational_invariant_training:  
                    # rotate the batch 90, 180 and 270 degrees
                    batch90 = self.rotate_patches_90_degrees(batch,1)
                    batch180 = self.rotate_patches_90_degrees(batch,2)
                    batch270 = self.rotate_patches_90_degrees(batch,3)
                    
                    util.update_progress((i+(it*maxIterations))/(maxIterations*self.nr_it))
                    kmeans.partial_fit(batch, batch90, batch180, batch270)
                    
                else:
                 # Normal training   
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
        
    def test_rotation(self):
        batches = randbr.RandomBatchReader()
        batch = batches.next()
        
        batch90 = self.rotate_patches_90_degrees(batch,1)
        batch180 = self.rotate_patches_90_degrees(batch,2)
        batch270 = self.rotate_patches_90_degrees(batch,3)
        
        first = np.array_equal(batch180, self.rotate_patches_90_degrees(batch90,1))
        second = np.array_equal(batch270, self.rotate_patches_90_degrees(batch180,1))
        third = np.array_equal(batch, self.rotate_patches_90_degrees(batch270,1))
        fourth = np.array_equal(batch, self.rotate_patches_90_degrees(batch180,2))
        fifth = np.array_equal(batch, self.rotate_patches_90_degrees(batch180,1))
        sixth = np.array_equal(batch, self.rotate_patches_90_degrees(batch90,2))
        
        print first     # True
        print second    # True
        print third     # True
        print fourth    # True
        print fifth     # False
        print sixth     # False
    
if __name__ == '__main__':    
    km = kMeansTrainer(100, 1)
    km.pipeline()