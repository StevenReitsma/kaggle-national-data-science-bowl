# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 21:49:01 2015

@author: Luc and Tom
"""

class supervisedKmeans():
    
    def __init__(self, centroids):
        self.centroids = centroids
        
    def distance_to_centroids(self, patches):
        product = []#initialize all the things
        tproduct = []
    
        for patch in patches:   #for each patch
            patch = np.asarray(patch)#convert to np.array
            for centroid in this.centroids:  #for each centroid
                centroid = np.asarray(centroid)
                tproduct.append(np.linalg.norm(patch-centroid))#calc euclidian distance
            product.append(tproduct)# add the list of all the distance from a patch to all centroids to product
            tproduct = []
    
    return product

    

