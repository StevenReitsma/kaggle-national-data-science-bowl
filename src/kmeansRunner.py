# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 20:42:04 2015

@author: Luc
"""

import kmeans
import numpy as np
import activationCalculation as act
import randombatchreader as randbr
import train_classifier as train
import predict_classifier as classifier
import h5py
import time


def singlePipeline(nr_centroids, label_path = "../data/preprocessed.h5"):
    start_time = time.time()
    batches = randbr.RandomBatchReader()
    kmTrainer = kmeans.kMeansTrainer(nr_centroids = nr_centroids)
    
    centroids = kmTrainer.fit(batches)
    kmTrainer.save_centroids(centroids)
    
    act_calc = act.ActivationCalculation()
    features = act_calc.pipeline(centroids)
    
    f = h5py.File(label_path)
    labels = f["labels"]
    
#    feature_data = h5py.File("../data/activations/100activationkmeans.h5")
#    features = feature_data["activations"]
#    
#    
#    train.trainSGD(features, labels, nr_centroids)

    classified = classifier.predict(features, nr_centroids)
    
   
    summing = 0

    for i, label in enumerate(labels):
        summing+= np.log(classified[i][label])

        
    summing = -summing/len(labels)
    
    print summing
    f.close()
    total_time = time.time() - start_time
    print "Total time " + str(nr_centroids) + " is: " + str(total_time)
#    feature_data.close()       
    #start = 14:05
       # prev score = 3.56




if __name__ == '__main__':
    for nr_centroids in range(200, 800, 100):
        singlePipeline(nr_centroids)