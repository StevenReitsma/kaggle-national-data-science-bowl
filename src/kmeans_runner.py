# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 20:42:04 2015

@author: Luc
"""
from __future__ import division
import kmeans
import numpy as np
import activationCalculation as act
import randombatchreader as randbr
import train_classifier as train
import predict_classifier as classifier
import train_svc as svc
import h5py
import time
import sys


def singlePipeline(nr_centroids, label_path = "../data/preprocessed.h5"):
    
    
    #Finds the features using kmeans
    kmTrainer = kmeans.kMeansTrainer(nr_centroids = nr_centroids, nr_it = 10)    
    centroids = kmTrainer.fit()
    kmTrainer.save_centroids(centroids)
    
    #Calculates the activaiton of the test set
    act_calc = act.ActivationCalculation()
    features = act_calc.pipeline(centroids)  
    
    
    print "Getting labels..."
    #get the labels
    f = h5py.File(label_path)
    labels = f["labels"]

    #loads feature data
#    feature_data = h5py.File("../data/activations/200activationkmeans.h5")
#    features = feature_data["activations"]
    
    print "Begin training"
    #Train the SGD classifier
    train.trainSGD(features, labels, nr_centroids)
    
    #Train SVC classifier
    #model = svc.train_svc(features,labels, nr_centroids)

    print "begin prediction"
    #    Classify the testset (the same as the training set in this case)
    classified = classifier.predict(features, nr_centroids)
    
    #classified = model.predict_proba(features)
    print "done"
    
   
    summing = 0
    correct = 0
    
    #calculate the log loss
    for i, label in enumerate(labels):
        if(classified[i][label] == 0):
            summing+= np.log(sys.float_info.min)
        else:
            summing+= np.log(classified[i][label])
        if labels[i] == np.argmax(classified[i]):
#            print classified[i][np.argmax(classified[i])]
            correct += 1
      

       
    summing = -summing/len(labels)
    print summing 
    print correct/len(labels)
    print np.min(classified)
   
#    print summing
#    np.savetxt( "realLabel.csv", labels, delimiter=";")
#    np.savetxt( "SGD_label.csv", max_SGD, delimiter=";")  
    
    f.close()

#    feature_data.close()       

    



if __name__ == '__main__':
    nr_centroids = 500
    singlePipeline(nr_centroids)
