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
import util
import sys


def singlePipeline(nr_centroids, nr_it, label_path = "../data/preprocessed.h5", clsfr = "SGD", calc_centroids = True):
    
    
    
    if calc_centroids:
        print "calculating centroids..."
        #Finds the features using kmeans
        kmTrainer = kmeans.kMeansTrainer(nr_centroids = nr_centroids, nr_it = nr_it)    
        centroids = kmTrainer.fit()
        kmTrainer.save_centroids(centroids)
        
        print "calculating activations..."
        #Calculates the activaiton of the test set
        act_calc = act.ActivationCalculation()
        features = act_calc.pipeline(centroids)  
    else:
        print "loading centroids from file..."
        #loads feature data
        feature_data = h5py.File("../data/activations/"+str(nr_centroids)+"activationkmeans.h5")
        features = feature_data["activations"]
    
    print "Getting labels..."
    #get the labels
    labels = util.load_labels(label_path)

    print "Got labels"

    

    
    
    
    if clsfr == "SGD": 
        #Train the SGD classifier
        print "Begin training of SGD..."
        train.trainSGD(features, labels, nr_centroids)
        print "Training done"
        
        #Predict based on SGD training
        print "Begin SGD predictions..."
        classified = classifier.predict(features, nr_centroids)
        print "Predicting done"        
        
    elif clsfr == "SVC": 
        #Train SVC classifier
        print "Begin training of SVC..."
        model = svc.train_svc(features, labels, nr_centroids)
        print "Training done"
        
        #Predict based on SVC training
        print "Begin SVC predictions..."
        classified = model.predict_proba(features)
        print "Predicting done"
        
    else:
        print "Selected classifier not available, please use an available classifier"
        return
       
    print "Calculating log loss..."
    summing = 0
    correct = 0
    
    #calculate the log loss
    for i, label in enumerate(labels):
        if(classified[i][label] == 0):
            summing+= np.log(10e-15)
        else:
            summing+= np.log(classified[i][label])
        if labels[i] == np.argmax(classified[i]):
#            print classified[i][np.argmax(classified[i])]
            correct += 1
    print "Calculation finished"  

       
    summing = -summing/len(labels)
    print "log loss: ", summing 
    print "correct/amount_of_labels: ", correct/len(labels)
    print "lowesr classification score: ", np.min(classified)
   
#    print summing
#    np.savetxt( "realLabel.csv", labels, delimiter=";")
#    np.savetxt( "SGD_label.csv", max_SGD, delimiter=";")  
    

    feature_data.close()       

    



if __name__ == '__main__':
    nr_centroids = 100  
    nr_it = 2           # Only used when calc_centroids is True
    clsfr = "SGD"       # Choice between SVC and SGD
    calc_centroids = False # Whether to calculate the centroids, 
                          # do NOT forget to set the nr_centroids to the desired centroidactivation file if False is selected.
    singlePipeline(nr_centroids, nr_it, clsfr=clsfr, calc_centroids = calc_centroids)
