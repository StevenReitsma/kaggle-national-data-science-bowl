from __future__ import print_function

#Author: Robbert v.d. Gugten & Inez Wijnands

#Calculate feature vectors for training set by using an unsupervised RBM

import batchreader
import randombatchreader
import numpy as np
import RBM
import pooling
import os
import util


#from preprocess import preprocess #For using code from different branch

_hidden_units = 300

def RBMtraining():
    counter = 1
    for data in randombatchreader.RandomBatchReader(batchsize = 200*529):
    #for data in batchreader.BatchReader():
        if counter == 1:
            rbm = RBM.RBM(len(data[0]), _hidden_units)
        print ("FITTING CHUNK :" + str(counter))
    #    data2 = np.array([1-(x/float(255)) for x in data])
    #    print(data)
        rbm.train(data,max_epochs=5)
        counter+=1

    print ("DONE")
    #print(rbm.weights)
    plotweights = np.transpose(rbm.weights[1:,1:])
    save_weights(plotweights, file_path = "../data/weightsrbm/")
    util.plot_centroids(plotweights, file_path = "../data/weightsrbm/" )

    return rbm

def save_weights(weights, file_path = "../data/weightsrbm/"):
    if not os.path.exists(file_path):
        os.makedirs(file_path)       
    np.savetxt(file_path + str(len(weights)) +  "weights.csv", weights, delimiter=",")
    
#Finds the hidden layer probability values belonging to data
def transform(rbm,data):
    return rbm.run_visible(np.array(data))

def logistic(x):
    return 1.0 / (1 + np.exp(-x))

def train():
    #Probably load data here!
    #patched, labels, flattened = preprocess("../data/train")
    rbm = RBMtraining()
    pooled_images = []
    imagesdone = 0
    for data in batchreader.BatchReader(batchsize=50*529):
    #    print(data)
#        X_train = [1-(x/float(255)) for x in data]
        weights = transform(rbm,data)
        #mean = np.mean(weights, axis=0)
        #std = np.std(weights, axis=0)
        #weights = (weights -  mean) / std
        for i in range(0,len(data)/529):
            dim = weights[i*529:i*529+529,1:]
            #print(dim.shape)
            feature_vector = pooling.pool(dim)
            pooled_images.append(feature_vector)
#            print(feature_vector)
            if i % 499 == 0 and i != 0:
                print(str(imagesdone) + " images done")
                imagesdone+=500
#        break
    return pooled_images
