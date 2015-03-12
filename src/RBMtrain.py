from __future__ import print_function

#Author: Robbert v.d. Gugten & Inez Wijnands

#Calculate feature vectors for training set by using an unsupervised RBM

import batchreader
import randombatchreader
import os
import util
import numpy as np
import RBM

#from preprocess import preprocess #For using code from different branch

_hidden_units = 100

#
def RBMtraining():
    counter = 1
    for data in randombatchreader.RandomBatchReader():
    #for data in batchreader.BatchReader():
        if counter == 1:
            rbm = RBM.RBM(len(data[0]), _hidden_units)
        print ("FITTING CHUNK :" + str(counter))
        data2 = np.array([1-(x/float(255)) for x in data])
    #    print(data)
        rbm.train(data2,max_epochs=10)
        counter+=1
        if counter == 3:
            break
    print ("DONE")
    print(rbm.weights)
    rbm.save_weights(rbm.weights)
    #util.plot_centroids(centroids = rbm.weights, file_path = "../data/weightsrbm/")

    return rbm

#Finds the hidden layer probability values belonging to data
#TODO find the correct bias!!
def transform(rbm,data):
    X = []
    for dat in data:
        Y = np.zeros(_hidden_units)
        for i in range(_hidden_units):
            Y[i] = logistic(rbm.weights[0,i] + sum(rbm.weights[1:,i] * dat))
#           print(rbm.weights[1:,i])
#           print(dat)
        X.append(Y)
    return X

def logistic(x):
    return 1.0 / (1 + np.exp(-x))
    
def save_weights(weights, file_path = "../data/weightsrbm/"):
    if not os.path.exists(file_path):
        os.makedirs(file_path)       
    np.savetxt(file_path + "weights.csv", weights, delimiter=",")

if __name__ == "__main__":
    #Probably load data here!
    #patched, labels, flattened = preprocess("../data/train")
    rbm = RBMtraining()
    for data in batchreader.BatchReader():
    #    print(data)
        X_train = [1-(x/float(255)) for x in data]
        weights = transform(rbm,X_train)
#       now pooling! TODO
        print (weights)
#        break






