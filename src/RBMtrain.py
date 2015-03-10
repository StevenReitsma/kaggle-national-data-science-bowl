from __future__ import print_function

#Author: Robbert v.d. Gugten & Inez Wijnands

#Calculate feature vectors for training set by using an unsupervised RBM

import batchreader
import randombatchreader
import numpy as np
import RBM
import pooling

#from preprocess import preprocess #For using code from different branch

_hidden_units = 150

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
        rbm.train(data2,max_epochs=4)
        counter+=1
        if counter == 3:
            break
    print ("DONE")
    print(rbm.weights)

    return rbm

#Finds the hidden layer probability values belonging to data
#TODO find the correct bias!!
def transform(rbm,data):
#    X = []

#     for dat in data:
#         Y = np.zeros(_hidden_units)
#         for i in range(_hidden_units):
#             Y[i] = logistic(rbm.weights[0,i] + sum(rbm.weights[1:,i] * dat))
# #           print(rbm.weights[1:,i])
# #           print(dat)
#         X.append(Y)
    return rbm.run_visible(np.array(data))

def logistic(x):
    return 1.0 / (1 + np.exp(-x))

if __name__ == "__main__":
    #Probably load data here!
    #patched, labels, flattened = preprocess("../data/train")
    rbm = RBMtraining()
    for data in batchreader.BatchReader():
    #    print(data)
        X_train = [1-(x/float(255)) for x in data]
        weights = transform(rbm,X_train)
        mean = np.mean(weights, axis=0)
        std = np.std(weights, axis=0)
        weights = weights -  mean / std
        pooled_images = []
        for i in range(0,len(data)/729):
#           now pooling! TODO
            dim = weights[i*729:i*729+729,:]
            feature_vector = pooling.pool(dim)
            pooled_images.append(feature_vector)
#            print(feature_vector)
#        break






