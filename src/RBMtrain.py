from __future__ import print_function

#Author: Robbert v.d. Gugten & Inez Wijnands

#Calculate feature vectors for training set by using an unsupervised RBM

import batchreader
import randombatchreader
import numpy as np
import RBM

#from preprocess import preprocess #For using code from different branch

###############################################################################
# Setting up

def RBMtraining():
    counter = 1
    for data in randombatchreader.RandomBatchReader():
    #for data in batchreader.BatchReader():
        if counter == 1:
            rbm = RBM.RBM(len(data[0]), 100)
        print ("FITTING CHUNK :" + str(counter))
        rbm.train(data,max_epochs=5)
        counter+=1

    print ("DONE")
    print(rbm.weights)

    return rbm

def getWeights(rbm,data):
    return rbm.run_visible(data)


if __name__ == "__main__":
    #Probably load data here!
    #patched, labels, flattened = preprocess("../data/train")
    rbm = RBMtraining()
    for data in batchreader.BatchReader():
    #    print(data)
    #    X_train = [1-(x/float(255)) for x in data]
        weights = getWeights(rbm,data)
        print (weights)
        break






