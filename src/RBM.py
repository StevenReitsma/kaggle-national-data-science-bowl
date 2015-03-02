from __future__ import print_function

#Author: Robbert v.d. Gugten & Inez Wijnands

#Calculate feature vectors for training set by using an unsupervised RBM

import batchreader
import randombatchreader
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM

from sklearn.pipeline import Pipeline
#from preprocess import preprocess #For using code from different branch

###############################################################################
# Setting up

def RBMtraining():
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = 5
    rbm.n_components = 100
    counter = 1
    for data in randombatchreader.RandomBatchReader():
        #Scale all grey values to probabilities between 0 and 1
        X_train = [1-(x/float(255)) for x in data]
    #    print(X_train)
        print ("FITTING CHUNK :" + str(counter))
        rbm.partial_fit(X_train)
        counter+=1

    print ("DONE")
    return rbm

def getWeights(rbm,data):
    return rbm.transform(data)


if __name__ == "__main__":
    #Probably load data here!
    #patched, labels, flattened = preprocess("../data/train")
    rbm = RBMtraining()
    for data in batchreader.BatchReader():
    #    print(data)
        X_train = [1-(x/float(255)) for x in data]
        weights = getWeights(rbm,X_train)
        print (weights)






