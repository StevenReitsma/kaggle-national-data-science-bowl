from __future__ import print_function

#Author: Robbert v.d. Gugten & Inez Wijnands

#Calculate feature vectors for training set by using an unsupervised RBM


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

def RBMtraining(X_train):


    #Scale all grey values to probabilities between 0 and 1
    X_train = [x/float(255) for x in X_train]
    #Flatten images, probably not necessary after new preprocess function
    X = []
    for x in X_train:
        X.append(np.ravel(x))


    #print (X_train)
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = 3
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    #logistic.C = 6000.0
    print ("FITTING")
    rbm.fit(X)
    print ("DONE")

    ###############################################################################
    #supervised:

    #classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])



    # Training RBM-Logistic Pipeline
    #classifier.fit(X_train, Y_train)

    #print(classifier.predict_proba(X_test)) #Outputs probability for every class in matrix/array
    #return(classifier.predict_proba(X_test))

    ###############################################################################

    #For every patch the belonging feature vector.
    return rbm.transform(X)


if __name__ == "__main__":
    #Probably load data here!
    patched, labels, flattened = preprocess("../data/train")
    weights = RBMtraining(flattened)
    print (len(weights))
    print (weights)






