from __future__ import print_function

#Author: Robbert v.d. Gugten & Inez Wijnands

#Returns probability according to RBM for every class.
#print(__doc__)


import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


###############################################################################
# Setting up


def RBMtraining(X_train):


    # Models we will use
    #logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    #logistic.C = 6000.0
    rbm.fit(X_train)
    #classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    ###############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.


    # Training RBM-Logistic Pipeline
    #classifier.fit(X_train, Y_train)

    #print(classifier.predict_proba(X_test)) #Outputs probability for every class in matrix/array
    #return(classifier.predict_proba(X_test))
    rbm.gibbs(X_train)
    return rbm.intercept_hidden_




