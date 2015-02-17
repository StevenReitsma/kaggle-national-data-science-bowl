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

# def nudge_dataset(X, Y):
#     """
#     This produces a dataset 5 times bigger than the original one,
#     by moving the 8x8 images in X around by 1px to left, right, down, up
#     """
#     direction_vectors = [
#         [[0, 1, 0],
#          [0, 0, 0],
#          [0, 0, 0]],
#
#         [[0, 0, 0],
#          [1, 0, 0],
#          [0, 0, 0]],
#
#         [[0, 0, 0],
#          [0, 0, 1],
#          [0, 0, 0]],
#
#         [[0, 0, 0],
#          [0, 0, 0],
#          [0, 1, 0]]]
#
#     shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
#                                   weights=w).ravel()
#     X = np.concatenate([X] +
#                        [np.apply_along_axis(shift, 1, X, vector)
#                         for vector in direction_vectors])
#     Y = np.concatenate([Y for _ in range(5)], axis=0)
#     return X, Y

# Load Data

def RBMtrain(X_train, X_test, Y_train):
    # digits = datasets.load_digits()
    # X = np.asarray(digits.data, 'float32')
    # Y = digits.target
    # X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    #
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
    #                                                     test_size=0.2,
    #                                                     random_state=0)

    # Models we will use
    logistic = linear_model.LogisticRegression()
    rbm = BernoulliRBM(random_state=0, verbose=True)

    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    ###############################################################################
    # Training

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    logistic.C = 6000.0

    # Training RBM-Logistic Pipeline
    classifier.fit(X_train, Y_train)

    print(classifier.predict_proba(X_test)) #Outputs probability for every class in matrix/array
    return(classifier.predict_proba(X_test))



