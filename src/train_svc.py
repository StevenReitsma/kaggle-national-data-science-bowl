# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:44:05 2015

@author: Luc
"""

from sklearn import svm
from sklearn.externals import joblib
import os

def train_svc(samples, labels, nr_centroids, degree=3, cache_size=40000):
    clf = svm.SVC(degree=degree, cache_size = cache_size, probability = True, verbose = 1)
    clf.fit(samples, labels)
    
    file_path = '../models/svc' + str(nr_centroids) + '/'
    if not os.path.exists(file_path):
              os.makedirs(file_path)
          
    joblib.dump(clf, file_path + '/classifier.pkl')
    return clf 

    