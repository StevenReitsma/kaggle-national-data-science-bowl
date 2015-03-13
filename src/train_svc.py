# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:44:05 2015

@author: Luc
"""

from sklearn import svm

def train_svc(samples, labels):
    clf = svm.SVC(degree=1, cache_size = 4000, probability = True, verbose = 2)
    clf.fit(samples, labels)
    return clf 


def train_linearSVC(samples, labels):
    clf = svm.LinearSVC(fit_intercept = False, verbose = 1)
    