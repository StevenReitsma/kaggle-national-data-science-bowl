# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 20:42:04 2015

@author: Luc
"""

import kmeans
import activationCalculation as act

if __name__ == '__main__':
    kmTrainer = kmeans.kMeansTrainer()
    centroids = kmTrainer.fit()
    kmTrainer.save_centroids(centroids)
    actCalc = act.ActivationCalculation()
    activation = 