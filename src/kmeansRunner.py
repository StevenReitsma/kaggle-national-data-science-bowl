# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 20:42:04 2015

@author: Luc
"""

import kmeans
import activationCalculation as act
import randombatchreader as randbr



def singelPipeline():
    batches = randbr.RandomBatchReader()
    kmTrainer = kmeans.kMeansTrainer()
    centroids = kmTrainer.fit(batches)
    kmTrainer.save_centroids(centroids)
    act_calc = act.ActivationCalculation()
    act_calc.pipeline(centroids)
    




if __name__ == '__main__':
