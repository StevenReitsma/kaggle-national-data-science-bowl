# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:19:54 2015

@author: Luc
"""
from __future__ import division
from pooling import pool
import activationCalculation as ac
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import util


list0 = []
list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []
list9 = []


def append_0(image):
    list0.append(image)

def append_1(image):
    list1.append(image)

def append_2(image):
    list2.append(image)

def append_3(image):
    list3.append(image)
    
def append_4(image):
    list4.append(image)
    
def append_5(image):
    list5.append(image)
    
def append_6(image):
    list6.append(image)
    
def append_7(image):
    list7.append(image)
    
def append_8(image):
    list8.append(image)
    
def append_9(image):
    list9.append(image)

store = {
0 : append_0,
1 : append_1,
2 : append_2,
3 : append_3,
4 : append_4,
5 : append_5,
6 : append_6,
7 : append_7,
8 : append_8,
9 : append_9,
}    


def save_as_image(image, label, nr, file_path = '../mnist/'):
    if not os.path.exists(file_path + str(label)):
        os.makedirs(file_path + str(label))
    
    image = np.reshape(image, (28,28))
    plt.gray()
    plt.imsave(file_path + str(label) + "/" + str(nr), image)   

        

def read_csv(file_path = '../data/mnist/train.csv'):
    f = open(file_path)
    try:
        reader = csv.reader(f)
        reader.next() # skip header
        util.update_progress(0)
        for i, row in enumerate(reader):
            label = int(row[0])
            image = np.delete(row, [0])
            save_as_image(image, label, i)
            util.update_progress(i/42000)
    finally:
        f.close()
        util.update_progress(1)
        
def read_test_csv(file_path = '../data/test.csv'):
    f = open(file_path)
    try:
        reader = csv.reader(f)
        reader.next()
        util.update_progress(0)
        for i, row in enumerate(reader):
            image = np.array(row)
            save_as_image(image, "testset",i , file_path = '../data/')
            util.update_progress(i/28000)
            
    finally:
        f.close()
        util.update_progress(1)


if __name__ == '__main__':
    
    calculator = ac.ActivationCalculation() 
    patch = np.array(range(1,5))
    patcheslist = np.zeros((16,4))
    patches = np.zeros((16,4))
    for i in range(16):
        patcheslist[i] = patch
        patch+=4
    
    patches[0] = patcheslist[0]
    patches[1] = patcheslist[1]
    patches[2] = patcheslist[4]
    patches[3] = patcheslist[5]
    patches[4] = patcheslist[2]
    patches[5] = patcheslist[3]
    patches[6] = patcheslist[6]
    patches[7] = patcheslist[7]
    patches[8] = patcheslist[8]
    patches[9] = patcheslist[9]
    patches[10] = patcheslist[12]
    patches[11] = patcheslist[13]
    patches[12] = patcheslist[10]
    patches[13] = patcheslist[11]
    patches[14] = patcheslist[14]
    patches[15] = patcheslist[15]
    
    print patches
    
    centroids = np.zeros((2,4))

    centroids[0] = patches[7]
    centroids[1] = patches[11]

    
    print centroids
    
    activations = calculator.distance_to_centroids(patches, centroids)
    pooled = pool(activations)
    print pooled
    
    
#    patch = np.array(range(1,5))
#    patches = np.zeros((5,4))
#    for i in range(4):
#        patches[i] = patch
#        patch+=4
#    
#    centroids = np.zeros((3,4))
#    centroid = np.array(range(5,9))        
#    for i in range(3):
#        centroids[i] = centroid
#        centroid+=4
#    
#    activations = calculator.distance_to_centroids(patches, centroids)
#    pool(activations)
#    print activations
#    array = np.array(range(16))+1
#    array = np.array(range(64))+1    
#
#    matrix = np.zeros((16,16))
#    for i in range(16):
#        matrix[i] = array
#        array+=1
#
#    pooled = pool(matrix)
#    print pooled
#    save_as_image(list0, 0)
#    save_as_image(list1, 1)
#    save_as_image(list2, 2)
#    save_as_image(list3, 3)
#    save_as_image(list4, 4)
#    save_as_image(list5, 5)
#    save_as_image(list6, 6)
#    save_as_image(list7, 7)
#    save_as_image(list8, 8)
#    save_as_image(list9, 9)
    
    
        
#    f = open('../data/mnist/train.csv', 'rt')
#    try:
#        reader = csv.reader(f)
#        reader.next() #skip header
#        reader.next()
#        data = reader.next()
#        data = np.array(data)
#        print data[0]
#        image = np.delete(data, [0])
#        print image[0]
#        image.resize(28,28)
#        print image
#        plt.gray()
#        plt.imsave('../2', image)
#        
#    finally:
#        	f.close()    
