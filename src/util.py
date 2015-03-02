# -*- coding: utf-8 -*-
import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import math

def load_unsupervised(filepath="../data/preprocessed.h5", shuffle=True):
    f = h5py.File(filepath)
    dset = f["data"]    
   # rdata = []
    dimensions = (len(dset), len(dset[0]))
    #dimensions = len(dset)
    rdata = np.zeros(dimensions, dtype=np.uint8)
    
    dset.read_direct(rdata)
    
    if shuffle:
        np.random.shuffle(rdata)
    
    f.close()    
    
    return rdata;


def flatten(collection):
    return [item for sublist in collection for item in sublist]


def normalize(images):
    return [image/float(255) for image in images]
    
    
def plot_mean_image():
    meta = load_metadata()
    mean = meta['mean_image']
    size = int(math.sqrt(len(mean)))
    mean = np.reshape(mean, (size,size))
    plt.imshow(mean)
    

def plot_centroids(centroids, file_path, im_size = (6,6)): 
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print "start plotting"        
    for i, centroid in enumerate(centroids):
        update_progress(i/len(centroids))
        centroid_matrix = np.reshape(centroid, im_size)
        plt.gray()
        plt.imsave(file_path + "centroid" + str(i) + ".png", centroid_matrix)
         
    update_progress(1.0)
    print "plotting done"

def load_metadata(filepath="../data/preprocessed.h5"):
    f = h5py.File(filepath)
    dset = f["data"]  
    
    attributes = {}
    
    # Copy into an in-memory dict
    for key, val in dset.attrs.items():
        attributes[key] = val
    
    f.close()
    return attributes

def load_labels(filepath="../data/preprocessed.h5"):
    f = h5py.File(filepath)
    dset = f["labels"]
    
    return [label for label in dset]


# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
# From http://stackoverflow.com/questions/3160699/python-progress-bar
def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "="*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()
    