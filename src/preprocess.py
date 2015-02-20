# -*- coding: utf-8 -*-

from __future__ import division
import os
import sys

import imsquare
import impatch
import imutil

from scipy import misc
import numpy as np
import h5py


"""
Preprocessing script

    1. Load all images into memory.
    
    2. For each image:
        1. Pad or stretch image into squar
        2. Resize (downsize probably) to common size
        3. Patch image
        4. Flatten image from 2D to 1D    
    
    3. Write the results to a file

    Note that patching images eats a lot of memory.

"""

def preprocess(path='../data/train', writeToFile=True):
    
    labels = getimagepaths(path)
    processedimages = []    
    allpatches = []
    
    n = len(labels)
    
    for i, (classname, filename, filepath) in enumerate(labels):
        image = misc.imread(filepath)
        image, patches = process(image)
        processedimages.append(image)
        allpatches.append(patches)
        
        if i % 20 == 0:
            update_progress(i/n)
        
    update_progress(1.0)
   
    

    
def process(image, squarefunction=imsquare.squarepad):
    """
        Process a single image (make square, resize, extract patches, flatten patches)
    """
    
    image = squarefunction(image)
    image = imutil.resizeimage(image)
    
    patches = impatch.patch(image)
    patches = [imutil.flattenimage(patch) for patch in patches]
    
    return image, patches
    
    

def save(patches, labels, unordered, filepath="../data/preprocessed.h5"):
    f = h5py.File(filepath, 'w')

    # Dimensions
    dUnordered = (len(unordered), len(unordered[0]))
    
    dsetP = f.create_dataset("unordered",dUnordered, dtype=np.uint8)
    dsetP[...] = unordered

def getimagepaths(path):
    
    labels = []
    
    # The classes are the folders in which the images reside
    classes = os.listdir(path)
    
    for classname in classes:
        for filename in os.listdir(os.path.join(path, classname)):
                filepath = os.path.join(path, classname, filename)
                labels.append((classname, filename, filepath))
    
    return labels


# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
# From http://stackoverflow.com/questions/3160699/python-progress-bar
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
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
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


if __name__ == '__main__':
    preprocess()
