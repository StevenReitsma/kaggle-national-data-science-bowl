# -*- coding: utf-8 -*-

from __future__ import division
import os

import util
import imsquare
import impatch
import imutil

from scipy import misc
import numpy as np
import h5py


"""
Preprocessing script

    1. Load all image paths into memory.
    2. Generate label tuple <classname (plankton type), filename, filepath>
    
    3. For each image:
        1. Load image from disk
        2. Pad or stretch image into squar
        3. Resize (downsize probably) to common size
        4. Patch image
        5. Flatten image from 2D to 1D    
        6. Write the results to file

"""

def preprocess(path='../data/train', 
               outpath="../data/preprocessed.h5", patchsize=6, imagesize=32):
    
    
    print "Patch size: {0}x{0} = {1}".format(patchsize, patchsize*patchsize)
    print "Image size: {0}".format(imagesize)
    
    
    labels = getimagepaths(path) 
    n = len(labels)
    
    print "Amount of images: {0}".format(n)
    
    
    f = h5py.File(outpath, 'w')
    
    # Calculate some dimensions
    patchesperimage = impatch.npatch(imagesize, patchsize)
    patchestotal = n*patchesperimage
    
    print "Patches per image: {0}".format(patchesperimage)
    print "Patches total: {0}".format(patchestotal)
    
    #Dimension of what will be written to file
    dimallpatches = (patchestotal, patchsize*patchsize)
    dsetunordered = f.create_dataset('data', dimallpatches)
    
    for i, (classname, filename, filepath) in enumerate(labels):
        
        image = misc.imread(filepath)
        image, patches = process(image, patchsize=patchsize, imagesize=imagesize)
         
        dsetunordered[i*patchesperimage:i*patchesperimage+len(patches)] = patches
        
        if i % 20 == 0:
            util.update_progress(i/n)
    
    f.close()
        
    util.update_progress(1.0)
   
    

def process(image, squarefunction=imsquare.squarepad, patchsize=6, imagesize=32):
    """
        Process a single image (make square, resize, extract patches, flatten patches)
    """
    
    image = squarefunction(image)
    image = imutil.resizeimage(image, imagesize)
    
    patches = impatch.patch(image, patchsize = patchsize)
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


if __name__ == '__main__':
    preprocess()
