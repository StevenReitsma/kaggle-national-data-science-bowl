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
    patches_per_image = impatch.npatch(imagesize, patchsize)
    patches_total = n*patches_per_image
    
    print "Patches per image: {0}".format(patches_per_image)
    print "Patches total: {0}".format(patches_total)
    
    #Dimension of what will be written to file
    dimallpatches = (patches_total, patchsize*patchsize)
    
    #Create dataset (in file)
    dset = f.create_dataset('data', dimallpatches)

    #Write metadata to file (options)
    write_metadata(dset, 
                   patch_size = patchsize, 
                   image_size = imagesize, 
                   patches_per_image=patches_per_image)
    
    
    for i, (classname, filename, filepath) in enumerate(labels):
        
        image = misc.imread(filepath)
        image, patches = process(image, patchsize=patchsize, imagesize=imagesize)
         
        dset[i*patches_per_image:i*patches_per_image+len(patches)] = patches
        
        if i % 20 == 0:
            util.update_progress(i/n)
    
    f.close()
    
    util.update_progress(1.0)
   
def write_metadata(dataset, patch_size=6, image_size=32, patches_per_image=729):
    dataset.attrs['patch_size'] = patch_size
    dataset.attrs['image_size'] = image_size
    dataset.attrs['patches_per_image'] = patches_per_image

def process(image, squarefunction=imsquare.squarepad, patchsize=6, imagesize=32, ):
    """
        Process a single image (make square, resize, extract patches, flatten patches)
    """
    
    image = squarefunction(image)
    image = imutil.resizeimage(image, imagesize)
    
    patches = impatch.patch(image, patchsize = patchsize)
    patches = [imutil.flattenimage(patch) for patch in patches]
    return image, patches
    

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
