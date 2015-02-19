# -*- coding: utf-8 -*-

import os
import sys
import imsquare
import impatch
import scipy
from scipy import misc
import numpy as np
from numpy import random
import h5py

"""
Preprocessing script

    1. Load all images into memory.
    2. Pad or stretch images into square images
    3. Resize (downsize probably) to common size
    4. Patch images

    TODO
    5. Write the results to a file

    Note that patching images eats a lot of memory.

"""

def preprocess(path='../data/train'):


    images, labels = loadimages(path)
    squared = squareimages(images, 'pad')
    resized = resizeimages(squared)
    patched = patchimages(resized)


        
    flattened = flatten(patched)
    print "Length before flattening: {0}, after: {1}".format(len(patched), len(flattened))
    
    print "SHUFFLING"
    random.shuffle(flattened)
    print "DONE"
    return flattened, labels


    print "DONE"
    save(patched, labels, flattened)
    return patched, labels, flattened
    
def save(patches, labels, unordered, filepath="../data/preprocessed.h5"):
    f = h5py.File(filepath, 'w')

    # Dimensions
    dUnordered = (len(unordered), len(unordered[0]), len(unordered[0][0]))
    
    dsetP = f.create_dataset("unordered",dUnordered, dtype=np.int8)
    dsetP[...] = unordered
    
    
def loadunsupervised(filepath="../data/preprocessed.h5"):
    f = h5py.File(filepath)
    dset = f["unordered"]    
    rdata = []
    dset.read_direct(rdata)
    
    return rdata;
    
    
    




def flatten(collection):
    print "FLATTENING"
    return [item for sublist in collection for item in sublist]



def loadimages(path):
    print "LOADING IMAGES INTO MEMORY"

    # Images and labels are kept in seperate lists, this is done for efficiency
    # Being able to process the images sequentially in memory decreases the
    # chance of cache misses
    images = []
    labels = []

    # The classes are the folders in which the images reside
    classes = os.listdir(path)

    for classname in classes:
        print "LOADING ", classname

        for filename in os.listdir(os.path.join(path, classname)):
            image = misc.imread( os.path.join(path, classname, filename) )
            images.append(image)
            labels.append((classname, filename))

    return images, labels




def squareimages(images, approach='pad'):
    print "SQUARING IMAGES"
    squaredimages = []

    if approach == 'pad':
        squarefunction = imsquare.squarepad
    elif approach == 'stretch':
        squarefunction = imsquare.squarestretch
    else:
        raise Exception('Unknown square method!')

    step = len(images)//10

    for i, image in enumerate(images):

        squared = squarefunction(image)
        squaredimages.append(squared)

        if (i % step) == 0: # Print progress
            print '\b.',
            sys.stdout.flush()

    print "DONE"
    return squaredimages


def resizeimages(images, size=32, interp='bilinear'): #Default width and height is 32
    print "RESIZING IMAGES TO", size, "x", size

    resizedimages = []

    step = len(images)//10

    for i, image in enumerate(images):

        resizedimages.append( scipy.misc.imresize(image, (size, size), interp))

        if (i % step) == 0: # Print progress
            print '\b.',
            sys.stdout.flush()

    print "DONE"
    return resizedimages

def patchimages(images, patchsize=6):
    print "PATCHING IMAGES"

    patchedimages = []

    step = len(images)//10

    for i, image in enumerate(images):

        patches = impatch.patch(image)
        patchedimages.append(patches)

        if (i % step) == 0: # Print progress
            print '\b.',
            sys.stdout.flush()

    print "DONE"
    return patchedimages

if __name__ == '__main__':
    preprocess()
