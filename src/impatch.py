# -*- coding: utf-8 -*-
import random

def patch(image, n=0, patchsize=6):    
    """
       Patches an image (samples sub-images)
    """
    
    patches = []
    
    xlength = len(image[0])
    ylength = len(image)
    
    if patchsize > xlength or patchsize > ylength:
        raise Exception("Patchsize too big for given image")


    # Max top left index from which patches are taken        
    xindexmax = xlength - patchsize    
    yindexmax = ylength - patchsize 
    
    nmaxpatches = (xindexmax+1) * (yindexmax+1)
    
    
    
    if n > nmaxpatches:
        raise Exception("Impossible to extract this many patches from image")
        
    if n == 0:
        n = nmaxpatches
        
    coords = [(x,y) for x in range(xindexmax+1) for y in range(yindexmax+1)]
    
    # Shuffle list of coords
    random.shuffle(coords)
    
    
    for i, coord in enumerate(coords):
        if i >= n:
            break
        
        x, y = coord

        patch = image[x:(x+patchsize),y:(y+patchsize)]       
        patches.append(patch)
    
    return patches
    
    
def npatch(imagesize, patchsize):
    return (imagesize + 1 - patchsize) * (imagesize + 1 - patchsize)
    """
        Maximum amount of patches extracted from image given size
    """