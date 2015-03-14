# -*- coding: utf-8 -*-
import random

def patch(image, n=0, patch_size=6):    
    """
       Patches an image (samples sub-images)
    """
    
    patches = []
    
    xlength = len(image[0])
    ylength = len(image)
    
    if patch_size > xlength or patch_size > ylength:
        raise Exception("Patchsize too big for given image")


    # Max top left index from which patches are taken        
    xindexmax = xlength - patch_size    
    yindexmax = ylength - patch_size 
    
    nmaxpatches = (xindexmax+1) * (yindexmax+1)
    
    
    
    if n > nmaxpatches:
        raise Exception("Impossible to extract this many patches from image")
        
    if n == 0:
        n = nmaxpatches
        
    coords = [(x,y) for x in range(xindexmax+1) for y in range(yindexmax+1)]
    
    # Shuffle list of coords
    #random.shuffle(coords)
    
    
    for i, coord in enumerate(coords):
        if i >= n:
            break
        
        x, y = coord

        patch = image[x:(x+patch_size),y:(y+patch_size)]       
        patches.append(patch)
    
    return patches
    
    
def npatch(imagesize, patchsize):
    return (imagesize + 1 - patchsize)**2
    """
        Maximum amount of patches extracted from image given size
    """