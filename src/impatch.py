# -*- coding: utf-8 -*-
import random

# Hack to improve memory usage for 32x32sized images
coords32 = [(x,y) for x in range(32-6+1) for y in range(32-6+1)]


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
    
    maxpatches = (xindexmax+1) * (yindexmax+1)
    
    
    
    if n > maxpatches:
        raise Exception("Impossible to extract this many patches from image")
        
    if n == 0:
        n = maxpatches
        
    # Hack to improve memory usage for 32x32sized images
    # Coord list doesn't have to be created for every image
    # TODO: Work for lengths that are not 32
    if xlength == 32 and ylength == 32:
        coords = coords32
    else:
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
    
    
    
    
    
    
    