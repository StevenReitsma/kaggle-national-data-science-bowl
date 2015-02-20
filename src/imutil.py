# -*- coding: utf-8 -*-
import numpy as np
import scipy

def flattenimage(image):
    return np.ravel(image)    
    
    
def resizeimage(image, size=32, interp='bilinear'):
    return ( scipy.misc.imresize(image, (size, size), interp))
