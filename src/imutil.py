# -*- coding: utf-8 -*-
import numpy as np
from scipy import misc
# Misc image functions

def flattenimage(image):
    return np.ravel(image)    
    
    
def resizeimage(image, size=32, interp='bilinear'):
    return ( misc.imresize(image, (size, size), interp))
