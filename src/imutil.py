# -*- coding: utf-8 -*-
import numpy as np
from scipy import misc
# Misc image functions

def flatten_image(image):
    return np.ravel(image)    
    
    
def resize_image(image, size=32, interp='bilinear'):
    return ( misc.imresize(image, (size, size), interp))

def sum_images(sequence_of_images):
    return sum(sequence_of_images)