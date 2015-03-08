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
               outpath="../data/preprocessed.h5", **kwargs):
    """
    Preprocesses given folder, parameters (all optional):

    Args:
        path (str): Path to folder with plankton images
        outpath (str): File to write to (.h5 file)
        patch_size (int): Width and length of patches
        image_size (int): Width and length to resize images to
        square_method (str): 'pad' or 'stretch', method to make images square.
    """    
    
    
    patch_size = kwargs.get('patch_size', 6)
    image_size = kwargs.get('image_size', 32)
    square_method = kwargs.get('square_method', 'pad')
        
    square_function = imsquare.get_square_function_by_name(square_method)
    
    
    file_metadata = get_image_paths(path)   
    classnames, filenames, filepaths = zip(*file_metadata)  
    
    
    
    label_dict = gen_label_dict(classnames)
    labels = [label_dict[c] for c in classnames]
    class_count = len(label_dict)
    
    # Amount of images
    n = len(file_metadata)
    
    
    # Calculate some dimensions
    patches_per_image = impatch.npatch(image_size, patch_size)
    patches_total = n*patches_per_image
    
    
    print "Patch size: {0}x{0} = {1}".format(patch_size, patch_size**2)
    print "Image size: {0}".format(image_size)
    print "Square method: {0}".format(square_method)
    print "Amount of images: {0}".format(n)
    print "Patches per image: {0}".format(patches_per_image)
    print "Patches total: {0}".format(patches_total)
    print "Labels: {0}".format(len(labels))
    print "Classes count: {0}".format(class_count)
    
    
    metadata = {}
    metadata['patch_size'] = patch_size
    metadata['image_size'] = image_size
    metadata['patches_per_image'] = patches_per_image
    metadata['square_method'] = square_method
    metadata['class_count'] = class_count
    
    if preprocessing_is_already_done(outpath, metadata):
        return
    
    #Dimension of what will be written to file
    dim_all_patches = (patches_total, patch_size**2)
    
    
    f = h5py.File(outpath, 'w')
    #Create dataset (in file)
    dset = f.create_dataset('data', dim_all_patches)

    print "-----------------------------------------"
    print "Writing labels"
    write_labels(labels, f)
    
    
    print "Processing and writing..."
    
    #Running total (sum) of all images
    sum_image = np.zeros(image_size**2)
    
    for i, filepath in enumerate(filepaths):
        
        image = misc.imread(filepath)
        image, patches = process(image, square_function, patch_size, image_size)
        sum_image += imutil.flatten_image(image)
        
        start_index = i*patches_per_image
        dset[start_index:start_index+len(patches)] = patches
        
        if i % 20 == 0:
            util.update_progress(i/n)
    
    util.update_progress(1.0)
    
    mean_image = sum_image/n
    
    
    
    metadata['mean_image'] = mean_image 
    
    print "Writing metadata (options used)" 
    write_metadata(dset, metadata)
    
    f.close()

    
# Returns a dictionary from plankton name to index in ordered, unique set
# of plankton names
def gen_label_dict(classnames):
    unique_list = list(set(classnames));
    unique_list = sorted(unique_list)
    
    label_dict = {cname:i    for i, cname in enumerate(unique_list)}
    
    return label_dict

    
def write_labels(labels, h5py_file):
    h5py_file.create_dataset('labels', data=labels)
    
    

def write_metadata(dataset, metadata):
    for attr in metadata:
        dataset.attrs[attr] = metadata[attr]


def process(image, squarefunction, patch_size, image_size):
    """
        Process a single image (make square, resize, extract patches, flatten patches)
    """
    
    image = squarefunction(image)
    image = imutil.resize_image(image, image_size)
    
    patches = impatch.patch(image, patch_size = patch_size)
    patches = [imutil.flatten_image(patch) for patch in patches]
    return image, patches
    
    
def preprocessing_is_already_done(filepath, metadata):
    print "Checking whether preprocess is already done for given settings"    
    
    f = h5py.File(filepath)
    attrs = f['data'].attrs
    
    for key in metadata:
        
        inFile = attrs.get(key, None)
        inOptions = metadata.get(key, None)   
        
        if not inFile == inOptions:
            print "Found a different setting between file and given options"
            print "Key {0} has value {1} in file, and {2} in options".format(key, inFile, inOptions)
            f.close()
            return False
        
    print "Match between given options and data in file {0}".format(filepath)
    print "Not preprocessing again"
    f.close()
    return True

def get_image_paths(path):
    
    metadata = []    
    
    # The classes are the folders in which the images reside
    classes = os.listdir(path)
    
    for classname in classes:
        for filename in os.listdir(os.path.join(path, classname)):
                filepath = os.path.join(path, classname, filename)
                metadata.append((classname, filename, filepath))
    
    return metadata



if __name__ == '__main__':
    preprocess()
