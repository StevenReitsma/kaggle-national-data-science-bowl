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

__PREPROCESS_VERSION__ = 4


"""
Preprocessing script

    1. Load all image paths into memory.
    2. Generate label tuple <classname (plankton type), filename, filepath>
    
    3. Determine mean, std and variance of all images
    4. Write labels to file
    
    5. For each image:
        1. Load image from disk
        2. Pad or stretch image into squar
        3. Resize (downsize probably) to common size
        4. Normalize image
        5. Patch image
        6. Flatten patches from 2D to 1D    
        7. Write the results to file
        
    6. Write metadata

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
    train_data_file = kwargs.get('train_data_file', '../data/preprocessed.h5')
        
    square_function = imsquare.get_square_function_by_name(square_method)
    
    
    file_metadata, is_train = get_image_paths(path)   
    classnames, filenames, filepaths = zip(*file_metadata)  
    
    
    if is_train:
        label_dict = gen_label_dict(classnames)
        labels = [label_dict[c] for c in classnames]
        class_count = len(label_dict)
    else:
        label_dict = {'UNLABELED':-1}
        labels = [-1 for _ in range(len(classnames))]
        class_count = 0
   
    
    label_names = [key for key in label_dict]
    label_names = np.sort(label_names)
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
    
    metadata['patches_count'] = patches_total
    metadata['image_count'] = n
    metadata['class_count'] = class_count
    
    metadata['train_data'] = is_train
    metadata['version'] = __PREPROCESS_VERSION__
    
    
    
    if preprocessing_is_already_done(outpath, metadata):
        print "----------------------------------------"
        return
    print "----------------------------------------"
    
    # Extract statistics such as the mean/std of image
    # Necessary for normalization of images

    if is_train:
        mean_image, variance_image, std_image = extract_stats(filepaths, image_size, square_function)

    else:
        meta = util.load_metadata(train_data_file)
        mean_image = meta['mean_image']
        std_image = meta['std_image']
        variance_image = meta['var_image']
    
    
    metadata['mean_image'] = mean_image 
    metadata['std_image' ] = std_image
    metadata['var_image' ] = variance_image
    
    print "---"
    #Dimension of what will be written to file
    dim_all_patches = (patches_total, patch_size**2)
    
    #Create file and dataset in file
    f = h5py.File(outpath, 'w')
    dset = f.create_dataset('data', dim_all_patches)

    print "-----------------------------------------"
    print "Writing labels"
    write_labels(labels, f)
    
    print "Writing label names"
    write_label_names(label_names, f)
    
    print "Processing and writing..."
    
    for i, filepath in enumerate(filepaths):
        
        image = misc.imread(filepath)
        image = process(image, square_function, image_size)
        
        # Normalize image     
        image = imutil.normalize(image, mean_image, std_image)
        
        patches = extract_patches(image, patch_size)
        
        
        for j, patch in enumerate(patches):
            mean = np.mean(patch)
            std = np.std(patch)
            patches[j] = imutil.normalize(patch, mean, std)
        
        
        patches = np.nan_to_num(patches)
        
        start_index = i*patches_per_image
        dset[start_index:start_index+len(patches)] = patches
        
        if i % 20 == 0:
            util.update_progress(i/n)
    
    util.update_progress(1.0)
    
    
    
    print "Writing metadata (options used)" 
    write_metadata(dset, metadata)
    
    f.close()
  


def extract_stats(filepaths, image_size, square_function):
    print "Calculating mean, std and var of all images"
    
    #Running total (sum) of all images
    count_so_far = 0
    mean = np.zeros((image_size,image_size))
    M2 = np.zeros((image_size,image_size))    
    
    n = len(filepaths)    
    
    for i, filepath in enumerate(filepaths):
        
        image = misc.imread(filepath)
        image = process(image, square_function, image_size)
        
        # Online statistics
        count_so_far = count_so_far+1
        delta = image - mean
        mean = mean + delta/count_so_far
        M2 = M2 + delta * (image-mean )
    
        if i % 50 == 0:
            util.update_progress(i/n)

    util.update_progress(1.0)
    
    mean_image = mean
    variance_image = M2/(n-1)
    std_image = np.sqrt(variance_image)
    
    print "Plotting mean image (only shows afterwards)"
    util.plot(mean_image, invert=True)
    
    return mean_image, variance_image, std_image
    
    
# Returns a dictionary from plankton name to index in ordered, unique set
# of plankton names
def gen_label_dict(classnames):
    unique_list = list(set(classnames));
    unique_list = sorted(unique_list)
    
    label_dict = {cname:i    for i, cname in enumerate(unique_list)}
    
    return label_dict

    
def write_labels(labels, h5py_file):
    h5py_file.create_dataset('labels', data=labels)
    
def write_label_names(label_names, h5py_file):
    h5py_file.create_dataset('label_names', data=label_names)
    
    

def write_metadata(dataset, metadata):

    for attr in metadata:
        dataset.attrs[attr] = metadata[attr]


def process(image, squarefunction, image_size):
    """
        Process a single image 
        - make horizontal by rotating 90 degrees if necessary
        - make square
        - resize
    """
    image = imutil.image_horizontal(image)
    image = squarefunction(image)
    image = imutil.resize_image(image, image_size)
    
    return image
    
def extract_patches(image, patch_size):
    """
     From image: extract patches, flatten patches
    """
    patches = impatch.patch(image, patch_size = patch_size)
    patches = [imutil.flatten_image(patch) for patch in patches]
    patches = np.array(patches)
    return patches
    
    
def preprocessing_is_already_done(filepath, metadata):
    print "----------------------------------------"
    print "Checking whether preprocess is already done for given settings"    
    
    if not os.path.exists(filepath):
        print "File {0} not found!".format(filepath)
        return False
    
    f = h5py.File(filepath)
    
    if not 'data' in f:
        print "Dataset not found in file"
        f.close()
        return False
        
    
    attrs = f['data'].attrs
    
    for key in metadata:
        
        inFile = attrs.get(key, None)
        inOptions = metadata.get(key, None)   
        
        if not inFile == inOptions:
            print "Found a different setting between file and given options"
            print "Key \"{0}\" has value \"{1}\" in file, and \"{2}\" in options".format(key, inFile, inOptions)
            f.close()
            return False
        
    print "Match between given options and data in file {0}".format(filepath)
    print "Not preprocessing again"
    f.close()
    return True


# Determines whether folder is train or test data
# Returns list of tuples of
# <classname of plankton, image filename, path to file>
#
# This classname is "UNLABELED" for test data
def get_image_paths(path):
    
    is_train = False
    
    for file_or_folder in os.listdir(path):
        if os.path.isdir(os.path.join(path,file_or_folder)):
            is_train = True
            break
    
    if is_train:
        print "Specified folder is train data"
        return get_image_paths_train(path), is_train
    else:
        print "Specified folder is test data"
        return get_image_paths_test(path), is_train

def get_image_paths_test(path):
    metadata = []
    
    classname = "UNLABELED"

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        metadata.append((classname, filename, filepath) )
        
    return metadata
        
    

def get_image_paths_train(path):
    
    metadata = []    
    
    # The classes are the folders in which the images reside
    classes = os.listdir(path)
    
    
    for classname in classes:
        for filename in os.listdir(os.path.join(path, classname)):
                filepath = os.path.join(path, classname, filename)
                metadata.append((classname, filename, filepath))
    
    return metadata


if __name__ == '__main__':
    preprocess(patch_size = 4, image_size = 16, path='../data/test', outpath='../data/preprocessed_test.h5')
