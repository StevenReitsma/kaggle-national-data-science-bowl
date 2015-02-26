import numpy as np

__author__ = 'Robbert'


#creates an N size array with 3D matrices, K by W by W where K is number of features and W the number of patches per row
#in an image
#patches should be a an ordered raveled list of patches
def extract(patches,W):
    # K = len(patches[0])
    # conv_vector = [] #np.empty(len(patches)/W^2)
    # i = 0
    # j = 0
    # image = np.zeros((W,W,K))
    # for patch in patches:
    #     if i % W == 0 and i != 0:
    #         i = 0
    #         j += 1
    #     if j % W == 0 and j != 0:
    #         j = 0
    #         conv_vector.append(image)
    #         image = np.zeros((W,W,K))
    #     image[j,i,:] = patch
    #     i += 1
    # conv_vector.append(image)
    conv_vector = np.reshape(patches,(W,W,len(patches[0]))) #For one image..
    print conv_vector
    return conv_vector


#For dimension reduction
def pool(convolutional_vector, quadrants):
   pass

#sums parts of image from i to i+quadrant and j to j+quadrant
def poolimage(i,j,image,quadrants):
   pass

