import numpy as np

__author__ = 'Robbert'


#creates an N size array with 3D matrices, K by W by W where K is number of features and W the number of patches per row
#in an image
#patches should be a an ordered raveled list of patches
def extract(patches,W):
    K = len(patches[0])
    conv_vector = [] #np.empty(len(patches)/W^2)
    i = 0
    j = 0
    image = np.zeros((W,W,K))
    for patch in patches:
        if i % W == 0 and i != 0:
            i = 0
            j += 1
        if j % W == 0 and j != 0:
            j = 0
            conv_vector.append(image)
            image = np.zeros((W,W,K))
        image[j,i,:] = patch
        i += 1
    conv_vector.append(image)
    return pool(conv_vector,4)

    #return conv_vector

#For dimension reduction
def pool(convolutional_vector, quadrants):
    pool_vector = []
    for image in convolutional_vector:
        newimage = np.empty((quadrants,quadrants,len(image[0,0])))
        for i in xrange(0,quadrants):
            for j in xrange(0,quadrants):
                newimage[i,j] = poolimage(i,j,image, quadrants)
        pool_vector.append(newimage)
    return pool_vector

#sums parts of image from i to i+quadrant and j to j+quadrant
def poolimage(i,j,image,quadrants):
    step = len(image)/(quadrants/2)
    i = i * step
    j = j * step
    X = np.empty(len(image[0,0]))
    for x in xrange(i,step+i):
        for y in xrange(j,step+j):
            X = np.sum([X,image[x,y]],axis=0)
    return X

