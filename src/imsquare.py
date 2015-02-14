import numpy as np
import scipy

# Square an image by stretching the shorter dimension
# Possible interpolation values: 'nearest', 'bilinear', 'bicubic'
# or 'cubic'
def squarestretch(image, interp= 'bilinear'):
    
    height = len(image)
    width = len(image[0])    
    
    #Desired width and height length.
    desiredsize = max([height, width])  
    
    return scipy.misc.imresize(image, (desiredsize, desiredsize), interp)
    


# Square an image by padding its sides
def squarepad(image, padvalue=255):
    
    height = len(image)
    width = len(image[0])    
    
    #Desired width and height length.
    desiredsize = max([height, width])
    
    
    if width < desiredsize : # Pad to the left and right
        
        leftlength, rightlength = calcpadsize(width, desiredsize)
        
        
        lpad = np.empty([height, leftlength], dtype=int)
        rpad = np.empty([height, rightlength], dtype=int)          
        
        rpad.fill(padvalue)        
        lpad.fill(padvalue)
        
        # Horizontally stack the paddings around the image
        image = np.hstack((lpad, image, rpad))
        
    if height < desiredsize :  # Pad to the top and bottom
       
        
        toplength, bottomlength = calcpadsize(height, desiredsize)

        tpad = np.empty([toplength, width], dtype=int)
        bpad = np.empty([bottomlength, width], dtype=int)

        tpad.fill(padvalue)
        bpad.fill(padvalue)

        # Vertically stack the paddings around the image
        image = np.vstack((tpad, image, bpad))        
        
    return image

# Returns the pad sizes for either side of the image
def calcpadsize(currentlength, desiredlength):
    padlength = desiredlength - currentlength
    
    l = padlength // 2
    r = padlength - l
    
    
    return l, r
    