# -*- coding: utf-8 -*-
import unittest
import numpy as np
import sys
sys.path.append('../src')
import impatch

class TestImageSquaring(unittest.TestCase):
    
    def setUp(self):
        # Testing variables setup
        self.testImage = np.array([
            [1,2,3,4,5],
            [6,7,8,9,0],
            [0,0,0,0,0],
            [0,9,8,7,6],
            [5,4,3,2,1]
           ])
           
                   
    
    
    def test_image_patch_amount(self):
        patches = impatch.patch(self.testImage, n=7, patchsize=2)        
        
        #Requested 7 patches
        self.assertEqual(len(patches), 7)
        
    def test_image_patch_all(self):
        patches = impatch.patch(self.testImage, patchsize=2)
        
        #16 patches of size 2x2 in test image
        self.assertEqual(len(patches), 16) 
       
    def test_image_patch_size(self):
        patches = impatch.patch(self.testImage, patchsize=3)
       
        for patch in patches:
            # X, correct width
            self.assertEqual(len(patch[0]), 3) 
            
            # Y, correct height
            self.assertEqual(len(patch), 3)       
       

if __name__ == '__main__':
    unittest.main()
