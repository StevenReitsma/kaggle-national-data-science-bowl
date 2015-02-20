# -*- coding: utf-8 -*-

import unittest
import numpy as np
import sys
sys.path.append('../src')
import imutil

class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        # Testing variables setup
        self.images = np.array([[
            [1,2,3],
            [0,5,0],
            [2,3,1]
           ],[
            [1,0,1],
            [0,1,0],
            [1,0,1]
           ]])
        
    
    def test_flatten_images(self):
        
        flat = imutil.flattenimages(self.images)  
        expected = np.array( [ [1,2,3,0,5,0,2,3,1],[1,0,1,0,1,0,1,0,1] ])
        
        self.assertTrue(np.array_equal(flat, expected))         
        

if __name__ == '__main__':
    unittest.main()