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
        
        flat = imutil.flatten_image(self.images[0])  
        expected = np.array( [1,2,3,0,5,0,2,3,1])
        
        self.assertTrue(np.array_equal(flat, expected))         
        
    def test_resize_images(self):
        
        im = self.images[0]
        
        resized = imutil.resize_image(im, 64)
        
        # Square
        self.assertEqual(len(resized), len(resized[0]))
        
        # Right size
        self.assertEqual(len(resized), 64)

if __name__ == '__main__':
    unittest.main()