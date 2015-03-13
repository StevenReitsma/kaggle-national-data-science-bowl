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
           
        self.h_image = np.array([
            [0,1,4,3,0],
            [5,0,0,0,2]
        ])
        
        self.v_image = np.array([
            [1,0],
            [0,2],
            [3,4]
        ])
        
    
    def test_horizontalize_image(self):
        p_h_image = imutil.image_horizontal(self.h_image)
        
        expected_p_h_image = self.h_image
        
        self.assertTrue(np.array_equal(p_h_image, expected_p_h_image))    
        
        p_v_image = imutil.image_horizontal(self.v_image)
        
        expected_p_v_image = np.array([
            [0,2,4],
            [1,0,3]
        ])
        
        self.assertTrue(np.array_equal(p_v_image, expected_p_v_image)) 
        
    
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

    def test_sum_images(self):
        images = self.images
        total = imutil.sum_images(images)
        
        expected = np.array([
            [2,2,4],
            [0,6,0],
            [3,3,2]
           ])
        
        self.assertTrue(np.array_equal(total, expected))        
        

if __name__ == '__main__':
    unittest.main()