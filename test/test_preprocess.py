# -*- coding: utf-8 -*-

import unittest
import sys
sys.path.append('../src')
import preprocess
import imsquare
import numpy as np

class TestPreprocess(unittest.TestCase):
    
    def setUp(self):
        # Testing variables setup
        self.classnames = ['a', 'b', 'c']
        
        self.image = np.array([
            [0,1,2,3,4],
            [5,6,7,8,9]
        ])
        
    
    def test_generate_label_dict(self):
        
        label_dict = preprocess.gen_label_dict(self.classnames)
        
        self.assertTrue(label_dict['a'] == 0)
        self.assertTrue(label_dict['b'] == 1)
        
        self.assertEqual(len(label_dict), 3)

    #Mini integration test
    def test_process_image(self):
        patch_size = 8
        resize_size = 64     
        square_function = imsquare.square_pad 
        
        expected_patches_amount = (resize_size-patch_size+1)**2 # = 3249
        
        processed = preprocess.process(self.image, 
                                                square_function, 
                                                resize_size)
                                                
        patches = preprocess.extract_patches(processed, patch_size)
        
        self.assertEqual(expected_patches_amount, len(patches))
        self.assertEqual( (64,64), np.shape(processed))
        
        for patch in patches:
            #Patches should be onedimensional
            self.assertEqual(patch_size**2, len(patch))

if __name__ == '__main__':
    unittest.main()