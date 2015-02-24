# -*- coding: utf-8 -*-

import unittest
import sys
sys.path.append('../src')
import preprocess

class TestPreprocess(unittest.TestCase):
    
    def setUp(self):
        # Testing variables setup
        self.classnames = ['a', 'b', 'c']
        
    
    def test_generate_label_dict(self):
        
        label_dict = preprocess.gen_label_dict(self.classnames)
        
        self.assertTrue(label_dict['a'] == 0)
        self.assertTrue(label_dict['b'] == 1)
        
        self.assertEqual(len(label_dict), 3)

if __name__ == '__main__':
    unittest.main()