__author__ = 'Robbert'
# -*- coding: utf-8 -*-
import unittest
import numpy as np
import sys
sys.path.append('../src')
import convextraction

class TestConvExtraction(unittest.TestCase):

    def setUp(self):
        # Testing variables setup
        self.testImage = np.array([
            [1,2,3,4,5],
            [6,7,8,9,0],
            [0,9,8,7,6],
            [5,4,3,2,1],
            [1,2,3,4,5],
            [6,7,8,9,0],
            [0,9,8,7,6],
            [5,4,3,2,1]
           ])




    def test_conv(self):
        self.test = convextraction.extract(self.testImage,2)
        result = np.zeros((2,2,5))
        result[0,0,:] = np.array([1,2,3,4,5])
        result[0,1,:] = np.array([6,7,8,9,0])
        result[1,0,:] = np.array([0,9,8,7,6])
        result[1,1,:] = np.array([5,4,3,2,1])
        self.assertEqual(self.test, result)

    def test_pool(self):
        pass


if __name__ == '__main__':
    unittest.main()
