import unittest
import numpy as np
import sys
sys.path.append('../src')
import pooling

class TestPooling(unittest.TestCase):

    def setUp(self):
        # Testing variables setup
        # Each row is a patch, each column a feature
        self.testPatches = np.array([
            [1,2,3,4,5],
            [6,7,8,9,10],
            [11,12,13,14,15],
            [16,17,18,19,20],
            [21,22,23,24,25],
            [26,27,28,29,30],
            [31,32,33,34,35],
            [36,37,38,39,40],
            [41,42,43,44,45],
            [46,47,48,49,50],
            [51,52,53,54,55],
            [56,57,58,59,60],
            [61,62,63,64,65],
            [66,67,68,69,70],
            [71,72,73,74,75],
            [76,77,78,79,80]
           ])

        self.result = np.array([
            54, 58, 62, 66, 70, 94, 98, 102, 106, 110, 214, 218, 222, 226, 230, 254, 258, 262, 266, 270
        ])

    def test_pooling(self):
        pooled = pooling.pool(self.testPatches, n_pool_regions = 4, operator = np.sum)
        self.assertEquals(pooled.shape, (20L,))
        self.assertTrue(np.array_equal(pooled, self.result))

if __name__ == '__main__':
    unittest.main()