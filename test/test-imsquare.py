import unittest
import numpy as np
import sys
sys.path.append('../src')
import imsquare as imsquare

class TestImageSquaring(unittest.TestCase):
    
    def setUp(self):
        # Testing variables setup
        self.alreadySquare = [
            [0,0,0],
            [0,0,0],
            [0,0,0]
           ]
        
        self.widerect = [
            [0,0,0],
            [0,0,0]
        ]
        
        self.verywiderect = [
            [0,0,0,0,0],
            [0,0,0,0,0]
        ]
        
        self.tallrect = [
            [0,0],
            [0,0],
            [0,0]
        ]
        
        self.verytallrect = [
            [0,0],
            [0,0],
            [0,0],
            [0,0],
            [0,0]
        ]
    
    
    def test_stretch_square_image(self):
        
        testimages = [self.alreadySquare, self.widerect, self.tallrect]        
        
        for image in testimages:
            self.stretch_test_image(image)
        
        
        
    def stretch_test_image(self, image):
        width = len(image[0])
        height = len(image)
        
        squared = imsquare.squarestretch(image)
        
        expectedsize = max([width, height])

        # Assert that dimensions are correct
        self.assertEquals(len(squared), expectedsize)
        self.assertEquals(len(squared[0]), expectedsize)
        
        
    
        
    def test_pad_square_image(self):
        #Image is already square, output should not change
        
        squared = imsquare.squarepad(self.alreadySquare)
        expected = self.alreadySquare
                
        
        self.assertTrue(np.array_equal(squared, expected))     

    def test_pad_wide_image(self):
        # Pad small non-square image with 1s
        squared = imsquare.squarepad(self.widerect, 1)
        expected = [
            [0,0,0],
            [0,0,0],
            [1,1,1]
        ]      
        
        self.assertTrue(np.array_equal(squared, expected))          
        
        # Test very wide image (requires padding to both top and bottom)
        squared = imsquare.squarepad(self.verywiderect, 1)
        expected = [
            [1,1,1,1,1],
            [0,0,0,0,0],
            [0,0,0,0,0],
            [1,1,1,1,1],
            [1,1,1,1,1]
        ]      
        
        self.assertTrue(np.array_equal(squared, expected))   
        
        
    def test_pad_tall_image(self):
        # Pad small non-square image with 1s
        squared = imsquare.squarepad(self.tallrect, 1)
        expected = [
            [0,0,1],
            [0,0,1],
            [0,0,1]
        ]
        
        self.assertTrue(np.array_equal(squared, expected))          
    
    
        # Pad very tall image (requries padding to both left and right)
        squared = imsquare.squarepad(self.verytallrect, 1)
        expected = [
            [1,0,0,1,1],
            [1,0,0,1,1],
            [1,0,0,1,1],
            [1,0,0,1,1],
            [1,0,0,1,1]
        ]
        
        self.assertTrue(np.array_equal(squared, expected))  
    
    
    def test_calc_pad_size(self):
        # Current width is 1, desired width is 4
        l, r = imsquare.calcpadsize(1, 4)
        # Expected pad sizes
        expectedl = 1
        expectedr = 2
        
        self.assertEquals(expectedl, l)
        self.assertEquals(expectedr, r)
        

if __name__ == '__main__':
    unittest.main();