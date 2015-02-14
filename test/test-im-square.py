import unittest
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
        
        self.tallrect = [
            [0,0],
            [0,0],
            [0,0]
        ]
        
        
        
    def test_pad_square_image(self):
        #Image is already square, output should not change
        
        squared = imsquare.squarepad(self.alreadySquare)
        self.assertEqual(self.alreadySquare, squared)

    def test_pad_wide_image(self):
        # Pad non-square image with 1s
        squared = imsquare.squarepad(self.widerect, 1)
        
        expected = [
            [0,0,0],
            [0,0,0],
            [1,1,1]
        ]      
        
        self.assertEqual(squared, expected)        
        
    def test_pad_tall_image(self):
        # Pad non-square image with 1s
        squared = imsquare.squarepad(self.tallrect, 1)
        
        expected = [
            [0,0,1],
            [0,0,1],
            [0,0,1]
        ]      
        
        self.assertEqual(squared, expected)     
    
    




if __name__ == '__main__':
    unittest.main();