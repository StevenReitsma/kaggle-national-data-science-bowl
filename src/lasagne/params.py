import glob
import os

IMAGE_SOURCE = "../../data/lasagne"
IM2BIN_OUTPUT = "../../data/lasagne/images.hdf5"

PIXELS = 64
USE_GPU = True

AUGMENTATION_PARAMS = {
			'zoom_range': (1.0, 1.0),
			'rotation_range': (0, 360),
			'shear_range': (0, 0),
			'translation_range': (-5, 5),
			'do_flip': True
		}
		
BATCH_SIZE = 64
START_LEARNING_RATE = 0.01
MOMENTUM = 0.9
