import glob
import os

IMAGE_SOURCE = "../../data/lasagne"
IM2BIN_OUTPUT = "../../data/lasagne/images.hdf5"

PIXELS = 64
USE_GPU = True

AUGMENTATION_PARAMS = {
			'zoom_range': (1/1.1, 1.1),
			'rotation_range': (0, 360),
			'shear_range': (0, 0),
			'translation_range': (-10, 10),
			'do_flip': True
		}
