from lasagne import layers
import numpy as np
import theano.tensor as T
from params import *

class SliceLayer(layers.MultipleInputsLayer):
	"""
	This layer optionally takes multiple input layers but it also works with just a single layer.
	This layer augments the batch with the square rotation of every image, optionally also flipped variants.
	The layer multiplies the batch size by a factor of 4, or a factor of 8 when the flip parameter is set to True.

	The part_size arguments configures how large the output images should be.
	"""
	def __init__(self, incomings, part_size, flip, **kwargs):
		if not isinstance(incomings, (list, tuple)):
			incomings = [incomings]

		super(SliceLayer, self).__init__(incomings, **kwargs)

		self.part_size = part_size
		self.flip = flip

	def get_output_shape_for(self, input_shape):
		return (None, 1, self.part_size, self.part_size)

	def get_output_for(self, input, *args, **kwargs):
		parts = []
		ps = self.part_size

		for layer_input in input:
			if self.flip:
				# Second element in the array flips the input image vertically
				flips = [layer_input, layer_input[:, :, :, ::-1]]
			else:
				flips = [layer_input]

			for flip in flips:
				part0 = flip[:, :, :ps, :ps] # 0 degrees
				part1 = flip[:, :, :ps, :-ps-1:-1].dimshuffle(0, 1, 3, 2) # 90 degrees
				part2 = flip[:, :, :-ps-1:-1, :-ps-1:-1] # 180 degrees
				part3 = flip[:, :, :-ps-1:-1, :ps].dimshuffle(0, 1, 3, 2) # 270 degrees

				parts.extend([part0, part1, part2, part3])

		return T.concatenate(parts, axis=0)

class MergeLayer(layers.Layer):
	"""
	This layer merges features from several images in the batch into one.
	Useful for combining the slices that were generated in the SliceLayer.

	The nr_views parameter should be set to 16 if flips are used, 8 if no flips are used.
	Divide these values by two if no 45 degree rotations are used.
	"""
	def __init__(self, incoming, nr_views, **kwargs):
		super(MergeLayer, self).__init__(incoming, **kwargs)

		self.nr_views = nr_views

	def get_output_shape_for(self, input_shape):
		return (None, self.nr_views*int(np.prod(input_shape[1:])))

	def get_output_for(self, input, *args, **kwargs):
		feature_count = int(np.prod(self.input_shape[1:]))
		new_mb_size = input.shape[0] // self.nr_views

		# Extract slices
		input_r = input.reshape((self.nr_views, new_mb_size, feature_count))
		# Shuffle dimensions and create feature vector
		return input_r.dimshuffle(1, 0, 2).reshape((new_mb_size, self.nr_views*feature_count))