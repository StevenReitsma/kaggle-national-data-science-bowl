import numpy as np
from nolearn.lasagne import BatchIterator
from skimage import transform
import skimage
from params import *
import random

class ShufflingBatchIterator(object):
	"""
	Shuffles minibatches.
	"""
	def __init__(self, batch_size):
		self.batch_size = batch_size

	def __call__(self, X, y=None):
		self.X, self.y = X, y
		return self

	def __iter__(self):
		n_samples = self.X.shape[0]
		bs = self.batch_size
		v = range((n_samples + bs - 1) // bs)
		random.shuffle(v)
		for i in v:
			sl = slice(i * bs, (i + 1) * bs)
			Xb = self.X[sl]
			if self.y is not None:
				yb = self.y[sl]
			else:
				yb = None
			yield self.transform(Xb, yb)

	def transform(self, Xb, yb):
		return Xb, yb

class DataAugmentationBatchIterator(ShufflingBatchIterator):
	"""
	Randomly changes images in the batch. Behaviour can be defined in params.py.
	Give mean and std of training set.

	Note: Shuffles minibatches.
	"""
	def __init__(self, batch_size, mean, std):
		super(DataAugmentationBatchIterator, self).__init__(batch_size)
		self.mean = mean
		self.std = std

	def transform(self, Xb, yb):
		Xb, yb = super(DataAugmentationBatchIterator, self).transform(Xb, yb)

		Xbb = np.zeros((Xb.shape[0], Xb.shape[1], Xb.shape[2], Xb.shape[3]), dtype=np.float32)

		IMAGE_WIDTH = PIXELS
		IMAGE_HEIGHT = PIXELS

		random_flip = np.random.randint(2)

		def fast_warp(img, tf, output_shape=(PIXELS,PIXELS), mode='nearest'):
			"""
			This wrapper function is about five times faster than skimage.transform.warp, for our use case.
			"""
			return skimage.transform._warps_cy._warp_fast(img, tf.params, output_shape=output_shape, mode=mode)

		def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=True):
			shift_x = np.random.uniform(*translation_range)
			shift_y = np.random.uniform(*translation_range)

			translation = (shift_x, shift_y)
			rotation = np.random.uniform(*rotation_range)
			shear = np.random.uniform(*shear_range)
			log_zoom_range = [np.log(z) for z in zoom_range]
			zoom = np.exp(np.random.uniform(*log_zoom_range))

			if do_flip and random_flip > 0: # flip half of the time
				shear += 180
				rotation += 180

			return build_augmentation_transform(zoom, rotation, shear, translation)

		center_shift = np.array((IMAGE_HEIGHT, IMAGE_WIDTH)) / 2. - 0.5
		tform_center = transform.SimilarityTransform(translation=-center_shift)
		tform_uncenter = transform.SimilarityTransform(translation=center_shift)

		def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
			tform_augment = transform.AffineTransform(scale=(1/zoom, 1/zoom), 
													  rotation=np.deg2rad(rotation), 
													  shear=np.deg2rad(shear), 
													  translation=translation)
			tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
			return tform

		tform_augment = random_perturbation_transform(**AUGMENTATION_PARAMS)
		tform_identity = skimage.transform.AffineTransform()
		tform_ds = skimage.transform.AffineTransform()

		for i in range(Xb.shape[0]):
			Xbb[i, 0, :, :] = fast_warp(Xb[i][0], tform_ds + tform_augment + tform_identity, output_shape=(PIXELS,PIXELS), mode='nearest').astype('float32')

			# Subtract mean and divide by std
			Xbb[i, 0, :, :] -= self.mean
			Xbb[i, 0, :, :] /= self.std

		return Xbb, yb

class ScalingBatchIterator(BatchIterator):
	"""
	Scales images by subtracting mean and dividing by standard deviation.
	Note: Does not shuffle data.
	"""
	def __init__(self, batch_size, mean, std):
		super(ScalingBatchIterator, self).__init__(batch_size)
		self.mean = mean
		self.std = std

	def transform(self, Xb, yb):
		Xb, yb = super(ScalingBatchIterator, self).transform(Xb, yb)

		Xbb = np.zeros((Xb.shape[0], Xb.shape[1], Xb.shape[2], Xb.shape[3]), dtype=np.float32)

		for i in range(Xb.shape[0]):
			Xbb[i, 0, :, :] = Xb[i, 0, :, :] - self.mean
			Xbb[i, 0, :, :] /= self.std

		return Xbb, yb
