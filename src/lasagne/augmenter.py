import pickle
from pandas import DataFrame
import numpy as np
import copy
from skimage import transform
import skimage
from params import *

class Augmenter():
	def __init__(self, X, rotation = 0, translation = (0,0), zoom = 1.0, shear = 0, do_flip = False, mean = None, std = None):
		self.X = copy.copy(X)

		self.IMAGE_WIDTH = PIXELS
		self.IMAGE_HEIGHT = PIXELS
		self.rotation = rotation
		self.translation = translation
		self.zoom = zoom
		self.shear = shear
		self.do_flip = do_flip
		self.mean = mean
		self.std = std

	def fast_warp(self, img, tf, output_shape=(PIXELS,PIXELS), mode='nearest'):
		"""
		This wrapper function is about five times faster than skimage.transform.warp, for our use case.
		"""
		m = tf.params
		img_wf = np.empty((output_shape[0], output_shape[1]), dtype='float32')
		img_wf = skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode)
		return img_wf

	def _transform(self):
		rotation = self.rotation
		shear = self.shear

		if self.do_flip:
			shear += 180
			rotation += 180

		return self.build_augmentation_transform(self.zoom, rotation, shear, self.translation)


	def build_augmentation_transform(self, zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
		center_shift = np.array((self.IMAGE_HEIGHT, self.IMAGE_WIDTH)) / 2. - 0.5
		tform_center = transform.SimilarityTransform(translation=-center_shift)
		tform_uncenter = transform.SimilarityTransform(translation=center_shift)

		tform_augment = transform.AffineTransform(scale=(1/zoom, 1/zoom), 
												  rotation=np.deg2rad(rotation), 
												  shear=np.deg2rad(shear), 
												  translation=translation)
		tform = tform_center + tform_augment + tform_uncenter 
		return tform


	def transform(self):
		tform_augment = self._transform()
		tform_identity = skimage.transform.AffineTransform()
		tform_ds = skimage.transform.AffineTransform()
		
		for i in range(self.X.shape[0]):
			new = self.fast_warp(self.X[i][0], tform_ds + tform_augment + tform_identity, 
								 output_shape=(PIXELS,PIXELS), mode='nearest').astype('float32')
			if self.mean is not None:
				new -= self.mean
			if self.std is not None:
				new /= self.std
			self.X[i, 0, :, :] = new

		return self.X