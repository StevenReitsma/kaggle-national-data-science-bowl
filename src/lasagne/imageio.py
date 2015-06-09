import numpy as np
from skimage.io import imread
from sklearn import preprocessing
import h5py
from params import *
from sklearn.utils import shuffle
from augmenter import Augmenter
import glob
import os

class ImageIO():
	def __init__(self):
		self.scaler = preprocessing.StandardScaler()

	def _load_train_images_from_disk(self):
		#get the total training images

		train_subdirectories = list(set(glob.glob(os.path.join(IMAGE_SOURCE, "train", "*"))\
		 ).difference(set(glob.glob(os.path.join(IMAGE_SOURCE,"train","*.*")))))

		numberofImages = 0
		for folder in train_subdirectories:
			for fileNameDir in os.walk(folder):
				for fileName in fileNameDir[2]:
					 # Only read in the images
					if fileName[-4:] != ".jpg":
					  continue
					numberofImages += 1

		maxPixel = PIXELS
		imageSize = maxPixel * maxPixel * CHANNELS
		num_rows = numberofImages # one row for each image in the training dataset
		num_features = imageSize 

		# X is the feature vector with one row of features per image
		# consisting of the pixel values and our metric
		X = np.zeros((num_rows, num_features), dtype=np.float32)
		# y is the numeric class label 
		y = np.zeros((num_rows))

		files = []
		# Generate training data
		i = 0    
		label = 0
		# List of string of class names
		namesClasses = list()

		print "Reading images"
		# Navigate through the list of directories
		for folder in train_subdirectories:
			# Append the string class name for each class
			currentClass = folder.split(os.pathsep)[-1]
			namesClasses.append(currentClass)
			for fileNameDir in os.walk(folder):   
				for fileName in fileNameDir[2]:
					# Only read in the images
					if fileName[-4:] != ".jpg":
					  continue
					
					# Read in the images and create the features
					nameFileImage = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)            
					image = imread(nameFileImage, as_grey=False)
					image = image.transpose(2, 0, 1) # c01 instead of 01c

					# Store the rescaled image pixels
					X[i, :imageSize] = np.reshape(image, (1, imageSize))
					
					# Store the classlabel
					y[i] = label
					i += 1
					# report progress for each 5% done  
					report = [int((j+1)*num_rows/20.) for j in range(20)]
					if i in report: print np.ceil(i *100.0 / num_rows), "% done"
			label += 1

		labels = map(lambda s: s.split('/')[-1], namesClasses)

		return X,y,labels

	def _load_test_images_from_disk(self):
		maxPixel = PIXELS
		imageSize = maxPixel * maxPixel * CHANNELS
		num_features = imageSize
		fnames = glob.glob(os.path.join(IMAGE_SOURCE, "test", "*.jpg"))

		numberofTestImages = len(fnames)
		X_test = np.zeros((numberofTestImages, num_features), dtype=np.float32)
		images = map(lambda fileName: fileName.split('/')[-1], fnames)
		i = 0
		# report progress for each 5% done  
		report = [int((j+1)*numberofTestImages/20.) for j in range(20)]

		for fileName in fnames:
			# Read in the images and create the features
			image = imread(fileName, as_grey=False)
			image = image.transpose(2, 0, 1) # c01 instead of 01c
		   
			# Store the rescaled image pixels and the axis ratio
			X_test[i, :imageSize] = np.reshape(image, (1, imageSize))
			
			i += 1
			if i in report: print np.ceil(i *100.0 / numberofTestImages), "% done"

		return X_test, images

	def _get_variants(self, X):
		"""
		Returns all possible rotations and translations of a certain image.
		Useful for generating an augmented mean and variance tensor.
		"""
		flips = [False, True]
		rotations = range(0, 360, 2)
		translations = range(-5, 5, 2)
		stack_pred = []

		for tranX in translations:
			for tranY in translations:
				for flip in flips:
					for rot in rotations:
						aug = Augmenter(np.array([[X]]), rot, (tranX, tranY), 1.0, 0, flip)
						augmented = aug.transform()
						stack_pred.append(augmented)

		return np.array(stack_pred)

	def _augment_mean_std(self, mean, std):
		"""
		This method augments the mean and variance to consider data augmentation.
		"""

		augmented_mean_channel_1 = self._get_variants(mean.reshape(CHANNELS, PIXELS, PIXELS)[0])
		mean_1 = augmented_mean_channel_1.mean(axis=0).reshape((PIXELS*PIXELS))

		augmented_std_channel_1 = self._get_variants(std.reshape(CHANNELS, PIXELS, PIXELS)[0])
		std_1 = np.sqrt((np.square(augmented_std_channel_1) + np.square(augmented_mean_channel_1)).mean(axis=0) - np.square(mean_1.reshape(PIXELS, PIXELS)))
		std_1 = std_1.reshape((PIXELS*PIXELS))

		if CHANNELS == 3:
			augmented_mean_channel_2 = self._get_variants(mean.reshape(CHANNELS, PIXELS, PIXELS)[1])
			mean_2 = augmented_mean_channel_2.mean(axis=0).reshape((PIXELS*PIXELS))
			augmented_mean_channel_3 = self._get_variants(mean.reshape(CHANNELS, PIXELS, PIXELS)[2])
			mean_3 = augmented_mean_channel_3.mean(axis=0).reshape((PIXELS*PIXELS))

			augmented_std_channel_2 = self._get_variants(std.reshape(CHANNELS, PIXELS, PIXELS)[1])
			std_2 = np.sqrt((np.square(augmented_std_channel_2) + np.square(augmented_mean_channel_2)).mean(axis=0) - np.square(mean_2.reshape(PIXELS, PIXELS)))
			std_2 = std_1.reshape((PIXELS*PIXELS))

			augmented_std_channel_3 = self._get_variants(std.reshape(CHANNELS, PIXELS, PIXELS)[2])
			std_3 = np.sqrt((np.square(augmented_std_channel_3) + np.square(augmented_mean_channel_3)).mean(axis=0) - np.square(mean_3.reshape(PIXELS, PIXELS)))
			std_3 = std_1.reshape((PIXELS*PIXELS))

			mean = np.hstack([mean_1, mean_2, mean_3])
			std = np.hstack([std_1, std_2, std_3])
		else:
			mean = mean_1
			std = std_1

		return mean, std

	def im2bin_full(self):
		"""
		Writes all images to a binary file.
		"""
		X,y,labels = self._load_train_images_from_disk()

		print "Computing mean..."
		self.scaler.fit(X)

		print "Augmenting mean..."
		mean,std = self._augment_mean_std(self.scaler.mean_, self.scaler.std_)

		print "Writing to file..."

		f = h5py.File(IM2BIN_OUTPUT, "w")
		dset = f.create_dataset("X_train", X.shape, dtype=np.float32, compression="gzip")
		dset[...] = X

		dset = f.create_dataset("y_train", y.shape, dtype=np.float32, compression="gzip")
		dset[...] = y

		dset = f.create_dataset("labels", (len(labels),), dtype=h5py.special_dtype(vlen=bytes), compression="gzip")
		dset[...] = labels

		X = None
		y = None

		print("Saved train images")

		X_test, images = self._load_test_images_from_disk()

		print("Writing test images...")

		dset = f.create_dataset("X_test", X_test.shape, dtype=np.float32, compression="gzip")
		dset[...] = X_test

		dset = f.create_dataset("image_names", (len(images),), dtype=h5py.special_dtype(vlen=bytes), compression="gzip")
		dset[...] = images

		# Save mean and std
		dset = f.create_dataset("mean", mean.shape, dtype=np.float32, compression="gzip")
		dset[...] = mean

		dset = f.create_dataset("std", std.shape, dtype=np.float32, compression="gzip")
		dset[...] = std

		print "Done!"

	def load_train_full(self):
		f = h5py.File(IM2BIN_OUTPUT, "r")
		X = f['X_train']
		y = f['y_train']

		X = X[:].astype(np.float32, copy=False)
		y = y[:].astype(np.int32, copy=False)

		X,y = shuffle(X, y, random_state = 42)
		X = X.reshape(-1, CHANNELS, PIXELS, PIXELS)

		return X,y

	def load_test_full(self):
		f = h5py.File(IM2BIN_OUTPUT, "r")
		X = f['X_test']

		X = X[:].astype(np.float32, copy=False)
		X = X.reshape(-1, CHANNELS, PIXELS, PIXELS)

		images = f['image_names']

		return X,images

	def load_labels(self):
		f = h5py.File(IM2BIN_OUTPUT, "r")
		labels = f['labels']

		return labels

	def load_mean_std(self):
		f = h5py.File(IM2BIN_OUTPUT, "r")
		mean = f['mean']
		std = f['std']

		return mean, std

if __name__ == "__main__":
	ImageIO().im2bin_full()
