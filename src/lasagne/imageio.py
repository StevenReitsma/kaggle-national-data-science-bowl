import numpy as np
from skimage.io import imread
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import h5py
from params import *
from sklearn.utils import shuffle

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
		imageSize = maxPixel * maxPixel
		num_rows = numberofImages # one row for each image in the training dataset
		num_features = imageSize 

		# X is the feature vector with one row of features per image
		# consisting of the pixel values and our metric
		X = np.zeros((num_rows, num_features), dtype=float)
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
					image = imread(nameFileImage, as_grey=False) # why not true?
					files.append(nameFileImage)
					
					# Store the rescaled image pixels and the axis ratio
					X[i, 0:imageSize] = np.reshape(image, (1, imageSize))
					
					# Store the classlabel
					y[i] = label
					i += 1
					# report progress for each 5% done  
					report = [int((j+1)*num_rows/20.) for j in range(20)]
					if i in report: print np.ceil(i *100.0 / num_rows), "% done"
			label += 1

		return X,y

	def _load_test_images_from_disk(self):
		maxPixel = PIXELS
		imageSize = maxPixel * maxPixel
		num_features = imageSize
		fnames = glob.glob(os.path.join(IMAGE_SOURCE, "test", "*.jpg"))

		numberofTestImages = len(fnames)
		X_test = np.zeros((numberofTestImages, num_features), dtype=float)
		images = map(lambda fileName: fileName.split('/')[-1], fnames)
		i = 0
		# report progress for each 5% done  
		report = [int((j+1)*numberofTestImages/20.) for j in range(20)]

		for fileName in fnames:
			# Read in the images and create the features
			image = imread(fileName, as_grey=False)
		   
			# Store the rescaled image pixels and the axis ratio
			X_test[i, 0:imageSize] = np.reshape(image, (1, imageSize))
			
			i += 1
			if i in report: print np.ceil(i *100.0 / numberofTestImages), "% done"

		return X_test

	def im2bin_cv(self):
		X,y = self._load_train_images_from_disk()
		X = self.scaler.fit_transform(X)

		### Create the training set with 90% of data
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

		### Split the testing set to 50/50 for validation and testing
		X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

		f = h5py.File(IM2BIN_OUTPUT, "w")
		dset = f.create_dataset("X_train", X_train.shape, dtype=np.float64, compression="gzip")
		dset[...] = X_train
		dset = f.create_dataset("y_train", y_train.shape, dtype=np.float64, compression="gzip")
		dset[...] = y_train
		dset = f.create_dataset("X_val", X_val.shape, dtype=np.float64, compression="gzip")
		dset[...] = X_val
		dset = f.create_dataset("y_val", y_val.shape, dtype=np.float64, compression="gzip")
		dset[...] = y_val
		dset = f.create_dataset("X_test", X_test.shape, dtype=np.float64, compression="gzip")
		dset[...] = X_test
		dset = f.create_dataset("y_test", y_test.shape, dtype=np.float64, compression="gzip")
		dset[...] = y_test

	def im2bin_full(self):
		X,y = self._load_train_images_from_disk()
		X = self.scaler.fit_transform(X)

		f = h5py.File(IM2BIN_OUTPUT, "w")
		dset = f.create_dataset("X_train", X.shape, dtype=np.float64, compression="gzip")
		dset[...] = X

		dset = f.create_dataset("y_train", y.shape, dtype=np.float64, compression="gzip")
		dset[...] = y

		X = None
		y = None

		print("Done loading train images")

		X_test = self._load_test_images_from_disk()
		X_test = self.scaler.transform(X_test)

		print("Done loading test images")

		dset = f.create_dataset("X_test", X_test.shape, dtype=np.float64, compression="gzip")
		dset[...] = X_test

	def load_data_cv(self):
		f = h5py.File(IM2BIN_OUTPUT, "r")
		X = np.hstack(([f['X_train']], [f['X_val']], [f['X_test']]))
		y = np.hstack(([f['y_train']], [f['y_val']], [f['y_test']]))

		X = X[0].astype(np.float32, copy=False)
		Y = y[0].astype(np.int32, copy=False)

		X,y = shuffle(X, y, random_state = 42)
		X = X.reshape(-1, 1, PIXELS, PIXELS)

		return X,y

	def load_train_full(self):
		f = h5py.File(IM2BIN_OUTPUT, "r")
		X = f['X_train']
		y = f['y_train']

		X = X[:].astype(np.float32, copy=False)
		y = y[:].astype(np.int32, copy=False)

		X,y = shuffle(X, y, random_state = 42)
		X = X.reshape(-1, 1, PIXELS, PIXELS)

		return X,y

	def load_test_full(self):
		f = h5py.File(IM2BIN_OUTPUT, "r")
		X = f['X_test']

		X = X.astype(np.float32, copy=False)
		X = X.reshape(-1, 1, PIXELS, PIXELS)

		return X

if __name__ == "__main__":
	ImageIO().im2bin_full()