from lasagne import layers
from lasagne.layers import dnn
from lasagne import nonlinearities
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
import theano
from params import *
from util import *
from iterators import DataAugmentationBatchIterator, ScalingBatchIterator
from learning_rate import AdjustVariable
from early_stopping import EarlyStopping, EarlyStoppingNoValidation
from imageio import ImageIO
from lasagne.objectives import multinomial_nll
import pickle
from custom_layers import SliceLayer, MergeLayer
import random
from modelsaver import ModelSaver
from custom_nonlinearities import *

if USE_GPU:
	Conv2DLayer = layers.dnn.Conv2DDNNLayer
	MaxPool2DLayer = layers.dnn.MaxPool2DDNNLayer
else:
	Conv2DLayer = layers.Conv2DLayer
	MaxPool2DLayer = layers.MaxPool2DLayer

Maxout = layers.pool.FeaturePoolLayer

# Fix seeds
np.random.seed(42)
random.seed(42)

def fit(output):
	X, y = ImageIO().load_train_full()
	mean, std = ImageIO().load_mean_std()

	net = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			('slicer', SliceLayer),
			('conv1', Conv2DLayer),
			('pool1', MaxPool2DLayer),
			('conv2', Conv2DLayer),
			('pool2', MaxPool2DLayer),
			('conv3', Conv2DLayer),
			('conv4', Conv2DLayer),
			('pool3', MaxPool2DLayer),
			('merger', MergeLayer),
			('hidden1', layers.DenseLayer),
			('maxout1', Maxout),
			('dropouthidden1', layers.DropoutLayer),
			('hidden2', layers.DenseLayer),
			('maxout2', Maxout),
			('dropouthidden2', layers.DropoutLayer),
			('output', layers.DenseLayer),
			],

		input_shape=(None, CHANNELS, PIXELS, PIXELS),

		slicer_part_size = 48, slicer_flip = True,
		merger_nr_views = 8,

		conv1_num_filters=32, conv1_filter_size=(6, 6), conv1_pad = 0, pool1_ds=(2, 2), pool1_strides = (2, 2),
		conv2_num_filters=64, conv2_filter_size=(5, 5), conv2_pad = 0, pool2_ds=(2, 2), pool2_strides = (2, 2),
		conv3_num_filters=128, conv3_filter_size=(3, 3), conv3_pad = 0,
		conv4_num_filters=128, conv4_filter_size=(3, 3), conv4_pad = 0, pool3_ds=(2, 2), pool3_strides = (2, 2),

		hidden1_num_units=4096,
		dropouthidden1_p=0.5,
		maxout1_ds=2,
		hidden2_num_units=4096,
		dropouthidden2_p=0.5,
		maxout2_ds=2,

		output_num_units=121,
		output_nonlinearity=nonlinearities.softmax,

		update_learning_rate=theano.shared(float32(START_LEARNING_RATE)),
		update_momentum=theano.shared(float32(MOMENTUM)),

		regression=False,
		batch_iterator_train=DataAugmentationBatchIterator(batch_size=BATCH_SIZE, mean=np.reshape(mean, (CHANNELS, PIXELS, PIXELS)), std=np.reshape(std, (CHANNELS, PIXELS, PIXELS))),
		batch_iterator_test=ScalingBatchIterator(batch_size=BATCH_SIZE, mean=np.reshape(mean, (CHANNELS, PIXELS, PIXELS)), std=np.reshape(std, (CHANNELS, PIXELS, PIXELS))),
		on_epoch_finished=[
			AdjustVariable('update_learning_rate', start=START_LEARNING_RATE),
			EarlyStopping(patience=20),
			ModelSaver(epochs=10, output=output), # saves model every X epochs
		],
		max_epochs=300,
		verbose=1,
		eval_size=0.1,
	)

	net.fit(X, y)

	with open(output, 'wb') as f:
		pickle.dump(net, f, -1)

if __name__ == "__main__":
	fit('model.pkl')
