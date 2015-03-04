from lasagne import layers
from lasagne.layers import cuda_convnet
from lasagne import nonlinearities
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
import theano
from params import *
from util import *
from iterators import DataAugmentationBatchIterator
from learning_rate import AdjustVariable
from early_stopping import EarlyStopping
from imageio import ImageIO
from lasagne.objectives import multinomial_nll
import pickle

if USE_GPU:
	Conv2DLayer = layers.cuda_convnet.Conv2DCCLayer
	MaxPool2DLayer = layers.cuda_convnet.MaxPool2DCCLayer
else:
	Conv2DLayer = layers.Conv2DLayer
	MaxPool2DLayer = layers.MaxPool2DLayer

Maxout = layers.pool.FeaturePoolLayer

net = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			('conv1', Conv2DLayer),
			('pool1', MaxPool2DLayer),
			('dropout1', layers.DropoutLayer),
			('conv2', Conv2DLayer),
			('pool2', MaxPool2DLayer),
			('dropout2', layers.DropoutLayer),
			('conv3', Conv2DLayer),
			('pool3', MaxPool2DLayer),
			('dropout3', layers.DropoutLayer),
			('hidden4', layers.DenseLayer),
			('maxout4', Maxout),
			('dropout4', layers.DropoutLayer),
			('hidden5', layers.DenseLayer),
			('maxout5', Maxout),
			('dropout5', layers.DropoutLayer),
			('output', layers.DenseLayer),
			],

		input_shape=(None, 1, PIXELS, PIXELS),
		conv1_num_filters=96, conv1_filter_size=(5, 5), pool1_ds=(3, 3), pool1_strides = (2,2),
		dropout1_p=0.2,
		conv2_num_filters=256, conv2_filter_size=(3, 3), pool2_ds=(3, 3), pool2_strides = (2,2),
		dropout2_p=0.2,
		conv3_num_filters=256, conv3_filter_size=(3, 3), pool3_ds=(3, 3), pool3_strides = (2,2),
		dropout3_p=0.2,
		hidden4_num_units=2048,
		dropout4_p=0.5,
		maxout4_ds=2,
		hidden5_num_units=2048,
		dropout5_p=0.5,
		maxout5_ds=2,

		output_num_units=121,
		output_nonlinearity=nonlinearities.softmax,

		update_learning_rate=theano.shared(float32(0.03)),
		update_momentum=theano.shared(float32(0.9)),

		regression=False,
		batch_iterator_train=DataAugmentationBatchIterator(batch_size=128),
		on_epoch_finished=[
			AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
			#EarlyStopping(patience=20),
		],
		max_epochs=110,
		verbose=1,
		eval_size=None,
	)

def fit(output):
	X, y = ImageIO().load_train_full()
	net.fit(X, y)

	with open(output, 'wb') as f:
		pickle.dump(net, f, -1)

if __name__ == "__main__":
	fit('model.pkl')
