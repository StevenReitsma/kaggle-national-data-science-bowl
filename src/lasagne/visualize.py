import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def visualize_first_layer(filename):
	with open(filename, 'rb') as f:
		model = pickle.load(f)

	W = model.layers_['conv1'].W.get_value()
	length = int(np.sqrt(W.shape[0]))

	f, ax = plt.subplots(length, length)

	for i in range(0, length):
		for j in range(0, length):
			ax[i, j].imshow(W[i*length+j][0], cmap = cm.Greys, interpolation = 'nearest')
			ax[i, j].axis('off')

	plt.show()



if __name__ == "__main__":
	visualize_first_layer('MODEL_NAIVE_RESIZING.pkl')
