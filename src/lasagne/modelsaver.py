import numpy as np
import pickle

class ModelSaver(object):
    def __init__(self, output, epochs = 10):
    	self.output = output
        self.epochs = epochs

    # Executed at end of epoch
    def __call__(self, nn, train_history):
    	epoch = train_history[-1]['epoch']

    	if epoch % self.epochs == 0:
    		with open(self.output + '_' + str(epoch), 'wb') as f:
				pickle.dump(nn, f, -1)