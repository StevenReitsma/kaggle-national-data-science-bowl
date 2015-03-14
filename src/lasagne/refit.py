from params import *
from util import *
from learning_rate import AdjustVariable
from early_stopping import EarlyStopping, EarlyStoppingNoValidation
from imageio import ImageIO
import pickle
import random
from modelsaver import ModelSaver

# Fix seeds
np.random.seed(42)
random.seed(42)

# This refits a loaded model and retrains it with the complete dataset (including validation).
# It should train until the same training loss is reached that was reached for the best validation loss.
# Edit it here:

TRAINING_LOSS = 0.640368

def fit(output):
	X, y = ImageIO().load_train_full()
	mean, std = ImageIO().load_mean_std()

	with open(output, 'rb') as f:
		net = pickle.load(f)

	net.on_epoch_finished = [
			AdjustVariable('update_learning_rate', start=START_LEARNING_RATE*10e-2), # Reduced learning rate for this step
			EarlyStoppingNoValidation(training_loss_threshold=TRAINING_LOSS),
		]

	net.eval_size = None

	net.fit(X, y)

	with open(output + "_full", 'wb') as f:
		pickle.dump(net, f, -1)

if __name__ == "__main__":
	fit('model.pkl')
