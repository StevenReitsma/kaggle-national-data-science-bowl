import numpy as np
import pandas as pd
import pickle
from predict import Augmenter, diversive_augment
from imageio import ImageIO
from sklearn.svm import SVC

# Averages over Kaggle submission files
def combine_mean(filenames):
	dfs = []
	for f in filenames:
		dfs.append(pd.DataFrame.from_csv(f))

	# Start averaging
	df_concat = pd.concat(dfs)
	df_mean = df_concat.groupby(level=0).mean()
	
	df_mean.to_csv('out_averaged.csv')

# Doesn't work at all
def combine_rf(model_names, filenames):
	X, y = ImageIO().load_train_full()
	models = []

	X = X[0:1024]
	y = y[0:1024]

	for m in model_names:
		with open(m, 'rb') as f:
			model = pickle.load(f)
			models.append(model)

	vectors = []

	for m in models:
		avg = diversive_augment(X, m)
		vectors.append(avg)

	vectors = np.hstack(vectors)

	svc = SVC(probability = True, verbose = 1)
	svc.fit(vectors, y)

	# RF fitted, now feed through test samples
	dfs = []
	df = None
	for f in filenames:
		df = pd.DataFrame.from_csv(f)
		dfs.append(df.values)

	test_preds = np.hstack(dfs)

	new_preds = svc.predict_proba(test_preds)

	for i in range(0, new_preds.shape[0]):
		df.ix[i] = new_preds[i]

	df.to_csv('out_averaged_rf.csv')

if __name__ == "__main__":
	#combine_mean(['MODEL_BEST.csv', 'MODEL_NAIVE_RESIZING.csv'])
	combine_rf(['MODEL_BEST.pkl', 'MODEL_NAIVE_RESIZING.pkl'], ['MODEL_BEST.csv', 'MODEL_NAIVE_RESIZING.csv'])