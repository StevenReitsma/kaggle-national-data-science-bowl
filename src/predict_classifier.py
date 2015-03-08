from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np

def create_csv_submission(predictions, filelist):
	# Read sample submission.
	sample_submission = pd.DataFrame.from_csv('../data/sampleSubmission.csv')

	# Set data
	for i, filename in enumerate(filelist):
		sample_submission.loc[filename] = np.round(predictions[i], 8) # round to 8 decimals to save upload time

	# Save file to disk.
	sample_submission.to_csv('../data/submission.csv')
 
 
def predict(features, nr_centroids):
    clf = joblib.load('../models/sgd' + str(nr_centroids) + '/classifier.pkl')
    predictions = clf.predict_proba(features)
    return predictions

if __name__ == "__main__":
	# TODO load test set feature vectors into memory.
	# X = 

	# TODO load filelist into memory.
	# F = 

	# Load the trained model into memory
	clf = joblib.load('../models/sgd/classifier.pkl')

	# Predict samples
	predictions = clf.predict_proba(X)

	# Create CSV file for submission to Kaggle
	create_csv_submission(predictions, F)