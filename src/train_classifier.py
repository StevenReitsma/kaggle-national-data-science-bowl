from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
import os


def trainSGD(features, labels, nr_centroids):
    	samples = features.shape[0]

	# Bottou recommends showing at least 10e06 samples to the SGD classifier.
	iters = 10e06 / samples

	# Using an elasticnet penalty introduces a regularizer (originally for SVM) that introduces sparsity.
	penalty = 'elasticnet'

	# How much sparsity do we need? 0 is no sparsity (L2 penalty, the default), 1 is super sparse.
	l1_ratio = 0.15

	# Specifies on how many threads you wish to run the classifier, defaults to the amount of cores you have.
	n_jobs = -1

	# This parameter multiplies the regularization term, should be tested through cross-validation to obtain an optimal result.
	alpha = 5*10e-1

	# Initialize classifier and fit on data
	clf = SGDClassifier(loss = 'log', penalty = penalty, n_iter = iters, l1_ratio = l1_ratio, shuffle = True, n_jobs = n_jobs, alpha = alpha, verbose = 1)
	clf.fit(features, labels)

	# Save model to file.
	file_path = '../models/sgd' + str(nr_centroids) + '/'
	if not os.path.exists(file_path):
              os.makedirs(file_path)
          
    	joblib.dump(clf, file_path + '/classifier.pkl')
