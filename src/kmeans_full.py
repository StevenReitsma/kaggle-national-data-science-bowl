# -*- coding: utf-8 -*-

import preprocess
import kmeans
import activationCalculation as ac
import kmeans_runner
import predict_classifier as pc
import util
import RBM
import RBMtrain
import train_classifier as train
import predict_classifier as classifier
import numpy as np
import scipy
from sklearn import metrics
#
# This file contains all steps for training k-means with a SVC classifier.
#

#----------------------------------------------------
#OPTIONS

#Classifier
#classifier="SVC" #'SGD', 'SVC', 'NUSVR', only tested for SVC

# NUSVR does not seem to work :c
# Can output likelihood-order, but not probabilities, meh.

#Preprocessing options
square_method = 'pad' #Either 'pad' or 'stretch'
patch_size = 6
image_size = 28 #Common size for all images to be resized to

train_folder = '../mnist/'
test_folder = '../data/testset/'

processed_train_filename = '../data/preprocessed.h5'
#processed_test_filename = '../data/preprocessed_test.h5'
processed_test_filename = '../data/preprocessed_test.h5'



#K-Means options
nr_iterations = 10
nr_centroids = 100
centroids_folder = "../data/centroidskmeans/"
activations_folder_test = "../data/activations_test/"
activations_folder_train = "../data/activations_train/"

#Misc options
nr_pool_regions = 4

#Classifier options
degree = 3
cache_size = 3000 #In MB
max_iter = 5000
tol = 1e-3
kernel = 'poly'


#----------------------------------------------------
# PREPROCESS
# Filesize with default settings above 2.96GB and 12.7GB

def one():
    #Train images, takes 5 min
    preprocess.preprocess(path=train_folder, 
                          outpath=processed_train_filename, 
                          patch_size=patch_size,
                          image_size=image_size)

# You can do this after step 6
def two():
    #Test images, takes 30 min+
    preprocess.preprocess(path=test_folder, 
                          outpath=processed_test_filename, 
                          patch_size=patch_size,
                          image_size=image_size,
                          train_data_file=processed_train_filename)

#----------------------------------------------------
# K-MEANS 

# CREATE CENTROIDS
# 30 min with default settings

def three():
    #km_trainer = kmeans.kMeansTrainer(nr_centroids = nr_centroids,
    #                                  nr_it = nr_iterations)
    pooledImages = RBMtrain.train()
    pooledImages = np.array(pooledImages)
    print "Creating centroids"
#    centroids = km_trainer.fit()
#    print "Saving centroids to file"
#    km_trainer.save_centroids(centroids,
#                              file_path=centroids_folder)
    label_path = "../data/preprocessed.h5"
    labels = util.load_labels(label_path)
    label_names = util.load_label_names(label_path)
    print "Begin training of SGD..."
    train.trainSGD(pooledImages, labels, nr_centroids)
    print "Training done"
    print "Dogfeeding"
    #Predict based on SGD training
    print "Begin SGD predictions..."
    classified = classifier.predict(pooledImages, nr_centroids)
    print "Predicting done"
    print "Calculating log loss..."
    summing = 0
    correct = 0

    np.savetxt("meuk.csv", classified, delimiter=";")

    loss = metrics.log_loss(labels, classified)
    print loss

    print -np.mean(np.log(classified)[np.arange(len(labels)), labels])

    #calculate the log loss
    for i, label in enumerate(labels):

        actual = labels[i]


        if(classified[i][label] == 0):
            summing+= np.log(10e-15)
        else:
            summing+= np.log(classified[i][label])
        if actual == np.argmax(classified[i]):
            correct += 1

    image = np.zeros((len(label_names),len(labels)))

    for j, label_index in enumerate(labels):
        image[label_index,j] = 1

    scipy.misc.imsave('correct.png', image)
    scipy.misc.imsave('predicted.png', classified.T)

    error = image - classified.T

    scipy.misc.imsave('error.png', error)


    print "Calculation finished"

    summing = -summing/len(labels)
    print "log loss: ", summing
    print "correct/amount_of_labels: ", correct/len(labels)
    print "lowest classification score: ", np.min(classified)

#    print summing
    np.savetxt( "realLabel.csv", labels, delimiter=";")
   # np.savetxt( "SGD_label.csv", max_SGD, delimiter=";")
    
    
# ACTIVATIONS OF TRAIN SET
# 5 min
def four():
    calculator = ac.ActivationCalculation()
    
    km = kmeans.kMeansTrainer()
    print "Loading centroids"
    centroids = km.get_saved_centroids(nr_centroids, 
                                       file_path=centroids_folder)
    
    print "Calculating activations of train data"
    calculator.pipeline(centroids, 
                        n_pool_regions = nr_pool_regions,
                        file_path = activations_folder_train,
                        data_file = processed_train_filename)
    print "Done"

#----------------------------------------------------
# TRAIN CLASSIFIER (SVC, NUSVR OR SGD)
def five():
    kmeans_runner.singlePipeline(nr_centroids, 
                                 nr_iterations, 
                                 label_path = processed_train_filename,
                                 clsfr = classifier,
                                 calc_centroids = False,
                                 dogfeed = False,
                                 train_model = True,
                                 cache_size = cache_size,
                                 degree = degree,
                                 tol = tol,
                                 max_iter = max_iter)

#!!!
# OPTIONAL, Predict train data to see some performance measure
# Useful to get estimate of performance before creating submission
#!!!

def six():
    
    model_filename = '../models/'+classifier.lower()+str(nr_centroids)+'/classifier.pkl'
    #Also creates nice visualizations of predictions as png files
    #in src folder
    kmeans_runner.singlePipeline(nr_centroids, 
                                 nr_iterations, 
                                 label_path = processed_train_filename,
                                 clsfr = classifier,
                                 calc_centroids = False,
                                 dogfeed = True,
                                 train_model = False,
                                 model_file = model_filename)
    
    
#----------------------------------------------------
# CALCULATE ACTIVATIONS OF TEST DATA
def steven():
    calculator = ac.ActivationCalculation()
    
    km = kmeans.kMeansTrainer()
    print "Loading centroids"
    centroids = km.get_saved_centroids(nr_centroids, 
                                       file_path=centroids_folder)
    print "Calculating activations of test data"
    calculator.pipeline(centroids, 
                        n_pool_regions = nr_pool_regions,
                        file_path = activations_folder_test,
                        data_file = processed_test_filename)
    print "Done"
    
    
    
#----------------------------------------------------
# USE TRAINED MODEL TO PREDICT TEST SETS

def eight():
    model_filename = '../models/'+classifier.lower()+str(nr_centroids)+'/classifier.pkl'
    

    pc.predict_classifier(model=model_filename,
                          activations_folder = activations_folder_test,
                          nr_centroids=nr_centroids)
    print "Done"


def testing():
    km = kmeans.kMeansTrainer()
    centroids = km.get_saved_centroids(nr_centroids, file_path=centroids_folder)
    util.plot_centroids(centroids, centroids_folder)

#one()
#two()
three()
