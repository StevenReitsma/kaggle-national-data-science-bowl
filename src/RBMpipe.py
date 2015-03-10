from __future__ import division

__author__ = 'Robbert'


import numpy as np
import train_classifier as train
import predict_classifier as classifier
import h5py
import sys
import RBMtrain


def singlePipeline(hidden_units,label_path = "../data/preprocessed.h5"):


    features = RBMtrain.train()
    print ("Boltzmann trained, loading labels..")

    #get the labels
    f = h5py.File(label_path)
    labels = f["labels"]
    print ("Labels loaded!, training classifier..")

#    feature_data = h5py.File("../data/activations/200activationkmeans.h5")
#    features = feature_data["activations"]
    features = np.array(features)

    #Train the SGD classifier
    train.trainSGD(features, labels, hidden_units)
    print ("Classifier trained, predicting..")
    #Classify the testset (the same as the training set in this case)
    classified = classifier.predict(features, hidden_units)



    summing = 0
    correct = 0

    #calculate the log loss
    for i, label in enumerate(labels):
        if(classified[i][label] == 0):
            summing+= np.log(sys.float_info.min)
        else:
            summing+= np.log(classified[i][label])
        if labels[i] == np.argmax(classified[i]):
#            print classified[i][np.argmax(classified[i])]
            correct += 1
#


    summing = -summing/len(labels)
    print summing
    print correct/len(labels)
#    print np.min(classified)

#    print summing
#    np.savetxt( "realLabel.csv", labels, delimiter=";")
#    np.savetxt( "SGD_label.csv", max_SGD, delimiter=";")

    f.close()

#    feature_data.close()





if __name__ == '__main__':
    singlePipeline(100)
