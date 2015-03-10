import numpy as np

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class EarlyStoppingTraining(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.best_train = np.inf
        self.best_train_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_train = train_history[-1]['train_loss']
        current_epoch = train_history[-1]['epoch']
        if current_train < self.best_train:
            self.best_train = current_train
            self.best_train_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_train_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best train loss was {:.6f} at epoch {}.".format(
                self.best_train, self.best_train_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()