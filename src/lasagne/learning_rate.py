import numpy as np

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    # Executed at end of epoch on learning rate and such
    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']

        ls_tmp = np.linspace(0.03, 0.0001, 100)
        ls_tmp_2 = np.linspace(0.0001, 0.000001, 50)
        ls_tmp_3 = np.linspace(0.000001, 0.0000001, 50)

        if epoch <= 100:
            new_value = np.cast['float32'](ls_tmp[epoch - 1])
        elif epoch > 100 and epoch <= 150:
            new_value = np.cast['float32'](ls_tmp_2[epoch - 1 - 100])
        else:
            new_value = np.cast['float32'](ls_tmp_3[epoch - 1 - 100])

        getattr(nn, self.name).set_value(new_value)