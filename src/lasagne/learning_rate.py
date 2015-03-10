import numpy as np

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.0003):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    # Executed at end of epoch on learning rate and such
    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']

        ls_tmp = np.linspace(self.start, self.stop, 100)
        ls_tmp_2 = np.linspace(self.stop, self.stop * 10e-2, 50)
        ls_tmp_3 = np.linspace(self.stop * 10e-2, self.stop * 10e-4, 150)

        if epoch <= 100:
            new_value = np.cast['float32'](ls_tmp[epoch - 1])
        elif epoch > 100 and epoch <= 150:
            new_value = np.cast['float32'](ls_tmp_2[epoch - 1 - 100])
        else:
            new_value = np.cast['float32'](ls_tmp_3[epoch - 1 - 150])

        getattr(nn, self.name).set_value(new_value)