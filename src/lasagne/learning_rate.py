import numpy as np

class AdjustVariable(object):
	"""
	Adjusts the learning rate according to a set scheme.
	"""
	def __init__(self, name, start=0.03):
		self.name = name
		self.start = start

	# Executed at end of epoch on learning rate and such
	def __call__(self, nn, train_history):
		epoch = train_history[-1]['epoch']

		stop = self.start * 10e-2

		ls_tmp = np.linspace(self.start, stop, 50)
		ls_tmp_2 = np.linspace(stop, stop * 10e-2, 50)
		ls_tmp_3 = np.linspace(stop * 10e-2, stop * 10e-4, 200)

		if epoch <= 50:
			new_value = np.cast['float32'](ls_tmp[epoch - 1])
		elif epoch > 50 and epoch <= 200:
			new_value = np.cast['float32'](ls_tmp_2[epoch - 1 - 50])
		else:
			new_value = np.cast['float32'](ls_tmp_3[epoch - 1 - 200])

		getattr(nn, self.name).set_value(new_value)
