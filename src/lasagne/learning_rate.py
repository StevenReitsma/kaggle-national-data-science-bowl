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

		stop = self.start * 0.25

		ls_tmp = np.linspace(self.start, stop, 30)
		ls_tmp_2 = np.linspace(stop, stop * 10e-3 * 4, 70)
		ls_tmp_3 = np.linspace(stop * 10e-3 * 4, stop * 10e-4 * 4, 200)

		if epoch <= 30:
			new_value = np.cast['float32'](ls_tmp[epoch - 1])
		elif epoch > 30 and epoch <= 70:
			new_value = np.cast['float32'](ls_tmp_2[epoch - 1 - 30])
		else:
			new_value = np.cast['float32'](ls_tmp_3[epoch - 1 - 70])

		getattr(nn, self.name).set_value(new_value)
