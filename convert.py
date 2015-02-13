import numpy as np
import os
from scipy import misc
import scipy.io as sio

if __name__ == "__main__":
	dirs = sorted(os.listdir("data/train"))
	
	classes = {}
	j = 0
	for dir in dirs:
		classes[dir] = j
		j += 1
	
	labels = []
	images = np.zeros((30336, 1, 64, 64))
	
	i = 0
	for dir in dirs:
		files = sorted(os.listdir("data/train/" + dir))
		for img in files:
			image = misc.imread("data/train/" + dir + "/" + img)

			image_zoom = misc.imresize(image, (64, 64), 'bicubic')
			image_zoom = image_zoom.astype(float)
			image_zoom /= 255.

			images[i, 0, :, :] = image_zoom
			labels += [classes[dir]]
			
			i += 1
			
			if i % 1000 == 0:
				print i / 30336. * 100., "%"
		
	sio.savemat("/vol/temp/sreitsma/training.mat", {"data": images}, False)
	sio.savemat("/vol/temp/sreitsma/labels.mat", {"labels": labels}, False)