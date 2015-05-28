import os
import sys
import numpy as np
from params import *
from skimage import io
from skimage.util import img_as_ubyte
from skimage.morphology import black_tophat, white_tophat, disk

if len(sys.argv) < 3:
	print "Usage: python gen_train.py input_folder output_folder"
	exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

classes = os.listdir(fi)

if NAIVE:
	cmd = "-resize 64x64\! -quality 100 "
else:
	cmd = "-resize 64x64 -gravity center -background white -extent 64x64 -quality 100 "

os.chdir(fo)
for i, cls in enumerate(classes):
	print "{}%".format(100 * i / float(len(classes)))
	try:
		os.mkdir(cls)
	except:
		pass
	imgs = os.listdir(fi + cls)
	for img in imgs:

		img_orig = img_as_ubyte(io.imread(fi + cls + "/" + img, as_grey=True))
		img_btop = 255-black_tophat(img_orig, disk(1))
		img_wtop = 255-white_tophat(img_orig, disk(1))

		img_out = np.zeros((img_orig.shape[0], img_orig.shape[1], 3), dtype=np.uint8)
		img_out[:, :, 0] = img_orig
		img_out[:, :, 1] = img_btop
		img_out[:, :, 2] = img_wtop

		io.imsave("/dev/shm/train_tmp.jpg", img_out)

		md = "convert /dev/shm/train_tmp.jpg " + cmd
		md += " " + fo + cls + "/" + img
		os.system(md)