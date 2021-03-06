import os
import sys
from params import *
import numpy as np
from skimage import io
from skimage.util import img_as_ubyte
from skimage.morphology import black_tophat, white_tophat, disk

if len(sys.argv) < 3:
	print "Usage: python gen_test.py input_folder output_folder"
	exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

if NAIVE:
	cmd = "-resize 64x64\! -quality 100 "
else:
	cmd = "-resize 64x64 -gravity center -background white -extent 64x64 -quality 100 "
	
imgs = os.listdir(fi)

for i, img in enumerate(imgs):
	if i % 1000 == 0:
		print "{}%".format(100 * i / float(len(imgs)))

	img_orig = img_as_ubyte(io.imread(fi + img, as_grey=True))
	img_btop = 255-black_tophat(img_orig, disk(1))
	img_wtop = 255-white_tophat(img_orig, disk(1))

	img_out = np.zeros((img_orig.shape[0], img_orig.shape[1], 3), dtype=np.uint8)
	img_out[:, :, 0] = img_orig
	img_out[:, :, 1] = img_btop
	img_out[:, :, 2] = img_wtop

	io.imsave("/dev/shm/test_tmp.jpg", img_out)
		
	md = "convert /dev/shm/test_tmp.jpg " + cmd
	md += fo + img
	os.system(md)