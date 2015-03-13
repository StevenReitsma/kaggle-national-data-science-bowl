import os
import sys
from params import *

if len(sys.argv) < 3:
	print "Usage: python gen_train.py input_folder output_folder"
	exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

classes = os.listdir(fi)

if NAIVE:
	cmd = "convert -resize 64x64\! -quality 100 "
else:
	cmd = "convert -resize 64x64 -gravity center -background white -extent 64x64 -quality 100 "

os.chdir(fo)
for cls in classes:
	try:
		os.mkdir(cls)
	except:
		pass
	imgs = os.listdir(fi + cls)
	for img in imgs:
		md = cmd
		md += fi + cls + "/" + img
		md += " " + fo + cls + "/" + img
		os.system(md)
