import os
import sys
import subprocess

if len(sys.argv) < 3:
	print "Usage: python gen_train.py input_folder output_folder"
	exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

classes = os.listdir(fi)

os.chdir(fo)
for cls in classes:
	try:
		os.mkdir(cls)
	except:
		pass
	imgs = os.listdir(fi + cls)
	for img in imgs:
		cmds = []
		for r in range(0, 360, 45):
			cmds.append("convert -rotate " + str(r) + " -trim -resize 64x64 -gravity center -background white -extent 64x64 -quality 100 ")

		for i, md in enumerate(cmds):
			md += fi + cls + "/" + img
			md += " " + fo + cls + "/" + img[:-4] + "_" + str(i) + ".jpg"
			os.system(md)