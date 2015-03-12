import os
import sys
import subprocess
from params import *

if len(sys.argv) < 3:
    print "Usage: python gen_test.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

if NAIVE:
	cmd = "convert -resize 64x64\! -quality 100 "
else:
	cmd = "convert -resize 64x64 padding: -gravity center -background white -extent 64x64 -quality 100 "
	
imgs = os.listdir(fi)


for img in imgs:
    md = ""
    md += cmd
    md += fi + img
    md += " " + fo + img
    os.system(md)



