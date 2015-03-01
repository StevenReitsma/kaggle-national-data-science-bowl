import os
import sys
import subprocess

if len(sys.argv) < 3:
    print "Usage: python gen_train.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

cmd = "convert -resize 64x64\! -quality 100 "
classes = os.listdir(fi)

os.chdir(fo)
for cls in classes:
    try:
        os.mkdir(cls)
    except:
        pass
    imgs = os.listdir(fi + cls)
    for img in imgs:
        md = ""
        md += cmd
        md += fi + cls + "/" + img
        md += " " + fo + cls + "/" + img
        os.system(md)

        # DATA AUGMENTATION
        # ROTATIONS
        md = ""
        md += cmd
        md += "-rotate 90 "
        md += fi + cls + "/" + img
        md += " " + fo + cls + "/" + img[:-4] + "_90.jpg"
        os.system(md)

        md = ""
        md += cmd
        md += "-rotate 180 "
        md += fi + cls + "/" + img
        md += " " + fo + cls + "/" + img[:-4] + "_180.jpg"
        os.system(md)

        md = ""
        md += cmd
        md += "-rotate 270 "
        md += fi + cls + "/" + img
        md += " " + fo + cls + "/" + img[:-4] + "_270.jpg"
        os.system(md)

        # FLIPS AND FLOPS

        md = ""
        md += cmd
        md += "-flip "
        md += fi + cls + "/" + img
        md += " " + fo + cls + "/" + img[:-4] + "_fli.jpg"
        os.system(md)

        md = ""
        md += cmd
        md += "-flop "
        md += fi + cls + "/" + img
        md += " " + fo + cls + "/" + img[:-4] + "_flo.jpg"
        os.system(md)

	# TRANSPOSES AND TRANSVERSE
        md = ""
        md += cmd
        md += "-transpose "
        md += fi + cls + "/" + img
        md += " " + fo + cls + "/" + img[:-4] + "_tsp.jpg"
        os.system(md)

        md = ""
        md += cmd
        md += "-transverse "
        md += fi + cls + "/" + img
        md += " " + fo + cls + "/" + img[:-4] + "_tsv.jpg"
        os.system(md)
