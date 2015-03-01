#!/bin/bash

CXXNET_DIR="/home/steven/cxxnet/cxxnet"
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

if [ "$1" == train ]; then
	# Cleanup
	rm -rf $DIR/../../data/cxxnet/train/*
	rm -f $DIR/../../data/cxxnet/train.lst
	rm -f $DIR/../../data/cxxnet/train.bin

	# Generate new data
	python gen_train.py $DIR/../../data/train/ $DIR/../../data/cxxnet/train/
	python gen_img_list.py train $DIR/../../data/sampleSubmission.csv $DIR/../../data/cxxnet/train/ $DIR/../../data/cxxnet/train.lst
	$CXXNET_DIR/tools/im2bin $DIR/../../data/cxxnet/train.lst / $DIR/../../data/cxxnet/train.bin
elif [ "$1" == test ]; then
	# Cleanup
	rm -rf $DIR/../../data/cxxnet/test/*
	rm -f $DIR/../../data/cxxnet/test.lst
	rm -f $DIR/../../data/cxxnet/test.bin

	# Generate new data
	python gen_test.py $DIR/../../data/test/ $DIR/../../data/cxxnet/test/
	python gen_img_list.py test $DIR/../../data/sampleSubmission.csv $DIR/../../data/cxxnet/test/ $DIR/../../data/cxxnet/test.lst
	$CXXNET_DIR/tools/im2bin $DIR/../../data/cxxnet/test.lst / $DIR/../../data/cxxnet/test.bin
fi