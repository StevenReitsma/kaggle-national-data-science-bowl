#!/bin/bash

CXXNET_DIR="/home/steven/cxxnet/cxxnet"
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

$CXXNET_DIR/bin/cxxnet $1
$CXXNET_DIR/bin/cxxnet $2

python make_submission.py $DIR/../../data/sampleSubmission.csv $DIR/../../data/cxxnet/test.lst $DIR/../../data/cxxnet/test.txt $DIR/../../data/cxxnet/out.csv