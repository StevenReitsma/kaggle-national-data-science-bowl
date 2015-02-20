# -*- coding: utf-8 -*-
import h5py
import numpy as np

def loadunsupervised(filepath="../data/preprocessed.h5"):
    f = h5py.File(filepath)
    dset = f["unordered"]    
   # rdata = []
    dimensions = (len(dset), len(dset[0]))
    #dimensions = len(dset)
    rdata = np.zeros(dimensions, dtype=np.uint8)
    dset.read_direct(rdata)
    
    return rdata;


def flatten(collection):
    return [item for sublist in collection for item in sublist]
