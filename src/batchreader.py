# -*- coding: utf-8 -*-

from __future__ import division
import h5py
import math


#Usage:
#
# for chunk_of_data in BatchReader()
#     print chunk_of_data
#
# In the last iteration the remainder chunk is returned
# (which may be smaller than batchsize)

class BatchReader:
    
    def __init__(self, filepath="../data/preprocessed.h5", batchsize=10000, dataset="data"):
        self.path = filepath;
        self.batchsize = batchsize
        
        self.file = h5py.File(filepath)
        self.dset = self.file[dataset]    
        self.dimensions = (len(self.dset), len(self.dset[0]))    
        

        # Iteration index
        self.current = 0
        # Max iterations
        self.nbatches = math.ceil(self.dimensions[0]/batchsize)

            
    def __iter__(self):
        return self
        
    def next(self):
        if self.current >= self.nbatches:
            self.file.close()
            raise StopIteration
        else:
            fromIndex = self.current*self.batchsize
            toIndex = fromIndex + self.batchsize
            
            dat = self.dset[fromIndex : toIndex]

            self.current += 1           
            return dat;
            
if __name__ == '__main__':
    for i, x in enumerate( BatchReader() ):
        print i, len(x), x