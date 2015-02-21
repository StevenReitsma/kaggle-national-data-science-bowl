# -*- coding: utf-8 -*-

import h5py


class BatchReader:
    
    def __init__(self, filepath="../data/preprocessed.h5", batchsize=1000000, dataset="unordered"):
        self.path = filepath;
        self.batchsize = batchsize
        
        self.file = h5py.File(filepath)
        self.dset = self.file[dataset]    
        self.dimensions = (len(self.dset), len(self.dset[0]))    
        

        # Iteration index
        self.current = 0
        # Max iterations
        self.nbatches = self.dimensions[0]//batchsize
        
        print self.nbatches
            
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
                
                
                
#a = BatchReader()
#for x in a:
#    print x