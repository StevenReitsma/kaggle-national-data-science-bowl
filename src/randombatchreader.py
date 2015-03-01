# -*- coding: utf-8 -*-
import batchreader as br
import numpy as np

class RandomBatchReader (br.BatchReader):
    
    def __init__(self, filepath="../data/preprocessed.h5", batchsize=10000, dataset="data"):
       # super(br.BatchReader, self).__init__(self, filepath, batchsize, dataset)
        br.BatchReader.__init__(self, filepath, batchsize, dataset)
        n = self.dimensions[0]
        print n
        
        self.order = np.arange(n)
        np.random.shuffle(self.order)
    
    def next(self):
        if self.current >= self.nbatches*self.batch_size:
            self.file.close()
            raise StopIteration
        else:
            dat = np.zeros( (self.batch_size, self.dimensions[1]))            
            
            for index in xrange(self.batch_size):
                dat[index] = self.dset[self.order[self.current]]
                self.current += 1;
 
            return dat;
    
if __name__ == "__main__":
    for chunk_of_data in RandomBatchReader():
        print chunk_of_data