'''@file delta.py
delta of features'''

import numpy as np

deriv = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

class Delta1(object):
    def __init__(self, dim):
        self.buffer = np.zeros((deriv.size, dim))
        self.i = 0 # where we write in the circular buffer
        self.filter=deriv
        self.delay = int((deriv.size-1)/2)

    def __call__(self, x):
        # returns the input delayed over group delay and the derivative
        self.buffer[self.i,:] = x
        self.filter = np.roll(self.filter,1)
        y = np.dot(self.filter, self.buffer)
        iOut=self.i-self.delay
        if iOut<0:
            iOut += deriv.size
        self.i += 1
        if self.i >= deriv.size:
            self.i = 0
        return np.concatenate((np.squeeze(self.buffer[iOut, :]), y))