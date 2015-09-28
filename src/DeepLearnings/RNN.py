'''
Created on 19 Aug 2015

@author: Dirk
'''
import numpy as np


class RNN:
  
    def step(self, x):
        # update the hidden state
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        # compute the output vector
        y = np.dot(self.W_hy, self.h)
        return y



vocab = ['h','e','l','o']
train = ['hello']

if __name__ == '__main__':
    rnn = RNN()