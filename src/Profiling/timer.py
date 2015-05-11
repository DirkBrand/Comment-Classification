'''
Created on 14 Apr 2014

@author: Dirk
'''

import time


def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        #print f.__name__, 'took', end - start, 'time'
        return result
    return f_timer