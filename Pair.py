import time 
import numpy as np
import weakref


class Pair(object):
    #data structure for point and furthur manipulation
    def __init__(self, x_init:float, y_init:float):
        self.x = x_init
        self.y = y_init
        self.w = x_init
        self.h = y_init
        self.pair = (x_init , y_init)
    
    def __getitem__(self, idx):
        try:
            if(idx >1):
                raise IndexError
            return self.pair[idx]
        except IndexError:
            print("Idx out of bound")

        