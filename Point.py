import time 
import numpy as np
import weakref


class Point(object):
    #data structure for point and furthur manipulation
    def __init__(self, x_init:float, y_init:float):
        self.x = x_init
        self.y = y_init
        self.w = x_init
        self.h = y_init
        self.Point= (x_init , y_init)
    
    def __getitem__(self, idx):
        try:
            if(idx >1):
                raise IndexError
            return self.Point[idx]
        except IndexError:
            print("Idx out of bound")

    def __str__(self):
        return "{},{}".format(self.x, self.y)
    

        