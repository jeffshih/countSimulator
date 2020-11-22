import numpy as np
import scipy as sp
from typing import Callable, Any, Iterable
from Util import *
from Pair import Pair


class rect_(object):

    def __init__(self, center:Pair, wh:Pair):
        self.width = wh[0]
        self.height = wh[1]
        self.center = center 
        self.wh = wh
        self.LeftUpper = convertLU(center,wh)
        self.area = wh[0]*wh[1]
