import numpy as np
from detection import detectionResult
from Pair import Pair
import string
from Util import *


class detectionParser(object):

    def __init__(self, imgSize:Pair):
        
        self.frame = []
        self.width = imgSize[0]
        self.height = imgSize[1]
        self.resolution = imgSize
        self.frameNum = 0

    def stringToDet(self, inputString):
        
        strSplit = inputString.split(",")
        TimeStamp = int(strSplit[0])
        currentFrame = int(strSplit[1])
        catagory = int(strSplit[2])
        confidence = float(strSplit[3])
        cx = float(strSplit[4])
        cy = float(strSplit[5])
        w = float(strSplit[6])
        h = float(strSplit[7])
        center = Pair(cx,cy)
        wh = Pair(w,h)
        relCenter = ratioToAbs(center)
        relWH = ratioToAbs(wh)
        det = detectionResult(TimeStamp, catagory, currentFrame, relWH\
            ,relCenter, self.resolution, confidence=confidence)
        return det