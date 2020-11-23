import time 
import numpy as np
import weakref
from Pair import Pair
from Util import *
from config import *

class detectionResult(object):
    #every param is imgSize when init a det
    def __init__(self, timestamp, catagory, id, wh:Pair, center:Pair, imgSize:Pair, stride = None, confidence=None):
        
        #in ratio
        self.wh = absToRatio(imgSize, wh)
        self.center = absToRatio(imgSize, center)

        #Left upper point
        self.relLU = convertLU(center, wh)
        self.LeftUpper = absToRatio(imgSize, self.relLU)
        self.areaRatio = self.wh[0]*self.wh[1]
        self.br = Pair((center.x+wh.w/2), (center.y+wh.h/2))

        #img size
        self.imgSize = imgSize
        
        #convert coordinate relative to imgsize
        self.relativeWH = wh
        self.relArea = wh.w*wh.h
        self.relativeCenter = center

        #unchange variable
        self.frameID = id 
        self.catagory = catagory

        #change by time
        self.timestamp = timestamp

        #confidence change simulation can be relative to category
        if confidence is not None:
            self.confidence = confidence
        else:
            self.confidence = 1-(np.random.rand()*0.2)

        #belt move in constant speed
        #when every frame move, the object move in x direction with 
        #1/5 of the frame width and y theoratically don't move
        #but introducing 5% of variation, we give it 10% of variation
        strideRatio = stride if stride is not None else np.random.randint(3,6)
        self.stride = Pair(imgSize.w/strideRatio, imgSize.h*0.01)
        self.existTime = 1
        self.originAbsY = center.y
        self.orW = wh.w
        self.orH = wh.h
        self.isDetected = True
    
    def updateConfidence(self):
        self.confidence = 1-(np.random.rand()*0.02)


        #input is in pixel format
    def updatePosition(self,timestamp:int, relCenter:Pair, relWH:Pair):
        self.timestamp = timestamp
        
        self.existTime +=1

        self.center = absToRatio(self.imgSize, relCenter)
        self.wh = absToRatio(self.imgSize, relWH)
        
        self.relLU = convertLU(relCenter, relWH)
        self.LeftUpper = absToRatio(self.imgSize, self.relLU)

        self.areaRatio = self.wh[0]*self.wh[1]
        self.relArea = relWH.w*relWH.h
        
        self.relativeCenter = relCenter
        self.relativeWH = relWH
        self.updateConfidence()
        if (np.random.rand() < catagoryMissRate[self.catagory]):
            self.isDetected = False
        else:
            self.isDetected = True

    def getDet(self):
        return ("{},{},{},{},{},{},{},{}".format(self.timestamp, self.frameID, self.catagory, self.confidence\
                , self.center.x, self.center.y, self.wh.x, self.wh.y))

    def getRect(self):
        return [self.relativeCenter.x, self.relativeCenter.y, \
            self.relativeWH.w, self.relativeWH.h]


def overlap(A:detectionResult, B:detectionResult):
    x1 = max(A.relLU.x, B.relLU.x)
    y1 = max(A.relLU.y, B.relLU.y)
    x2 = min(A.relLU.x+A.relativeWH.w, B.relLU.x+B.relativeWH.w)
    y2 = min(A.relLU.y+A.relativeWH.h, B.relLU.y+B.relativeWH.h)
    W = x2-x1
    H = y2-y1 
    overlapArea = H*W
    if W < 0 or H < 0 :
        return 0
    else:
        return overlapArea

def union(A:detectionResult, B:detectionResult):
    unionArea = A.relArea+B.relArea - overlap(A,B)
    #print("A area {} B area {}".format(A.relArea, B.relArea))
    return unionArea

def IOU(A:detectionResult, B:detectionResult):
    #print("IOU :",overlap(A,B)/union(A,B))
    return overlap(A,B)/union(A,B)