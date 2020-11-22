import time 
import numpy as np
import weakref
from Pair import Pair
from Util import *


class detectionResult(object):
    #every param is imgSize when init a det
    def __init__(self, timestamp, catagory, id, wh:Pair, center:Pair, imgSize:Pair, confidence=None):
        
        #in ratio
        self.wh = absToRatio(imgSize, wh)
        self.center = absToRatio(imgSize, center)

        #Left upper point
        self.relLU = convertLU(center, wh)
        self.LeftUpper = absToRatio(imgSize, self.relLU)
        self.areaRatio = self.wh[0]*self.wh[1]

        #img size
        self.imgSize = imgSize
        
        #convert coordinate relative to imgsize
        self.relativeWH = wh
        self.relArea = wh[0]*wh[1]
        self.relativeCenter = center

        #unchange variable
        self.obj = id 
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
        self.stride = Pair(imgSize[0]/5.0,imgSize[1]*0.1)
        self.existTime = 1
    
    def updateConfidence(self):
        self.confidence = 1-(np.random.rand()*0.01)

    def updatePosition(self,timestamp:int, relCenter:Pair, relWH:Pair):
        self.timestamp = timestamp
        
        self.existTime +=1
        
        self.center = absToRatio(self.imgSize, relCenter)
        self.wh = absToRatio(self.imgSize, relWH)
        
        self.relLU = convertLU(relCenter, relWH)
        self.LeftUpper = absToRatio(relCenter, relWH)

        self.areaRatio = self.wh[0]*self.wh[1]
        self.absArea = relWH[0]*relWH[1]
        
        self.relativeCenter = relCenter
        self.relativeWH = relWH
        self.updateConfidence()

    def getDet(self):
        return ("{},{},{},{},{},{},{},{}".format(self.timestamp, self.objId, self.catagory, self.confidence\
                , self.center.x, self.center.y, self.wh.x, self.wh.y))

    def getRect(self):
        return [self.relativeCenter.x, self.relativeCenter.y, \
            self.relativeWH.w, self.relativeWH.h]

def overlap(A:detectionResult, B:detectionResult):
    x1 = max(A.relLU.x, B.relLU.x)
    y1 = min(A.relLU.y, B.relLU.y)
    x2 = max(A.relLU.x+A.relativeWH.x, B.relLU.x+B.relativeWH.x)
    y2 = min(A.relLU.y+A.relativeWH.y, B.relLU.y+B.relativeWH.y)
    overlapW = x2-x1
    overlapH = y2-y1 
    overlapArea = overlapH*overlapW
    #print("overlap :", overlapArea)
    return overlapArea

def union(A:detectionResult, B:detectionResult):
    unionArea = A.relArea+B.relArea - overlap(A,B)
    #print("union :", unionArea)
    return unionArea

def IOU(A:detectionResult, B:detectionResult):
    #print("IOU :",overlap(A,B)/union(A,B))
    return overlap(A,B)/union(A,B)