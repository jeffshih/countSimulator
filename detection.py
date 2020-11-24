import time 
import numpy as np
import weakref
from Point import Point
from Util import *
from config import *
from dataStructure import *

class detectionResult(object):
    #every param is imgSize when init a det
    def __init__(self, timestamp, catagory, id, wh:Point, center:Point, imgSize:Point, confidence=None):
        
        #in ratio
        self.wh = absToRatio(imgSize, wh)
        self.center = absToRatio(imgSize, center)
        self.absCenter = center
        self.absWH = wh
        self.absW = wh.w 
        self.absH = wh.h
        self.rect = rect_(center,wh)

        #Left upper point
        self.LU = calcLU(center, wh)
        self.BR = Point((center.x+wh.w/2), (center.y+wh.h/2))
        
        #img size
        self.imgSize = imgSize
        
        #convert coordinate ans to imgsize
        self.absArea = wh.w*wh.h
        self.areaRatio = self.wh[0]*self.wh[1]
        
        #unchange variable
        self.objId = id 
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
        self.xStride = imgSize.w/5
        
        self.existTime = 1
        self.lifespan = np.random.randint(3,6)
        
        #object original y while object wont move in y direction drastically,
        #keep the original w and h
        self.originAbsY = center.y
        self.orW = wh.w
        self.orH = wh.h
        self.isDetected = True
    

    def mover(self):
        pass

    def updateConfidence(self):
        self.confidence = 1-(np.random.rand()*0.02)


        #input is in pixel format
    def updatePosition(self,timestamp:int, absCenter:Point, absWH:Point):
        
        self.rect = rect_(absCenter,absWH)

        self.timestamp = timestamp
        self.existTime +=1

        self.center = absToRatio(self.imgSize, absCenter)
        self.wh = absToRatio(self.imgSize, absWH)
        
        self.absW, self.absH = absWH.w , absWH.h
        self.LU = calcLU(absCenter, absWH)
        self.BR = Point(self.LU.x+absWH.w, self.LU.y+absWH.h)

        self.areaRatio = self.wh[0]*self.wh[1]
        self.absArea = absWH.w*absWH.h
        
        self.absCenter = absCenter
        self.absWH = absWH
        self.updateConfidence()
        if (np.random.rand() < catagoryMissRate[self.catagory]):
            self.isDetected = False
        else:
            self.isDetected = True

    def getDet(self):
        return ("{},{},{},{},{},{},{},{}".format(self.timestamp, self.frameID, self.catagory, self.confidence\
                , self.center.x, self.center.y, self.wh.x, self.wh.y))

    def getRect(self):
        return [self.absCenter.x, self.absCenter.y, \
            self.absWH.w, self.absWH.h]