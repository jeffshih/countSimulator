import numpy as np
from numpy.lib.function_base import average
import scipy as sp
from typing import Callable, Any, Iterable
from math import sqrt
from Pair import Pair



class trackerMessage(object):
    
    def __init__(self, trk):
        self.trkId = trk.trackerId
        sumConf = 0
        for c in trk.confidence:
            sumConf += c
        self.confidence = sumConf/len(trk.confidence)
        self.catagory = trk.catagory
        
    def __str__(self):
        return "catagory: {}, confidence: {:.4f}, trackerId: {}".format(self.catagory, \
            self.confidence, self.trkId)

class msgForRender(object):
    def __init__(self, frameNum, trkList, detList):
        self.estimateBoxs = []
        self.catagories = []
        self.lastUpdateRects = []
        self.confidence = []
        self.frameNum = frameNum
        self.historyList = []
        self.trackIds = []
        self.detList = []
        for trkId, trk in trkList.items():
            self.estimateBoxs.append(trk.estimateBox)
            self.lastUpdateRects.append(trk.lastUpdateRect)
            self.historyList.append(trk.history)
            self.trackIds.append(trkId)
            sumConf = 0
            for c in trk.confidence:
                sumConf += c 
            self.confidence.append(sumConf/len(trk.confidence))

        for det in detList:
            detBox = rect_(det.relativeCenter, det.relativeWH)
            self.detList.append(detBox)
            self.catagories.append(det.catagory)

class rect_(object):
    #input are pixels
    def __init__(self, center:Pair, wh:Pair):
        self.width = wh[0]
        self.height = wh[1]
        self.center = center 
        self.wh = wh
        self.LeftUpper = convertLU(center,wh)
        self.area = wh[0]*wh[1]
    
    def __str__(self):
        return "{},{},{},{}".format(self.center.x, self.center.y, self.width, self.height)
    

class state(object):
    
    def __init__(self, bbox:rect_):
        self.measurement = np.zeros((4,1))
        self.measurement[0][0] = bbox.center.x
        self.measurement[1][0] = bbox.center.y
        self.measurement[2][0] = bbox.area
        self.measurement[3][0] = bbox.width/bbox.height


    def getMeasurement(self):
        return np.array(self.measurement)

    def toRect(self):
        x = self.measurement[0][0]
        y = self.measurement[1][0]
        w = sqrt(self.measurement[2][0]*self.measurement[3][0])
        h = sqrt(self.measurement[2][0]/self.measurement[3][0])
        return rect_(Pair(x,y), Pair(w,h))

def convertLU(center:Pair, wh:Pair):
    nx, ny = center.x - wh.w/2, center.y - wh.h/2
    nx = 0 if nx < 0 else nx 
    ny = 0 if ny < 0 else ny
    return Pair(nx,ny)