import numpy as np
from numpy.lib.function_base import average
import scipy as sp
from typing import Callable, Any, Iterable
from math import sqrt
from Point import Point
from config import *



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
    def __init__(self, frameNum, trkList, detList, pair=None):
        self.pair = []
        self.estimateBoxs = []
        self.catagories = []
        self.lastUpdateRects = []
        self.confidence = []
        self.frameNum = frameNum
        self.historyList = []
        self.trackIds = []
        self.detList = []
        self.colors = []
        self.detColors = []
        self.trackStatus = []
        for trkId, trk in trkList.items():
            self.estimateBoxs.append(trk.estimateBox)
            self.lastUpdateRects.append(trk.lastUpdateRect)
            self.historyList.append(trk.history)
            self.trackIds.append(trkId)
            self.colors.append(trk.color)
            self.trackStatus.append(trk.status)
            sumConf = 0
            for c in trk.confidence:
                sumConf += c 
            self.confidence.append(sumConf/len(trk.confidence))

        for det in detList:
            detBox = det.rect
            self.detList.append(detBox)
            self.catagories.append(det.catagory)
           
#basic rect box for passing, x and y denote center
class rect_(object):
    #input are pixels
    def __init__(self, center:Point, wh:Point):
        self.width = wh.w
        self.height = wh.h
        self.center = center 
        self.x = center.x 
        self.y = center.y
        self.wh = wh
        self.rx = center.x/resolution[0]
        self.LU = calcLU(center,wh)
        self.BR = Point(self.LU.x + wh.w, self.LU.y+wh.h)
        self.area = wh.w*wh.h 

    def __str__(self):
        return "{},{},{},{}".format(self.center.x, self.center.y, self.width, self.height)

    def getXYSR(self):
        return np.array([self.x,self.y,self.area,self.width/self.height])
    



#for kalman filter tracking
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
        return rect_(Point(x,y), Point(w,h))

    def __str__(self):
        return "{}".format(self.measurement)

def calcLU(center:Point, wh:Point):
    nx, ny = center.x - wh.w/2, center.y - wh.h/2
    nx = 0 if nx < 0 else nx 
    ny = 0 if ny < 0 else ny
    return Point(nx,ny)