from detectionParser import detectionParser
from Util import makeValidRange, measurementToRect, transform, wapperM2R
import numpy as np
import cv2
from dataStructure import rect_, state
from kalmanFilter import kalmanFilter
from Point import Point
from math import sqrt
from Generator import detGenerator
from config import *
from numpy.random import randint as rnd
from kalmanWrapper import kalmanWrapper


class Tracker(object):

    ##tracker were init in pixel cooridinate

    def __init__(self, id, rect:rect_, catagory:int, confidence:float):

        self.trackerId = id
        self.initRect = rect
        self.estimateBox = rect
        self.lifespan = 0
        self.updateTimes = 0
        self.lastUpdateRect = rect
        self.catagory = catagory
        self.lastUpdateTime = 0
        self.controlMatrix = np.array([[rect.width//2],[0],[0],[0]])

        self.color = (rnd(255), rnd(255), rnd(255))

        self.confidence = [confidence]
        self.status = 0
        #kalman filter matrix
        
        #self.kf = cv2.KalmanFilter()
        self.initTracker(rect)
        #print("create new tracker {}".format(id))
        self.history = [rect]

    def initTracker(self, det:rect_):
        measurement = state(det).getMeasurement()
        #self.kf = kalmanWrapper(measurement)
        self.kf = kalmanFilter(x0=measurement)

    def setTracked(self, det:rect_, confidence:float):
        self.update(det)
        self.lastHistMat = det 
        self.confidence.append(confidence)


    def predict(self):

        #bbox = self.kf.predict(u=self.controlMatrix)
        bbox = self.kf.predict()
        #print(bbox)
        
        self.lifespan +=1
        if bbox[2] > 0 and bbox[3] > 0:
            self.estimateBox = measurementToRect(bbox)
            #self.estimateBox = wapperM2R(bbox)
        else:
            bbox = rect_(Point(10,10),Point(10,10))
            self.status = 2
            self.estimateBox = bbox 
            return self.estimateBox
    
        #update 3 times is healthy enough        
        if self.updateTimes == 3:
            self.status = 1        
        #if prediction is already out of boundary
        elif self.estimateBox.center.x > resolution[0] and self.updateTimes >1:
            self.status = 1
        elif self.estimateBox.center.y > resolution[1] or self.estimateBox.center.y < 0:
            self.status = 2
        elif self.lifespan > 2 :
            self.status = 1
        #if the tracker move 3 frame but no update-> unhealthy
        elif self.lifespan - self.lastUpdateTime > 2:
            self.status = 2
        # healthy and young tracker, keep tracking
        else:
            self.status = 0

        makeValidRange(self.estimateBox)
        self.history.append(self.estimateBox)

        return self.estimateBox
        
    def update(self, det:rect_):

        self.updateTimes +=1
        self.lastUpdateTime = self.lifespan
        measurement = state(det).getMeasurement()
        self.kf.update(measurement)
        self.lastUpdateRect = det

    def getRect(self):
        return self.estimateBox


class history(object):
    
    def __init__(self, frameNum, trk:Tracker):
        self.frameNum = frameNum
        self.rects = {trk.trackerId:[trk.estimateBox]}
        self.colors = {trk.trackerId:trk.color}

    def add(self, trk:Tracker):
        self.rects[trk.trackerId].append(trk.estimateBox)

    def __str__(self):
        return ' '.join(['{}'.format(i) for i in self.rects])



if __name__=="__main__":
    detGen = detGenerator(minObj=1,maxObj=2)
    res = detGen.getDetectionRes()
    transformedData = transform(res)

    detParser = detectionParser()
    start = detParser.stringToDet(res[0])
    detBox = rect_(start.relativeCenter, start.relativeWH)
    catagory = start.catagory
    trk = Tracker(1, detBox, catagory)
    p = trk.predict()
    print(p)

    for idx, detLine in enumerate(res):
        det = detParser.stringToDet(detLine)
        detBox = rect_(det.relativeCenter, det.relativeWH)
        catagory = det.catagory
        trk.setTracked(detBox)
        p = trk.predict()
        print(p)