from detectionParser import detectionParser
from Util import makeValidRange, measurementToRect, transform
import numpy as np
import cv2
from dataStructure import rect_, state
from kalmanFilter import kalmanFilter
from Point import Point
from math import sqrt
from Generator import detGenerator
from config import *
from numpy.random import randint as rnd



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
        self.history = []

    def initTracker(self, det:rect_):
        st = state(det)
        measurement = st.getMeasurement()
        self.kf = kalmanFilter(x0=measurement)


    def setTracked(self, det:rect_, confidence:float):
        self.update(det)
        self.lastHistMat = det 
        self.confidence.append(confidence)


    def checkStatus(self):
        return self.status

    def predict(self):

        bbox = self.kf.predict(u=self.controlMatrix)
        
        self.lifespan +=1
        if bbox[2][0] > 0 and bbox[3][0] > 0:
            self.estimateBox = measurementToRect(bbox)
        else:
            bbox = rect_(Point(10,10),Point(10,10))
            self.status = 2
            self.estimateBox = bbox 
            return self.estimateBox
    

        #update 3 times is healthy enough
        
        if self.updateTimes == 3:
            self.status = 1
        
        #if prediction is already out of boundary
        elif self.estimateBox.center.x > resolution[0] or self.estimateBox.center.x < 0:
            self.status = 2

        elif self.estimateBox.center.y > resolution[1] or self.estimateBox.center.y < 0:
            if (self.updateTimes > 2):
                self.status = 2
            else:
                self.status = 4

        elif self.lifespan > 3 :
            self.status = 3

        #if the tracker move 3 frame but no update-> unhealthy
        elif self.lifespan - self.lastUpdateTime > 3:
            self.status = 4
        # healthy and young tracker, keep tracking
        else:
            self.status = 0

        makeValidRange(self.estimateBox)
        self.history.append(self.estimateBox.center)

        return self.estimateBox
        
    def update(self, det:rect_):

        self.updateTimes +=1
        #every frame tracker will update
        self.lastUpdateTime = self.lifespan
        st = state(det)
        measurement = st.getMeasurement()
        self.kf.update(measurement)
        self.lastUpdateRect = det

    def getRect(self):
        return self.estimateBox


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