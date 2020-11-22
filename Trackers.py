import numpy as np
import cv2
from numpy.testing._private.utils import measure
from dataStructure import rect_
from kalmanFilter import kalmanFilter
from Pair import Pair
from math import sqrt

class state(object):
    
    def __init__(self, bbox:rect_):
        self.measurement = np.zeros((4,1))
        self.measurement[0][0] = bbox.center.x
        self.measurement[1][0] = bbox.center.y
        self.measurement[2][0] = bbox.area
        self.measurement[3][0] = bbox.width/bbox.height


    def getMeasurement(self):
        return self.getMeasurement

    def toRect(self):
        x = self.measurement[0][0]
        y = self.measurement[1][0]
        w = sqrt(self.measurement[2][0]*self.measurement[3][0])
        h = sqrt(self.measurement[2][0]/self.measurement[3][0])
        return rect_(Pair(x,y), Pair(w,h))

class Tracker(object):

    ##tracker were init in pixel cooridinate

    def __init__(self, id, rect:rect_, catagory):

        self.trackerId = id
        self.initRect = rect
        self.estimateBox = rect
        self.lifespan = 0
        self.updateTimes = 0
        self.lastHistMatrix = rect
        self.lastUpdateRect = rect
        self.catagory = catagory

        #kalman filter matrix
        
        #self.kf = cv2.KalmanFilter()
        self.initTracker()

    def initTracker(self, det:rect_):
        st = state(det)
        measurement = st.getMeasurement()
        self.kf = kalmanFilter(x0=measurement)


    def setTracked(self, det:rect_):
        self.update(det)
        self.lastHistMat = det 



    def predict(self):
        bbox = self.kf.predict()
        self.lifespan +=1
        if bbox[2][0] > 0 and bbox[3][0] > 0:
            self.estimateBox = bbox
        else:
            tempBox = rect_(Pair(10,10),Pair(10,10))
            st = state(tempBox)
            bbox = st.getMeasurement()
            self.estimateBox = bbox 
        

    def update(self, det:rect_):

        self.updateTimes +=1
        measurement = state(det)
        self.kf.update(x0=measurement)
