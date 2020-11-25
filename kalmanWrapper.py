import numpy as np
from filterpy.kalman import KalmanFilter
import math

class kalmanWrapper(object):
    
    def __init__(self,bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        
        self.kf.R[2:, 2:] *= 10
        self.kf.Q[-1, -1] *= 0.01 
        self.kf.Q = np.array([[0.01, 0, 0, 0, 0.01, 0, 0],
                              [0, 0.01, 0, 0, 0, 0.01, 0],
                              [0, 0, 0.01, 0, 0, 0, 0.01],
                              [0, 0, 0, 0.01, 0, 0, 0],
                              [0.01, 0, 0, 0, 0.1, 0, 0],
                              [0, 0.01, 0, 0, 0, 0.1, 0],
                              [0, 0, 0.01, 0, 0, 0, 0.1]])

        self.kf.x[:4] = self.get_xysr_rect(bbox) 
        

    def get_rect_xysr(self,p):
          
        cx, cy, s, r = p[0], p[1], p[2], p[3]
        w = math.sqrt(abs(s*r))
        h = abs(float(s/w))
        x = float(cx-w/2.)
        y = float(cy-h/2.)

        if (x < 0 and cx > 0):
            x = 0
        
        if (y < 0 and cy > 0):
            y = 0

        return np.array([x, y, w, h])

    def get_xysr_rect(self, bbox):
        cx = bbox[0]+bbox[2]/2. 
        cy = bbox[1]+bbox[3]/2. 
        s = bbox[2]*bbox[3]
        r = float(bbox[2])/bbox[3]
        
        return np.array([cx, cy, s, r])


    def predict(self): 
        self.kf.predict()
        predictBox = self.getState()
        nextEstimatedRect = predictBox
        return nextEstimatedRect

    def update(self, bbox):
    
        measurement = self.get_xysr_rect(bbox)
        self.kf.update(measurement)
        self.lastUpdateRect = bbox


        
    def getState(self):
        return self.get_rect_xysr(self.kf.x)
