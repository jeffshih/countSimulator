import numpy
from Generator import detGenerator, detectionResult
from Util import *
from trackerManager import trackerManager
from Point import Point
from detectionParser import detectionParser
from config import *

class mainProgram(object):

    def __init__(self, detRes:list, imgSize:Point):

        self.objectCount = 0
        self.currentFrame = 1
        self.width = imgSize.w
        self.width = imgSize.h
        self.detParser = detectionParser()
        self.detectionSequence = self.detParser.getDetSequence(transform(detRes))
        self.momTracker = trackerManager()
    
    def doCounting(self):
        for frameNum, dets in self.detectionSequence.items():
            print(frameNum)
            self.momTracker.doTracking(dets)
            messages = self.momTracker.predict()
            for trkId, msg in messages.items():
                print("At frameNum: {}, trkId {} was counted as type {} with confidence {:.4f}".\
                    format(frameNum, trkId, msg.catagory, msg.confidence))



if __name__ == "__main__":
    detGen = detGenerator(minObj=5, maxObj=10)
    res = detGen.getDetectionRes()
    #print(res)
    mainProgram(res,resolution).doCounting()
    




