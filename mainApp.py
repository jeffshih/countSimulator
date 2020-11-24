import numpy
from Generator import detGenerator, detectionResult
from Util import *
from trackerManager import trackerManager
from Point import Point
from detectionParser import detectionParser
from config import *

class mainProgram(object):

    def __init__(self, imgSize=resolution):

        self.objectCount = 0
        self.currentFrame = 1
        self.width = imgSize.w
        self.width = imgSize.h
        self.detParser = detectionParser()
        self.detectionSequence = {}
        self.momTracker = trackerManager()
        self.countingResult = {}

    def initWithData(self, detRes:list):
        self.detectionSequence = self.detParser.getDetSequence(transform(detRes))


    def doCounting(self):
        for frameNum, dets in self.detectionSequence.items():
            self.momTracker.doTracking(dets, frameNum)
            messages = self.momTracker.predict()
            for trkId, msg in messages.items():
                print("At frameNum: {}, trkId {} was counted as type {} with confidence {:.4f}".\
                    format(frameNum, trkId, msg.catagory, msg.confidence))
           

    def genCountingResult(self):
        for frameNum, dets in self.detectionSequence.items():
            self.momTracker.doTracking(dets, frameNum)
            messages = self.momTracker.predict()
            self.countingResult[frameNum] = []
            for trkId, msg in messages.items():
                self.countingResult[frameNum].append(msg.catagory)
        return self.countingResult

    def printSequence(self):
        for frameNum, dets in self.detectionSequence.items():
            print(frameNum, len(dets))

if __name__ == "__main__":
    detGen = detGenerator(minObj=5, maxObj=10,framenum=10)
    res = detGen.getDetectionRes()
    App = mainProgram()
    App.initWithData(res)
    #print(res)
    App.doCounting()

    




