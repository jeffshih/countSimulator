import numpy
from Generator import detectionResult
from Util import *
from trackerManager import trackerManager
from Pair import Pair
from detectionParser import detectionParser

class mainProgram(object):

    def __init__(self, detRes:list, imgSize:Pair):
        
        self.detectionSequence = transform(detRes)
        self.objectCount = 0
        self.currentFrame = 1
        self.width = imgSize.w
        self.width = imgSize.h
        self.detParser = detectionParser(imgSize)
        self.objectCounter = trackerManager()
    
    def populateFrameDetection(self):
        currentFrameRes = []
        frameData = self.detectionSequence[self.currentFrame]
        for detString in frameData :
            det = self.detParser.stringToDet(detString)
            currentFrameRes.append(det)
        self.currentFrame +=1
        return currentFrameRes
    
    def setSpecificFrame(self, frameNum:int):
        self.currentFrame = frameNum



