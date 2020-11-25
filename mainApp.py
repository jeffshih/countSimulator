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
            
            self.momTracker.update(dets)
            messages,renderData = self.momTracker.predict()
            
            for trkId, msg in messages.items():
                print("At frameNum: {}, captured type {} with confidence {:.4f}".\
                    format(frameNum, msg.catagory, msg.confidence))
            
    def genCountingResult(self):
        for frameNum, dets in self.detectionSequence.items():
            
            self.momTracker.update(dets)
            messages, renderData = self.momTracker.predict()
            
            self.countingResult[frameNum] = []
            for trkId, msg in messages.items():
                self.countingResult[frameNum].append(msg.catagory)
        self.momTracker.reset()
        return self.countingResult

    def printSequence(self):
        for frameNum, dets in self.detectionSequence.items():
            print(frameNum, len(dets))

    def printTrackerHistory(self):
        for frameNum, dets in self.detectionSequence.items():
            self.momTracker.update(dets)
            messages,renderData = self.momTracker.predict()    
            
        trackerHistory = self.momTracker.getTrackerHistory()
        transpose = {}
        colors = {}
        for idx, history in trackerHistory.items():
           transpose[history.frameNum]=history.rects
           colors[history.frameNum] = history.colors
           
        for frameNum, rectsPair in transpose.items():
            backGround = np.zeros((720, 1080, 3), np.uint8)
            backGround[:,:,:] = (255,255,255)
            for id, rects in rectsPair.items():
                for idx, rect in enumerate(rects):
                    color = colors[frameNum][id]
                    renderRectWithColor(rect,backGround,color)
                    renderTextUnderRect(rect,backGround,idx,color)
            cv2.imshow("blank", backGround)
            k = cv2.waitKey(0)
            if k == 'n':                       
                continue
            elif k == 27:  #escape key 
                break

if __name__ == "__main__":
    detGen = detGenerator(minObj=5, maxObj=10,framenum=100)
    res = detGen.getDetectionRes()
    App = mainProgram()
    App.initWithData(res)
    #print(res)
    App.doCounting()
    #App.printTrackerHistory()

    




