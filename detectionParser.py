import numpy as np
from detection import detectionResult
from Pair import Pair
import string
from Util import *
from config import *
from Generator import detGenerator

class detectionParser(object):

    def __init__(self):
        
        self.frame = []
        self.detectionSequence = {}
        self.frameNum = 0

    def stringToDet(self, inputString):
        
        strSplit = inputString.split(",")
        TimeStamp = int(strSplit[0])
        currentFrame = int(strSplit[1])
        catagory = int(strSplit[2])
        confidence = float(strSplit[3])
        cx = float(strSplit[4])
        cy = float(strSplit[5])
        w = float(strSplit[6])
        h = float(strSplit[7])
        center = Pair(cx,cy)
        wh = Pair(w,h)
        relCenter = ratioToAbs(resolution, center)
        relWH = ratioToAbs(resolution, wh)
        det = detectionResult(TimeStamp, catagory, currentFrame, relWH\
            ,relCenter, resolution, confidence=confidence)
        return det

    def getDetSequence(self, detDict:dict):
        for frameNum, detList in detDict.items():
            self.detectionSequence[frameNum] = list()
            for detLines in detList:
                self.detectionSequence[frameNum].append(self.stringToDet(detLines))
        return self.detectionSequence



if __name__ == "__main__":
    detGen = detGenerator(minObj=5,maxObj=10)
    res = detGen.getDetectionRes()
    transformedData = transform(res)

    detParser = detectionParser()
    detectionSequence = detParser.getDetSequence(transformedData)

    for frameNum, dets in detectionSequence.items():
        print(frameNum, len(dets))