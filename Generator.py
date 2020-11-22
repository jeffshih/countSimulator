from typing import Generator
import numpy as np
import scipy as sp
from Util import *
import datetime
import time 
from detection import detectionResult, IOU
from datetime import datetime
import csv



class detGenerator(object):
    
    def __init__(self, size=200, resolution = (1080,720), minObj=5, maxObj=10):
        self.objSize = size
        self.width = resolution[0]
        self.height = resolution[1]
        self.wh = Pair(resolution[0],resolution[1])
        self.maxObj = maxObj
        self.minObj = minObj
        self.imgId = 0
        self.objId = 0
        self.objectHolds = {}
        self.initObjectCount = np.random.randint(minObj,maxObj+1)
        self.objectCount = self.initObjectCount
        self.detectionResultList = []
        self.statistics = [0 for _ in range(100)]
        self.initFrame()
        
    def initFrame(self):
        for i in range(self.initObjectCount):
            det = self.generateObj(isFirst=True)
            #print(i)
            if self.objId == 0:
                self.objectHolds[self.objId] = det
                self.objId+=1
                self.statistics[self.imgId]+=1
                continue
            holdDet = self.objectHolds[i-1]
            tempDet = self.generateObj(isFirst=True)
            while IOU(tempDet, holdDet) < 0.3:
                tempDet = self.generateObj(isFirst=True)
                if IOU(tempDet, holdDet) < 0.3:
                    break
            self.objectHolds[self.objId] = tempDet
            self.objId+=1
            self.statistics[self.imgId]+=1
        self.imgId+=1
            

    def generateObj(self, isFirst=False):
        catagory = np.random.randint(1,6)
        now = datetime.now()
        currentTimestamp = int(datetime.timestamp(now))
        sizeVariation = [i+0.95 for i in np.random.rand(2)/10]
        relWH = Pair(catagorySize[catagory][0],catagorySize[catagory][1])
        #random position of x, appear from the left
        if isFirst:
            relCenterX = np.random.rand()*(self.width)
        else:
            relCenterX = np.random.rand()*(self.width/5)
        #make object only appear between 0.1*h and 0.9*h
        relCenterY = np.random.rand()*(self.height*9/10)+self.height/10
        #make variation of detection bot
        varWH = Pair(relWH.x*sizeVariation[0],relWH.y*sizeVariation[1])
        relC = Pair(relCenterX, relCenterY)
        det = detectionResult(currentTimestamp, catagory, self.objId, varWH, relC, self.wh)
        return det


    def move(self, det: detectionResult):
        variationX, variationY = [i+0.95 for i in np.random.rand(2)/10]
        variationW, variationH = [i+0.95 for i in np.random.rand(2)/10]
        newCx = det.relativeCenter.x + det.stride.x*variationX
        newCy = det.relativeCenter.y + det.stride.y*(1-variationY)
        newW = det.relativeWH[0]*variationW
        newH = det.relativeWH[1]*variationH
        now = datetime.now()
        currentTimestamp = int(datetime.timestamp(now))
        ncr = Pair(newCx, newCy)
        nwhr = Pair(newH,newW)
        det.updatePosition(currentTimestamp, ncr, nwhr)

    def printCurrentFrame(self):
        for key, det in self.objectHolds.items():
            print(self.constructDet(det))

    def createNewObject(self, genCount:int):
        for i in range(genCount):
            tempDet = self.generateObj()
            for k, holdDet in self.objectHolds.items():
                while IOU(tempDet, holdDet) < 0.2:
                    tempDet = self.generateObj()
                    if IOU(tempDet, holdDet) < 0.2:
                        break
            self.objId+=1
            self.objectHolds[self.objId] = tempDet 
            self.statistics[self.imgId]+=1

    def populateFrame(self):
        toRemove = []
        for key, det in self.objectHolds.items():
            self.move(det)
            if det.existTime >= 5 :
                toRemove.append(key)

        for k in toRemove:
            del self.objectHolds[k]
        self.objectCount = len(self.objectHolds)
        complement = self.maxObj-len(self.objectHolds)

        if self.objectCount < self.maxObj:
            objToGen = np.random.randint(complement)
            self.createNewObject(objToGen)
        self.imgId +=1
        
    
    def run(self):
        self.printCurrentFrame()
        while(self.imgId < 100):
            self.populateFrame()
            self.printCurrentFrame()

    def getDetectionRes(self):
        for key, det in self.objectHolds.items():
            self.detectionResultList.append(self.constructDet(det))
        while(self.imgId < 100):
            self.populateFrame()
            for key, det in self.objectHolds.items():
                self.detectionResultList.append(self.constructDet(det))
        return self.detectionResultList

    def constructDet(self, det:detectionResult):
        return ('{},{},{},{:.4f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(det.timestamp, self.imgId, det.catagory, det.confidence\
                    , det.center.x, det.center.y, det.wh.x, det.wh.y))


    def toCsv(self, filename="detResult.csv"):
        with open(filename, 'a') as f:
            for key, det in self.objectHolds.items():
                f.writelines(self.constructDet(det))
            while(self.imgId < 100):
                self.populateFrame()
                for key, det in self.objectHolds.items():
                    f.writelines(self.constructDet(det))           


if __name__=="__main__":
    detGen = detGenerator()
    #detGen.run()
    res = detGen.getDetectionRes()
    #lstIdx = "1"
    print(detGen.statistics)
    #print(res) 
    transformedRes = transform(res)
    detGen.toCsv()
    #print(transformedRes)

        

