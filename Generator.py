from typing import Generator
import numpy as np
import scipy as sp
from Util import *
from Util import overlap as ovlp
import datetime
import time 
import csv
from detection import detectionResult, IOU
from datetime import datetime
import csv
from config import *


class detGenerator(object):
    
    def __init__(self, resolution = (1080,720), minObj=5, maxObj=10):
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
        self.bg = rect_(Pair(self.width/2, self.height/2), self.wh)
        self.initFrame()
        
    def initFrame(self):
        det = self.generateObj(isFirst=True)
        self.addObj(det)    
        for i in range(1,self.initObjectCount): 
            tempDet = self.generateObj(isFirst=True)
            while self.checkDetValid(tempDet, 0.2) is not True:
                tempDet = self.generateObj(isFirst=True)
            self.addObj(tempDet)
        self.imgId+=1
    
    def addObj(self, det:detectionResult):
        self.objectHolds[self.objId] = det 
        self.objId += 1
        self.statistics[self.imgId]+=1

    def generateObj(self, isFirst=False):
        now = datetime.now()
        currentTimestamp = int(datetime.timestamp(now))
        objAppearFrameCnt = np.random.randint(3,6)
        stride = objAppearFrameCnt
        leftBound = 1/(stride*2)
        catagory, absC, absWH = self.constructRect(leftBound, isFirst)
        tmp = rect_(absC, absWH)
        if ovlp(tmp,self.bg) <= (tmp.area*0.4) :
            catagory, absC, absWH = self.constructRect(leftBound, isFirst)
            tmp = rect_(absC, absWH)
        det = detectionResult(currentTimestamp, catagory, self.objId, absWH, absC, resolution, stride)
        return det

    def constructRect(self,lb, isFirst):
        catagory = np.random.randint(1,6)
        sizeVariation = [i+0.99 for i in np.random.rand(2)*0.2]
        centerRangeX = lb+np.random.rand()*lb
        #print("sizeVar: ", sizeVariation)
        relWH = Pair(catagorySize[catagory][0],catagorySize[catagory][1])
        #random position of x, appear from the left
        absCenterX = np.random.rand()*self.width if isFirst==True else centerRangeX*self.width
        #make object only appear between 0.1*h and 0.9*h
        absCenterY = generateNotSoRandomY()*self.height
        #make variation of detection bot
        absWH = Pair(relWH.w*sizeVariation[0],relWH.h*sizeVariation[1])
        absC = Pair(absCenterX, absCenterY)
        return catagory, absC, absWH


    def move(self, det: detectionResult):
        variationX, variationY = [i+0.95 for i in np.random.rand(2)/10]
        variationW, variationH = [i+0.99 for i in np.random.rand(2)*0.2]
        Cx = det.relativeCenter.x + det.stride.x*variationX
        Cy = det.originAbsY + det.stride.y*(1-variationY)
        newW = det.orW*variationW
        newH = det.orH*variationH
        now = datetime.now()
        currentTimestamp = int(datetime.timestamp(now))
        ncr, nwhr = Pair(Cx, Cy), Pair(newW, newH)
        det.updatePosition(currentTimestamp, ncr, nwhr)

    def createNewObject(self, genCount:int):
        for i in range(genCount):
            tempDet = self.generateObj()
            retryCounter = 0
            while self.checkDetValid(tempDet, 0.15) is not True:
                tempDet = self.generateObj()
                retryCounter +=1
                if self.objectCount > self.minObj and retryCounter > 100:
                    return
            self.addObj(tempDet)

    def checkDetValid(self, det:detectionResult, thresh):
        for k, holdDet in self.objectHolds.items():
            if IOU(det, holdDet) > thresh:
                return False 
        return True 

    def populateFrame(self):
        toRemove = []
        for key, det in self.objectHolds.items():
            self.move(det)
            xb, yb = resolution[0], resolution[1]
            tc = det.relativeCenter
            if det.existTime > 5 or tc.x > xb or tc.y > yb or tc.y < 0:
                toRemove.append(key)

        for k in toRemove:
            del self.objectHolds[k]

        self.objectCount = len(self.objectHolds)
        complement = self.maxObj-self.objectCount
        l = self.minObj - self.objectCount

        objToGen = 0
        if self.objectCount < self.maxObj:
            if self.objectCount > self.minObj:
                if np.random.rand() < 0.3:
                    objToGen = np.random.randint(complement)
            else:
                objToGen = np.random.randint(l, complement)
            self.createNewObject(objToGen)
        self.imgId +=1
        
    
    def run(self):
        self.printCurrentFrame()
        while(self.imgId < 100):
            self.populateFrame()
            self.printCurrentFrame()
    
    def printCurrentFrame(self):
        for key, det in self.objectHolds.items():
            print(self.constructDet(det))

    def getDetectionRes(self, useObj=False):
        for key, det in self.objectHolds.items():
            if useObj:
                self.detectionResultList.append(det)
            else:
                self.detectionResultList.append(self.constructDet(det))
        while(self.imgId < 100):
            self.populateFrame()
            for key, det in self.objectHolds.items():
                if useObj:
                    self.detectionResultList.append(det)
                else:
                    self.detectionResultList.append(self.constructDet(det))
        return self.detectionResultList


    def constructDet(self, det:detectionResult):
        return ('{},{},{},{:.4f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(det.timestamp, self.imgId, det.catagory, det.confidence\
                    , det.center.x, det.center.y, det.wh.x, det.wh.y))


    def toCsv(self, filename="detResult.csv"):
        with open(filename, 'a') as f:
            for key, det in self.objectHolds.items():
                f.write("{}\n" .format(self.constructDet(det)))
            while(self.imgId < 100):
                self.populateFrame()
                for key, det in self.objectHolds.items():
                    f.write("{}\n".format(self.constructDet(det)))        

    def display(self):
        while(self.imgId < 100):
            backGround = np.zeros((720, 1080, 3), np.uint8)
            backGround[:,:,:] = (255,255,255)
            self.populateFrame()
            for key, det in self.objectHolds.items():
                rect = rect_(det.relativeCenter, det.relativeWH)
                renderRect(rect, backGround)
            cv2.imshow("background", backGround)
            cv2.waitKey(1000)
            


if __name__=="__main__":
    detGen = detGenerator(minObj=5,maxObj=10)
    detGen.display()
    #detGen.run()
    #res = detGen.getDetectionRes()
    #lstIdx = "1"
    #print(detGen.statistics)
    #print(res) 
    #transformedRes = transform(res)
    #detGen.toCsv()
    #print(transformedRes)

        

