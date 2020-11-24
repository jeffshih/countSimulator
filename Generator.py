from typing import Generator
import numpy as np
import scipy as sp
from Util import *
import datetime
import time 
import csv
from detection import detectionResult
from datetime import datetime
import csv
from config import *


class detGenerator(object):
    
    def __init__(self, resolution = (1080,720), minObj=5, maxObj=10):
        self.width = resolution[0]
        self.height = resolution[1]
        self.wh = Point(resolution[0],resolution[1])
        self.maxObj = maxObj
        self.minObj = minObj
        self.imgId = 0

        #for validation
        self.objId = 0
        self.objCatagories = {1:0,2:0,3:0,4:0,5:0}
        
        self.objectHolds = {}
        
        self.initObjectCount = np.random.randint(minObj,maxObj+1)
        self.objectCount = self.initObjectCount
        
        #result to return
        self.detectionResultList = []
        self.statOfFrame = np.zeros(100)

        #background image
        self.bg = rect_(Point(self.width/2, self.height/2), self.wh)

        self.validGenAreaY = np.ones(5)
        self.objToCreate = []
    
        #self.validGenAreaX = np.ones((5,2))
        #init first frame
        self.initFrame()
        
    def initFrame(self):
        #first frame the object can be anywhere, just avoid to much overlapping
        det = self.generateObj(isFirst=True)
        self.addObj(det)    
        for i in range(1,self.initObjectCount): 
            tempDet = self.generateObj(isFirst=True)
            while self.checkDetOverLap(tempDet, 0.2) is not True:
                tempDet = self.generateObj(isFirst=True)
            self.addObj(tempDet)
        self.imgId+=1
    
    def addObj(self, det:detectionResult, isFirst=False):
        
        if not isFirst:
            #occupiedX = int(det.center.x// 0.2)
            occupiedY = int(det.center.y // 0.2)
            self.validGenAreaY[occupiedY] = 0
            #self.validGenAreaX[occupiedX][occupiedY] = 0
        self.objectHolds[self.objId] = det 
        self.objId += 1
        self.statOfFrame[self.imgId]+=1

    def generateObj(self, isFirst=False):
        now = datetime.now()
        currentTimestamp = int(datetime.timestamp(now))
        catagory, absC, absWH = self.constructRect(isFirst)
        det = detectionResult(currentTimestamp, catagory, self.objId, absWH, absC, resolution)
        return det

    def constructRect(self, isFirst):
        catagory = np.random.randint(1,6)
        sizeVariation = [i+0.99 for i in np.random.rand(2)*0.2]    
        
        #Object always appear from left except first frame, 
        #I assume it center appear from 1/10 to 6/10 of the frame
        #each object might appear 3~5 frame
        centerRangeX = 0.1+np.random.rand()*0.2
        absCenterX = np.random.rand()*self.width if isFirst==True else centerRangeX*self.width  

        #get object width and height
        w, h = catagorySize[catagory][0], catagorySize[catagory][1]
        
        #Object won't directly has large overlap, it might continuously appear
        #but not overlap

        
        possibleIdx, = np.nonzero(self.validGenAreaY)
        possibleY = np.random.randint(5) if possibleIdx.size == 0 else np.random.choice(possibleIdx)
        
        #possibleIdx, = np.nonzero(self.validGenAreaY)
        #possibleY = np.random.choice(possibleIdx)
        centerRangeY = 0.2*possibleY+(np.random.rand()-0.5)*0.1
        absCenterY = centerRangeY*self.height
        
        #make variation of detection bot
        absWH = Point(w*sizeVariation[0],h*sizeVariation[1])
        absC = Point(absCenterX, absCenterY)
        return catagory, absC, absWH


    def move(self, det: detectionResult):

        #each move has 5% variation
        variation = 0.95 + np.random.rand()*0.1 
        #w and h might change due to detection 
        variationW, variationH = [i+0.99 for i in np.random.rand(2)*0.2]
        stride = det.xStride*variation
        #x move horizontal, with same speed and 5% variation
        Cx = det.absCenter.x + stride
        #print("obj {} move from {} to {} with life {}".format(det.objId,det.absCenter.x, Cx, det.existTime))
        #theoretically y won't move but i give it a same variation range
        #corespond to x
        Cy = det.originAbsY + (det.xStride-stride)
        newW = det.orW*variationW
        newH = det.orH*variationH
        now = datetime.now()
        currentTimestamp = int(datetime.timestamp(now))
        ncr, nwhr = Point(Cx, Cy), Point(newW, newH)
        det.updatePosition(currentTimestamp, ncr, nwhr)

    def createNewObject(self, genCount:int):
        #print("Frame : {}".format(self.imgId))
        #print("generating {} object".format(genCount))
        #print(self.validGenAreaY)
        objToCreate = []
        for i in range(genCount):
            tempDet = self.generateObj()
            while self.checkDetOverLap(tempDet, 0.1) is not True:
                tempDet = self.generateObj()
        
        self.addObj(tempDet)
            

    def checkDetOverLap(self, det:detectionResult, thresh):
        for k, holdDet in self.objectHolds.items():
            if IOU(det.rect, holdDet.rect) > thresh:
                return False 
        return True 

    def populateFrame(self):

        #each Frame reset the valid y range
        self.validGenAreaY = np.ones(5)
        toRemove = []
        for key, det in self.objectHolds.items():
            self.move(det)
            xb, yb = resolution[0], resolution[1]
            tc = det.absCenter
            if det.existTime == det.lifespan or overlap(self.bg, det.rect) < 0.2:
                toRemove.append(key)

        for k in toRemove:
            del self.objectHolds[k]

        self.objectCount = len(self.objectHolds)
        l = abs(self.minObj - self.objectCount)
        complement = self.maxObj - self.objectCount
        objToGen = 0
        if self.objectCount < self.minObj:
            objToGen = np.random.randint(l, self.minObj+1)
        elif np.random.rand() < 0.7:
            objToGen = np.random.randint(complement)
        
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
                renderRect(det.rect, backGround, det.catagory-1)
            cv2.imshow("background", backGround)
            cv2.waitKey(300)
            


if __name__=="__main__":
    detGen = detGenerator(minObj=5,maxObj=10)
    detGen.display()
    #detGen.run()
    #res = detGen.getDetectionRes()
    #lstIdx = "1"
    #print(detGen.statOfFrame)
    #print(res) 
    #transformedRes = transform(res)
    #detGen.toCsv()
    #print(transformedRes)

        

