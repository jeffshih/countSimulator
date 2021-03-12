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
import sys

class detGenerator(object):
    
    def __init__(self, resolution = (1080,720), minObj=5, maxObj=10, framenum=100):
        self.width = resolution[0]
        self.height = resolution[1]
        self.wh = Point(resolution[0],resolution[1])
        self.maxObj = maxObj
        self.minObj = minObj
        self.imgId = 0
        self.framenum = framenum
        #for validation
        self.objId = 0
        self.objCatagories = {1:0,2:0,3:0,4:0,5:0}
        
        self.lastCreateTime = 0
        self.objectHolds = {}
        
        self.leftMost = resolution[0]

        self.initObjectCount = np.random.randint(minObj,maxObj+1)
        self.objectCount = self.initObjectCount
        
        #result to return
        self.detectionResultList = []
        self.statOfFrame = np.zeros(100)
        self.countResult = {}
        self.totalObj = 0
        #background image
        self.bg = rect_(Point(self.width/2, self.height/2), self.wh)

        self.validGenAreaY = np.ones(5)
        
        #init first frame
        self.batchAddObject(self.initObjectCount)
        


    
    def addObj(self, det:detectionResult):        
        if self.imgId != 0:
            occupiedY = int(det.center.y // 0.2)
            self.validGenAreaY[occupiedY] = 0
        self.objectHolds[self.objId] = det 
        self.objId += 1
        self.statOfFrame[self.imgId-1]+=1
        
       

    def genObject(self):
        now = datetime.now()
        currentTimestamp = int(datetime.timestamp(now))
        
        catagory = np.random.randint(1,6)
        wVar, hVar  = [i+0.99 for i in np.random.rand(2)*0.2]    
        
        #Object always appear from left except first frame, 
        #each object might appear 3~5 frame
        #gen object width and height
        
        w, h = np.random.randint(100,250), np.random.randint(70, 170)
        
        if self.imgId == 0:
            absCenterX = np.random.rand()*self.width 
        else:
            xMid = self.leftMost/2
            absCenterX = min((0.5-np.random.rand())*w + xMid, self.width/5)
            
        #Object won't directly has large overlap, it might continuously appear
        #but not overlap
        
        possibleIdx, = np.nonzero(self.validGenAreaY)
        possibleY = np.random.randint(5) if possibleIdx.size == 0 else np.random.choice(possibleIdx)
        
        centerRangeY = 0.2*possibleY+0.1+(np.random.rand()-0.5)*0.1
        absCenterY = centerRangeY*self.height
        
        
        #make variation of detection bot
        absWH = Point(w*wVar, h*hVar)
        absC = Point(absCenterX, absCenterY)
        return detectionResult(currentTimestamp, catagory, self.objId, absWH, absC, resolution)
       

    def move(self, det: detectionResult):

        #each move has 5% variation
        variation = 0.95 + np.random.rand()*0.1 
        #w and h might change due to detection 
        variationW, variationH = [i+0.99 for i in np.random.rand(2)*0.2]
        stride = det.xStride*variation
        #x move horizontal, with same speed and 5% variation
        Cx = det.absCenter.x + stride
        
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
        self.batchAddObject(genCount)   
        self.lastCreateTime = 0
        self.leftMost = self.width

    def batchAddObject(self, num):
        for i in range(num):
            tempDet = self.genObject()
            while self.checkDetOverLap(tempDet, 0.3) is not True:
                tempDet = self.genObject()
            self.addObj(tempDet)

    def updateLeftMostPoint(self):
        for key, det in self.objectHolds.items():
            self.leftMost = min(det.LU.x, self.leftMost)
            

    def checkDetOverLap(self, det:detectionResult, thresh):
        if len(self.objectHolds) == 0:
            return True
        for k, holdDet in self.objectHolds.items():
            if IOU(det.rect, holdDet.rect) > thresh:
                return False 
        return True 

    def populateFrame(self):

        #each Frame reset the valid y range
        self.validGenAreaY = np.ones(5)
        toRemove = []
        self.countResult[self.imgId] = []        
        for key, det in self.objectHolds.items():
            self.move(det)
            if det.existTime == det.lifespan :
                toRemove.append(key)
                self.countResult[self.imgId].append(det.catagory)       

        for k in toRemove:
            del self.objectHolds[k]

        self.updateLeftMostPoint()

        self.objectCount = len(self.objectHolds)
        l = abs(self.minObj - self.objectCount)
        complement = self.maxObj - self.objectCount
        #print("lower bound {}".format(l))
        #print("object num {} and complement {}".format(self.objectCount, complement))

        #must create new object 
        if self.objectCount < self.minObj:
            objToGen = np.random.randint(l, complement+1)
        else:
            objToGen = np.random.randint(complement+1)
        #print("Current object hold are {} and generating {}".format(self.objectCount, objToGen))
        self.createNewObject(objToGen)
        
        self.lastCreateTime +=1    
        self.imgId +=1
    
    def constructDet(self, det:detectionResult):
        return ('{},{},{},{:.4f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(det.timestamp, self.imgId, det.catagory, det.confidence\
                    , det.center.x, det.center.y, det.wh.x, det.wh.y))



    def reset(self):
        self.imgId = 0
        self.objId = 0
        self.objCatagories = {1:0,2:0,3:0,4:0,5:0}
        
        self.lastCreateTime = 0
        self.objectHolds.clear()
        
        self.leftMost = self.width

        self.initObjectCount = np.random.randint(self.minObj,self.maxObj+1)
        self.objectCount = self.initObjectCount
        
        #result to return
        self.detectionResultList.clear()
        self.statOfFrame = np.zeros(100)
        self.countResult.clear()
        self.batchAddObject(self.initObjectCount)


        

    '''
    below are method for output
    '''

    def printCurrentFrame(self, Output=True):
        if not Output :
            return
        for key, det in self.objectHolds.items():
            print(self.constructDet(det))


    def getDetectionRes(self, useObj=False):
        while(self.imgId < self.framenum):
            for key, det in self.objectHolds.items():
                if useObj:
                    self.detectionResultList.append(det)
                else:
                    self.detectionResultList.append(self.constructDet(det))
            self.populateFrame()       
        return self.detectionResultList


    def toCsv(self, filename="detResult.csv"):
        with open(filename, 'a') as f:
            while(self.imgId < self.framenum):
                for key, det in self.objectHolds.items():
                    f.write("{}\n".format(self.constructDet(det)))        
                self.populateFrame()
            for key, det in self.objectHolds.items():
                f.write("{}\n" .format(self.constructDet(det)))
            
    def display(self):
        while(self.imgId < self.framenum):
            backGround = np.zeros((720, 1080, 3), np.uint8)
            backGround[:,:,:] = (255,255,255)
            self.populateFrame()
            for key, det in self.objectHolds.items():
                renderRect(det.rect, backGround, det.catagory-1)
            cv2.imshow("background", backGround)
            k = cv2.waitKey(0)
            if k == 'n':                       
                continue
            elif k == 27:  #escape key 
                break
        cv2.destroyAllWindows()

    def run(self, Output=True):
        while(self.imgId <= self.framenum):
            self.printCurrentFrame(Output)
            self.populateFrame()

    def getGroundTruth(self):
        return self.countResult


if __name__=="__main__":
    arg = str(sys.argv)
    detGen = detGenerator(minObj=5,maxObj=10, framenum = 100)
    if "display" in arg:
        detGen.display()
    elif "save" in arg:
        detGen.toCsv()
    else:
        detGen.run()
    print(detGen.getGroundTruth())

