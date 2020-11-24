from numpy.lib.function_base import average
from numpy.testing._private.utils import break_cycles
from Generator import detGenerator
from Util import getMatchingCost, renderRectWithColor, renderTextUnderRect, transform, renderRect
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import *
from Trackers import Tracker
from dataStructure import rect_, trackerMessage, msgForRender
from Point import Point
from detectionParser import detectionParser
import cv2


class trackerManager(object):

    def __init__(self):
        self.trackerList = {}
        self.detectionBoxes = []
        self.predictionBoxes = []
        self.assignment = []
        self.currentTrackerID = 0
        self.objectCnt = {}
        self.frameNum = 0
        for i in range(1,6):
            self.objectCnt[i] = 0


    def doTracking(self, detections):
        self.frameNum +=1
        self.currentDet = detections
        self.detectionBoxes = []
        self.predictionBoxes = []
        self.assignment = []

        for det in detections:
            self.detectionBoxes.append(det.rect)
            #print("At frame:{}, create detection {}".format(frameNum,detBox))

        matchedPoints = {}
        trkIdInList = 0

        for trkId, t in self.trackerList.items():
            predBox = t.estimateBox
            self.predictionBoxes.append(predBox)
            #print("At frame:{}, hold prediction num {} ,{}".format(frameNum,trkId,predBox))
            matchedPoints[trkIdInList] = trkId
            trkIdInList +=1

        detCnt = len(self.detectionBoxes)
        predCnt = len(self.predictionBoxes)

        costMatrix = np.zeros((detCnt,predCnt),dtype=float)
        
        for i, detection in enumerate(self.detectionBoxes):
            for j, prediction in enumerate(self.predictionBoxes):
                cost = getMatchingCost(detection, prediction)
                costMatrix[i][j] = cost 
            

        self.assignment, col_ind = linear_sum_assignment(costMatrix)
        unmatchPred = set()
        unmatchDet = set()
        allItems = set()
        matchedItems = set()
        
        

        if(detCnt > predCnt):
            for i in range(detCnt):
                allItems.add(i)
            for i in range(predCnt):
                matchedItems.add(self.assignment[i])
            unmatchDet = allItems-matchedItems

        elif(detCnt < predCnt):
            for i in range(predCnt):
                if (i not in self.assignment):
                    unmatchPred.add(i)
        
        #print(self.assignment)
        for idx, assign in enumerate(self.assignment):
            confidence = self.currentDet[assign].confidence
            
            '''
            print("assign det {} to pred {}".format(assign, matchedPoints[idx]))
            
            print(self.detectionBoxes[assign])
            print(len(self.trackerList))
            print(self.trackerList[matchedPoints[idx]].getRect())
            '''

            self.trackerList[matchedPoints[idx]].setTracked(self.detectionBoxes[assign], confidence)
            #print(self.trackerList[matchedPoints[idx]])
            #pairDP

        for detIdx in unmatchDet:
            '''
            print("umatched det at idx {}".format(detIdx))
            '''
            newDet = self.currentDet[detIdx]
            
            newTracker = Tracker(self.currentTrackerID, newDet.rect, newDet.catagory, newDet.confidence)
            self.trackerList[self.currentTrackerID] = newTracker
            self.currentTrackerID += 1


    def checkStatus(self, tracker):
        statusCode = tracker.status
        id = tracker.trackerId
        '''
        if statusCode == 1:
            print("very healthy tracker tracker {}".format(id))
        elif statusCode == 2:
            print("Tracker {} travel through belt".format(id))
        elif statusCode == 3:
            print("long live tracker {}".format(id))
        elif statusCode == 4:
            print("{} is a dead tracker, bad".format(id))
        else:
            print("{} is a good tracker doing well".format(id))
        '''
        return statusCode != 0

    def predict(self):
        trkToKill = []
        currentMessage = {}
        for trkId, t in self.trackerList.items():
            #if tracker is still young, keep track
            #print("trkId from trackerListDict {}".format(trkId))
            if t.lifespan <= 5:
                #print("trkId from tracker {}".format(t.trackerId))
                bbox = t.predict()
            #check tracker healthy status
            if self.checkStatus(t):
                self.objectCnt[t.catagory] += 1
                trkMsg = trackerMessage(t)
                currentMessage[t.trackerId] = trkMsg
                trkToKill.append(trkId)
                
        for id in trkToKill:
            #print("removing {}".format(id))
            del self.trackerList[id]
        
        renderData = msgForRender(self.frameNum, self.trackerList, self.currentDet)
        return currentMessage, renderData
    

if __name__ == "__main__":
    detGen = detGenerator(minObj=5,maxObj=10)
    res = detGen.getDetectionRes()
    transformedData = transform(res)

    detParser = detectionParser()
    detectionSequence = detParser.getDetSequence(transformedData)
    
    momTracker = trackerManager()
    for frameNum, dets in detectionSequence.items():
        momTracker.doTracking(dets)
        currentFrameCounted, renderData = momTracker.predict()
       
        backGround = np.zeros((720, 1080, 3), np.uint8)
        backGround[:,:,:] = (255,255,255)
        
        for idx, est in enumerate(renderData.estimateBoxs):
            color = renderData.colors[idx]
            text = "{}, status :{}".format(renderData.trackIds[idx],renderData.trackStatus[idx])
            renderRectWithColor(est, backGround, color)
            renderTextUnderRect(est, backGround, text, color)

        '''
        for idx, det in enumerate(renderData.detList):
            catagory = renderData.catagories[idx]-1
            renderRect(det, backGround,catagory)
        '''

        cv2.imshow("blank", backGround)
        k = cv2.waitKey(0)
        if k == 'n':                       
            continue
        elif k == 27:  #escape key 
            break

    cv2.destroyAllWindows()