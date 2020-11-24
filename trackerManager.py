from numpy.lib.function_base import average
from numpy.testing._private.utils import break_cycles
from Generator import detGenerator
from Util import getMatchingCost, renderRectWithColor, transform, renderRect
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
        for i in range(1,6):
            self.objectCnt[i] = 0


    def doTracking(self, detections, frameNum):
        self.detectionBoxes = []
        self.predictionBoxes = []
        self.assignment = []
        meta = []


        for det in detections:
            self.detectionBoxes.append(det.rect)
            meta.append([det.catagory, det.confidence])
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
                #print("At i:{}, j:{}, cost is {}".format(i,j,cost))

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
            confidence = meta[assign][1]
            '''
            print("assign det {} to pred {}".format(assign, matchedPoints[idx]))
            print(self.detectionBoxes[assign])
            print(len(self.trackerList))
            print(self.trackerList[matchedPoints[idx]].getRect())
            '''
            self.trackerList[matchedPoints[idx]].setTracked(self.detectionBoxes[assign], confidence)

        for detIdx in unmatchDet:
            '''
            print("umatched det at idx {}".format(detIdx))
            '''
            newDet = self.detectionBoxes[detIdx]
            data = meta[detIdx]
            newTracker = Tracker(self.currentTrackerID, newDet, data[0], data[1])
            self.trackerList[self.currentTrackerID] = newTracker
            self.currentTrackerID += 1

        renderData = msgForRender(frameNum, self.trackerList, detections)
        return renderData


    def predict(self):
        trkToKill = []
        currentMessage = {}
        for trkId, t in self.trackerList.items():
            #if tracker is still young, keep track
            #print("trkId from trackerListDict {}".format(trkId))
            if t.lifespan <= 5:
                #print("trkId from tracker {}".format(t.trackerId))
                t.predict()
            #check tracker healthy status
            if t.checkStatus() == 1:
                self.objectCnt[t.catagory] += 1
                trkMsg = trackerMessage(t)
                currentMessage[t.trackerId] = trkMsg
                trkToKill.append(trkId)
                #print("trkId :{} die in glory".format(trkId))
            #unhealthy death, do not count
            elif t.checkStatus() == 2:
                trkToKill.append(trkId)
                #print("trkId :{} die in vain".format(trkId))

        for id in trkToKill:
            #print("removing {}".format(id))
            del self.trackerList[id]

        return currentMessage
    

if __name__ == "__main__":
    detGen = detGenerator(minObj=5,maxObj=10)
    res = detGen.getDetectionRes()
    transformedData = transform(res)

    detParser = detectionParser()
    detectionSequence = detParser.getDetSequence(transformedData)
    
    momTracker = trackerManager()
    for frameNum, dets in detectionSequence.items():
        #print(frameNum)
        renderData = momTracker.doTracking(dets,frameNum)
        currentFrameCounted = momTracker.predict()
        #for key, msg in currentFrameCounted.items():
        #    print(msg)

        backGround = np.zeros((720, 1080, 3), np.uint8)
        backGround[:,:,:] = (255,255,255)
        
        for idx, est in enumerate(renderData.estimateBoxs):
            color = renderData.colors[idx]
            renderRectWithColor(est, backGround,color)
        
        for idx, det in enumerate(renderData.detList):
            catagory = renderData.catagories[idx]-1
            renderRect(det, backGround,catagory)

        cv2.imshow("blank", backGround)
        key = cv2.waitKey(1000)