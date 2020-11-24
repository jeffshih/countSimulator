from numpy.lib.function_base import average
from numpy.testing._private.utils import break_cycles
from Generator import detGenerator
from Util import getMatchingCost, transform
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
            detBox = rect_(det.relativeCenter, det.relativeWH)
            self.detectionBoxes.append(detBox)
            meta.append([det.catagory, det.confidence])
            #print("At frame:{}, create detection {}".format(frameNum,detBox))

        matchedPoints = {}
        trkIdInList = 0

        for trkId, t in self.trackerList.items():
            predBox = t.estimateBox
            self.predictionBoxes.append(predBox)
            print("At frame:{}, hold prediction num {} ,{}".format(frameNum,trkId,predBox))
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
        
        print(self.assignment)
        for idx, assign in enumerate(self.assignment):
            confidence = meta[assign][1]
            print("assign det {} to pred {}".format(assign, matchedPoints[idx]))
            print(self.detectionBoxes[assign])
            print(len(self.trackerList))
            print(self.trackerList[matchedPoints[idx]].getRect())
            self.trackerList[matchedPoints[idx]].setTracked(self.detectionBoxes[assign], confidence)

        for detIdx in unmatchDet:
            print("umatched det at idx {}".format(detIdx))
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
                print("trkId :{} die in glory".format(trkId))
            #unhealthy death, do not count
            elif t.checkStatus() == 2:
                trkToKill.append(trkId)
                print("trkId :{} die in vain".format(trkId))

        for id in trkToKill:
            print("removing {}".format(id))
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
        
        
        '''
        #draw det box
        detCnt = len(renderData.detList)
        for det in renderData.detList:
            #print(det)
            LU = (int(det.LeftUpper.x), int(det.LeftUpper.y))
            BR = (int(det.LeftUpper.x+det.width), int(det.LeftUpper.y+det.height))
            #print(LU)
            #print(BR)
            cv2.rectangle(backGround, LU,BR , (255,0,0), 3)
            count = "current det box count are {}".format(detCnt)
            cv2.putText(backGround, count, (50,50),fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3 , color=(255,0,0), thickness=3)
        '''
        '''
        #draw pred box
        for est in renderData.estimateBoxs:
            LU = (int(est.LeftUpper.x), int(est.LeftUpper.y))
            BR = (int(est.LeftUpper.x+est.width), int(est.LeftUpper.y+est.height))
            cv2.rectangle(backGround, LU, BR, colorMap['est'], 3)
        
        
        for history in renderData.historyList:
            for idx, p in enumerate(history):
                #print("history : ",p.x, p.y)
                th = 0 if idx == 0 else (idx -1)
                p1 = (int(history[th].x),int(history[th].y))
                p2 = (int(p.x), int(p.y))
                cv2.line(backGround, p1, p2, (0,100,150), 1)
        '''
        
        '''
        #catagory
        for idx, cat in enumerate(renderData.catagories):
            est = renderData.detList[idx]
            rendStr = "catagory {}".format(cat)
            BR = (int(est.LeftUpper.x+est.width)-50, int(est.LeftUpper.y+est.height)-20)
            cv2.putText(backGround, rendStr, BR, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0,0,255),thickness=3)
        '''

        for est, trkId in zip(renderData.estimateBoxs,renderData.trackIds):
            rendStr = "trk id: {}".format(trkId)
            BR2 = (int(est.LeftUpper.x+est.width)-50, int(est.LeftUpper.y+est.height)-20)
            LU = (int(est.LeftUpper.x), int(est.LeftUpper.y))
            BR = (int(est.LeftUpper.x+est.width), int(est.LeftUpper.y+est.height))
            cv2.rectangle(backGround, LU, BR, colorMap['est'], 3)
            cv2.putText(backGround, rendStr, BR2, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=colorMap['est'],thickness=3)
        


        cv2.imshow("blank", backGround)
        key = cv2.waitKey(1000)