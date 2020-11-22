from Util import getMatchingCost
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from Trackers import Tracker
from dataStructure import rect_
from Pair import Pair


class trackerManager(object):

    def __init__(self):
        self.trackerList = []
        self.detectionBoxes = []
        self.predictionBoxes = []
        self.assignment = []

    def doTracking(self, detections):
        self.detectionBoxes = []
        self.predictionBoxes = []
        self.assignment = []

        for det in detections:
            detBox = rect_(det.relativeCenter, det.relativeWH)
            self.detectionBoxes.append(detBox)
        
        for t in self.trackerList:
            predBox = t.estimateBox
            self.predictionBoxes.append(predBox)
        
        detCnt = len(self.detectionBoxes)
        predCnt = len(self.predictionBoxes)

        costMatrix = np.zeros((predCnt,detCnt),dtype=float)
        
        for i, detection in enumerate(self.detectionBoxes):
            for j, prediction in enumerate(self.predictionBoxes):
                costMatrix[i][j] = getMatchingCost(detection, prediction)

        self.assignment, col_ind = linear_sum_assignment(costMatrix)
        unmatchPred = set()
        unmatchDet = set()
        allItems = set()
        matchedItems = set()
        matchedPairs = []
        
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
        for idx, assign in enumerate(self.assignment):
            matchedPairs.append(Pair(idx,assign))
             



    def update(self,assignment):
        
        pass 

    def predict(self):
        pass

