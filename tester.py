from Util import transform
from mainApp import mainProgram
import numpy as np
from Generator import detGenerator
import matplotlib.pyplot as plt
from Util import *
import timeit
from datetime import datetime
from config import *






class evaluation(object):

    def __init__(self):
        self.dataGen = detGenerator(minObj=5,maxObj=10, framenum = 100)
        self.App = mainProgram()
        self.genDetList = self.dataGen.getDetectionRes()
        self.App.initWithData(self.genDetList)
        self.groundTruth = self.dataGen.getGroundTruth()
        self.countingRes = self.App.genCountingResult()

    def extractType(self, res:dict):
        countType = [0 for _ in range(5)]
        for val in res.values():
            for r in val:
                if r is not None:
                    countType[r-1]+=1    
        return countType 

    def runCountingAndOutput(self):
        self.App.doCounting()
    
    def printGroundTruth(self):
        print(self.extractType(self.groundTruth))

    def printCountingResult(self):
        print(self.extractType(self.countingRes))

    def resetData(self):
        self.dataGen.reset()
        self.genDetList = self.dataGen.getDetectionRes()
        self.App.initWithData(self.genDetList)
        self.groundTruth = self.dataGen.getGroundTruth()
        
    def getGroundTruth(self):
        return self.groundTruth
    
    def getCountResult(self):
        return self.countingRes

    def updateCountingResult(self):
        self.countingRes = self.App.genCountingResult()

    def compareResult(self):
        return self.extractType(self.countingRes), self.extractType(self.groundTruth)


    def extractCountForPlot(self, res:dict):
        x , y = [], []
        for frame, cnts in res.items():
            if len(cnts) == 0:
                cnts = [0]
            for element in cnts:
                x.append(int(frame))
                y.append(element)
        
        return np.array(x), np.array(y)

    def plotGenDistribution(self, trial):
        distribution = []
        for _ in range(trial):
            l = self.dataGen.getDetectionRes()
            distribution.append(len(l))
            self.dataGen.reset()
        plt.hist(distribution, bins=20)
        plt.show()
    

    def plotCountResult(self):
        #f = plt.figure()
        plt.figure(21)

        GT = self.extractCountForPlot(self.groundTruth)
        CNT = self.extractCountForPlot(self.countingRes)
        plt.subplot(211)
        plt.scatter(GT[0], GT[1], color='C0', marker="o")
        plt.subplot(212)
        plt.scatter(CNT[0], CNT[1], color='C1',marker="o")
        plt.suptitle("Comparison of counted frame")
        plt.show()
        #print(GT, CNT)    



if __name__ == "__main__":
    eval = evaluation()
    #countResult = eval.getCountResult()
    #groundTruth = eval.getGroundTruth()
    #print(countResult,groundTruth)
    eval.plotCountResult()
    #eval.printGroundTruth()
    #eval.printCountingResult()
    print(eval.compareResult())
    print(eval.extractCountForPlot)

    