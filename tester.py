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
        self.grountTruth = self.dataGen.getGroundTruth()
        self.countingRes = self.App.genCountingResult()
        self.dataGen.reset()

    def extractType(self, res:dict):
        countType = [0 for _ in range(5)]
        for val in res.values():
            for r in val:
                if r is not None:
                    countType[r-1]+=1    
        return countType 

    def runCountingAndOutput(self):
        self.App.doCounting()

    def resetData(self):
        self.dataGen.reset()
        self.genDetList = self.dataGen.getDetectionRes()
        self.App.initWithData(self.genDetList)
        self.grountTruth = self.dataGen.getGroundTruth()
        

    def updateCountingResult(self):
        self.countingRes = self.App.genCountingResult()

    def compareResult(self):
        return self.extractType(self.countingRes), self.extractType(self.grountTruth)


    def extractCountForPlot(self, res:dict):
        x , y = [], []
        for frame, cnts in res.items():
            if len(cnts) == 0:
                cnts = [0]
            for element in cnts:
                x.append(int(frame))
                y.append(element)
        
        return np.array(x), np.array(y)

    def plotDistribution(self, trial):
        distribution = []
        for _ in range(trial):
            l = self.dataGen.getDetectionRes()
            distribution.append(len(l))
            self.dataGen.reset()
        plt.hist(distribution, bins=20)
        plt.show()
    
    

    



if __name__ == "__main__":
    dataGen = detGenerator(framenum=30)
    detectionResList = dataGen.getDetectionRes()
    dataGenRes = dataGen.getGroundTruth()
    dataGen.reset()

    #App = mainProgram(detectionResList, resolution)
    #res = App.genCountingResult()

    plotDistribution(dataGen, 1000)

    