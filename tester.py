from Util import transform
from mainApp import mainProgram
import numpy as np
from Generator import detGenerator
import matplotlib.pyplot as plt
from Util import *
import timeit
from datetime import datetime
from config import *



def extractCountedType(res:dict):
    countType = [0 for _ in range(5)]
    for val in res.values():
        for r in val:
            if r is not None:
                countType[r-1]+=1    
    return countType 

def compareResult(res:dict, truth:dict):
    return extractCountedType(res), extractCountedType(truth)


def extractCountForPlot(res:dict):
    x , y = [], []
    for frame, cnts in res.items():
        x.append(int(frame))
        y.append(np.array(cnts if cnts is not None else [0]))
    return np.array(x), np.array(y)

if __name__ == "__main__":
    dataGen = detGenerator()
    detectionResList = dataGen.getDetectionRes()
    dataGenRes = dataGen.getGroundTruth()
    #dataGen.reset()

    App = mainProgram(detectionResList, resolution)
    res = App.genCountingResult()

    x , y = extractCountForPlot(res)
    print(x, y)
#    print(compareResult(res, dataGenRes))
    #plt.plot(x, y, )   