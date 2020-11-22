from typing import Tuple
import numpy as np
import scipy as sp
from Pair import Pair
from dataStructure import rect_
from math import exp, sqrt
catagorySize = {1:(300,200), 2:(180,300), 3:(400, 320), 4:(520, 200), 5:(300, 400)}


def convertLU(center:Pair, wh:Pair):
    nx, ny = center.x - wh.x/2, center.y - wh.y/2
    if (center.x-wh.x/2) < 0:
        nx = 0
    if (center.y-wh.y/2) < 0:
        ny = 0
    return Pair(nx,ny)

def areaOfTwoPair(lu:Pair, br:Pair):
    w = br.x - lu.x 
    h = br.y - lu.y
    return w*h 

def distanceOfTwoPair(src:Pair, dst:Pair):
    src = np.array(src)
    dst = np.array(dst)
    return np.linalg.norm(src-dst)


def absToRatio(imgSize:Pair, input:Pair):
    x_ratio = input.x/imgSize.x
    y_ratio = input.y/imgSize.y
    return Pair(x_ratio, y_ratio)

def ratioToAbs(imgSize:Pair, ratio:Pair):
    xAbs = imgSize.x*ratio.x
    yAbs = imgSize.y*ratio.y
    return Pair(xAbs, yAbs)

def transform(l : list):
    res = {}
    for det in l:
        imgIdx = det.split(",")[1]
        if imgIdx not in res:
            res[imgIdx] = [det]
        else:
            res[imgIdx].append(det)
    return res

def overlap(A:rect_, B:rect_):
    x1 = max(A.LeftUpper.x, B.LeftUpper.x)
    y1 = min(A.LeftUpper.y, B.LeftUpper.y)
    x2 = max(A.LeftUpper.x+A.width, B.LeftUpper.x+B.width)
    y2 = min(A.LeftUpper.y+A.height, B.LeftUpper.y+B.height)
    overlapW = x2-x1
    overlapH = y2-y1 
    overlapArea = overlapH*overlapW
    #print("overlap :", overlapArea)
    return overlapArea

def union(A:rect_, B:rect_):
    unionArea = A.area+B.area - overlap(A,B)
    #print("union :", unionArea)
    return unionArea

def IOU(A:rect_, B:rect_):
    #print("IOU :",overlap(A,B)/union(A,B))
    return overlap(A,B)/union(A,B)

def getMatchingCost(det, pred):
    IOUCost = IOU(det, pred)
    shapeCost = getShapeSizeCost(det, pred)
    return 0.5*IOUCost+0.5*shapeCost


def getShapeSizeCost(det:rect_, pred:rect_):

    normalizedDx = (det.LeftUpper.x-pred.LeftUpper.x)/min(det.width, pred.width)
    normalizedDy = (det.LeftUpper.y-pred.LeftUpper.y)/min(det.height, pred.height)
    costA = exp(-0.5*(normalizedDx**2 + normalizedDy**2))

    normalizedDw = (det.width-pred.width)/(det.width+pred.width)
    normalizedDh = (det.height-pred.height)/(det.height+pred.height)
    costB = exp(-0.5*(normalizedDw**2+normalizedDh**2))

    return 1-(costA*costB)

def measurementToRect(measurement):
    x = measurement[0][0]
    y = measurement[1][0]
    w = sqrt(measurement[2][0]*measurement[3][0])
    h = sqrt(measurement[2][0]/measurement[3][0])
    return rect_(Pair(x,y), Pair(w,h))