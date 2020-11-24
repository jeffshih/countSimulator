from typing import Tuple
import numpy as np
import scipy as sp
from Point import Point
from dataStructure import rect_
from math import exp, sqrt
import cv2 
from config import *


def renderRect(rect:rect_, backGround, catagory):
    LU = (int(rect.LU.x), int(rect.LU.y))
    BR = (int(rect.LU.x+rect.width), int(rect.LU.y+rect.height))
    cv2.rectangle(backGround, LU, BR, catagoryColor[catagory], 3)

def renderRectWithColor(rect:rect_, bg, color):
    LU = (int(rect.LU.x), int(rect.LU.y))
    BR = (int(rect.LU.x+rect.width), int(rect.LU.y+rect.height))
    cv2.rectangle(bg, LU, BR, color, 3)

        
def renderTextUnderRect(rect:rect_, backGround, text, trkColor):
    rendStr = "trk id: {}".format(text)
    BR = (int(rect.LU.x+rect.width), int(rect.LU.y+rect.height))
    cv2.putText(backGround, rendStr, BR, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=trkColor ,thickness=3)



def calcLU(center:Point, wh:Point):
    nx, ny = center.x - wh.x/2, center.y - wh.y/2
    if (center.x-wh.x/2) < 0:
        nx = 0
    if (center.y-wh.y/2) < 0:
        ny = 0
    return Point(nx,ny)    

def areaOfTwoPoint(lu:Point, br:Point):
    w = br.x - lu.x 
    h = br.y - lu.y
    return w*h 

def distanceOfTwoPoint(src:Point, dst:Point):
    src = np.array(src)
    dst = np.array(dst)
    return np.linalg.norm(src-dst)


#if the center is out of background, resize it to proper place
def adjustCenter(x, y, w, h):
    xb = resolution[0]
    yb = resolution[1]
    if x < 0 :
        newW = x + w/2
        newX = newW/2
    elif x > xb:
        newW = w/2-(x-xb)
        newX = xb-newW/2
    else:
        newW, newX = w, x 

    if y < 0:
        newH = y + h/2
        newY = newH/2
    elif y > yb:
        newH = h/2-(y-yb)
        newY = yb-newH/2
    else:
        newH, newY = h, y
    return Point(newX, newY), Point(newW, newH)

def makeValidRange(det:rect_):
    x = det.center.x
    y = det.center.y
    h = det.height
    w = det.width

    if x < 0 :
        det.center.x = 0
        det.width = x+w/2
    elif x > resolution[0]:
        det.center.x = resolution[0]
        det.width = w/2-(x-resolution[0])
    
    if y < 0 :
        det.center.y = 0
        det.height = y+h/2
    elif y > resolution[1]:
        det.center.y = resolution[1]
        det.height = h/2-(y-resolution[1])
    
def generateNotSoRandomY():
    piv = [0, 0.2, 0.4, 0.6, 0.8]
    start = np.random.choice(piv)
    return start+np.random.rand()*0.2
    

def boundPixelSize(imgSize, input):
    if input >= imgSize:
        return imgSize
    elif input < 0:
        return 1
    else:
        return input

def boundRatioSize(input):
    if input >=1:
        return 1
    elif input < 0:
        return 0
    else:
        return input

def absToRatio(imgSize:Point, input:Point):
    
    x = boundPixelSize(imgSize.x, input.x)
    y = boundPixelSize(imgSize.y, input.y)
    x_ratio = x/imgSize.x
    y_ratio = y/imgSize.y
    return Point(x_ratio, y_ratio)

def ratioToAbs(imgSize:Point, ratio:Point):
    x = boundRatioSize(ratio.x)
    y = boundRatioSize(ratio.y)
    xAbs = imgSize.x*x
    yAbs = imgSize.y*y
    return Point(xAbs, yAbs)

#transform output of data generator from list to dict
def transform(l : list):
    res = {}
    for det in l:
        imgIdx = det.split(",")[1]
        if imgIdx not in res:
            res[imgIdx] = [det]
        else:
            res[imgIdx].append(det)
    return res

#calculation of overlapping

def overlap(A:rect_, B:rect_):
    x1 = max(A.LU.x, B.LU.x)
    y1 = max(A.LU.y, B.LU.y)
    x2 = min(A.LU.x+A.width, B.LU.x+B.width)
    y2 = min(A.LU.y+A.height, B.LU.y+B.height)
    width = x2-x1
    height = y2-y1 
    if width < 0 or height < 0:
        return 0
    overlapArea = width*height
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

    normalizedDx = (det.LU.x-pred.LU.x)/min(det.width, pred.width)
    normalizedDy = (det.LU.y-pred.LU.y)/min(det.height, pred.height)
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
    return rect_(Point(x,y), Point(w,h))