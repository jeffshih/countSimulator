import numpy as np
from Point import Point

catagorySize = {1:(180, 100), 2:(220, 70), 3:(250, 80), 4:(230, 120), 5:(20, 140)}
catagoryColor = [(200,0,0), (0, 200, 0), (0, 0, 200), (200, 200, 0), (0, 200, 200)]
validSize = {1:(200, 400), 2:(200, 100), 3:(250, 110), 4:(230, 120), 5:(130, 100)}



catagoryMissRate = {}
for i in range(0,6):
    catagoryMissRate[i] = np.random.rand()*0.4
catagoryConfidence = [1-np.random.rand()*0.2 for _ in range(5)]

resolution = Point(1080,720)

colorMap = {'det':(255,0,0), "est":(0,255,0), "update":(0,0,255), "text":(255,255,0)}