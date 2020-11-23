import numpy as np
from Pair import Pair

catagorySize = {1:(200, 130), 2:(170, 100), 3:(250, 110), 4:(230, 120), 5:(130, 100)}
catagoryMissRate = {}
for i in range(0,6):
    catagoryMissRate[i] = np.random.rand()*0.4

resolution = Pair(1080,720)

colorMap = {'det':(255,0,0), "est":(0,255,0), "update":(0,0,255), "text":(255,255,0)}