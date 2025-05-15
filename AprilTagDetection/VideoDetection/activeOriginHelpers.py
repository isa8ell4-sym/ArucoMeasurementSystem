import cv2
import numpy as np
import time, keyboard, os, statistics
import cv2.aruco as aruco
from vision.video import *
import pandas as pd
from pandas import DataFrame


def avgFilter(vals, numFrames=10, threshold=0.66): # default parameters are for degrees
    """get average from valid values that are within std"""

    avg = statistics.mean(vals)
    std = statistics.stdev(vals)

    validVals = [v for v in vals if v < avg+std and v > avg-std]
    newAvg = statistics.mean(validVals)
    if len(validVals) / numFrames >= threshold:
        reliable = True
    else:
        reliable = False

    return reliable, newAvg

def liveDifference(undImg, origin, markerID, pixCoord, worldCoord, theta):
    cv2.circle(undImg, (origin[1][0],origin[1][1]), radius=5, color=(0,255,0), thickness=10) # mark origin
    originTheta = origin[3]
    originWorld = origin[2]
    cv2.circle(undImg, (pixCoord[0],pixCoord[1]), radius=5, color=(0,0,255), thickness=10)
    diffTheta = theta - originTheta
    diffWorld = np.subtract(worldCoord, originWorld)
    thetaText = f"{round(diffTheta, 3)} deg"

    idText = f"tagID: {markerID}"
    cv2.putText(undImg, "Difference:", (25, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(undImg, str(diffWorld), (25, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(undImg, thetaText, (25, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(undImg, idText, (25, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(230, 70, 173), thickness=6, lineType=cv2.LINE_AA)
