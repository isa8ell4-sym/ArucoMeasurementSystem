import cv2
import numpy as np
import time, keyboard, os, statistics
import cv2.aruco as aruco
from vision.video import *
import pandas as pd
from pandas import DataFrame
from collections import deque


def avgFilter(vals, numFrames=10, threshold=0.66): # default parameters are for degrees
    """get average from valid values that are within std"""

    avg = statistics.mean(vals)
    std = statistics.stdev(vals)

    validVals = [v for v in vals if v <= avg+std and v >= avg-std]
    # print(validVals)

    newAvg = statistics.mean(validVals)
    # print(len(validVals)/numFrames)

    if len(validVals) / numFrames >= threshold:
        reliable = True
    else:
        reliable = False

    return reliable, newAvg

def avgFilterNested(lst, numFrames=10, threshold=0.66):
    grouped = list(map(list, zip(*lst)))
    # print(grouped)
    results = []
    # print(f'grouped: {grouped}')
    for d in grouped:
        rel, avg = avgFilter(d, numFrames, threshold)
        results.append(avg)
    
    return results
    
def liveDifference(undImg, origin, markerID, pixCoord, worldCoord, theta):
    # cv2.circle(undImg, (origin[1][0],origin[1][1]), radius=5, color=(0,255,0), thickness=10) # mark origin
    originTheta = origin[3]
    originWorld = origin[2]
    # print(f'pixCoord: {pixCoord}')
    # cv2.circle(undImg, (int(pixCoord[0]),int(pixCoord[1])), radius=5, color=(0,0,255), thickness=10)
    diffTheta = theta - originTheta
    diffWorld = np.subtract(worldCoord, originWorld)


    return undImg, diffWorld, diffTheta

def textPos(img, title, color, id=None, pos=None, theta=None, posRel = True, thetaRel=True):

    cv2.putText(img, title, (25, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=color, thickness=5, lineType=cv2.LINE_AA)
    if id is not None and theta is not None and pos is not None:
        # print(id)
        thetaText = f"{round(theta, 3)} deg"
        idText = f"tagID: {id}"
        posText = f'[{round(pos[0], 1)}, {round(pos[1], 1)}, {round(pos[2], 1)}]'
        
        if thetaRel: 
            thetaColor = color
        else: 
            thetaColor = (0,0,255) # if not reliable make color red

        if posRel: 
            posColor = color
        else: 
            posColor = (0,0,255) # if not reliable make color red
        cv2.putText(img, posText, (25, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=posColor, thickness=5, lineType=cv2.LINE_AA)
        cv2.putText(img, thetaText, (25, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=thetaColor, thickness=5, lineType=cv2.LINE_AA)
        cv2.putText(img, idText, (25, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(230, 70, 173), thickness=5, lineType=cv2.LINE_AA)

    return img


def expMovingAvg(newPos, prevPos, alpha = 0.3):

    smoothedPos = alpha * newPos + (1-alpha) * prevPos # EMA formula

    return smoothedPos

def confidenceFilter(history: deque, window: int, threshold=1.0, columns=None) -> bool:

    if len(history) < window:
        return False

    data = np.array(history)

    if columns is not None:
        data = data[:, columns]

    std_devs = np.std(data, axis=0)

    if isinstance(threshold, (int, float)):
        return np.all(std_devs < threshold)
    else:
        return np.all(std_devs < np.array(threshold))