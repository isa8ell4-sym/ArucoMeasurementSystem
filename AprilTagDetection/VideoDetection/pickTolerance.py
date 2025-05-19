import cv2
import numpy as np
import time, keyboard, os, statistics
import cv2.aruco as aruco
from vision.video import *
import pandas as pd
from pandas import DataFrame
from activeOriginHelpers import avgFilter, liveDifference, avgFilterNested, textPos

class PositionData:
    def __init__(self, names, ids, x, y, z, theta):
        self.names = names
        self.ids = ids
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta

def activePosUserInput(cap, resolution, tagSize, K, D, displayScale = 0.5):
    """gives current position of tag until user takes a "screenshot" of the starting position"""

    if cap is None:
        print("Could not open GoPro camera.")
        exit()

    frameCount = 0

    while cap.isOpened():
        ret, frame = cap.read()
        scale = 1
        
        if frameCount == 0 : 
            # displayScale = .3
            h, w = frame.shape[:2]
            # print(h, w)
            h = int(h * displayScale)
            w = int(w * displayScale)

        if not ret:
            print('failed to open frame')
            break

        if frameCount % resolution == 0:
            #undistort
            undImg = cv2.undistort(frame, K, D, None) 

            # ask for user input
            # user = input("Hit Enter key to record start position...")
            cv2.putText(undImg, "Hit Enter key to record start position", (0,1200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)

            # get tag position
            success, markerID, pixCoord, worldCoord, theta = findTag(undImg, tagSize, K, D)

            if success: 
                
                # # draw center point of tags
                cv2.circle(undImg, (pixCoord[0],pixCoord[1]), radius=5, color=(0,0,255), thickness=10)
                thetaText = f"{theta} deg"
                worldCoordText = f"({worldCoord}) mm"
                idText = f"tagID: {markerID}"

                cv2.putText(undImg, str(worldCoord), (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                cv2.putText(undImg, thetaText, (50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                cv2.putText(undImg, idText, (50, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(230, 70, 173), thickness=6, lineType=cv2.LINE_AA)
            

            cv2.namedWindow('video feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('video feed', w, h)
            cv2.imshow('video feed', undImg)
            key = cv2.waitKey(1)
            # print(key) 

            if key == 13 and success:
                # print(key)
                return markerID, pixCoord, worldCoord, theta
            elif key == 13:
                print('could not detect tag. try again')

        frameCount += 1

def diffFromOrigin(cap, resolution, tagSize, K, D, origin, displayScale=0.5):
    origPix = origin[0]
    origWorld = origin[1]
    origTheta = origin[2]
    
    if cap is None:
        print("Could not open GoPro camera.")
        exit()

    frameCount = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        if frameCount == 0 : 
            # displayScale = .3
            h, w = frame.shape[:2]
            # print(h, w)
            h = int(h * displayScale)
            w = int(w * displayScale)

        if not ret:
            print('failed to open frame')
            break

        if frameCount % resolution == 0:
            #undistort
            undImg = cv2.undistort(frame, K, D, None) 

            # original position of tag
            cv2.circle(undImg, (origPix[0],origPix[1]), radius=5, color=(0,255,0), thickness=10)

            # communicate user input
            # Set your message
            message = "Hit Enter key to record difference from start position"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 4

            # Set bottom-left corner of text
            x = 10  # some padding from the left
            y = h + 50  # some padding from the bottom

            # Draw text
            cv2.putText(undImg, message, (x, y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
            # cv2.putText(undImg, "Hit Enter key to record difference from start position", (0,185), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)

            # get tag position
            success, markerID, pixCoord, worldCoord, theta = findTag(undImg, tagSize, K, D)

            if success: 
                
                # # draw center point of tags
                cv2.circle(undImg, (pixCoord[0],pixCoord[1]), radius=5, color=(0,0,255), thickness=10)
                diffTheta = theta - origTheta
                diffWorld = np.subtract(worldCoord, origWorld)
                thetaText = f"{diffTheta} deg"

                idText = f"tagID: {markerID}"

                cv2.putText(undImg, str(diffWorld), (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                cv2.putText(undImg, thetaText, (50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                cv2.putText(undImg, idText, (50, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(230, 70, 173), thickness=6, lineType=cv2.LINE_AA)
            

            cv2.namedWindow('video feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('video feed', w, h)
            cv2.imshow('video feed', undImg)
            key = cv2.waitKey(1)

            if key == 13 and success:
                # print(key)
                return diffWorld, diffTheta
            elif key == 13:
                print('could not detect tag. try again')

        frameCount += 1

def activeOrigin(cap, tagSize, K, D, savePath, resolution=20, displayScale=0.5):
    i=0
    origin = None
    frameCount = 0
    relTheta = True

    measuredTheta = []
    measuredWC = []
    measuredPixC = []

    currWorld = []
    currTheta = None
    currPix = []

    nameData = []
    idData = []
    xData = []
    yData = []
    zData = []
    tData = []

    
    if cap is None:
        print("Could not open GoPro camera.")
        exit()


    while cap.isOpened():
        ret, frame = cap.read()

        if frameCount % 3 ==0:        
            # display scale initialization
            if frameCount == 0 : 
                # displayScale = .3
                h, w = frame.shape[:2]
                # print(h, w)
                h = int(h * displayScale)
                w = int(w * displayScale)

            if not ret:
                print('failed to open frame')
                break


            #undistort
            undImg = cv2.undistort(frame, K, D, None) 
            key = cv2.waitKey(1)

            # ask for user input
            cv2.putText(undImg, f"Spacebar to record start position, Enter to take a photo, C to exit", (0,1050), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            # get tag position
            success, markerID, pixCoord, worldCoord, theta = findTag(undImg, tagSize, K, D)

            if success: 

                cv2.circle(undImg, (int(pixCoord[0]),int(pixCoord[1])), radius=5, color=(0,0,255), thickness=10) # always mark center of tag
            
                if origin: # origin exists, delta readings
                    # print(origin)
                    cv2.circle(undImg, (int(origin[1][0]),int(origin[1][1])), radius=5, color=(0,255,0), thickness=10) # mark origin
                    
                    if len(measuredTheta) >= resolution: 

                        relTheta, currTheta = avgFilter(measuredTheta, numFrames=resolution)
                        currWorld = avgFilterNested(measuredWC, numFrames=resolution)
                        currPix = avgFilterNested(measuredPixC, numFrames=resolution)

                        undImg, diffWorld, diffTheta = liveDifference(undImg, origin, markerID, currPix, currWorld, currTheta, relTheta)

                        measuredTheta = []
                        measuredWC = []
                        measuredPixC = []

                    elif len(measuredTheta) < resolution: 

                        measuredTheta.append(theta)
                        measuredWC.append(worldCoord)
                        measuredPixC.append(pixCoord)

                    try: 
                        undImg = textPos(undImg, "Delta:", (0,0,255), markerID, diffWorld, diffTheta, relTheta)
                    except:
                        undImg = textPos(undImg, "Current Position:", (0,255,0), markerID, currWorld, currTheta, relTheta)

                else:

                    if len(measuredTheta) >= resolution: 

                        relTheta, currTheta = avgFilter(measuredTheta, numFrames=resolution)
                        currWorld = avgFilterNested(measuredWC, numFrames=resolution)
                        currPix = avgFilterNested(measuredPixC, numFrames=resolution)

                        measuredTheta = []
                        measuredWC = []
                        measuredPixC = []

                    elif len(measuredTheta) < resolution: 

                        measuredTheta.append(theta)
                        measuredWC.append(worldCoord)
                        measuredPixC.append(pixCoord)

                    undImg = textPos(undImg, "Current Position:", (0,255,0), markerID, currWorld, currTheta, relTheta)


            if key == 32: # spacebar to record origin
                print("record origin")
                origin = [markerID, currPix, currWorld, currTheta]

            if key == 13: # enter to save frame
                filename = os.path.join(savePath, f"img{i}.jpg")
                cv2.imwrite(filename, undImg)

                if origin is not None:
                
                    nameData.append(i)
                    idData.append(markerID)
                    xData.append(diffWorld[0])
                    yData.append(diffWorld[1])
                    zData.append(diffWorld[2])
                    tData.append(diffTheta)  
                else:
                    nameData.append(i)
                    idData.append(markerID)
                    xData.append(None)
                    yData.append(None)
                    zData.append(None)
                    tData.append(None)                     
                i+=1 
                            
            if key == ord('c'): # press c to exit
                positionData = PositionData(nameData, idData, xData, yData, zData, tData)        
                return positionData



            cv2.namedWindow('video feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('video feed', w, h)
            cv2.imshow('video feed', undImg)
    frameCount += 1

   


if __name__ == '__main__':

    # open camera
    # undistort footage
    # get starting position from user
    # get april tag position
    # get difference from datum/starting position

    tagSize = 50 # mm
    goProSettings = ["photo_linear", "photo_fisheye", "video_fisheye"]
    setting = goProSettings[2]
    cam = 1
    resolution = 15
    displayScale = 0.5
    savePath = f'C:/Users/irosenstein/Documents/Vision/Results/StretchBot/pickTolerance'
    name= 'thetaExp'

    cap = open_camera(cam)
    K, D, DIM, R = getIntrinsicParams(setting)
    # print(f'K: \n {K}')
    # print(f'D: \n {D}')
    deltaPos = activeOrigin(cap, tagSize, K, D, savePath, resolution) #TODO: make theta colored if unreliable

    df = DataFrame({
        "Img Name": deltaPos.names,
        "Tag ID": deltaPos.ids,
        "x": deltaPos.x,
        "y": deltaPos.y,
        "z": deltaPos.z,
        "theta": deltaPos.theta   
    })
    df.to_excel(f'{savePath}/{name}.xlsx', sheet_name='sheet1', index=False)

    cap.release()
    cv2.destroyAllWindows()
    