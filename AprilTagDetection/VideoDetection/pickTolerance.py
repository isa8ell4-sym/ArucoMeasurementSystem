import cv2
import numpy as np
import time, keyboard
import cv2.aruco as aruco
from vision.video import *

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


def activeOrigin(cap, tagSize, K, D, resolution=3, displayScale=0.5):
    origin = None
    if cap is None:
        print("Could not open GoPro camera.")
        exit()

    frameCount = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        # display scale
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
            cv2.putText(undImg, "Hit Spacebar to record start position Enter to exit", (0,1050), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)

            # get tag position
            success, markerID, pixCoord, worldCoord, theta = findTag(undImg, tagSize, K, D)

            if success: 
                if origin is None:
                    # # draw center point of tags
                    cv2.circle(undImg, (pixCoord[0],pixCoord[1]), radius=5, color=(0,0,255), thickness=10)
                    thetaText = f"{round(theta, 3)} deg"
                    worldCoordText = f"({worldCoord}) mm"
                    idText = f"tagID: {markerID}"

                    cv2.putText(undImg, "Current Position:", (25, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
                    cv2.putText(undImg, str(worldCoord), (25, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
                    cv2.putText(undImg, thetaText, (25, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
                    cv2.putText(undImg, idText, (25, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(230, 70, 173), thickness=6, lineType=cv2.LINE_AA)
                else: 
                    cv2.circle(undImg, (origin[1][0],origin[1][1]), radius=5, color=(0,255,0), thickness=10) # mark origin
                    
                    # # draw center point of tags
                    
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

            


            cv2.namedWindow('video feed', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('video feed', w, h)
            cv2.imshow('video feed', undImg)
            key = cv2.waitKey(1)
            # print(key) 

            if key == 32 and success: # spacebar to record origin
                origin = [markerID, pixCoord, worldCoord, theta]

            if key == 13 and success: # enter to exit
                return diffWorld, diffTheta

            if origin is not None: 
                cv2.circle(undImg, (origin[1][0],origin[1][1]), radius=5, color=(0,255,0), thickness=10) # mark origin



        frameCount += 1


if __name__ == '__main__':

    # open camera
    # undistort footage
    # get starting position from user
    # get april tag position
    # get difference from datum/starting position

    tagSize = 150 # mm
    goProSettings = ["photo_linear", "photo_fisheye", "video_fisheye"]
    setting = goProSettings[2]
    cam = 1
    resolution = 3
    displayScale = 0.5

    cap = open_camera(cam)
    K, D, DIM, R = getIntrinsicParams(setting)
    # print(f'K: \n {K}')
    # print(f'D: \n {D}')
    deltaWorld, deltaTheta = activeOrigin(cap, tagSize, K, D, resolution)

    cap.release()
    cv2.destroyAllWindows()
    