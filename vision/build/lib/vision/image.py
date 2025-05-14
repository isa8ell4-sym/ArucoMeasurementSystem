import cv2, json
import cv2.aruco as aruco
import numpy as np
import os
from tag import Tag
from undistort import *

class Image: 
    def __init__(self, image, type, path=None, tags=None):
        self.image = image
        self.type = type
        self.path = path
        self.tags = tags

    def getIntrinsicParams(self):
        data = json.load('intrinsics.json')
        if self.type == "photo_linear":
            data = data["photo_linear"]
            K = data["K"]
            D = data["D"]
            DIM = data["DIM"]
            R = data["R"]
        elif self.type == "photo_fisheye":
            data = data["photo_fisheye"]
            K = data["K"]
            D = data["D"]
            DIM = data["DIM"]
            R = data["R"]
        elif self.type == "video_fisheye":
            data = data["video_fisheye"]
            K = data["K"]
            D = data["D"]
            DIM = data["DIM"]
            R = data["R"]
        else: 
            K = 0
            D = 0
            DIM = 0
            R = 0
            print(f'cannot get intrinsic params of {self.type}')

        return K, D, DIM, R

    def undistort(self):
        dst = self.image
        if self.type == "photo_linear":
            udst = undstPhotoLinear(dst)  
        elif self.type == "photo_fisheye":
            udst = undstPhotoFisheye(dst)
        elif self.type == "video_fisheye":
            udst = undstVideoFisheye(dst)
        else: 
            print(f'cannot recognize {self.type}')

        self.image = dst
        return udst

    def detectTags(self, tagSize, K, D): #TODO: detect, get world position of all tags. eventually split up / reorganize

        # Load the ArUco dictionary
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
        parameters = aruco.DetectorParameters()

        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 50
        parameters.adaptiveThreshWinSizeStep = 5

        parameters.minMarkerPerimeterRate = .1
        parameters.maxMarkerPerimeterRate = 4.0
        parameters.errorCorrectionRate = 0.5  # Increase to allow noisy markers to be decoded

        detector = aruco.ArucoDetector(aruco_dict, parameters)
     
        tags = []

        # image = cv2.imread(self.path)
        # print(f"shape: {image.shape}")

        image = self.image

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)

        # If markers are detected
        if markerIds is not None:
            aruco.drawDetectedMarkers(image, markerCorners, markerIds)

            # each tag. center coordinates of tag
            for corners, marker_id in zip(markerCorners, markerIds.flatten()):

                # print(f'{avgXPix}, {avgYPix}')

                tag = Tag(marker_id, corners)

                pixCoord = tag.calcPixPos(corners)

                # Draw bounding box around tags
                cornersReshape = corners.reshape((4, 2)).astype(np.int32)  # Ensure integer type

                cv2.polylines(image, [cornersReshape], isClosed=True, color=(0, 255, 0), thickness=2)

                # draw center point of tags
                cv2.circle(image, pixCoord, radius=5, color=(0,0,255), thickness=10)
                pixCoordText = f"({pixCoord[0]}, {pixCoord[1]})"
                idText = f"tagID: {marker_id}"

                # Add the coordinates as text
                cv2.putText(image, pixCoordText, (pixCoord[0] + 10, pixCoord[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                cv2.putText(image, idText, (pixCoord[0] - 100, pixCoord[1] + 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(230, 70, 173), thickness=6, lineType=cv2.LINE_AA)

                success, worldPosMM, theta = tag.calcWorldPos(corners, tagSize, K, D)
                
                if success:

                    thetaText = f"{theta} deg"
                    worldCoordText = f"({worldPosMM}) mm"

                    cv2.putText(image, worldCoordText, (50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)
                    cv2.putText(image, thetaText, (50, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)
                    # cv2.putText(image, idText, (50, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(230, 70, 173), thickness=8, lineType=cv2.LINE_AA)

                tags.append(tag)

            self.tags = tags

        else: # No tag detected
            print(f"No markers detected in {self.path}.")
            aruco.drawDetectedMarkers(image, rejectedCandidates)
            
        return tags