import cv2, json
import cv2.aruco as aruco
import numpy as np
import os, time
from .image import *
# from .tag import Tag
# from .undistort import *


def open_camera(index, retries=10, delay=1.0):
    for i in range(retries):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.any():
                print(f"Camera opened on index {index} (try {i + 1})")
                return cap
        print(f"Retry {i + 1}/{retries}...")
        cap.release()
        time.sleep(delay)
    return None

def estimatePoseSingleMarkers(corners, marker_size, K, D):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    K - is the camera matrix
    D - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []

    for c in corners:
        # print("Detected image points:", c)
        # print("Shape of c:", c.shape)
        success, R, t = cv2.solvePnP(marker_points, c, K, D, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(success)
    return success, rvecs, tvecs, trash

def getIntrinsicParams(setting):

    here = os.path.dirname(__file__)  # directory of image.py
    path = os.path.join(here, 'intrinsics.json')
    with open(path, 'r') as f:
        data = json.load(f)
    if setting == "photo_linear":
        data = data["photo_linear"]
        K = data["K"]
        D = data["D"]
        DIM = data["DIM"]
        R = data["R"]
    elif setting == "photo_fisheye":
        data = data["photo_fisheye"]
        K = data["K"]
        D = data["D"]
        DIM = data["DIM"]
        R = data["R"]
    elif setting == "video_fisheye":
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
        print(f'cannot get intrinsic params of {setting}')

    return np.array(K), np.array(D), DIM, R

def findTag(frame, tagSize, K, D): #TODO: organize better, maybe use image class but might make slower
    # Load the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters()

    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 50
    parameters.adaptiveThreshWinSizeStep = 5

    parameters.minMarkerPerimeterRate = .1
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.errorCorrectionRate = 0.5  # 

    detector = aruco.ArucoDetector(aruco_dict, parameters)
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)


    # If markers are detected
    if markerIds is not None:
        aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        # each tag. center coordinates of tag
        for corners, marker_id in zip(markerCorners, markerIds.flatten()):

            cornersReshape = corners.reshape((4, 2)).astype(np.int32)  # Ensure integer type
            top_left, top_right, bottom_right, bottom_left = cornersReshape.astype(int)
            avgX = int(np.average([[c[0] for c in cornersReshape]]))
            avgY = int(np.average([[c[1] for c in cornersReshape]]))
            pixCoord = [avgX, avgY]
            # print(f'{avgX}, {avgY}')

            # Draw bounding box around tags
            # cv2.polylines(frame, [cornersReshape], isClosed=True, color=(0, 255, 0), thickness=2)

            # # draw center point of tags
            # cv2.circle(frame, (avgX,avgY), radius=5, color=(0,0,255), thickness=10)
            # pixCoord = [avgX, avgY]

            # world coordinates and angle of tag
            success, rvec, tvec, _ = estimatePoseSingleMarkers(corners, tagSize, K, D)
            if success:

                rvec = rvec[0]
                R, _ = cv2.Rodrigues(rvec)

                # Extract Euler angles (Theta is the yaw angle)
                theta_z = np.degrees(np.arctan2(R[1, 0], R[0, 0]))  # Yaw (Theta)

                worldCoord = [int(tvec[0][0]), int(tvec[0][1]), int(tvec[0][2])] # [round(c,2) for c in tvec[0]]
                

    else: # No tag detected
        success = 0
        pixCoord = worldCoord = marker_id = theta_z = None

    return success, marker_id, pixCoord, worldCoord, theta_z

