# Purpose of this is to detect payload bay of stretch bot detect if there is a package in bay and get the package's position

import cv2, json
import cv2.aruco as aruco
import numpy as np
import glob
import datetime
import os
import sys
import calibration
def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

def makeDir(env='CMD'):
    now = datetime.datetime.now()
    if env == 'CMD':
        basePath = f"C:/Users/irosenstein/Documents/VibrationTesting/VibeTestingTags/tagDetection/tagDetection{now.month}.{now.day}_{now.hour}.{now.minute}/"
        os.makedirs(basePath,  exist_ok=True)
        undistortedPath = os.path.join(basePath, f'undistorted{now.month}.{now.day}_{now.hour}.{now.minute}')
        detectionPath = os.path.join(basePath, f'detection{now.month}.{now.day}_{now.hour}.{now.minute}')
        os.makedirs(undistortedPath, exist_ok=True)
        os.makedirs(detectionPath, exist_ok=True)
        return basePath, undistortedPath, detectionPath
    elif env == 'WSL':
        basePath = f'/mnt/c/Users/irosenstein/Documents/VibrationTesting/VibeTestingTags/tagDetection/tagDetection{now.month}.{now.day}_{now.hour}.{now.minute}/'
        os.makedirs(basePath,  exist_ok=True)
        filteredPath = os.path.join(basePath, f'filter{now.month}.{now.day}_{now.hour}.{now.minute}')
        undistortedPath = os.path.join(basePath, f'filterStepper')
        detectionPath = os.path.join(basePath, f'detection{now.month}.{now.day}_{now.hour}.{now.minute}')
        os.makedirs(filteredPath, exist_ok=True)
        os.makedirs(undistortedPath, exist_ok=True)
        os.makedirs(detectionPath, exist_ok=True)
        return basePath, filteredPath, undistortedPath, detectionPath
    else:
        print(f"{env} is not a valid environment")
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

def processImages(images, filteredPath, stepperPath, env='CMD'):
    # process images for better detection
    for fname in images:
        # print(fname)
        _, tail = os.path.split(fname)
        name = tail.replace(".JPG", "")
        # print(name)
        if env == 'CMD':
            os.chdir('C:/Users/irosenstein/Documents/VibrationTesting/VibeTestingTags/')
        else:
            os.chdir('/mnt/c/Users/irosenstein/Documents/VibrationTesting/VibeTestingTags/')

        undImg = undistortPaddingWidePhotos(fname, 1.5)

        cv2.imwrite(os.path.join(stepperPath, f'{name}.JPG'), undImg)

    #TODO should make return images

def detectMarkers(image):
    # Load the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250) #DICT_4X4_100)
    parameters = aruco.DetectorParameters()

    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 50
    parameters.adaptiveThreshWinSizeStep = 5

    parameters.minMarkerPerimeterRate = .1
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.errorCorrectionRate = 0.5 

    detector = aruco.ArucoDetector(aruco_dict, parameters)
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)
    return markerCorners, markerIds, rejectedCandidates

def formatCornersCV(corners):
    corners = np.array(corners)
    corners = corners.reshape((-1, 1, 2)).astype(np.int32)
    return [corners]

def detectBay(image, bayCornerIDs, detectedIDs, detectedCorners):
    matchingTags = [i for i in bayCornerIDs if i in detectedIDs]

    
    if len(matchingTags) == 0:
        return False, image
    elif len(matchingTags) == 2:


        return True, image
    elif len(matchingTags) == 3:


        return True, image
    else:
        # TODO: Identify bays using average center points of tags, then get outter corner
        minBounds = np.min(detectedCorners, axis=0)
        maxBounds = np.max(detectedCorners, axis=0)
        print(f"minBounds: {minBounds}")
        print(f"maxBounds: {maxBounds}")
        cv2.polylines(image, formatCornersCV(minBounds), isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.polylines(image, formatCornersCV(maxBounds), isClosed=True, color=(0, 0, 255), thickness=2)


        return True, image
    

def payloadBayPackageDetection(images, bayCornerIDs, env='CMD'):

    # detect markers: ids, position in pixels, thetas
    for fname in images: 
        
        _, tail = os.path.split(fname)
        name = tail.replace(".JPG", "")

        image = cv2.imread(fname)
        markerCorners, markerIds, rejectedCandidates = detectMarkers(image)

        # If markers are detected
        if markerIds is not None:
            aruco.drawDetectedMarkers(image, markerCorners, markerIds)
            # # Print marker IDs
            # print("Detected marker IDs:", markerIds)

            bay, image = detectBay(image, bayCornerIDs, markerIds, markerCorners)
            print(bay)
            show("bay", image)

            # # each tag. center coordinates of tag
            # for corners, marker_id in zip(markerCorners, markerIds.flatten()):
            #     cornersReshape = corners.reshape((4, 2)).astype(np.int32)  # Ensure integer type
            #     top_left, top_right, bottom_right, bottom_left = cornersReshape.astype(int)
            #     avgX = int(np.average([[c[0] for c in cornersReshape]]))
            #     avgY = int(np.average([[c[1] for c in cornersReshape]]))
            #     # print(f'{avgX}, {avgY}')

            #     # Draw bounding box around tags
            #     cv2.polylines(image, [cornersReshape], isClosed=True, color=(0, 255, 0), thickness=2)

            #     # draw center point of tags
            #     cv2.circle(image, (avgX,avgY), radius=5, color=(0,0,255), thickness=10)
            #     pixCoordText = f"({avgX}, {avgY})"
            #     idText = f"tagID: {marker_id}"

            #     # Add the coordinates as text
            #     cv2.putText(image, pixCoordText, (avgX + 10, avgY - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
            #     cv2.putText(image, idText, (avgX - 100, avgY + 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(230, 70, 173), thickness=6, lineType=cv2.LINE_AA)
            #     # cv2.namedWindow("Detected Markers", cv2.WINDOW_NORMAL)

            #     # Resize the window to 600x400
            #     # cv2.resizeWindow("Detected Markers", 600, 400)
            #     # Display the image
            #     # cv2.imshow("Detected Markers", undImg)

            #     # theta angle of tag
            #     K=np.array([[2669.6160711764005, 5.984153675872468, 2573.893362213685], [0.0, 2672.716998014564, 2623.5130415465924], [0.0, 0.0, 1.0]])
            #     D =np.array([[0.01938121722377818], [-0.004488854452876614], [-0.0013977634268640517], [0.008871738034432555]])

            #     success, rvec, tvec, _ = estimatePoseSingleMarkers(corners, 120, K, D)
            #     if success:
            #         # print(f"rvec: {rvec}")
            #         # print(f"tvec: {tvec}")
            #         rvec = rvec[0]
            #         R, _ = cv2.Rodrigues(rvec)

            #         # Extract Euler angles (Theta is the yaw angle)
            #         # theta_x = np.arctan2(R[2, 1], R[2, 2])  # Roll
            #         # theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))  # Pitch
            #         theta_z = np.degrees(np.arctan2(R[1, 0], R[0, 0]))  # Yaw (Theta)

            #         thetaText = f"{theta_z} deg"
            #         cv2.putText(image, thetaText, (avgX + 50, avgY + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
            #         worldCoordText = f"({int(tvec[0][0])}, {int(tvec[0][1])}, {int(tvec[0][2])}) mm"
            #         cv2.putText(image, worldCoordText, (avgX - 150, avgY -150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                    

            # # save image
            # cv2.imwrite(os.path.join(detectionPath, f'{os.path.basename(name)}_Success.JPG'), image)

        else: # No tag detected
            print(f"No markers detected in {name}.")
            output_img = aruco.drawDetectedMarkers(image.copy(), rejectedCandidates)
            # cv2.imwrite(os.path.join(detectionPath, f'{os.path.basename(name)}_Failure.JPG'), output_img)
            # show("Rejected Candidates", output_img)

if __name__ == '__main__':

    # basePath, filteredPath, undistortedPath, detectionPath = makeDir()

    images = glob.glob('bayDetectionPhotos/img1.JPG')
    # images = processImages(images)


    payloadBayPackageDetection(images, [203,23,98,62])

    cv2.waitKey(0)
    cv2.destroyAllWindows()