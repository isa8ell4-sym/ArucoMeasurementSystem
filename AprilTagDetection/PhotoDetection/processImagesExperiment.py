
import cv2, json
import cv2.aruco as aruco
import numpy as np
import glob
import datetime
import os
# from undistort import *
import pandas as pd
from pandas import DataFrame

class PositionData:
    def __init__(self, names, ids, x, y, z, theta):
        self.names = names
        self.ids = ids
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta

def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

def makeDir(name): # only works for CMD
    now = datetime.datetime.now()

    basePath = f"C:/Users/irosenstein/Documents/Vision/Projects/AprilTagDetection/PhotoDetection/processing/{name}_{now.month}.{now.day}_{now.hour}.{now.minute}/"
    os.makedirs(basePath,  exist_ok=True)
    filteredPath = os.path.join(basePath, f'filter')
    stepperPath = os.path.join(basePath, f'filterStepper')
    detectionPath = os.path.join(basePath, f'tagDetection')
    os.makedirs(filteredPath, exist_ok=True)
    os.makedirs(stepperPath, exist_ok=True)
    os.makedirs(detectionPath, exist_ok=True)
    return basePath, filteredPath, stepperPath, detectionPath

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
        # Load the image
        # image = cv2.imread(fname)
        # undImg = undistortPaddingWidePhotos(fname, 1.5)
        undImg = undistortFisheyeCrop(fname)
        # undImg = undistortImagesLinear(fname)

        os.chdir(filteredPath)
        # print(os.listdir(filteredPath))  
        cv2.imwrite(f'{name}_undistorted.JPG', undImg)

        gray = cv2.cvtColor(undImg, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{name}_gray.JPG', gray)

        dialated = cv2.dilate(gray, np.ones((7,7), np.uint8))
        cv2.imwrite(f'{name}_dialted.JPG', dialated)
        # show("dialated", dialated)

        bg_img = cv2.medianBlur(dialated, 21)
        # show("bg", bg_img)
        cv2.imwrite( f'{name}_blur.JPG', bg_img)

        diff_img = 255 - cv2.absdiff(gray, bg_img)

        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # show("normalized", norm_img)
        cv2.imwrite(f'{name}_n1.JPG', norm_img)

        norm_img2 = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # show("normalizedGray", norm_img2)
        cv2.imwrite(f'{name}_n2.JPG', norm_img2)

        # thresh_img = cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                    cv2.THRESH_BINARY, 11, 2)
        # show("thresholding", thresh_img)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_img = clahe.apply(norm_img)
        cv2.imwrite(f'{name}_clache.JPG', enhanced_img)

        # os.chdir(stepperPath)
        cv2.imwrite(os.path.join(stepperPath, f'{name}.JPG'), undImg)
        # i+=1
        # show("clache", enhanced_img)
    #TODO should make return images

def undistortImages(images, stepperPath):
    for img in images:
        _, tail = os.path.split(img)
        name = tail.replace(".JPG", "")

        # undImg = undistortFisheyeCrop(img)
        undImg = undistortImagesLinear(img)
        cv2.imwrite(os.path.join(stepperPath, f'{name}.JPG'), undImg)

def detectTagsImages(images, detectionPath, tagSize, menv='CMD'):

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
    nameData = []
    idData = []
    xData = []
    yData = []
    zData = []
    tData = []

    # detect markers: ids, position in pixels, thetas
    for fname in images: 
        # print(fname)
        _, tail = os.path.split(fname)
        name = tail.replace(".JPG", "")



        image = cv2.imread(fname)
        # print(f"shape: {image.shape}")
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)

        # If markers are detected
        if markerIds is not None:
            aruco.drawDetectedMarkers(image, markerCorners, markerIds)
            # Print marker IDs
            # print("Detected marker IDs:", markerIds)

            # each tag. center coordinates of tag
            for corners, marker_id in zip(markerCorners, markerIds.flatten()):
                nameData.append(name)
                idData.append(marker_id)
                # print(f"corners: {corners}") # debugging / resolution purposes
                
                cornersReshape = corners.reshape((4, 2)).astype(np.int32)  # Ensure integer type
                top_left, top_right, bottom_right, bottom_left = cornersReshape.astype(int)
                avgX = int(np.average([[c[0] for c in cornersReshape]]))
                avgY = int(np.average([[c[1] for c in cornersReshape]]))
                # print(f'{avgX}, {avgY}')

                # Draw bounding box around tags
                cv2.polylines(image, [cornersReshape], isClosed=True, color=(0, 255, 0), thickness=2)

                # draw center point of tags
                cv2.circle(image, (avgX,avgY), radius=5, color=(0,0,255), thickness=10)
                pixCoordText = f"({avgX}, {avgY})"
                idText = f"tagID: {marker_id}"

                # Add the coordinates as text
                cv2.putText(image, pixCoordText, (avgX + 10, avgY - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                cv2.putText(image, idText, (avgX - 100, avgY + 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(230, 70, 173), thickness=6, lineType=cv2.LINE_AA)
                # cv2.namedWindow("Detected Markers", cv2.WINDOW_NORMAL)

                # theta angle of tag #TODO #########################################################################################################################################################################
                # K = np.array([[2669.6160711764005, 5.984153675872468, 2573.893362213685], [0.0, 2672.716998014564, 2623.5130415465924], [0.0, 0.0, 1.0]])
                # D = np.array([[0.01938121722377818], [-0.004488854452876614], [-0.0013977634268640517], [0.008871738034432555]])

                # linear K and D
                K = np.array([[2.27437163e+03, 0.00000000e+00, 2.78599839e+03],
                                [0.00000000e+00, 2.26778978e+03, 2.43566662e+03],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

                D = np.array([[ 2.58530987e-02, -3.93644636e-02, -3.73699642e-05, 
                                2.22695924e-04,  2.16061178e-02]])
                print(corners)
                success, rvec, tvec, _ = estimatePoseSingleMarkers(corners, tagSize, K, D)
                if success:
                    rvec = rvec[0]
                    R, _ = cv2.Rodrigues(rvec)

                    # Extract Euler angles (Theta is the yaw angle)
                    theta_z = np.degrees(np.arctan2(R[1, 0], R[0, 0]))  # Yaw (Theta)

                    thetaText = f"{theta_z} deg"
                    # cv2.putText(image, thetaText, (avgX + 50, avgY + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                    worldCoordText = f"({int(tvec[0][0])}, {int(tvec[0][1])}, {int(tvec[0][2])}) mm"
                    # cv2.putText(image, worldCoordText, (avgX - 150, avgY -150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                    
                    xData.append(int(tvec[0][0]))
                    yData.append(int(tvec[0][1]))
                    zData.append(int(tvec[0][2]))
                    tData.append(round(theta_z, 2))

                    cv2.putText(image, worldCoordText, (50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)
                    cv2.putText(image, thetaText, (50, 250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(0, 0, 255), thickness=8, lineType=cv2.LINE_AA)
                    cv2.putText(image, idText, (50, 400), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=4, color=(230, 70, 173), thickness=8, lineType=cv2.LINE_AA)

            # save image
            cv2.imwrite(os.path.join(detectionPath, f'{os.path.basename(name)}_Success.JPG'), image)

        else: # No tag detected
            print(f"No markers detected in {name}.")
            output_img = aruco.drawDetectedMarkers(image.copy(), rejectedCandidates)
            cv2.imwrite(os.path.join(detectionPath, f'{os.path.basename(name)}_Failure.JPG'), output_img)
            idData.append(None)
            xData.append(None)
            yData.append(None)
            zData.append(None)
            tData.append(None)
            # show("Rejected Candidates", output_img)
    positionData = PositionData(names=nameData, ids=idData, x=xData, y=yData, z=zData, theta=tData)
    return positionData

if __name__ == '__main__':
    name = 'CaseHandlingVariation_4_15_25'
    tagSize = 100 # mm

    basePath, filterPath, stepperPath, detectionPath = makeDir(name)

    # images = glob.glob('photos_2_28_25/1750 cycles/*.JPG')
    # images = glob.glob('positionalTesting/20mmXZ/*.JPG')

    images = glob.glob('C:/Users/irosenstein/Documents/StretchBotDVT/CaseHandlingVariation/photos/photos_4_15_25/*.JPG')
    undistortImages(images, stepperPath)

    undistortedImages = glob.glob(f'{stepperPath}/*.JPG')
    # undistortedImages = images

    pd = detectTagsImages(undistortedImages, detectionPath, tagSize)

    df = DataFrame({'Img Name': pd.names, 'Tag ID': pd.ids, 'x': pd.x, 'y': pd.y, 'z':pd.z, 'theta': pd.theta})
    df.to_excel(f'{basePath}/{name}_PositionData.xlsx', sheet_name='sheet1', index=False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()