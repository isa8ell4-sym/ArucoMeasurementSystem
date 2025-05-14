## accesses live footage, wired connection, find tag
import cv2
import numpy as np
import time
import cv2.aruco as aruco
from processImage import estimatePoseSingleMarkers
from kalman import kalman_predict


# original K and D for photos (which have a different aspect ratio)
# DIM = (1920,1080)
# CAMERA_MATRIX_0=np.array([[961.783242368072, 0, 971.4432373479299], [0.0, 961.902, 540.488], [0.0, 0.0, 1.0]])
# DISTORTION_COEFF_0= np.array([[0.0677954596296918], [-0.2673685939595122], [0.5510758796903314], [-0.3710967486321212]])

# DIM = (1920,1080)
CAMERA_MATRIX = np.array([
    [900.23427416,     0.0, 967.91825739],
    [    0.0,       902.59811495, 488.16118046],
    [    0.0,           0.0,        1.0]
], dtype=np.float64)
DISTORTION_COEFF = np.array(
    [-0.27516087, 0.11849932, 0.00274007, 0.00064677, -0.0268265],
    dtype=np.float64
)
scale = 1

def findTag(frame, tagSize):
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
            # print(f'{avgX}, {avgY}')

            # Draw bounding box around tags
            cv2.polylines(frame, [cornersReshape], isClosed=True, color=(0, 255, 0), thickness=2)

            # draw center point of tags
            cv2.circle(frame, (avgX,avgY), radius=5, color=(0,0,255), thickness=10)
            pixCoord = [avgX, avgY]

            # Add the coordinates as text
            # cv2.putText(frame, pixCoordText, (avgX + 10, avgY - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)


            # world coordinates and angle of tag
            success, rvec, tvec, _ = estimatePoseSingleMarkers(corners, tagSize, CAMERA_MATRIX, DISTORTION_COEFF)
            if success:

                rvec = rvec[0]
                R, _ = cv2.Rodrigues(rvec)

                # Extract Euler angles (Theta is the yaw angle)
                # theta_x = np.arctan2(R[2, 1], R[2, 2])  # Roll
                # theta_y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))  # Pitch
                theta_z = np.degrees(np.arctan2(R[1, 0], R[0, 0]))  # Yaw (Theta)
                worldCoord = [int(tvec[0][0]), int(tvec[0][1]), int(tvec[0][2])] # [round(c,2) for c in tvec[0]]
                

    else: # No tag detected
        success = 0
        pixCoord = worldCoord = marker_id = theta_z = None

    return success, marker_id, pixCoord, worldCoord, theta_z

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

cap = open_camera(1)  # Try different indices if needed
if cap is None:
    print("Could not open GoPro camera.")
    exit()

frameCount = 0

while cap.isOpened():
    ret, frame = cap.read()
    scale = 1
    
    if frameCount == 0 : 
        displayScale = .3
        h, w = frame.shape[:2]
        print(h, w)
        h = int(h * displayScale)
        w = int(w * displayScale)

    if not ret:
        print('failed to open frame')
        break

    if frameCount % 3 == 0:

        # Knew = CAMERA_MATRIX.copy()
        # Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
        # undImg0 = cv2.fisheye.undistortImage(frame, K=CAMERA_MATRIX_0, D=DISTORTION_COEFF_0, Knew=Knew)

        undImg1 = cv2.undistort(frame, CAMERA_MATRIX, DISTORTION_COEFF, None) ## trying this instead


        # success, markerID, pixCoord, worldCoord, theta_z = findTag(undImg1, tagSize=50)



        # # add text if tag found
        # if success: 
        #     x, y, z, theta = kalman_predict(worldCoord[0], worldCoord[1], worldCoord[2], theta_z)
        #     thetaText = f"{int(theta_z)} deg"
        #     idText = f"tagID: {markerID}"
        #     kText = f"kalman: {int(x), int(y), int(z), int(theta)}"
                        
        #     cv2.putText(undImg1, str(worldCoord), (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
        #     cv2.putText(undImg1, thetaText, (50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
        #     cv2.putText(undImg1, idText, (50, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(230, 70, 173), thickness=6, lineType=cv2.LINE_AA)
        #     cv2.putText(undImg1, kText, (50, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(230, 70, 173), thickness=6, lineType=cv2.LINE_AA)
        #     # cv2.putText(frame, worldCoord, (undImg.shape[0], undImg.shape[1]), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
            

        cv2.namedWindow('Distorted', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Distorted', w, h)
        cv2.imshow('Distorted', frame)

        # Display Undistorted 0
        # cv2.namedWindow('Undistorted_0', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Undistorted_0', w, h)
        # cv2.imshow('Undistorted_0', undImg0)

        # Display Undistorted 1
        cv2.namedWindow('Undistorted_1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Undistorted_1', w, h)
        cv2.imshow('Undistorted_1', undImg1)

    frameCount += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # print("Shapes:", frame.shape, undImg0.shape, undImg1.shape)
        break



cap.release()
cv2.destroyAllWindows()
