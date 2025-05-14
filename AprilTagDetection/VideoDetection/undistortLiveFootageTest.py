import cv2 as cv
import numpy as np

K=np.array([[2669.6160711764005, 5.984153675872468, 2573.893362213685], [0.0, 2672.716998014564, 2623.5130415465924], [0.0, 0.0, 1.0]])
D=np.array([[0.01938121722377818], [-0.004488854452876614], [-0.0013977634268640517], [0.008871738034432555]])

# Initialize video capture
cap = cv.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Get optimal new camera matrix and ROI
new_K, roi = cv.getOptimalNewCameraMatrix(K, D, (640, 480), 1, (640, 480))

# Generate undistortion map
mapx, mapy = cv.initUndistortRectifyMap(K, D, None, new_K, (640, 480), cv.CV_32FC1)

while(True):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Undistort the frame using remap
    undistorted_frame = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)

    # Crop the image
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]

    cv.imshow('Undistorted', undistorted_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()