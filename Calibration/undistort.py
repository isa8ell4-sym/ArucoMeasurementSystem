# from internalCalibration import *
import cv2, json
import cv2.aruco as aruco
import numpy as np
import glob
import datetime
import os

# case migration
def undistortCropWidePhotos(img):
    img = cv2.imread(img)


    with open('calibParams.json', 'r') as file:
        data = json.load(file)
    
    ret = data["ret"]
    mtx = np.array(data["mtx"])  # Convert camera matrix
    dist = np.array(data["dist"])  # Convert distortion coefficients
    rvecs = [np.array(rvec) for rvec in data["rvecs"]]  # Convert rotation vectors
    tvecs = [np.array(tvec) for tvec in data["tvecs"]]  # Convert translation vectors

    DIM=(5568, 4872) 
    K=np.array([[2669.6160711764005, 5.984153675872468, 2573.893362213685], [0.0, 2672.716998014564, 2623.5130415465924], [0.0, 0.0, 1.0]])
    D=np.array([[0.01938121722377818], [-0.004488854452876614], [-0.0013977634268640517], [0.008871738034432555]])


    # h,  w = img.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # # unImg = cv2.undistortCrop(img, mtx, dist, None, newcameramtx)
    
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    # unImg = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # TODO: change to tutorial 1
    Knew = K.copy()
    Knew[(0,1), (0,1)] = 1 * Knew[(0,1), (0,1)]
    unImg = cv2.fisheye.undistortImage(img, K, D, Knew=Knew)

    return unImg

# case migration
def undistortPaddingWidePhotos(img, scale_factor):
    DIM=(4872, 5568) 
    K=np.array([[2669.6160711764005, 5.984153675872468, 2573.893362213685], [0.0, 2672.716998014564, 2623.5130415465924], [0.0, 0.0, 1.0]])
    D=np.array([[0.01938121722377818], [-0.004488854452876614], [-0.0013977634268640517], [0.008871738034432555]])

    # print(img)
    img = cv2.imread(img)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    # print(dim1)
    dim2 = (int(dim1[1] * scale_factor), int(dim1[0] * scale_factor))
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], f"Image to undistort needs to have same aspect ratio as the ones used in calibration, {dim1[0], dim1[1]} != {DIM[0], DIM[1]}" 
    # print(dim1, DIM)

    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=1)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    ## Method 2
    # Knew = K.copy()
    # Knew[(0,1), (0,1)] = 1 * Knew[(0,1), (0,1)]
    # unImg = cv2.fisheye.undistortImage(img, K, D, Knew=Knew)

    
    return undistorted_img
