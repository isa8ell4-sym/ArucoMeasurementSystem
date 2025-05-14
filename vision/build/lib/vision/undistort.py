# from internalCalibration import *
import cv2, json
import cv2.aruco as aruco
import numpy as np
import glob
import datetime
import os

# class undistort: 
#     def __init__(self, K):
#         pass

def undstPhotoFisheye(img):
    # reprojection error = 0.45
   
    DIM=(5568, 4872) 
    K=np.array([[2498.69798, 0, 2773.89781], [0, 2497.4940, 2427.23503], [0.0, 0.0, 1.0]])
    D=np.array([[-0.26189899], [0.08908975], [-0.0015667], [-0.00032891], [-0.01505716]])

    img = cv2.imread(img)

    undImg1 = cv2.undistort(img, K, D, None) ## trying this instead


    return undImg1

def undstPhotoLinear(fname):
    mtx = np.array([[2.27437163e+03, 0.00000000e+00, 2.78599839e+03],
                    [0.00000000e+00, 2.26778978e+03, 2.43566662e+03],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist = np.array([[ 2.58530987e-02, -3.93644636e-02, -3.73699642e-05, 
                    2.22695924e-04,  2.16061178e-02]])


    img = cv2.imread(fname)
    # _, tail = os.path.split(fname)
    # name = tail.replace(".JPG", "")
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h)) 
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx) ## TODO: Try using remap
    
    # crop the image
    x, y, w, h = roi
    undistorted_img = dst[y:y+h, x:x+w]

    return undistorted_img

    # os.chdir(calibPath)

    # cv2.imwrite(f'{name}.JPG', undistorted_img)

def undstVideoFisheye(frame):
    CAMERA_MATRIX = np.array([
    [900.23427416,     0.0, 967.91825739],
    [    0.0,       902.59811495, 488.16118046],
    [    0.0,           0.0,        1.0]
    ], dtype=np.float64)
    DISTORTION_COEFF = np.array(
        [-0.27516087, 0.11849932, 0.00274007, 0.00064677, -0.0268265],
        dtype=np.float64
    )
    undImg1 = cv2.undistort(frame, CAMERA_MATRIX, DISTORTION_COEFF, None) ## trying this instead
    return undImg1

def undstPhotoFisheyePadding(img, scale_factor):
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

def undistortCropWidePhotos(img): #Old
    img = cv2.imread(img)


    # with open('calibParams.json', 'r') as file:
    #     data = json.load(file)
    
    # ret = data["ret"]
    # mtx = np.array(data["mtx"])  # Convert camera matrix
    # dist = np.array(data["dist"])  # Convert distortion coefficients
    # rvecs = [np.array(rvec) for rvec in data["rvecs"]]  # Convert rotation vectors
    # tvecs = [np.array(tvec) for tvec in data["tvecs"]]  # Convert translation vectors

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

#### from video detection
# # case migration
# def undistortCropWidePhotos(img):
#     img = cv2.imread(img)


#     with open('calibParams.json', 'r') as file:
#         data = json.load(file)
    
#     ret = data["ret"]
#     mtx = np.array(data["mtx"])  # Convert camera matrix
#     dist = np.array(data["dist"])  # Convert distortion coefficients
#     rvecs = [np.array(rvec) for rvec in data["rvecs"]]  # Convert rotation vectors
#     tvecs = [np.array(tvec) for tvec in data["tvecs"]]  # Convert translation vectors

#     DIM=(5568, 4872) 
#     K=np.array([[2669.6160711764005, 5.984153675872468, 2573.893362213685], [0.0, 2672.716998014564, 2623.5130415465924], [0.0, 0.0, 1.0]])
#     D=np.array([[0.01938121722377818], [-0.004488854452876614], [-0.0013977634268640517], [0.008871738034432555]])


#     # h,  w = img.shape[:2]
#     # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

#     # # unImg = cv2.undistortCrop(img, mtx, dist, None, newcameramtx)
    
#     # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
#     # unImg = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     # TODO: change to tutorial 1
#     Knew = K.copy()
#     Knew[(0,1), (0,1)] = 1 * Knew[(0,1), (0,1)]
#     unImg = cv2.fisheye.undistortImage(img, K, D, Knew=Knew)

#     return unImg

# # case migration
# def undistortPaddingWidePhotos(img, scale_factor):
#     DIM=(4872, 5568) 
#     K=np.array([[2669.6160711764005, 5.984153675872468, 2573.893362213685], [0.0, 2672.716998014564, 2623.5130415465924], [0.0, 0.0, 1.0]])
#     D=np.array([[0.01938121722377818], [-0.004488854452876614], [-0.0013977634268640517], [0.008871738034432555]])

#     # print(img)
#     img = cv2.imread(img)
#     dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
#     # print(dim1)
#     dim2 = (int(dim1[1] * scale_factor), int(dim1[0] * scale_factor))
#     assert dim1[0]/dim1[1] == DIM[0]/DIM[1], f"Image to undistort needs to have same aspect ratio as the ones used in calibration, {dim1[0], dim1[1]} != {DIM[0], DIM[1]}" 
#     # print(dim1, DIM)

#     scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
#     scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
#     # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
#     new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=1)
#     map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim1, cv2.CV_16SC2)
#     undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#     ## Method 2
#     # Knew = K.copy()
#     # Knew[(0,1), (0,1)] = 1 * Knew[(0,1), (0,1)]
#     # unImg = cv2.fisheye.undistortImage(img, K, D, Knew=Knew)

    
#     return undistorted_img
