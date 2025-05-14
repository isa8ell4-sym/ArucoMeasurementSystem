import numpy as np
import cv2 
import glob
import os
import datetime
import codecs, json,time

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1) # 0.001)


objp = np.zeros((1,7*9, 3), np.float32)
objp[0,:,:2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
shape = []

images = glob.glob('calibrationPhotos/*.JPG')
# print("Found images:", images)  # Debugging step


for fname in images:
    # head, tail = os.path.split(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)


    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,9), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # use for cv.calibrateCamera
        # corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        # imgpoints.append(corners2) 

        # use for cv.fisheye.calibrate
        corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
        imgpoints.append(corners2.reshape(-1, 1, 2))  


        # Draw and display the corners
        # cv2.drawChessboardCorners(img, (7,9), corners2, ret)
        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow('img', img)
        # cv2.imwrite(f'c_{fname}', img)
        # cv2.waitKey(100)
    else:
        print(f'no chessboard found in {fname}')


# cv2.destroyAllWindows()

# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        0,
        criteria
    )
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(img.shape[::-1]))
print("DIM=" + str(img.shape))
print("K=np.array(" + str(K.tolist()) + ")")
# print("K=" + K)
print("D=np.array(" + str(D.tolist()) + ")")#cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], None, None)
DIM=(4872,5568) ### might need to change this to be backwards
K=np.array([[2669.6160711764005, 5.984153675872468, 2573.893362213685], [0.0, 2672.716998014564, 2623.5130415465924], [0.0, 0.0, 1.0]])
D=np.array([[0.01938121722377818], [-0.004488854452876614], [-0.0013977634268640517], [0.008871738034432555]])

# save data to json
filename = 'calibParams.json'
data = {
    "ret": ret, 
    "mtx": mtx.tolist(), 
    "dist": dist.tolist(), 
    "rvecs": [rvec.tolist() for rvec in rvecs],  # Convert each rvecs entry to list
    "tvecs": [tvec.tolist() for tvec in tvecs],   # Convert each tvecs entry to list
    "DIM": (4872,5568), 
    "K": [K.tolist() for K in K], 
    "D": [D.tolist() for D in D]
}
with open(filename, 'w') as file:
    json.dump(data, file, indent=4) # indent for pretty printing

print(f"Data saved to {filename}")


now = datetime.datetime.now()
os.mkdir(f'CalibResults{now.month}.{now.day}_{now.hour}.{now.minute}')
calibPath = f'/mnt/c/Users/irosenstein/Documents/VibrationTesting/VibeTestingTags/CalibResults{now.month}.{now.day}_{now.hour}.{now.minute}/'
basePath = f'/mnt/c/Users/irosenstein/Documents/VibrationTesting/VibeTestingTags/'
i = 0


for fname in images:

    os.chdir(basePath)
    img = cv2.imread(fname)
    # print(img)
    
    h,  w = img.shape[:2]
    # print(h,w)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # print(newcameramtx)
    # undistort
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    os.chdir(calibPath)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # scale_factor = 1
    # new_w, new_h = int(w * scale_factor), int(h * scale_factor)

    # # Scale camera matrix to fit new size
    # new_K = K.copy()
    # new_K[:2] /= scale_factor  # Reduce focal length

    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(new_K, D, np.eye(3), new_K, (new_w, new_h), cv2.CV_16SC2)
    # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


    # new_mtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, (w, h), np.eye(3), balance=1.0)

    # # Undistort the image with black filling around the edges
    # undistorted_img = cv2.fisheye.undistortImage(img, mtx, dist, None, new_mtx)

    cv2.imwrite(f'{i}_uncropped.JPG', undistorted_img)
    i = i+1






# img = cv2.imread("images/oneBox021925.jpg")
# h,  w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# # undistort
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]



# cv2.namedWindow(f'c_{fname}', cv2.WINDOW_NORMAL)
# cv2.imwrite(os.path.join(path, f'c_GP0100{i}.JPG'), dst)