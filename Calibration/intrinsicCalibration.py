import numpy as np
import cv2 
import glob
import os
import datetime
import codecs, json,time
import matplotlib.pyplot as plt

# # from calibPhotosVideo2
# DIM = (1920,1080)
# CAMERA_MATRIX=np.array([[961.783242368072, 0, 971.4432373479299], [0.0, 961.3270106900933, 540.4568833909045], [0.0, 0.0, 1.0]])
# DISTORTION_COEFF= np.array([[0.0677954596296918], [-0.2673685939595122], [0.5510758796903314], [-0.3710967486321212]])
DIM=(4872, 5568)
CAMERA_MATRIX=np.array([[2669.6160711761095, 5.984153675882966, 2573.8933622139734], [0.0, 2672.7169980142894, 2623.513041546733], [0.0, 0.0, 1.0]]) #np.array([[2653.4452971622027, -3.1603263457583113, 2783.2573129364737], [0.0, 2651.97097890021, 2445.7552144485167], [0.0, 0.0, 1.0]])
DISTORTION_COEFF=np.array([[0.01938121722503705], [-0.0044888544596155605], [-0.0013977634137292841], [0.0088717380261235]]) #np.array([[0.03247919682991756], [-0.0587965706011023], [0.23088243787715307], [-0.2560286610203281]])

def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)

def getReprojectionError(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    
    print( "total error: {}".format(mean_error/len(objpoints)) )

def getParamsFisheye(images, basePath, calibPath):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1) # 0.001)

    objp = np.zeros((1,7*9, 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    shape = []

    # images = glob.glob('calibPhotosVideo2/*.JPG')
    # print("Found images:", images)  # Debugging step

    # read checkerboard
    for fname in images:
        
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,9), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE) #, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            # use for cv.fisheye.calibrate 
            corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
            imgpoints.append(corners2.reshape(-1, 1, 2))  

            print(f"len of img points: {len(imgpoints)}\nlen of obj points: {len(objpoints)}")


            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,9), corners2, ret)
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # cv2.imshow('img', img)
            os.chdir(calibPath)
            filename = os.path.basename(fname)

            cv2.imwrite(os.path.join(calibPath, f'c_{filename}'), img)
            os.chdir(basePath)

        else:
            print(f'no chessboard found in {fname}')

    print(f'objpoints: {type(objpoints)}\n\nimgpoints: {type(imgpoints)}')
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
    
    getReprojectionError(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(img.shape[::-1]))
    print("DIM=" + str(img.shape))
    print("K=np.array(" + str(K.tolist()) + ")")
    # print("K=" + K)
    print("D=np.array(" + str(D.tolist()) + ")") #cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], None, None)
    # DIM=(5568, 4872) ### might need to change this to be backwards

def getParamsLinear(images):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1) # 0.001)

    objp = np.zeros((1,7*9, 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:7, 0:9].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # read checkerboard
    for fname in images:
        
        # print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,9), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE) #, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            # use for cv.fisheye.calibrate #TODO: this might be messing up the calibration ###############################################################################
            corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
            imgpoints.append(corners2.reshape(-1, 1, 2))  


            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,9), corners2, ret)
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # cv2.imshow('img', img)
            os.chdir(calibPath)
            filename = os.path.basename(fname)
            # print(filename)
            cv2.imwrite(os.path.join(calibPath, f'c_{filename}'), img)
            os.chdir(basePath)
            # cv2.waitKey(100)
        else:
            print(f'no chessboard found in {fname}')

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("DIM=" + str(img.shape[::-1]))
    print("DIM=" + str(img.shape))
    print(f"ret={ret}\nmtx={mtx}\ndist={dist}\nrvecs={rvecs}\ntvecs={tvecs}")
    return mtx, dist, rvecs, tvecs

def getParamsFisheyeCharuco(images, basePath, calibPath):
    # ENTER YOUR REQUIREMENTS HERE:
    ARUCO_DICT = cv2.aruco.DICT_4X4_100
    SQUARES_VERTICALLY = 11
    SQUARES_HORIZONTALLY = 8
    SQUARE_LENGTH = 0.02
    MARKER_LENGTH = 0.015
    # ...
     # Define the aruco dictionary and charuco board
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    board.setLegacyPattern(True)
    parameters = cv2.aruco.DetectorParameters()

    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 50
    parameters.adaptiveThreshWinSizeStep = 5
    parameters.minMarkerPerimeterRate = .1
    parameters.maxMarkerPerimeterRate = 3.0
    parameters.errorCorrectionRate = 0.5  # Increase to allow noisy markers to be decoded

    arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)
    charucoParams = cv2.aruco.CharucoParameters()
    charucoDetector = cv2.aruco.CharucoDetector(board) #, charucoParams, parameters)

    # Prepare data
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    image_size = None

    all_charuco_corners = []
    all_charuco_ids = []

    # build up object and image points
    for fname in images:
        # print(f"img: {fname}")
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_copy = image.copy()
        image_size = gray.shape[::-1]  # (width, height)
        try:
            marker_corners, marker_ids, rejected_img_pts = arucoDetector.detectMarkers(image)
            # print(f"marker ids: {marker_ids}")
                  
            # print(f'marker ids: {marker_ids}')
            cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            show('charuco', image_copy)
            # Convert to proper format if needed
            marker_corners_charuco = np.array([corner[0] for corner in marker_corners])
            marker_ids_charuco = np.array(marker_ids, dtype=np.int32)

            # print(f'reformatted: \n{type(marker_corners_charuco)}')
            # print(f'reformatted: {type(marker_ids_charuco)}')
            charuco_corners, charuco_ids, recovered_corners, recovered_ids = charucoDetector.detectBoard(gray, marker_corners_charuco, marker_ids_charuco) #########################
            
            # print(f'corners: {charuco_corners}\nidentifiers: {charuco_ids}\nrecovered corners: {recovered_corners}\n recovered ids: {recovered_ids}\n')

            if charuco_corners is not None and len(charuco_corners) > 4:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

        except TypeError:
            print(f'could not identify charuco board in {fname}')


    objpoints_chessBoard = board.getChessboardCorners()

    count = 0 #TODO: formating incorrectly, appending weird
    for corners, ids in zip(all_charuco_corners, all_charuco_ids): 
        if len(corners) >= 4:
            count += 1
            obj_points = objpoints_chessBoard[ids]
            objpoints.append(obj_points)
            imgpoints.append(corners)
            # print(f'appended {obj_points} and {corners}')
    print(f"len of img points: {len(imgpoints)}\nlen of obj points: {len(objpoints)}")
        

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_CHECK_COND +
        cv2.fisheye.CALIB_FIX_SKEW
    )
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))

    # print(f'\nobject points\n {obj_points}')

    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            image_size,
            K,
            D,
            rvecs,
            tvecs,
            flags,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )
    
    getReprojectionError(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(image.shape[::-1]))
    print("DIM=" + str(image.shape))
    print("K=np.array(" + str(K.tolist()) + ")")
    # print("K=" + K)
    print("D=np.array(" + str(D.tolist()) + ")") #cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], None, None)

    cv2.destroyAllWindows()

def undistortImagesFisheye(images, basePath, calibPath):

    if calibPath is None:
        print("ERROR! Not using calib path, fix!")

    i = 0
    for fname in images:

        os.chdir(basePath)
        img = cv2.imread(fname)

        # h,w = img.shape[:2]
        # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, [x*2 for x in DIM], cv2.CV_16SC2)
        # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        ## Method 2 
        # scale_factor = 1

        # orig_extents = img.shape[:2][::-1]
        # final_extents = orig_extents * scale_factor
        # scaled_K = CAMERA_MATRIX * scale_factor
        # scaled_K[2][2] = 1.0

        # print(f"{orig_extents} => {final_extents} (sf = {scale_factor})")

        # # new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(CAMERA_MATRIX, DISTORTION_COEFF, orig_extents, np.eye(3))
        # new_camera_matrix = CAMERA_MATRIX.copy()
        # new_camera_matrix[0,0]=new_camera_matrix[0,0]/2
        # new_camera_matrix[1,1]=new_camera_matrix[1,1]/2

        # print(CAMERA_MATRIX)
        # print(new_camera_matrix)

        # map1, map2 = cv2.fisheye.initUndistortRectifyMap(CAMERA_MATRIX, DISTORTION_COEFF, np.eye(3), new_camera_matrix, orig_extents, cv2.CV_16SC2)

        # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        

        ## Method 3 
        # use Knew to scale the output
        Knew = CAMERA_MATRIX.copy()
        Knew[(0,1), (0,1)] = .7 * Knew[(0,1), (0,1)]
        undistorted_img = cv2.fisheye.undistortImage(img, CAMERA_MATRIX, D=DISTORTION_COEFF, Knew=Knew)

        # cv2.imshow("ud", undistorted_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()

        # dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        # print(dim1)
        # dim2 = (int(dim1[0] * scale_factor), int(dim1[1] * scale_factor))
        # assert dim1[0]/dim1[1] == DIM[0]/DIM[1], f"Image to undistort needs to have same aspect ratio as the ones used in calibration, {dim1[0], dim1[1]} != {DIM[0], DIM[1]}" 
        # print(dim2, DIM)

        # scaled_K = (K * (dim1[0] / DIM[0])).astype(np.float64)  # The values of K is to scale with image dimension.
        # scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!



        # plt.subplot(1, 2, 1)
        # plt.imshow(map1[:, :, 0], cmap="gray")  # X-coordinates
        # plt.title("M ap1 - X Coordinates")
        # plt.colorbar()

        # plt.subplot(1, 2, 2)
        # plt.imshow(map1[:, :, 1], cmap="gray")  # Y-coordinates
        # plt.title("Map1 - Y Coordinates")
        # plt.colorbar()

        # plt.show()

        os.chdir(calibPath)

        cv2.imwrite(f'{i}_uncropped.JPG', undistorted_img)
        i = i+1

def undistortImagesLinear(images, calibPath, mtx, dist):

    if calibPath is None:
        print("ERROR! Not using calib path, fix!")

    i = 0
    for fname in images:

        os.chdir(basePath)
        img = cv2.imread(fname)
        _, tail = os.path.split(fname)
        name = tail.replace(".JPG", "")
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h)) 
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx) ## TODO: Try using remap
        
        # crop the image
        x, y, w, h = roi
        undistorted_img = dst[y:y+h, x:x+w]

        os.chdir(calibPath)

        cv2.imwrite(f'{name}.JPG', undistorted_img)
        i = i+1




if __name__ == '__main__':
    basePath = f"C:/Users/irosenstein/Documents/Vision/Photos/CharucoCalibration2" 
    # basePath = f"C:/Users/irosenstein/Documents/Vision/Photos/fisheyeCalibration1" 
    now = datetime.datetime.now()
    os.mkdir(f'Results/CalibResults{now.month}.{now.day}_{now.hour}.{now.minute}')
    calibPath = f"C:/Users/irosenstein/Documents/Vision/Projects/Calibration/Results/CalibResults{now.month}.{now.day}_{now.hour}.{now.minute}/"
    calibPath = f"C:/Users/irosenstein/Documents/Vision/Projects/Calibration/Results"
    images = glob.glob(f"C:/Users/irosenstein/Documents/Vision/Photos/CharucoCalibration2/*.JPG")
    # images = glob.glob(f"C:/Users/irosenstein/Documents/Vision/Photos/fisheyeCalibration1/*.JPG")
    # getParamsFisheye(images, basePath, calibPath) # run this to get K, D
    getParamsFisheyeCharuco(images, basePath, calibPath)
    # undistortImagesFisheye(images, basePath, calibPath)
    # undistortImages(images, calibPath) # run this to undistort












# # save data to json
# filename = 'calibParams.json'
# data = {
#     "ret": ret, 
#     "mtx": mtx.tolist(), 
#     "dist": dist.tolist(), 
#     "rvecs": [rvec.tolist() for rvec in rvecs],  # Convert each rvecs entry to list
#     "tvecs": [tvec.tolist() for tvec in tvecs],   # Convert each tvecs entry to list
#     "DIM": (4872,5568), 
#     "K": [K.tolist() for K in K], 
#     "D": [D.tolist() for D in D]
# }
# with open(filename, 'w') as file:
#     json.dump(data, file, indent=4) # indent for pretty printing

# print(f"Data saved to {filename}")