import numpy as np
import cv2

class Tag: 
    def __init__(self, id, corners, pixCoord=None, worldCoord=None, theta=None):
        self.id = id
        self.corners = corners
        self.pixCoord = pixCoord
        self.worldCoord = worldCoord
        self.theta = theta

    def calcPixPos(self, corners): 
            cornersReshape = corners.reshape((4, 2)).astype(np.int32)  # Ensure integer type
            top_left, top_right, bottom_right, bottom_left = cornersReshape.astype(int)
            avgXPix = int(np.average([[c[0] for c in cornersReshape]]))
            avgYPix = int(np.average([[c[1] for c in cornersReshape]]))

            self.pixCoord = [avgXPix, avgYPix]
            return [avgXPix, avgYPix]

    def estimateRvecTvec(self, corners, marker_size, K, D):
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
            success, R, t = cv2.solvePnP(marker_points, c, K, D, False, cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R)
            tvecs.append(t)
            trash.append(success)
        return success, rvecs, tvecs, trash
    
    def calcWorldPos(self, corners, marker_size, K, D):

        success, rvec, tvec, _ = self.estimateRvecTvec(corners, marker_size, K, D)
        if success: 
            rvec = rvec[0]
            R, _ = cv2.Rodrigues(rvec)

            # Extract Euler angles (Theta is the yaw angle)
            theta_z = np.degrees(np.arctan2(R[1, 0], R[0, 0]))  # Yaw (Theta)

            
            worldPosMM = [int(tvec[0][0]), int(tvec[0][1]), int(tvec[0][2])]
            self.worldCoord = worldPosMM
            self.theta = theta_z
            return success, worldPosMM, theta_z
        else: 
            print(f'could not estimate position of tag {self.id}')
            return success, None, None
        

