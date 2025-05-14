import cv2
import numpy as np

import cv2
import numpy as np

# # Initialize Kalman Filter
# kalman = cv2.KalmanFilter(6, 3)  # 4 state vars (x, y, theta, dx, dy, dt) and 2 measurement vars (x, y, theta)
# kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
# kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
#                                     [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
# kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
# kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

# def kalman_predict(x, y, theta):
#     """Predict and update with the Kalman filter."""
#     prediction = kalman.predict()
    
#     # Measurement update
#     measurement = np.array([[np.float32(x)], [np.float32(y)], [np.float32(theta)]])
#     kalman.correct(measurement)
    
#     # Smoothed position
#     x, y, theta = kalman.statePost[0], kalman.statePost[1], kalman.statePost[2]
#     return x, y, theta


# Create Kalman filter with 8 states (x, y, z, theta, dx, dy, dz, dtheta) and 4 measurements (x, y, z, theta)
kalman = cv2.KalmanFilter(8, 4)

# State Transition Matrix F
kalman.transitionMatrix = np.array([
    [1, 0, 0, 0, 1, 0, 0, 0],   # x' = x + dx
    [0, 1, 0, 0, 0, 1, 0, 0],   # y' = y + dy
    [0, 0, 1, 0, 0, 0, 1, 0],   # z' = z + dz
    [0, 0, 0, 1, 0, 0, 0, 1],   # theta' = theta + dtheta
    [0, 0, 0, 0, 1, 0, 0, 0],   # dx' = dx
    [0, 0, 0, 0, 0, 1, 0, 0],   # dy' = dy
    [0, 0, 0, 0, 0, 0, 1, 0],   # dz' = dz
    [0, 0, 0, 0, 0, 0, 0, 1]    # dtheta' = dtheta
], np.float32)

# Measurement Matrix H
kalman.measurementMatrix = np.eye(4, 8, dtype=np.float32)  # Map measurements to states


# Process Noise Covariance
kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-3
# Measurement Noise Covariance
kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

# Error Covariance Matrix
kalman.errorCovPost = np.eye(8, dtype=np.float32) * 0.1

# Initial state
kalman.statePost = np.zeros((8, 1), dtype=np.float32)

def kalman_predict(x, y, z, theta):
    """Predict and update with the Kalman filter."""
    prediction = kalman.predict()

    # Measurement update
    measurement = np.array([[np.float32(x)], 
                            [np.float32(y)], 
                            [np.float32(z)],
                            [np.float32(theta)]])
    
    kalman.correct(measurement)

    # Return smoothed position and angle
    x, y, z, theta = kalman.statePost[0], kalman.statePost[1], kalman.statePost[2], kalman.statePost[3]
    return x, y, z, theta