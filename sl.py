import cv2
import cv2.aruco as aruco
import numpy as np

# Load default camera (webcam)
cap = cv2.VideoCapture(0)

# Load ArUco dictionary and parameters
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Sample camera matrix and distortion (normally you calibrate the camera to get this)
# These values are just examples; for accuracy, use your own calibration
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((5, 1))  # assuming no lens distortion

# Marker size in meters (example: 0.05 means 5 cm)
marker_length = 0.05

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw marker boundaries
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of each marker
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            # Draw axis on each marker
            aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
            print("Rotation Vector:", rvec)
            print("Translation Vector:", tvec)

    # Show the frame
    cv2.imshow('AR Marker Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()