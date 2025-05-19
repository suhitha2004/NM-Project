import cv2
import cv2.aruco as aruco
import numpy as np

# Camera calibration (replace with real values for accuracy)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # No distortion

# ArUco marker setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Draw detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Pose estimation
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            # Draw 3D axis
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

            # Get marker ID
            marker_id = ids[i][0]

            # Get corners (4 points: top-left, top-right, bottom-right, bottom-left)
            marker_corners = corners[i].reshape(4, 2)

            print("Marker Id",marker_id)
            print("Corners (bounding box):")
            for j, point in enumerate(marker_corners):
                x, y = point
                print(f" Corner {j+1}: (x={x:.2f}, y={y:.2f})")

            # Also print pose info
            print(f" Translation: {tvecs[i].flatten()}")
            print(f" Rotation: {rvecs[i].flatten()}")
            print('-' * 40)

    # Show output
    cv2.imshow('AR Marker Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()