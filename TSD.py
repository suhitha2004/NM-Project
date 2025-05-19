import cv2
import numpy as np

# Initialize the video capture (0 for webcam, or file path for video)
cap = cv2.VideoCapture(0)  # Change to your video source if needed

# Load pre-trained model for traffic sign detection
# For this example, we'll use a simple color/shape-based approach
# In a real application, you would use a trained CNN model

# Define color ranges for sign detection (in HSV)
color_ranges = {
    'red': ([0, 70, 50], [10, 255, 255]),
    'blue': ([100, 150, 0], [140, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
    'green': ([40, 70, 80], [80, 255, 255])
}

# Dictionary of sign types we want to detect
sign_types = {
    'stop': {'color': 'red', 'shape': 'octagon'},
    'yield': {'color': 'red', 'shape': 'triangle'},
    'speed_limit': {'color': 'white', 'shape': 'circle'},
    'no_entry': {'color': 'red', 'shape': 'circle'},
    'traffic_light': {'color': ['red', 'green', 'yellow'], 'shape': 'rectangle'}
}

def detect_shapes(contour):
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Triangle
    if len(approx) == 3:
        shape = "triangle"
    # Rectangle or square
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    # Octagon (stop sign)
    elif len(approx) == 8:
        shape = "octagon"
    # Circle
    else:
        shape = "circle"
    
    return shape

def detect_signs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    detected_signs = []
    
    # Check for each color range
    for color_name, (lower, upper) in color_ranges.items():
        # Create mask for the color
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each contour
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 1000:
                continue
                
            # Detect shape
            shape = detect_shapes(contour)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Try to identify the sign type
            sign_type = "unknown"
            for type_name, properties in sign_types.items():
                if (properties['color'] == color_name or 
                    (isinstance(properties['color'], list) and color_name in properties['color'])) and \
                   properties['shape'] == shape:
                    sign_type = type_name
                    break
            
            detected_signs.append({
                'type': sign_type,
                'bbox': (x, y, x+w, y+h),
                'color': color_name,
                'shape': shape
            })
    
    return detected_signs

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for webcam
    frame = cv2.flip(frame, 1)
    
    # Detect signs in the frame
    signs = detect_signs(frame)
    
    # Draw bounding boxes and labels
    for sign in signs:
        x1, y1, x2, y2 = sign['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        label = f"{sign['type']} ({sign['color']} {sign['shape']})"
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Traffic Sign Detection', frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()