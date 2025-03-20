import numpy as np
from robotpy_apriltag import AprilTagDetector
import cv2  # Required for camera access and display

# Constants
APRILTAG_SIZE = 0.0675  # AprilTag size in meters (e.g., 10 cm)
FOCAL_LENGTH_PIXELS = 25  # Approximate focal length in pixels (adjust for accuracy)

# Initialize AprilTag detector
detector = AprilTagDetector()
detector.addFamily("tag36h11")  # Choose the correct tag family

# Start Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (AprilTag detection requires grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    detections = detector.detect(gray)

    for detection in detections:
        # Get the homography matrix (approximate way to extract size in pixels)
        homography = detection.getHomography()

        if homography is not None:
            # The first two rows of homography give an approximation of size
            width_pixels = np.linalg.norm(homography[0])  # Rough pixel width

            if width_pixels > 0:
                # Compute distance using the pinhole camera model
                distance = (APRILTAG_SIZE * FOCAL_LENGTH_PIXELS) / width_pixels
                distance_text = f"ID: {detection.getId()} - Distance: {distance:.2f}m"

                # Draw text on the frame
                center_x, center_y = map(int, detection.getCenter())
                cv2.putText(frame, distance_text, (center_x - 50, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw a circle at the center of the detected AprilTag
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

    # Show the frame with detections
    cv2.imshow("AprilTag Distance Estimation", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
