#did not test this on the turtle bot yet
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import numpy as np
from robotpy_apriltag import AprilTagDetector
import cv2

class AprilTagDetectorNode(Node):
    def __init__(self):
        super().__init__('april_tag_detector')

        # Subscribe to the TurtleBot3 camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Change if using a different camera topic
            self.image_callback,
            10)
        
        # Publisher for detected tag distances
        self.distance_publisher = self.create_publisher(Float32, '/tag_distance', 10)

        # Publisher for modified image with annotations
        self.image_publisher = self.create_publisher(Image, '/tag_image', 10)

        self.bridge = CvBridge()
        self.detector = AprilTagDetector()
        self.detector.addFamily("tag36h11")  # Ensure this matches your AprilTag family
        
        # Constants
        self.APRILTAG_SIZE = 0.1  # AprilTag size in meters (e.g., 10 cm)
        self.FOCAL_LENGTH_PIXELS = 700  # Adjust this experimentally for best results

    def image_callback(self, msg):
        # Convert ROS2 Image message to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for detection

        # Detect AprilTags
        detections = self.detector.detect(gray)

        for detection in detections:
            homography = detection.getHomography()

            if homography is not None:
                width_pixels = np.linalg.norm(homography[0])  # Approximate width in pixels

                if width_pixels > 0:
                    # Compute distance using the pinhole camera model
                    distance = (self.APRILTAG_SIZE * self.FOCAL_LENGTH_PIXELS) / width_pixels
                    self.get_logger().info(f"AprilTag ID {detection.getId()} - Distance: {distance:.2f}m")

                    # Publish distance to ROS2 topic
                    distance_msg = Float32()
                    distance_msg.data = distance
                    self.distance_publisher.publish(distance_msg)

                    # Draw the detection on the frame
                    center_x, center_y = map(int, detection.getCenter())
                    text = f"ID: {detection.getId()} - {distance:.2f}m"
                    cv2.putText(frame, text, (center_x - 50, center_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Convert modified frame back to ROS2 Image and publish
        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_publisher.publish(ros_image)

        # Show the processed image with OpenCV
        cv2.imshow("AprilTag Detection", frame)
        cv2.waitKey(1)  # Required for OpenCV window updates

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
