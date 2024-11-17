#!/usr/bin/env python3

import os
import rospy
import cv2
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from duckietown_msgs.msg import WheelsCmdStamped


class CameraReaderNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"

        # bridge between OpenCV and ROS
        self._bridge = CvBridge()

        # create window for visualization
        self._window = "camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)

        # construct subscriber for camera
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        # construct publisher for wheel commands
        self._publisher = rospy.Publisher(self._wheels_topic, WheelsCmdStamped, queue_size=1)

        # Initialize the movement state
        self._is_stopped = False
        self._default_vel_left = 0.3
        self._default_vel_right = 0.3

    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)

        # detect stop sign
        stop_detected = self.detect_stop_sign(image)

        if stop_detected and not self._is_stopped:
            self.stop_wheels()
        elif not stop_detected and self._is_stopped:
            self.resume_wheels()

        # display frame (optional)
        cv2.imshow(self._window, image)
        cv2.waitKey(1)

    def detect_stop_sign(self, image):
        """Detects if there is a high red value area in the image."""
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define red color range in HSV
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Analyze a specific region (e.g., top center of the image)
        height, width = image.shape[:2]
        region_of_interest = red_mask[0:height // 3, width // 3:2 * width // 3]

        # Count the number of red pixels
        red_pixel_count = cv2.countNonZero(region_of_interest)

        # Define a threshold for detecting the stop sign
        red_pixel_threshold = 500

        # Return True if red pixels exceed the threshold
        return red_pixel_count > red_pixel_threshold

    def stop_wheels(self):
        """Stops the wheels by publishing a stop command."""
        stop_message = WheelsCmdStamped(vel_left=0.0, vel_right=0.0)
        self._publisher.publish(stop_message)
        self._is_stopped = True
        rospy.loginfo("Stop sign detected! Sending stop command to wheels.")

    def resume_wheels(self):
        """Resumes the car movement by publishing default velocity commands."""
        move_message = WheelsCmdStamped(vel_left=self._default_vel_left, vel_right=self._default_vel_right)
        self._publisher.publish(move_message)
        self._is_stopped = False
        rospy.loginfo("No stop sign detected. Resuming movement.")


if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()
