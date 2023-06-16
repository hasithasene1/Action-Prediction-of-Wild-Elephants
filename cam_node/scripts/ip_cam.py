#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class IPCamNode:
    def __init__(self):
        rospy.init_node('ip_cam_node')
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('image_topic', Image, queue_size=10)

    def start(self):
        cam_url = "http://192.168.8.105:4747/video"  # Replace with the URL of your IP camera stream
        cap = cv2.VideoCapture(cam_url)

        while not rospy.is_shutdown():
            ret, frame = cap.read()

            if ret:
                # Convert the frame to a ROS message
                ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

                # Publish the ROS image message
                self.image_pub.publish(ros_image)

if __name__ == '__main__':
    try:
        node = IPCamNode()
        node.start()
    except rospy.ROSInterruptException:
        pass
