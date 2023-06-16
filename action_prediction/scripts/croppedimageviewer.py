#!/usr/bin/env python

import os
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CroppedImageViewer:

    def __init__(self):
        rospy.init_node('cropped_image_viewer')
        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber('predicted_actions', Image, self.image_callback)

    def image_callback(self, image):
        frame = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

        # Display the received image


        cv2.imshow('Cropped Image Viewer', frame)
        rospy.logwarn("Viewed")
        cv2.waitKey(1)


if __name__ == '__main__':
    viewer = CroppedImageViewer()
    rospy.spin()