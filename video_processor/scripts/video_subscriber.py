#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def callback(msg):
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


    cv2.imshow("Video Subscriber", frame)
    cv2.waitKey(1) 

def subscriber():

    rospy.init_node('video_subscriber_node', anonymous=True)


    rospy.Subscriber('normal_stream', Image, callback)
    rospy.logerr('Hasitha')

    rospy.spin()

if __name__ == '__main__':
    subscriber()
