#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def publisher():
    rospy.init_node('video_publisher_node', anonymous=True)
    pub = rospy.Publisher('normal_stream', Image, queue_size=10)

    
    rate = rospy.Rate(30)

    video_path = '../data/Vid1.mp4'  
    cap = cv2.VideoCapture(video_path)

    bridge = CvBridge()

    loop_video = True

    while not rospy.is_shutdown() and loop_video:

        rospy.logwarn('Video being published.')
        ret, frame = cap.read()

        if ret:
            msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")

            pub.publish(msg)

        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass