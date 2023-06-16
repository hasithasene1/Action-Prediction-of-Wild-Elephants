#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriberNode:
    def __init__(self):
        rospy.init_node('image_subscriber_node')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('image_topic', Image, self.image_callback)

    def image_callback(self, ros_image):
        try:
            # Convert the ROS image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")

            # Display the image
            cv2.imshow("IP Camera Feed", cv_image)
            cv2.waitKey(1)  # Refresh display

        except Exception as e:
            rospy.logerr("Error processing image: %s", str(e))

if __name__ == '__main__':
    try:
        node = ImageSubscriberNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

# Clean up OpenCV windows
cv2.destroyAllWindows()
