#!/usr/bin/env python

import os
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from time import time
import numpy as np
import torch

class CroppedImageViewer:

    def __init__(self):
        rospy.init_node('cropped_image_viewer')
        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber('elephant_actions', Image, self.image_callback)

    def image_callback(self, image):
        frame = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')

        # Display the received image
        cv2.imshow('Cropped Image Viewer', frame)
        rospy.logwarn("Viewed")
        cv2.waitKey(1)
class ActionPrediction:

    def __init__(self):
        rospy.init_node('Action_Prediction_node')


        self.model = self.load_model('weights/best_Jun20.pt')
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)
        self.cv_bridge = CvBridge()


        self.crop_dir = 'cropped'
        if not os.path.exists(self.crop_dir):
            os.makedirs(self.crop_dir)


        self.image_sub = rospy.Subscriber('elephant_actions',Image,self.image_callback)
        self.image_pub = rospy.Publisher('predicted_actions', Image, queue_size=1)    

    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)

        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame):
        
        labels, cord = results
        n = len(labels)

        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            rospy.logwarn(f"{row}")
            file_path = "output.txt"  
            file = open(file_path, "a")
            file.write(f"{row}\n")
            file.close()
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, f"{torch.round(row[4]*100)} {self.class_to_label(labels[i])}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr)

                # Crop the rectangle
                cropped_frame = frame[y1:y2, x1:x2]

                # Save the cropped frame
                file_name = f"croped/Action{int(rospy.Time.now().to_sec())} .jpg"            
                result_image = cv2.imwrite(file_name, cropped_frame)
                rospy.logwarn(f"ImageAction{file_name} :{result_image}" )

                # Publish the cropped frame
                cropped_image_msg = self.cv_bridge.cv2_to_imgmsg(cropped_frame)
                self.image_pub.publish(cropped_image_msg)

        return frame
    
    def image_callback(self, image):
        frame = self.cv_bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
        #frame = cv2.imread('test_image2.jpg',cv2.IMREAD_COLOR) 
        frame = cv2.resize(frame, (416, 416))
        
        start_time = time()
        results = self.score_frame(frame)
        labels, cordinates = results
        rospy.logwarn(f"{labels}")
        frame = self.plot_boxes(results=results, frame=frame)
        end_time = time()

        fps = 1 / np.round(end_time - start_time, 2)
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        cv2.imshow('Yolov5 Detection', frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    viewer = ActionPrediction()
    rospy.spin()
