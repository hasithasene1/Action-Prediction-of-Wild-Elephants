import cv2
import os
from datetime import datetime


class VideoToFrames:
    def __init__(self,video_name) -> None:
        self.video_name = video_name
        

    def to_frames(self,length_min,length_sec):

        now = datetime.now()

        current_time = now.strftime("%H%M%S")
        self.crop_dir = current_time
        if not os.path.exists(self.crop_dir):
            os.makedirs(self.crop_dir)

        video = cv2.VideoCapture(self.video_name)
        FPS = video.get(cv2.CAP_PROP_FPS)
        print(f"FPS :{FPS}")

        for i in range(int((length_min*60+length_sec)*FPS)):
            video = cv2.VideoCapture(self.video_name)
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            done = cv2.imwrite(f'{self.crop_dir}/frame_{i}.png', frame)
            print(f'{self.crop_dir}/frame_{i}.png :{done}')


Vid = VideoToFrames('vid.mp4') #Video Name
Vid.to_frames(length_min=0,length_sec=2) #Length of the video Minitues and Seconds 