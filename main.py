import pandas as pd
import os
import re
import shutil

import cv2
from imutils.video import VideoStream

import time

from utils.preprocess import *

labels_path = "/mnt/d/University Stuff/OneDrive - SSN Trust/7th Semester/Capstone Project/Data/AI City Full Data/Labels/A1"
videos_path = "/mnt/d/Capstone/A1"
view_filter = "dashboard"
saved_output = "/mnt/d/Capstone/train_trimmed/"

missing_videos = []
frame_rate = 30


def main():
    user_labels = os.listdir(labels_path)

    replace_directory(saved_output)

    user_labels = os.listdir(labels_path)
    # user_labels_length = len(user_labels)

    for user in user_labels:
        # print(user)
        user_id = user[:-4]
        user_data_path = labels_path + "/" + user
        user_video_path = videos_path + "/" + user[:-4]
        user_data = pd.read_csv(user_data_path)

        print(user_video_path)
        if(os.path.isdir(user_video_path)):
            user_video_files = os.listdir(user_video_path)
            for user_video in user_video_files:
                if(re.search(view_filter, user_video, re.IGNORECASE) == None):
                    continue

                #String Preprocessing
                user_video_process = user_video[:-4]
                user_video_process = re.sub("_NoAudio_", "_", user_video_process)

                user_video_path_full = user_video_path + "/"+ user_video
                user_data_filtered = user_data[user_data['Filename'] == user_video_process]
                print(user_video_path_full)

                print()
                print(user_id)
                print(user_data_path)    
                image_processor(user_video_path_full, user_data_filtered, user_id, saved_output, frame_rate)
                print()


        else:
            missing_videos.append(user)

    with open("./missing_videos.txt", 'w') as file_object:
        file_object.writelines([name + '\n' for name in missing_videos])



if __name__ == "__main__":
    main()