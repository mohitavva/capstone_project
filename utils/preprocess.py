import pandas as pd
import os
import re
import shutil
import glob

import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time


def replace_directory(saved_output):
    if(os.path.exists(saved_output)):
        # replace = (input("Folder Exists. Replace? Y/n: ") or "Y")
        replace = "Y"
        if(replace == "Y"):
            shutil.rmtree(saved_output)
            os.mkdir(saved_output)
        elif(replace == "N"):
            print("Folder not Overwritten")
        else:
            print("Invalid Option, Folder not Overwritten")
            replace = "N"
    else:
        os.mkdir(saved_output)
    return 


def crop_frame(image, left_fraction=0.3, output_size=(224, 224)):

    if image is None:
        print("Error: Image not loaded.")
        return None
    h, w, _ = image.shape

    x_end = int(w * left_fraction)  
    cropped_image = image[:, x_end:] 

    resized_cropped_image = cv2.resize(cropped_image, output_size)

    return resized_cropped_image

def play_video_frame_by_frame(video_path, start_time_seconds, end_time_seconds, file_save_name, activity_label, saved_output, user_id, frame_rate):
    cap = cv2.VideoCapture(video_path)
    fps = frame_rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    
    start_frame = int(start_time_seconds * fps)
    end_frame = int(end_time_seconds * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #Frame delay
    delay = 1 / frame_rate
    
    current_frame = start_frame
    chdir_flag = False

    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
    
        save_name = file_save_name + "_" + str(current_frame) +".jpg"

        if(os.path.exists(saved_output+"/"+activity_label) == False):
            os.mkdir(saved_output+activity_label)

        if(os.path.exists(saved_output+activity_label+"/"+user_id) == False):
            os.mkdir(saved_output+activity_label+"/"+user_id)

        
        if(os.path.exists(saved_output+"/"+activity_label+"/"+user_id) and chdir_flag==False):
            os.chdir(saved_output+"/"+activity_label+"/"+user_id)
            os.system("pwd")
            chdir_flag= True

        elif(chdir_flag==False):
            os.system("pwd")
            os.chdir(saved_output+"/"+activity_label+"/"+user_id)
            os.system("pwd")
            chdir_flag=True

        cropped_frame = crop_frame(frame)
        if(cropped_frame is not None):
            cv2.imwrite(save_name, cropped_frame)
        else:
            cv2.imwrite(save_name, frame)

        start_time_seconds = time.time()
        
        current_frame += 5

    os.chdir("../../")
    cap.release()
    cv2.destroyAllWindows()
    return

def image_processor(user_video_path_full, user_data_filtered, user_id, saved_output, frame_rate):
    for i in range(len(user_data_filtered)):
        
        index_value = user_data_filtered.index[i]
        time_str = user_data_filtered.loc[index_value, 'Start Time']
        hours, minutes, seconds = map(int, time_str.split(":"))

        print("Start Time:", time_str)
        
        start_time_seconds = seconds + minutes*60 + hours*60*60

        time_str = user_data_filtered.loc[index_value, 'End Time']

        print("End Time:", time_str)

        hours, minutes, seconds = map(int, time_str.split(":"))
        end_time_seconds = seconds + minutes*60 + hours*60*60

        activity_label = user_data_filtered.loc[index_value, "Label (Primary)"]
        # print("------")
        activity_label = "class_" + activity_label.split(" ")[-1]
        print(activity_label)
        print(user_id)

        file_save_name = user_data_filtered.loc[index_value, 'Filename']

        play_video_frame_by_frame(user_video_path_full, start_time_seconds, end_time_seconds, file_save_name, activity_label, saved_output, user_id, frame_rate)
        os.chdir("../../")
     
    return
