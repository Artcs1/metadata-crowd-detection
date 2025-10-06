import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
#import supervision as sv
import math
import time
import requests
import torch
import os
import json
import base64
import io
import shutil
import logging
import glob
import copy


from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
from qwen_vl_utils import process_vision_info
from our_utils import *
from tqdm import tqdm
from pathlib import Path
#from unidepth.models import UniDepthV1, UniDepthV2



def main():
    
    #model_v1 = UniDepthV1.from_pretrained(f"lpiccinelli/{name}")
        
    #model_v2 = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")


    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",torch_dtype=torch.bfloat16,device_map="cuda:0")
    #qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",torch_dtype=torch.bfloat16)#,attn_implementation="flash_attention_2",device_map="cuda:2",)
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    qwen.eval()
    
    pose = YOLO("yolo11m-pose.pt", 0.15)
    pose.to('cuda:0')
    print(pose.device)
    
    dirs = [d for d in glob.glob("VBIG_dataset/jsons_step4/*") if os.path.isfile(d)]

    #print(dirs)
    parts  = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder","Left Elbow","Right Elbow","Left Wrist","Right Wrist","Left Hip","Right Hip","Left Knee","Right Knee","Left Ankle","Right Ankle"]
    
    for json_path4 in dirs:

        print(json_path4)

        try:

            p = Path(json_path4)
            current_dir = Path(str(p).replace("jsons_step4", "videos_frames"))
            current_dir = str(current_dir.with_suffix(""))
    
            with open(json_path4, "r") as f:
                json_data_s4 = json.load(f)
    
            json_data_s5 = copy.deepcopy(json_data_s4)
    
            #print(json_data_s3)
            json_data_s5['body_parts'] = parts
            for ind, frame in enumerate(json_data_s4['frames']):
    
                #print(f'{current_dir}/{str(ind+1).zfill(5)}.jpeg')
                image = cv2.imread(f'{current_dir}/{str(ind+1).zfill(5)}.jpeg')
                
                opencv_frames = []
                for det_id, detection in enumerate(frame['detections']):
                    #if detection == {}:
                    #    continue
    
                    track_id = int(detection['track_id'])
                    o1_x1, o1_y1, o1_x2, o1_y2 = map(int, detection['bbox'])
                    o1_mid = ((o1_x1+o1_x2)//2, (o1_y1+o1_y2)//2)
                
                    opencv_frame = image[int(o1_y1):int(o1_y2),int(o1_x1):int(o1_x2)]
                    opencv_frames.append(opencv_frame)
                    
                    pil_frame = opencv_to_pil(opencv_frame)            
                    label = 'person'
                    rgb_tuple = (255, 0, 0)
                    
                    prob_male, answer_male = vqa_yes_prob(qwen, processor, pil_frame, 'Is the person a male?')
                    prob_female, answer_female = vqa_yes_prob(qwen, processor, pil_frame, 'Is the person a female?')
                    prob_child, answer_child = vqa_yes_prob(qwen, processor, pil_frame, 'Is the person a child?')
                    prob_nbin, answer_nbin = vqa_yes_prob(qwen, processor, pil_frame, 'Is the person non-binary?')
            
                    person = np.array([prob_male, prob_female, prob_child, prob_nbin])
                    sex = 'unknown'
                    if np.argmax(person)   == 0:
                        sex='male'
                    elif np.argmax(person) == 1:
                        sex='female'
                    elif np.argmax(person) == 2:
                        sex='child'
                    elif np.argmax(person) == 3:
                        sex='non binary'
    
                    json_data_s5['frames'][ind]['detections'][det_id]['sex'] = sex
                    
                    #draw_tracking(frame=image, bbox = detection['bbox'], label=label, color=rgb_tuple)
    
                #print(len(opencv_frames))
                results = pose(opencv_frames)
    
                for det_id, pose_result in enumerate(results):
    
                    
                    #print(len(pose_result))
                    #print(pose_result)
                    #print('_________________________________')
                    if len(pose_result) == 0:
                        json_data_s5['frames'][ind]['detections'][det_id]['direction'] = direction
                        json_data_s5['frames'][ind]['detections'][det_id]['visible'] = visible
                        continue
                    
                    direction, visible   = 'unknown', 'unknown'
                
                    if len(pose_result[0].boxes) > 0:
            
                        confs = pose_result[0].boxes.conf.cpu().numpy()
                        best_idx = confs.argmax()
                        
                        highest_conf_result = pose_result[0][best_idx:best_idx+1]            
                        confidence = highest_conf_result.keypoints.conf
                        #print(confidence)
                        
                        values = highest_conf_result.keypoints.conf>0.3
                        parts  = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder","Left Elbow","Right Elbow","Left Wrist","Right Wrist","Left Hip","Right Hip","Left Knee","Right Knee","Left Ankle","Right Ankle"]
                        n_val = values.cpu().detach().numpy()[0]
                        #print(highest_conf_result)
                        key_values = highest_conf_result.keypoints.conf.cpu().detach().numpy()[0]
                        #print(highest_conf_result.keypoints.xy)
                        pose_position = highest_conf_result.keypoints.xy.cpu().detach().numpy()[0]
                        #print(pose_position)
                        counts_points_body = np.sum(np.array(n_val[5:]))
            
                        if counts_points_body:
                            visible = 'not occluded'
                        else:
                            visible = 'occluded'
            
                        if n_val[0] == True and n_val[1] == True and n_val[2] == True and n_val[3] == True and n_val[4] == True:
                            direction = 'front'
                        elif n_val[0] == True and n_val[1] == True and n_val[2] == True and n_val[3] == True and n_val[4] == False:
                            direction = 'front right'
                        elif n_val[0] == True and n_val[1] == True and n_val[2] == False and n_val[3] == True and n_val[4] == False:
                            direction = 'front rright'
                        elif n_val[0] == True and n_val[1] == True and n_val[2] == True and n_val[3] == False and n_val[4] == True:
                            direction = 'front left'
                        elif n_val[0] == True and n_val[1] == False and n_val[2] == True and n_val[3] == False and n_val[4] == True:
                            direction = 'front lleft'
                        elif n_val[0] == False and n_val[1] == False and n_val[2] == False and n_val[3] == True and n_val[4] == True:
                            direction = 'back'
                        elif n_val[0] == False and n_val[1] == False and n_val[2] == True and n_val[3] == True and n_val[4] == True:
                            direction = 'back right'
                        elif n_val[0] == False and n_val[1] == True and n_val[2] == False and n_val[3] == True and n_val[4] == True:
                            direction = 'back left'
        
                        json_data_s5['frames'][ind]['detections'][det_id]['xy_body_parts'] = pose_position.tolist()
                        json_data_s5['frames'][ind]['detections'][det_id]['conf_body_parts'] = key_values.tolist()
                    #print(json_data_s5['frames'][ind]['detections'][det_id])
                    
                    json_data_s5['frames'][ind]['detections'][det_id]['direction'] = direction
                    json_data_s5['frames'][ind]['detections'][det_id]['visible'] = visible
    
                #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                #plt.show()
                
                    #print(direction)
    
            p2 = Path(json_path4)         
            new_path2 = Path(str(p2).replace("jsons_step4", "jsons_step5"))
        
            p3 = Path(new_path2)
            json5_path = p3.parent
    
            os.makedirs(json5_path, exist_ok=True)
            with open(f"{new_path2}", "w") as fp: 
                json.dump(json_data_s5, fp, indent=4)
    
            #print(json_data_s5)
    
   
        except Exception as e:
            pass 
if __name__ == '__main__':
    main()

