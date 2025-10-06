import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
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
import argparse


logging.getLogger("ultralytics").setLevel(logging.ERROR)

from transformers import pipeline
from unidepth.models import UniDepthV1, UniDepthV2
from PIL import Image
from our_utils import *
from tqdm import tqdm
from pathlib import Path


def get_3d_with_fov(depth_array, fov):
            
    height, width = depth_array.shape
    camera_intrinsics=None
    if camera_intrinsics is None:
        camera_intrinsics = get_intrinsics(width, height, fov)
    
    depth_image = np.maximum(depth_array, 1e-5)
    depth_image = 100.0 / depth_image
    X, Y, Z = pixel_to_point(depth_image, True, camera_intrinsics)
    return (X,Y,Z, camera_intrinsics)


def main():

    parser = argparse.ArgumentParser(description="Simple argparse example")

    parser.add_argument("--dataset_path",  type=str, default='VBIG_dataset', help="Dataset Path")
    parser.add_argument("--device", type=int, default=0, help="Cuda Device")
    parser.add_argument("--start", type=int, default=0, help="Start of the worker")
    parser.add_argument("--end", type=int, help="Start of the worker")


    args = parser.parse_args()


    device = str(args.device)
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device="cuda:"+str(device))

    cmap = plt.get_cmap("tab20")
    colormap = plt.cm.get_cmap("tab20", 20)
    
    N = 20
    colors = [cmap(i) for i in range(N)]
    palette = (np.array(colors)[:, :3] * 255).astype(np.uint8)
    
    dirs = [d for d in glob.glob(args.dataset_path+"/jsons_step2/*") if os.path.isfile(d)]
    name = 'unidepth-v2-vitl14'
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")
    model.to('cuda:'+str(device))

    start = int(args.start)
    if args.end is None:
        end = len(dirs)
    else:
        end  = int(args.end)


    
    for json_path2 in tqdm(dirs[start:end]):

        try:

            p = Path(json_path2)
            current_dir = Path(str(p).replace("jsons_step2", "videos_frames"))
            current_dir = str(current_dir.with_suffix(""))
    
            with open(json_path2, "r") as f:
                json_data_s2 = json.load(f)
    
            json_data_s3 = copy.deepcopy(json_data_s2)
    
            #print(json_data_s3)
            for ind, frame in enumerate(json_data_s2['frames']):
    
                image_path = f'{current_dir}/{str(ind+1).zfill(5)}.jpeg'
    
                image = cv2.imread(image_path)
                
                #coord3 = np.zeros((1000,image.shape[1],3),dtype=np.uint8)+255
                #coord2 = np.zeros((1000,1000,3),dtype=np.uint8)+255
                #coord1 = np.zeros((1000,image.shape[1],3),dtype=np.uint8)+255
                
                rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) 
                
                predictions = model.infer(rgb)
                
                depth = predictions["depth"]
                xyz = predictions["points"]
                intrinsics = predictions["intrinsics"]
    
    
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
                result = pipe(opencv_to_pil(frame_rgb))
                depth_pil = result["depth"]
                depth_array = np.array(depth_pil)
    
                X_160, Y_160, Z_160, intrinsics_160 = get_3d_with_fov(depth_array, 160)
                X_110, Y_110, Z_110, intrinsics_110 = get_3d_with_fov(depth_array, 110)
                X_60, Y_60, Z_60, intrinsics_60 = get_3d_with_fov(depth_array, 60)
    
                intrinsics = intrinsics[0,:].detach().cpu().numpy()
                json_data_s3['frames'][ind]['unidepth_intrinsics'] = intrinsics.tolist()
                json_data_s3['frames'][ind]['naive_3D_intrinsics_FOV60'] = intrinsics_60.tolist()
                json_data_s3['frames'][ind]['naive_3D_intrinsics_FOV110'] = intrinsics_110.tolist()
                json_data_s3['frames'][ind]['naive_3D_intrinsics_FOV160'] = intrinsics_160.tolist()
    
                # print(intrinsics)
                # print(intrinsics_60)
                # print(intrinsics_110)
                # print(intrinsics_160)
                
                for det_id, detection in enumerate(frame['detections']):
    
    
                    track_id = int(detection['track_id'])
                    o1_x1, o1_y1, o1_x2, o1_y2 = map(int, detection['bbox'])
                    o1_mid = ((o1_x1+o1_x2)//2, (o1_y1+o1_y2)//2)
                
                    opencv_frame = image[int(o1_y1):int(o1_y2),int(o1_x1):int(o1_x2)]
                    if opencv_frame.size == 0:
                        continue
                    pil_frame = opencv_to_pil(opencv_frame)            
                    label     = 'person'
                    #rgb_tuple = (255, 0, 0)
                    
                    rgb_tuple = (int(palette[track_id%20][0]),int(palette[track_id%20][1]),int(palette[track_id%20][2]))
                    draw_tracking(frame=image, bbox = detection['bbox'], label=label, color=rgb_tuple)
    
                    d = depth[0,0, o1_mid[1],o1_mid[0]].detach().cpu().item()
                    x = xyz[0,0,o1_mid[1],o1_mid[0]].detach().cpu().item()
                    y = xyz[0,1,o1_mid[1],o1_mid[0]].detach().cpu().item()
                    z = xyz[0,2,o1_mid[1],o1_mid[0]].detach().cpu().item()
                    
                    json_data_s3['frames'][ind]['detections'][det_id]['naive_depth']     = int(depth_array[o1_mid[1],o1_mid[0]])
                    json_data_s3['frames'][ind]['detections'][det_id]['unidepth_depth']  = d
                    json_data_s3['frames'][ind]['detections'][det_id]['unidepth_3D']     = [x,y,z]
                    json_data_s3['frames'][ind]['detections'][det_id]['naive_3D_160FOV'] = [float(X_160[o1_mid[1],o1_mid[0]]),float(Y_160[o1_mid[1],o1_mid[0]]),float(Z_160[o1_mid[1],o1_mid[0]])] 
                    json_data_s3['frames'][ind]['detections'][det_id]['naive_3D_110FOV'] = [float(X_110[o1_mid[1],o1_mid[0]]),float(Y_110[o1_mid[1],o1_mid[0]]),float(Z_110[o1_mid[1],o1_mid[0]])] 
                    json_data_s3['frames'][ind]['detections'][det_id]['naive_3D_60FOV']  = [float(X_60[o1_mid[1],o1_mid[0]]),float(Y_60[o1_mid[1],o1_mid[0]]),float(Z_60[o1_mid[1],o1_mid[0]])] 
                    
    
                    #direction = 'unknow' #det['direction']
                    #offset = 25
                    #center1 = (int((x+5)*200),1000-int(d*10))
                    #print(center1)
                    
                    #plot_coord(coord1, center1, direction, offset, rgb_tuple)
                    #plot_coord(coord2, center2, direction, offset, rgb_tuple)
                    #plot_coord(coord3, center3, direction, offset, rgb_tuple)
    
                
    
                
                # plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                # plt.show()
    
                # plt.imshow(cv2.cvtColor(coord1,cv2.COLOR_BGR2RGB))
                # plt.show()
    
    
    
            #print(json_data_s3)
            p2 = Path(json_path2)         
            new_path2 = Path(str(p2).replace("jsons_step2", "jsons_step3"))
        
            p3 = Path(new_path2)
            json3_path = p3.parent
    
            os.makedirs(json3_path, exist_ok=True)
            with open(f"{new_path2}", "w") as fp: 
                json.dump(json_data_s3, fp, indent=4)

        except Exception as e:
            pass

        
            
    
if __name__ == '__main__':
    main()

