import os
import cv2
import torch
import numpy as np
import supervision as sv
import json
import glob
import copy
import argparse

from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Simple argparse example")

parser.add_argument("--dataset_path",  type=str, default='VBIG_dataset', help="Dataset Path")
parser.add_argument("--device", type=int, default=0, help="Cuda Device")
parser.add_argument("--start", type=int, default=0, help="Start of the worker")
parser.add_argument("--end", type=int, help="Start of the worker")


args = parser.parse_args()


# This demo shows the continuous object tracking plus reverse tracking with Grounding DINO and SAM 2
"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"


print("device", device)

#dirs = [d for d in glob.glob("/gpfs/scratch/jmurrugarral/VBIG_dataset/videos_frames/*") if os.path.isdir(d)]
dirs = [d for d in glob.glob(args.dataset_path+"/jsons_step1/*") if os.path.isfile(d)]


start = int(args.start)
if args.end is None:
    end = len(dirs)
else:
    end  = int(args.end)
print(len(dirs))

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)

# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    


for c_ind, new_path in enumerate(dirs[start:end]):

    print(len(dirs[start:end]))

    print(f'{start+c_ind} / {len(dirs)}')

    if True:
        p = Path(new_path)
        current_dir = Path(str(p).replace("jsons_step1", "videos_frames"))
        current_dir = str(current_dir.with_suffix(""))

        print(current_dir)
   
        
        # setup the input image and text prompt for SAM 2 and Grounding DINO
        # VERY important: text queries need to be lowercased + end with a dot
        text = "person."
        
        # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
        video_dir = current_dir
        # 'output_dir' is the directory to save the annotated frames
        output_dir = "outputs_"+str(args.device)
        # 'output_video_path' is the path to save the final video
        output_video_path = "./outputs/output.mp4"
        # create the output directory
        mask_data_dir = os.path.join(output_dir, "mask_data")
        json_data_dir = os.path.join(output_dir, "json_data")
        result_dir = os.path.join(output_dir, "result")
        CommonUtils.creat_dirs(mask_data_dir)
        CommonUtils.creat_dirs(json_data_dir)
        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
        # init video predictor state
        inference_state = video_predictor.init_state(video_path=video_dir)
        step = 20 # the step to sample frames for Grounding DINO predictor
        
        sam2_masks = MaskDictionaryModel()
        PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
        objects_count = 0
        frame_object_count = {}
        """
        Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
        """
        #print("Total frames:", len(frame_names))
        for start_frame_idx in range(0, len(frame_names), step):
        # prompt grounding dino to get the box coordinates on specific frame
            #print("start_frame_idx", start_frame_idx)
            # continue
            img_path = os.path.join(video_dir, frame_names[start_frame_idx])
            image = Image.open(img_path).convert("RGB")
            image_base_name = frame_names[start_frame_idx].split(".")[0]
            mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
        
            # run Grounding DINO on the image
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)
        
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.25,
                target_sizes=[image.size[::-1]]
            )
        
            # prompt SAM image predictor to get the mask for the object
            image_predictor.set_image(np.array(image.convert("RGB")))
       
            scores_boxes = results[0]["scores"]
            # process the detection results
            input_boxes = results[0]["boxes"] # .cpu().numpy()
            # print("results[0]",results[0])
            OBJECTS = results[0]["labels"]
            if input_boxes.shape[0] != 0:
        
                # prompt SAM 2 image predictor to get the mask for the object
                masks, scores, logits = image_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )
                # convert the mask shape to (n, H, W)
                if masks.ndim == 2:
                    masks = masks[None]
                    scores = scores[None]
                    logits = logits[None]
                elif masks.ndim == 4:
                    masks = masks.squeeze(1)
                """
                Step 3: Register each object's positive points to video predictor
                """
        
                # If you are using point prompts, we uniformly sample positive points based on the mask
                if mask_dict.promote_type == "mask":
                    mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS, score_list=scores_boxes)
                else:
                    raise NotImplementedError("SAM 2 video predictor only support mask prompts")
            else:
                #print("No object detected in the frame, skip merge the frame merge {}".format(frame_names[start_frame_idx]))
                mask_dict = sam2_masks
        
            """
            Step 4: Propagate the video predictor to get the segmentation results for each frame
            """
            objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count)
            frame_object_count[start_frame_idx] = objects_count
            #print("objects_count", objects_count)
            
            if len(mask_dict.labels) == 0:
                mask_dict.save_empty_mask_and_json(mask_data_dir, json_data_dir, image_name_list = frame_names[start_frame_idx:start_frame_idx+step])
                #print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
                continue
            else:
                video_predictor.reset_state(inference_state)
        
                for object_id, object_info in mask_dict.labels.items():
                    frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                            inference_state,
                            start_frame_idx,
                            object_id,
                            object_info.mask,
                        )
                
                video_segments = {}  # output the following {step} frames tracking masks
                for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                    frame_masks = MaskDictionaryModel()
                    
                    for i, out_obj_id in enumerate(out_obj_ids):
                        out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                        object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id), logit=mask_dict.get_target_logit(out_obj_id), score=mask_dict.get_target_score(out_obj_id))
                        object_info.update_box()
                        #print(object_info)
                        frame_masks.labels[out_obj_id] = object_info
                        image_base_name = frame_names[out_frame_idx].split(".")[0]
                        frame_masks.mask_name = f"mask_{image_base_name}.npy"
                        frame_masks.mask_height = out_mask.shape[-2]
                        frame_masks.mask_width = out_mask.shape[-1]
        
                    video_segments[out_frame_idx] = frame_masks
                    sam2_masks = copy.deepcopy(frame_masks)
        
                #print("video_segments:", len(video_segments))
            """
            Step 5: save the tracking masks and json files
            """
            for frame_idx, frame_masks_info in video_segments.items():
                mask = frame_masks_info.labels
                mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
                for obj_id, obj_info in mask.items():
                    mask_img[obj_info.mask == True] = obj_id
        
                mask_img = mask_img.numpy().astype(np.uint16)
                np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)
        
                json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
                #print(frame_masks_info)
                frame_masks_info.to_json(json_data_path)
               
        
        #CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)
        
        #print("try reverse tracking")
        start_object_id = 0
        object_info_dict = {}
        for frame_idx, current_object_count in frame_object_count.items():
            #print("reverse tracking frame", frame_idx, frame_names[frame_idx])
            if frame_idx != 0:
                video_predictor.reset_state(inference_state)
                image_base_name = frame_names[frame_idx].split(".")[0]
                json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
                json_data = MaskDictionaryModel().from_json(json_data_path)
                mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
                mask_array = np.load(mask_data_path)
                for object_id in range(start_object_id+1, current_object_count+1):
                    #print("reverse tracking object", object_id)
                    object_info_dict[object_id] = json_data.labels[object_id]
                    video_predictor.add_new_mask(inference_state, frame_idx, object_id, mask_array == object_id)
        
        
            if start_object_id == current_object_count:
                continue
            start_object_id = current_object_count
                
            
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step*2,  start_frame_idx=frame_idx, reverse=True):
                #print(out_frame_idx)
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
                json_data = MaskDictionaryModel().from_json(json_data_path)
                mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
                mask_array = np.load(mask_data_path)
                # merge the reverse tracking masks with the original masks
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu()
                    if out_mask.sum() == 0:
                        #print("no mask for object", out_obj_id, "at frame", out_frame_idx)
                        continue
                    object_info = object_info_dict[out_obj_id]
                    object_info.mask = out_mask[0]
                    object_info.update_box()
                    json_data.labels[out_obj_id] = object_info
                    mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                    mask_array[object_info.mask] = out_obj_id
                
                np.save(mask_data_path, mask_array)
                json_data.to_json(json_data_path)
        
        """
        Step 6: Draw the results and save the video
        """
        #CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir+"_reverse")
        #create_video_from_images(result_dir, output_video_path, frame_rate=15)

        with open(new_path, "r") as f:
            json_data_s1 = json.load(f)
        
        json_data_s1['frames'] = []
        idx = 0 

        json_files = glob.glob(f"outputs_{str(args.device)}/json_data/*.json")
        json_files.sort()
        
        for file in json_files:
            with open(file,"r") as f2:
                json_temp = json.load(f2)
            current_frame_info = {}
            idx+=1
            current_frame_info['frame_id'] = idx 
            current_frame_info['detections'] = []
            for key in json_temp['labels']:
                detection={}
                flag = False
                if not np.array_equal(np.array([0,0,0,0]), np.array([json_temp['labels'][key]['x1'], json_temp['labels'][key]['y1'], json_temp['labels'][key]['x2'], json_temp['labels'][key]['y2']])):
                    detection['track_id'] = json_temp['labels'][key]['instance_id']
                    detection['score'] = json_temp['labels'][key]['score']
                    detection['bbox'] = [json_temp['labels'][key]['x1'], json_temp['labels'][key]['y1'], json_temp['labels'][key]['x2'], json_temp['labels'][key]['y2']]
                    flag = True
                if flag:
                    current_frame_info['detections'].append(detection) 
            json_data_s1['frames'].append(current_frame_info)

        p2 = Path(new_path)     
        new_path2 = Path(str(p2).replace("jsons_step1", "jsons_step2"))

        p3 = Path(new_path2)
        json2_path = p3.parent

        CommonUtils.creat_dirs(json2_path)
        print(json2_path)

        with open(f"{new_path2}", "w") as fp:
            json.dump(json_data_s1, fp, indent=4)

        #print('success')
        #dir_path = "/ruta/al/directorio"

        #try:
        #    os.rmdir(mask_data_dir)
        #    os.rmdir(json_data_dir)
        #    print(f"Directorio {dir_path} eliminado")
        #except FileNotFoundError:
        
        #    print("El directorio no existe")

        torch.cuda.empty_cache()
    #except Exception as e:
    #    pass
