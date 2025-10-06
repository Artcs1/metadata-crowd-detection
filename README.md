# metadata-crowd-detection Setup Guide


## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git](https://github.com/Artcs1/metadata-crowd-detection.git
```
### 2. EXPECTED STRUCTURE

### 3. CREATE OUR ENVIORENMENT FOR STEP 1 and 5

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git](https://github.com/Artcs1/metadata-crowd-detection.git
```

### 4. PROCESS EACH STEP

Some of the step need you to install an indepent enviorenment due to potential conflicts between the tools. There is detailed information inside in directory file, note that some of them require NVCC (load it with "module load <package>"). 

In this folder with conda activate py10-video
```bash
python3 1_convert_videos_frames.py
```

Inside Grounded-SAM-2 and conda activate py10-gsam2
```bash
python3 2_detect_persons.py
```

Inside UniDepth and conda activate py11-unidepth
```bash
python3 3_process_depth.py
```

Inside DetAny3D and conda activate py08-detany
```bash
python3 4_predict_detany.py
```

In this folder with conda activate py10-video
```bash
python3 5_detect_metadata.py
```
