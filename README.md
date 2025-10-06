# metadata-crowd-detection Setup Guide


## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git](https://github.com/Artcs1/metadata-crowd-detection.git
```
### 2. CREATE OUR ENVIORENMENT FOR STEP 1 and 5

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git](https://github.com/Artcs1/metadata-crowd-detection.git
```

### 3. PROCESS EACH STEP

Some of the step need you to install an indepent enviorenment due to potential conflicts between the tools. There is detailed information inside in directory file, note that some of them require NVCC (load it with "module load <package>"). 

```bash
python3 1_convert_videos_frames.py
```

```bash
python3 2_detect_persons.py
```

```bash
python3 3_process_depth.py
```

```bash
python3 4_predict_detany.py
```

```bash
python3 5_detect_metadata.py
```
