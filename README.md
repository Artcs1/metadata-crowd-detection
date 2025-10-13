# metadata-crowd-detection Setup Guide

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Artcs1/metadata-crowd-detection.git
cd metadata-crowd-detection
```

### 2. Directory Structure
Your dataset folder should follow this structure:

```
VBIG_dataset/
├── videos_frames/
│   ├── video1/
│   │   ├── frame_00001.jpeg
│   │   ├── frame_00002.jpeg
│   │   └── ...
│   ├── video2/
│   │   ├── frame_00001.jpeg
│   │   └── ...
│   └── ...
└── jsons_step1/
    ├── video1.json
    ├── video2.json
    └── ...
```

**Important:** Each video folder under `videos_frames/` must have a corresponding folder with the same name under `jsons_step1/`.

### 3. Environment Setup
This project requires multiple conda environments to avoid dependency conflicts. Create the necessary environments:

```bash
# Base environment for steps 1 and 5
conda create -n py10-video python=3.10
conda activate py10-video
# Install requirements from requirements.txt

# Environment for Grounded-SAM-2
conda create -n py10-gsam2 python=3.10
# Install Grounded-SAM-2 requirements

# Environment for UniDepth
conda create -n py11-unidepth python=3.11
# Install UniDepth requirements

# Environment for DetAny3D
conda create -n py08-detany python=3.8
# Install DetAny3D requirements
```

**HPC Users:** Some steps require NVCC (NVIDIA CUDA Compiler). Load it with:
```bash
module load cuda  # Adjust module name for your cluster
```

## Processing Pipeline

Each step uses a dedicated conda environment and accepts the following arguments:
```bash
python3 <script_name> --dataset_path <path> --start <index> --end <index>
```

### Step 1: Convert Videos to Frames
**Environment:** `py10-video` | **Location:** Root directory

```bash
conda activate py10-video
python3 1_convert_videos_frames.py --dataset_path ./VBIG_dataset/
```

### Step 2: Detect Persons
**Environment:** `py10-gsam2` | **Location:** `Grounded-SAM-2/`

```bash
cd Grounded-SAM-2
conda activate py10-gsam2
python3 2_detect_persons.py --dataset_path ../../../VBIG_dataset/
cd ..
```

### Step 3: Process Depth Information
**Environment:** `py11-unidepth` | **Location:** `UniDepth/`

```bash
cd UniDepth
conda activate py11-unidepth
python3 3_process_depth.py --dataset_path ../../../VBIG_dataset/
cd ..
```

### Step 4: 3D Detection
**Environment:** `py08-detany` | **Location:** `DetAny3D/`

```bash
cd DetAny3D
conda activate py08-detany
python3 4_predict_detany.py --dataset_path ../../../VBIG_dataset/
cd ..
```

### Step 5: Extract Metadata
**Environment:** `py10-video` | **Location:** Root directory

```bash
conda activate py10-video
python3 5_detect_metadata.py --dataset_path ./VBIG_dataset/
```

## Notes

- Each tool directory contains detailed setup instructions and requirements files
- Adjust `--dataset_path` to match your dataset location
- Use `--start` and `--end` parameters to process specific ranges
- Ensure all environments are created before running the pipeline

## Troubleshooting

- Verify you're using the correct conda environment for each step
- Check individual directory READMEs for tool-specific issues
- Confirm NVCC is loaded if CUDA-related errors occur
