# DetAny3D Setup Guide

This guide provides step-by-step instructions for setting up DetAny3D.

## Installation Steps

### 1. Create and Activate Conda Environment

```bash
conda create -n detany3d python=3.8
conda activate detany3d
```

### 2. Install Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 3. Clone and Install GroundingDINO

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..
```

### 4. Install Core Dependencies

```bash
pip install termcolor
pip install einops
pip install shapely
pip install open3d
```

### 5. Install MMCV

```bash
pip install -U openmim  # pip usually does not work to install mmcv
mim install mmcv
```

### 6. Install Additional Packages

```bash
pip install xformers
pip install python-box
pip install gradio
pip install gradio_image_prompter
```

## Environment Summary

- **Python Version:** 3.8
- **Conda Environment:** detany3d
- **Key Dependencies:** 
  - Segment Anything (SAM)
  - GroundingDINO
  - Open3D
  - MMCV
  - Gradio
  - XFormers

## Notes

- Ensure you have conda installed before starting
- OpenMIM is required for proper MMCV installation
- GPU with CUDA support is recommended for optimal performance
