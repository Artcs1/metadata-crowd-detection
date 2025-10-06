# Grounded-SAM-2 Setup Guide

**Hint:** Each step has its on enviorenment to avoid conflicts. You should gitclone the GSAM2 project and then copypaste the current code to this folder.

This guide provides step-by-step instructions for setting up the Grounded-SAM-2 project.

## Source

**Repository:** [https://github.com/IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```

### 2. Create and Activate Conda Environment

```bash
conda create --name py10-gsam2 python=3.10
conda activate py10-gsam2
```

### 3. Navigate to Project Directory

```bash
cd Grounded-SAM-2/
```

### 4. Download SAM-2 Checkpoints

```bash
cd checkpoints/
bash download_ckpts.sh
cd ..
```

### 5. Download Grounding DINO Checkpoints

```bash
cd gdino_checkpoints/
bash download_ckpts.sh
cd ..
```

### 6. Install PyTorch and Dependencies

Install PyTorch with CUDA 12.1 support:

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### 7. Install Additional Python Packages

```bash
pip install opencv-python
pip install supervision
pip install transformers==4.53.0
```

## Environment Summary

- **Python Version:** 3.10
- **Conda Environment:** py10-gsam2
- **PyTorch Version:** 2.3.1 (CUDA 12.1)
- **Key Dependencies:** opencv-python, supervision, transformers==4.53.0

## Notes

- Ensure you have conda installed before starting
- CUDA 12.1 compatible GPU drivers are required for GPU acceleration
- The checkpoint downloads may take some time depending on your internet connection
