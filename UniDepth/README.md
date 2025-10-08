
# UniDepth Setup Guide

This guide provides step-by-step instructions for setting up UniDepth.

## Source

**Repository:** [https://github.com/lpiccinelli-eth/UniDepth](https://github.com/lpiccinelli-eth/UniDepth)

## Installation Steps

### 0. Clone the Repository and copy or replace the current files in the folder.

```bash
git clone https://github.com/lpiccinelli-eth/UniDepth.git
```

### 1. Create and Activate Conda Environment

```bash
conda create --name py11-unidepth python=3.11
conda activate py11-unidepth
```

### 2. Install UniDepth and Dependencies

Install UniDepth with CUDA 11.8 support:

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Additional Python Packages

```bash
pip install transformers
pip install qwen_vl_utils
```

## Environment Summary

- **Python Version:** 3.11
- **Conda Environment:** py11-unidepth
- **CUDA Version:** 11.8 or higher
- **Key Dependencies:** transformers, qwen_vl_utils

## Notes

- Ensure you have conda installed before starting
- CUDA 11.8 or higher compatible GPU drivers are required
- The installation uses PyTorch's CUDA 11.8 wheel repository
