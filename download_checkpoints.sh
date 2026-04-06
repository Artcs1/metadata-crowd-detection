#!/bin/bash

# Create the directory structure
mkdir -p checkpoints/{sam_ckpts,unidepth_ckpts,dino_ckpts,detany3d_ckpts}
mkdir -p GroundingDINO/weights


# Download the SAM checkpoint
echo "⬇️ Downloading SAM checkpoint..."
wget -O checkpoints/sam_ckpts/sam_vit_h_4b8939.pth \
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

echo "⬇️ Downloading UniDepth checkpoint..."
gdown --id 1VF6e4fgv5J7ULmCeZqWG-dqe2m6URa36 -O checkpoints/unidepth_ckpts/model.pth

echo "⬇️ Downloading DINO checkpoint..."
gdown --id 15llnMaUtj68cZgNnhHUb0O_GcqSpRXEj -O checkpoints/dino_ckpts/dinov2_vitl14_pretrain.pth

echo "⬇️ Downloading DetAny3D checkpoint..."
gdown --id 1t4Ps0JhLbYTdeeC_1mh2JG5q24MRPYtT -O checkpoints/detany3d_ckpts/other_exp_ckpt.pth

echo "⬇️ Downloading GroundingDINO checkpoint..."
wget -O GroundingDINO/weights/groundingdino_swinb_cogcoor.pth \
https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

echo "✅ All checkpoints downloaded and directory structure created successfully!"
