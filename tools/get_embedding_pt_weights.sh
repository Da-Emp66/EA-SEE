#!/bin/bash

# Retrieve environment variables
source .env

# Retrieve the files
wget http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz

# Unzip the files
tar -xzvf vgg_face_torch.tar.gz vgg_face_torch

# Convert weights from `.t7` format to `.pt` format
python3 tools/weight_conversion.py -t7 vgg_face_torch/VGG_FACE.t7 -pt ${WEIGHTS_FILE}

# Clean up
rm -rf vgg_face_torch
rm vgg_face_torch.tar.gz
