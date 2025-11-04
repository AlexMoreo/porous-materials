#!/bin/bash

# Exit immediately on error
set -e

conda create -n porous python=3.12
conda activate porous
pip install numpy pandas scikit-learn matplotlib

#cpu-only installation, only for inference
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
