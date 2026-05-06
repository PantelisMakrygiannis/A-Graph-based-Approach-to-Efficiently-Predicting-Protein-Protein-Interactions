#!/bin/bash
# Setup script for IJAIT - PPI Prediction with GNNs
# Requires: Python 3.10+, CUDA 12.1, RTX 4060 (or compatible)

python3 -m venv gnn
source gnn/bin/activate

pip install --upgrade pip

# PyTorch with CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric dependencies
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# PyTorch Geometric
pip install torch-geometric==2.5.0

# Remaining dependencies
pip install -r requirements.txt

echo "Done! Activate with: source gnn/bin/activate"
