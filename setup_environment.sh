#!/bin/bash

# Physics Simulation Environment Setup Script
# Recommended for Ubuntu 22.04+

set -e  # Exit on any error

echo "Setting up Physics Simulation Environment..."

# Check if running on Ubuntu
if ! command -v apt &> /dev/null; then
    echo "Warning: This script is designed for Ubuntu. Please install dependencies manually on other systems."
fi

# Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    gfortran \
    libboost-all-dev \
    libsuitesparse-dev \
    libeigen3-dev \
    libopenblas-dev \
    liblapack-dev \
    libomp-dev \
    pkg-config

# Create conda environment
ENV_NAME="fmbcui"
echo "Creating conda environment: $ENV_NAME"

# Remove existing environment if it exists
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# Create new environment with Python 3.9
conda create -n $ENV_NAME python=3.11 -y

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install pinocchio
conda install -c conda-forge pinocchio -y

# Install remaining packages via pip
pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "1. Build the C++/CUDA components with: ./build.sh"
echo "2. Run simulations with: python simuspclf1d_mfs_coins.py or others"

