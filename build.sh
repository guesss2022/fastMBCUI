#!/bin/bash

# Build script for mfs_torch C++/CUDA components

set -e  # Exit on any error

echo "Building mfs_torch components..."

# Check if in conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "Warning: No conda environment detected. Please activate the physics_sim environment first:"
    echo "  conda activate physics_sim"
    exit 1
fi

# Check for required tools
if ! command -v cmake &> /dev/null; then
    echo "Error: cmake not found. Please install cmake."
    exit 1
fi

if ! command -v make &> /dev/null; then
    echo "Error: make not found. Please install build-essential."
    exit 1
fi

# Check for CUDA
if command -v nvcc &> /dev/null; then
    echo "CUDA detected: $(nvcc --version)"
    CUDA_AVAILABLE=true
else
    echo "Warning: CUDA not detected. Building CPU-only version."
    CUDA_AVAILABLE=false
fi

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with cmake
###
# IMPORTANT: Sometimes there is problem finding module Torch.
# You can add the path to torch (usually path/to/your/env/lib/python3.xx/site-packages/torch) to CMAKE_PREFIX_PATH
###
echo "Configuring build with cmake..."
if [ "$CUDA_AVAILABLE" = true ]; then
    cmake .. -DCMAKE_BUILD_TYPE=Release
else
    echo "Error: Build failed. No CUDA detected."
    exit 1
fi

# Build
echo "Building (this may take several minutes)..."
make -j$(nproc)

cd ..

# Verify build
if [ -f "build/mfs_torch/csrc/libmfs_torch.so" ]; then
    echo ""
    echo "Build successful! ✓"
    echo "Shared library created: build/mfs_torch/csrc/libmfs_torch.so"
    
    # Test import
    echo "Testing Python import..."
    if python -c "import mfs_torch; print('mfs_torch import successful')" 2>/dev/null; then
        echo "Python import test passed! ✓"
    else
        echo "Warning: Python import test failed. Check library paths."
    fi
else
    echo "Error: Build failed. Shared library not found."
    exit 1
fi

echo ""
echo "Build complete! You can now run the simulation scripts:"
echo "  python simuspclf1d_mfs_coins.py"
echo "  python simuspclf1d_btPin_mfs_grasp.py"
echo "  python simuspclf1d_btPin_mfs_leaf_floor.py"

