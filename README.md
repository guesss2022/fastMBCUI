# Fast Multi-Body Coupling for Underwater Interactions

This repository contains the example code for the paper Fast Multi-Body Coupling for Underwater Interactions in PG 2025. A fast physical simulation framework for underwater rigid body interactions using preconditioned Method of Fundamental Solutions (MFS) inspired by [1].

## Requirements

- **OS**: Ubuntu 22.04+
- **GPU**: NVIDIA GPU with CUDA 11.8+

## Quick Start

### 1. Setup Environment

**Option A: Automatic setup (recommended)**
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

**Option B: Manual setup**

Manually setup the environment based on requirements in setup_environment.sh.

### 2. Build Components

```bash
conda activate fmbcui
chmod +x build.sh
./build.sh
```

***IMPORTANT NOTE***: Sometimes there is problem finding module Torch. You can add the path to torch (usually path/to/your/env/lib/python3.xx/site-packages/torch) to CMAKE_PREFIX_PATH in this case before configuring.

### 3. Run Simulations

```bash
# Coin simulation
python simuspclf1d_mfs_coins.py

# Robotic grasping simulation  
python simuspclf1d_btPin_mfs_grasp.py

# Leaf-floor interaction simulation
python simuspclf1d_btPin_mfs_leaf_floor.py
```

## System Dependencies

The setup script will install these automatically on Ubuntu:

- `build-essential`, `cmake`, `git`
- `libboost-all-dev` - Boost C++ libraries
- `libsuitesparse-dev` - Sparse matrix operations  
- `libeigen3-dev` - Linear algebra library
- `libopenblas-dev`, `liblapack-dev` - Optimized BLAS/LAPACK
- `libomp-dev` - OpenMP parallel computing

## Architecture

- **`mfs_torch/`** - CUDA-accelerated MFS solver
- **`core/`** - Physics simulation code
- **`utils/`** - Utility functions for visualization, I/O, etc.
- **`models/`** - Robot models and geometries data

## Output

Simulation results are saved to `./output/` with trajectory data and 3D mesh sequences for visualization.

## Citation

```latex
@inproceedings{10.2312:pg.20251268,
    booktitle = {Pacific Graphics Conference Papers, Posters, and Demos},
    editor = {Christie, Marc and Han, Ping-Hsuan and Lin, Shih-Syun and Pietroni, Nico and Schneider, Teseo and Tsai, Hsin-Ruey and Wang, Yu-Shuen and Zhang, Eugene},
    title = {{Fast Multi-Body Coupling for Underwater Interactions}},
    author = {Gao, Tianhong and Chen, Xuwen and Li, Xingqiao and Li, Wei and Chen, Baoquan and Pan, Zherong and Wu, Kui and Chu, Mengyu},
    year = {2025},
    publisher = {The Eurographics Association},
    ISBN = {978-3-03868-295-0},
    DOI = {10.2312/pg.20251268}
}
```

## References

[1] Chen, J., Schäfer, F. T., & Desbrun, M. (2024). Lightning-fast method of fundamental solutions. ACM Transactions on Graphics, 43(4), 77.

