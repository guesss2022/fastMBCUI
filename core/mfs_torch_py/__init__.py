import torch
from pathlib import Path

# Get the current file's directory
current_dir = Path(__file__).parent
# Construct the path to the shared library
lib_path = './build/mfs_torch/csrc/libmfs_torch.so'
torch.ops.load_library(lib_path)

from . import ops