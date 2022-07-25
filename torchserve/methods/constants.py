import os
import torch

IMG_SHAPE = 720  # (720, 720, 3)
MAP_LOCATION = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_ENABLED = os.environ.get("CUDA_ACTIVATED", "False") == "True"

assert CUDA_ENABLED == False or MAP_LOCATION == 'cuda', f"GPU IS REQUIRED ? {MAP_LOCATION} ?"
