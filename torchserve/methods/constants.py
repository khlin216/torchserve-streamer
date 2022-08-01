import os
import torch

IMG_WIDTH = 420  # MIN_FACE_SIZE cannot be more than 50
IMG_HEIGHT = int(IMG_WIDTH * 0.75)
MAP_LOCATION = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_ENABLED = os.environ.get("CUDA_ACTIVATED", "False") == "True"
MIN_FACE_SIZE = int(IMG_WIDTH * 0.1)
assert CUDA_ENABLED == False or MAP_LOCATION == 'cuda', f"GPU IS REQUIRED ? {MAP_LOCATION} ?"
