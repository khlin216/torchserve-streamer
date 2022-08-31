import os
import torch

W, H = 640, 640
MAP_LOCATION = 'cuda' if torch.cuda.is_available() else 'cpu'
CUDA_ENABLED = os.environ.get("CUDA_ACTIVATED", "False") == "True"
print("CUDA ENABLED", CUDA_ENABLED)
MAP_LOCATION = MAP_LOCATION if CUDA_ENABLED else "cpu"
assert CUDA_ENABLED == False or MAP_LOCATION == 'cuda', f"GPU IS REQUIRED ? {MAP_LOCATION} ?"


### triangle model constants

TRIANGLE_MODEL_PATH = os.environ.get("TRIANGLE_MODEL_PATH", "model-data/yolo.pt")
VOD_TRIANGLE_PATH = os.environ.get("VOD_TRIANGLE_PATH", "model-data/vod_triangle.ckpt")
VOD_TRIANGLE_BATCHES = int(os.environ.get("VOD_TRIANGLE_BATCHES", "50"))