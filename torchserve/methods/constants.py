import os
import torch

IMG_SHAPE = 720  # (720, 720, 3)
MAP_LOCATION = 'cuda' if torch.cuda.is_available() else 'cpu'
assert MAP_LOCATION == 'cuda', "GPU IS REQUIRED"
