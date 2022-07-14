import os
import torch

IMG_SHAPE = 1920 # (1080, 1920, 3)
MAP_LOCATION =  'cuda' if torch.cuda.is_available() else 'cpu'
