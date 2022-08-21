import os
import sys
import os.path as osp
from time import sleep
from einops import rearrange, repeat, reduce

import torch
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt

import imageio
import random
import kornia as kn
import numpy as np
from natsort import natsorted
from collections import defaultdict, Counter
import cv2
from tqdm import tqdm
from skimage import transform
runi = random.uniform
from PIL import Image
import PIL, argparse
from torch.utils.data import Dataset, DataLoader

this_path = osp.split(osp.abspath(__file__))[0]
utilspath = osp.join(this_path, '..')
sys.path.append(utilspath)
sys.path.append(this_path)

from train_dataset import SynthTrianglePatchDataset, get_train_val_split
from gen_utils import mkdir_if_missing, write_json

def create_patch_dataset(split, destdir, particulars):
    imgs_folder = osp.join(destdir, 'imgs')
    coords_folder = osp.join(destdir, 'coords')
    mkdir_if_missing(imgs_folder) # create the folder
    mkdir_if_missing(coords_folder) # create the folder

    patchyfier = SynthTrianglePatchDataset(triangle_patch_size=64, bgfiles=particulars[0], template_triangles=particulars[1], 
                                            random_triangle_transforms=particulars[2], num_htiles=2, num_wtiles=2, 
                                            return_strided_outputs=False, stride=4)
    dataloader = DataLoader(patchyfier, batch_size=16, drop_last=False,
                            shuffle=False, num_workers=4)

    fileindex = 0
    for res_triangles, res_seg_mask, res_corner_map, res_triangle_coords, res_rects in tqdm(dataloader):
        res_triangles = rearrange(res_triangles, 'b sb c h w -> (b sb) c h w')
        res_triangle_coords = rearrange(res_triangle_coords, 'b sb x y-> (b sb) x y')
        for img, coord in zip(res_triangles, res_triangle_coords):
            img = img.numpy()
            img = np.transpose(img, (1,2,0))
            coord = coord.numpy().reshape(-1)
            
            filename = f'{split}_synth_{fileindex:06d}'
            img_filepath = osp.join(imgs_folder, f'{filename}.jpg')
            coord_filepath = osp.join(coords_folder, f'{filename}.json')

            imageio.imwrite(img_filepath, img)
            write_json({'coords':coord.tolist()}, coord_filepath)
            fileindex += 1
            # break

        # input()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='patch dataset creation')
    parser.add_argument('--train_frames_root', type=str, default="../../data/train")
    parser.add_argument('--destdir', type=str, default="../../data/patch_training")
    args = parser.parse_args()

    train_particulars, val_particulars = get_train_val_split(root='../../data/train')
    torch.save({'train':train_particulars, 'val':val_particulars}, osp.join(args.destdir,'synth_config.pth'))

    create_patch_dataset('train', osp.join(args.destdir, 'train'), train_particulars)
    create_patch_dataset('val', osp.join(args.destdir, 'val'), val_particulars)