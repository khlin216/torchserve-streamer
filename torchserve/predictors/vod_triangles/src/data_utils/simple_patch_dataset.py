import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import coord_utils
import os, sys, os.path as osp
import numpy as np, cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as tv_transforms
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

this_path = osp.split(osp.abspath(__file__))[0]
utilspath = osp.join(this_path, '..')
sys.path.append(utilspath)
sys.path.append(this_path)

from gen_utils import get_immediate_filefolder_paths, read_json, read_image, show_figures, print_shapes
from augmentations import get_train_transform

class SimplePatchDataset(Dataset):
    def __init__(self, root, transform=None) -> None:
        super().__init__()

        self.transform = transform
        self.imgs_folder = osp.join(osp.abspath(root), 'imgs')
        self.coords_folder = osp.join(osp.abspath(root), 'coords')
        assert osp.exists(self.imgs_folder), f"{self.imgs_folder} does not exists"
        assert osp.exists(self.coords_folder), f"{self.coords_folder} does not exists"

        config_file = osp.join(root, 'config.pth')
        if osp.exists(config_file):
            print(f'previous config file exists in {config_file}. loading details from {config_file}')
            config_content = torch.load(config_file)
            self.filecoord_details = []
            for filename, coord in config_content:
                self.filecoord_details.append((osp.join(self.imgs_folder, filename), coord))
        else:
            print(f'previous config ({config_file}) file does not exist. Retrieving details from folders!')
            imgpaths, imgnames = get_immediate_filefolder_paths(self.imgs_folder)
            self.filecoord_details = []
            config_content = []

            # for each image, read the coords
            for filepath, filename_ext in tqdm(zip(imgpaths, imgnames)):
                filename = osp.splitext(filename_ext)[0]
                coordpath = osp.join(self.coords_folder, filename+'.json')
                assert osp.exists(coordpath), f'{coordpath} does not exist'

                filecontent = read_json(coordpath)
                self.filecoord_details.append((filepath, filecontent['coords']))
                config_content.append((filename_ext, filecontent['coords']))

            print(f'saving config to {config_file}')
            torch.save(config_content, config_file)

        print(f"{len(self.filecoord_details)} files found in {root}")


    def __len__(self):
        return len(self.filecoord_details)


    def __getitem__(self, index):
        filepath, coord = self.filecoord_details[index]
        img = read_image(filepath)
        img = np.array(img)

        h, w = img.shape[:2]

        coord = np.array(coord).reshape(3, 2).astype(np.int32)

        # apply transform
        if self.transform:
            # collect keypoints on a needed format
            kps = KeypointsOnImage([Keypoint(x=x, y=y) for x, y in coord], shape=img.shape)

            # apply transform
            img, kps = self.transform(image=img, keypoints=kps)

            # retrieve the coordinates back from iaa
            coord = np.array([[kp.x, kp.y] for kp in kps]).astype(np.int32)
            img = img.copy() # to avoid negative stride error after augmentation

        # workaround to clip coords >= 64
        if np.any(coord >= 64):
            #print(filepath)
            coord[coord >= 64] = 63

        # mask preparation
        h, w = img.shape[:2]
        mask = np.zeros((h, w))
        cv2.fillPoly(mask, pts=[coord], color=(1,))

        # corner map preparation
        cornermap = np.zeros((h, w))
        for x, y in coord: cornermap[y, x] = 1.

        bbox = np.vstack([coord.min(axis=0), coord.max(axis=0)])
        img = np.transpose(img, (2,0,1)) # channel-first for torch

        return img, mask, cornermap, coord, bbox


class SimplePatchDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        train_transform = get_train_transform()
        test_transform = None
        self.train = SimplePatchDataset(root=self.args.trainroot, transform=train_transform)
        self.val = SimplePatchDataset(root=self.args.valroot)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.train_batch, drop_last=True,
                          shuffle=True, num_workers=self.args.workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.val_batch,
                          num_workers=self.args.workers)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--trainroot", type=str,
                            default=None, help="training data root folder")
        parser.add_argument("--valroot", type=str, default=None,
                            help="validation data root folder")
        parser.add_argument("--workers", type=int, default=4,
                            help="num workers to load the data")
        parser.add_argument("--train_batch", type=int,
                            default=16, help="train batch size")
        parser.add_argument("--val_batch", type=int,
                            default=16, help="val/test batch size")
        return parser


if __name__ == '__main__':
    from augmentations import get_train_transform
    transform = get_train_transform()
    ds = SimplePatchDataset("../../data/patch_training/real1x_synth15x/val", transform=transform)
    print(len(ds))
    ds[0]

    for out in ds:
        out = [x for x in out]
        out[0] = np.transpose(out[0], (1,2,0))
        print_shapes(out)
        show_figures(out[:1])
        plt.pause(50)
