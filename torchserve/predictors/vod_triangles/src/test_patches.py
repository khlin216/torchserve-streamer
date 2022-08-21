import torch, os, sys, os.path as osp
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as tv_models
import timm
import pytorch_lightning as pl
import torch.optim as optim
from data_utils import SimplePatchDataModule
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from models import SimplePatchCornerModule
import numpy as np
import cv2
from PIL import Image
import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import NonMaxSuppression
from attrdict import AttrDict
import kornia as kn

this_path = osp.split(osp.abspath(__file__))[0]
datautils_path = osp.join(this_path, 'data_utils')
sys.path.append(datautils_path)

from gen_utils import (show_figures, print_shapes, add_general_argparse_args)

class PatchModelPostProcessor(object):
    def __init__(self, ckpt_path, args):
        """constructor

        Args:
            ckpt_path (str): checkpoint path to load the weights from
            args (dict): args from argparse to pass to the model
        """
        self.ckpt_path = ckpt_path
        ckpt = torch.load(ckpt_path)
        # print(list(ckpt.keys()))
        # print(ckpt['hyper_parameters'])
        # args.backbone = ckpt['hyper_parameters']['args'].backbone

        self.model = SimplePatchCornerModule.load_from_checkpoint(ckpt_path, args=args)
        self.model.eval()
        self.model.cuda()
        
        # prepare args
        self.nms = NonMaxSuppression()

        self.denormalize = kn.enhance.Denormalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                                        std=torch.tensor([0.229, 0.224, 0.225]))         

    def process_image(self, image):
        """to process the given image and return the triangle coords

        Args:
            image (tensor): imagenet normalized image of dimension bxcxhxw

        Returns:
            coords: triangle coords of dimension bx3x2 containing x, y coords
        """
        seg_mask_logits, corner_map_logits, coords = self.model(image)

        # return the triangle coordinates
        nms_seg, points = self.nms(corner_map_logits)
        seg_mask = seg_mask_logits.sigmoid()
        corner_map = corner_map_logits.sigmoid()

        # get segmentation mask, corner mask 
        img = (self.denormalize(image)[0] * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        nms_seg = nms_seg[0,0].cpu().detach().numpy()
        seg_mask = seg_mask[0,0].cpu().detach().numpy()
        corner_map = corner_map[0,0].cpu().detach().numpy()

        # fillPoly with the deduced coords
        nms_seg_mask = np.zeros_like(nms_seg)
        cv2.fillPoly(nms_seg_mask, pts=[points[0, :3].cpu().numpy()], color=(1,))
        show_figures([img, seg_mask, nms_seg, corner_map, nms_seg_mask])

        coords = points[:, :3].cpu().numpy()
        return coords
        

def main():
    parser = argparse.ArgumentParser(description="Triangle detection")
    parser = pl.Trainer.add_argparse_args(parser)

    parser = add_general_argparse_args(parser)
    parser = SimplePatchDataModule.add_argparse_args(parser)
    parser = SimplePatchCornerModule.add_argparse_args(parser)
    args = parser.parse_args()

    print(args)
    args.trainroot = '../data/patch_training/synth_train'
    args.valroot = '../data/patch_training/real_val'
    # args.valroot = '../data/patch_training/synth_val'
    args.dont_set_stride = False
   
    data_module = SimplePatchDataModule(args)

    data_module.prepare_data()
    val_dataset = data_module.val_dataloader()

    normalize = kn.enhance.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                                        std=torch.tensor([0.229, 0.224, 0.225])) 
    model = PatchModelPostProcessor(args.resume_from_checkpoint, args)


    for i, (img, mask, cornermap, coord, bbox) in enumerate(tqdm(val_dataset)):
        th_patch = normalize(img / 255.)
        
        model.process_image(th_patch.cuda())
        plt.pause(50)


if __name__ == '__main__':
    # execute the file as 
    # python test_patches.py --resume_from_checkpoint <checkpoint_path>
    main()

