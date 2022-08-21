import torch, os, sys, os.path as osp
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as tv_models
import timm
import pytorch_lightning as pl
import torch.optim as optim
from models import LightningTriangleModule, LightningTrianglePatchModule
from data_utils import TriangleDataModule, TrianglePatchesDataModule
import argparse
from gen_utils import add_general_argparse_args
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import numpy as np
import cv2
from PIL import Image
import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm

this_path = osp.split(osp.abspath(__file__))[0]
datautils_path = osp.join(this_path, 'data_utils')
sys.path.append(datautils_path)

from gen_utils import (show_figures, print_shapes)

def add_taskspecific_args(parser):
    parser.add_argument('--validate', action='store_true', help='whether to perform only validation')
    return parser

def extract_patch(img, mask, seg):
    # img = h,w,3, seg = 6-dim vector
    h, w = img.shape[:2]

    seg = np.reshape(seg, (3, 2))
    x1, y1 = seg.min(axis=0) # topleft
    x2, y2 = seg.max(axis=0) # bottom right

    y1 = max(y1 - 10, 0)
    x1 = max(x1 - 10, 0)
    y2 = min(y2 + 10, h-1)
    x2 = min(x2 + 10, w-1)
    # print(seg, x1, y1, x2, y2)
    patch = img[y1:y2+1, x1:x2+1]
    patch_mask = mask[y1:y2+1, x1:x2+1]

    # print(patch.shape)
    # show_figures([patch])
    # plt.pause(50)
    # input()
    return patch, patch_mask, (x1, y1)


def main():
    parser = argparse.ArgumentParser(description="Triangle detection")
    parser = pl.Trainer.add_argparse_args(parser)

    parser = add_general_argparse_args(parser)
    parser = TriangleDataModule.add_argparse_args(parser)
    parser = TrianglePatchesDataModule.add_argparse_args(parser)
    parser = LightningTriangleModule.add_argparse_args(parser)
    parser = LightningTrianglePatchModule.add_argparse_args(parser)
    parser = add_taskspecific_args(parser)
    args = parser.parse_args()

    print(args)

    m = LightningTrianglePatchModule(args)
    data_module = TrianglePatchesDataModule(args)

    if not args.validate:
        # trainer creation
        model_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_train_steps=1000)
        lr_logger = LearningRateMonitor(logging_interval='step')
        callbacks = [model_checkpoint, lr_logger]
        trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
        trainer.fit(m, data_module)
    else:
        # perform validation and store the output in a folder
        model = LightningTrianglePatchModule.load_from_checkpoint(args.resume_from_checkpoint, args=args)
        model.eval()
        model.cuda()

        data_module.prepare_data()
        val_dataset = data_module.val_dataloader()

        # create a folder 
        destdir = args.resume_from_checkpoint+'_val'
        os.makedirs(destdir, exist_ok=True)

        index = 0
        for i, (img, mask, triangle_coords, bbox_coords) in enumerate(tqdm(val_dataset)):
            predmask = np.zeros_like(mask)
            predcorners = np.zeros_like(mask)

            # for each bbox coordinate, pass the crop through the model and get the predictions
            for tricoord in triangle_coords:
                actual_patch, actual_mask, startpos = extract_patch(img, mask, tricoord)
                h, w = actual_patch.shape[:2]
                
                resized_patch = cv2.resize(actual_patch, (args.triangle_patch_size, args.triangle_patch_size), PIL.Image.BILINEAR)
                resized_mask = cv2.resize(actual_mask, (args.triangle_patch_size, args.triangle_patch_size), PIL.Image.BILINEAR)
                # print_shapes([actual_patch, resized_patch])

                th_resized_patch = torch.from_numpy(resized_patch).permute(2,0,1)[None]
                seg_mask_logits, corner_map_logits, coords = model(th_resized_patch.cuda())
                # print_shapes([actual_patch, resized_patch])
                # input()
                seg_mask_logits = seg_mask_logits.sigmoid()
                corner_map_logits = corner_map_logits.sigmoid()

                # get segmentation mask, corner mask 
                seg_mask_logits = seg_mask_logits[0,0].cpu().detach().numpy()
                corner_map_logits = corner_map_logits[0,0].cpu().detach().numpy()
                filepath = osp.join(destdir, f'frame_{index:05}.jpg')
                index += 1

                plot_imgs([resized_patch, resized_mask, seg_mask_logits, corner_map_logits], filepath)

                # res_segmask = cv2.resize(seg_mask_logits, (w, h), PIL.Image.BILINEAR)
                # res_cornermap = cv2.resize(corner_map_logits, (w, h))

                # predmask[startpos[1]:startpos[1]+h, startpos[0]:startpos[0]+w] = res_segmask
                # predcorners[startpos[1]:startpos[1]+h, startpos[0]:startpos[0]+w] = res_cornermap

            
            # print('done')
            # input()

def plot_imgs(imgarr, filepath):
    # plot the image along with target, predicted masks
    fig, axs = plt.subplots(nrows=1, ncols=len(imgarr), figsize=(12, 8))
    for p, plotimg in enumerate(imgarr):
        axs[p].imshow(plotimg)
        axs[p].axis('off')
    
    fig.tight_layout(pad=0)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.canvas.draw()            
    plt.savefig(filepath)
    plt.close()    

if __name__ == '__main__':
    main()
