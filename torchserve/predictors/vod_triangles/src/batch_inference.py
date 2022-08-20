from math import inf
import numpy as np

import sys
import os
import mmcv
import torch

#from methods.constants import *



vod_triangles_path = os.path.abspath("./predictors/vod_triangles/src")
sys.path.append(vod_triangles_path)

from models import SimplePatchCornerModule, NonMaxSuppression
import kornia as kn
from torchsummary import summary

sys.path.remove(vod_triangles_path)


def fetch_model(ckpt : str):
    MAP_LOCATION="cpu"
    model = SimplePatchCornerModule.load_from_checkpoint(ckpt)
    model.eval()
    model = model.to(MAP_LOCATION)
    
    nms = NonMaxSuppression()
    denormalize = kn.enhance.Denormalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                                        std=torch.tensor([0.229, 0.224, 0.225]))
    normalize = kn.enhance.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                                        std=torch.tensor([0.229, 0.224, 0.225])) 
    #print(summary(model, (3, 224, 224)))
    return model, nms, normalize, denormalize

def infer_batch(imgs: np.ndarray, model, nms, normalize, denormalize):
    imgs = torch.from_numpy(imgs)
    imgs = imgs.permute(0, 3, 1, 2)

    print(imgs.shape)
    imgs = normalize(imgs)
    seg_mask_logits, corner_map_logits, coords = model(imgs)

    # return the triangle coordinates
    nms_seg, points = nms(corner_map_logits)
    seg_mask = seg_mask_logits.sigmoid()
    corner_map = corner_map_logits.sigmoid()

    # get segmentation mask, corner mask 
    imgs = (denormalize(imgs)[0] * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
    nms_seg = nms_seg[0,0].cpu().detach().numpy()
    seg_mask = seg_mask[0,0].cpu().detach().numpy()
    corner_map = corner_map[0,0].cpu().detach().numpy()

    # fillPoly with the deduced coords
    # nms_seg_mask = np.zeros_like(nms_seg)
    # cv2.fillPoly(nms_seg_mask, pts=[points[0, :3].cpu().numpy()], color=(1,))
    # show_figures([img, seg_mask, nms_seg, corner_map, nms_seg_mask])

    coords = points[:, :3].cpu().numpy()
    return coords


if __name__=="__main__":
    print("sup")
    # run python batch_inference from here
    image = open("../../triangle/img.png","rb").read()
    imgs = mmcv.imfrombytes(image) 
    imgs = np.array([imgs] * 1)
    
    coords = infer_batch(imgs, * fetch_model("../../../model-data/vod_triangle.ckpt"))
    print(coords)