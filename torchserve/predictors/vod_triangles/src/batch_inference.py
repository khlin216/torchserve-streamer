from math import inf
import numpy as np

import sys
import os
import mmcv
import torch


vod_triangles_path = os.path.abspath("./predictors/vod_triangles/src")
print(sys.path)
sys.path.append(vod_triangles_path)
try:
    from .models import SimplePatchCornerModule, NonMaxSuppression
except ImportError:
    from models import SimplePatchCornerModule, NonMaxSuppression
import kornia as kn



sys.path.remove(vod_triangles_path)


def fetch_model(ckpt : str, device : str):
    
    model = SimplePatchCornerModule.load_from_checkpoint(ckpt)
    model.eval()
    model = model.to(device)
    
    nms = NonMaxSuppression()
    denormalize = kn.enhance.Denormalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                                        std=torch.tensor([0.229, 0.224, 0.225]))
    normalize = kn.enhance.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                                        std=torch.tensor([0.229, 0.224, 0.225])) 
    return model, nms, normalize, denormalize


def infer_batch(imgs: np.ndarray, model, nms, normalize, denormalize, device):

    imgs = imgs.permute(0, 3, 1, 2)
    imgs = normalize(imgs)
    with torch.no_grad():
        imgs = imgs.to(device)
        seg_mask_logits, corner_map_logits, coords = model(imgs)

        # return the triangle coordinates
        nms_seg, points = nms(corner_map_logits)
        coords = points[:, :3].cpu().numpy()
        return coords


def test_batch(batch_sz, model, nms, normalize, denormalize, device):
    image = open("../img.png","rb").read()
    imgs = mmcv.imfrombytes(image)[:, :, ::-1] 
    imgs = torch.from_numpy(np.array([imgs] * batch_sz)).to(device)
    tic = time.time()
    coords = infer_batch(imgs, model, nms, normalize, denormalize, device)
    print(coords)
    exit(0)
    toc = time.time()
    #print("Batchsz", batch_sz, "time", toc - tic)
    return toc - tic




if __name__=="__main__":
    print("sup")
    # run python batch_inference from here
    
    import time
    a = time.time()
    tp = fetch_model("../../../model-data/vod_triangle.ckpt", device="cuda")
    stats= ""
    for batch_sz in range(10, 100, 20):
        times = []
        for _ in range(5):
            times.append(test_batch(batch_sz, *tp, device="cuda"))
        stats += "\n" + str(int(np.array(times).mean()*1000))
    print(stats)
   