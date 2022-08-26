from math import inf
import numpy as np

import sys
import os
import mmcv
import torch
from torchvision import transforms


vod_triangles_path = os.path.abspath("./predictors/vod_triangles/src")
print(sys.path)
sys.path.append(vod_triangles_path)
try:
    from .models.triangle_cornermap_segment import TrianglePatchSegment
except ImportError:
    from models.triangle_cornermap_segment import TrianglePatchSegment
import kornia as kn


sys.path.remove(vod_triangles_path)


def fetch_model(ckpt: str, device: str):
    ckpt = torch.load(ckpt, map_location=device)
    model = TrianglePatchSegment(
        backbone=ckpt['backbone'],
        set_stride_to1=ckpt['set_stride_to1'],
        outlevel=ckpt['outlevel'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model = model.to(device)
    
    normalize = kn.enhance.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std =torch.tensor([0.229, 0.224, 0.225])
    )
    return model, normalize


def get_point_vectorized(corner_map_sigmoid):
    assert corner_map_sigmoid.ndim == 4
    min_dist = 30
    n_batch = corner_map_sigmoid.shape[0]
    corner_map_sigmoid = transforms.functional.gaussian_blur(corner_map_sigmoid, kernel_size=9)
    x_flat = torch.flatten(corner_map_sigmoid, 1)
    v, i = torch.sort(x_flat, dim=1, descending=True)
    pxy_bl2 = torch.dstack([torch.div(i, 64, rounding_mode="floor"), i % 64]).float()
    points = torch.ones(n_batch, 3, 2, device="cuda") * 1000
    points[:, 0] = pxy_bl2[0, 0]
    dist_bl = torch.cdist(pxy_bl2, points).min(dim=2).values

    i_found = 1
    range_batch = torch.arange(n_batch, device="cuda")
    points[range(n_batch), i_found] = pxy_bl2[list(range(n_batch)), (dist_bl > min_dist).int().argmax(dim=1).tolist()]
    dist_bl = torch.cdist(pxy_bl2, points).min(dim=2).values
    i_found = 2
    points[range(n_batch), i_found] = pxy_bl2[list(range(n_batch)), (dist_bl > min_dist).int().argmax(dim=1).tolist()]
    return points


def infer_batch(imgs: np.ndarray, model, normalize, device):
    imgs = imgs.permute(0, 3, 1, 2)
    imgs = normalize(imgs/255.)
    with torch.no_grad():
        imgs = imgs.to(device)
        _, _, coords = model(imgs)
        coords = coords.reshape(-1, 3, 2) * 64
        return coords


def test_batch(batch_sz, model, normalize, denormalize, device):
    image = open("../img.png","rb").read()
    imgs = mmcv.imfrombytes(image)[:, :, ::-1] 
    imgs = torch.from_numpy(np.array([imgs] * batch_sz)).to(device)
    tic = time.time()
    coords = infer_batch(imgs, model, normalize, denormalize, device)
    print(coords)
    exit(0)
    toc = time.time()
    #print("Batchsz", batch_sz, "time", toc - tic)
    return toc - tic


if __name__ == "__main__":
    print("sup")
    # run python batch_inference from here
    
    import time
    a = time.time()
    tp = fetch_model("../../../model-data/vod_triangle.ckpt", device="cuda")
    stats = ""
    for batch_sz in range(10, 100, 20):
        times = []
        for _ in range(5):
            times.append(test_batch(batch_sz, *tp, device="cuda"))
        stats += "\n" + str(int(np.array(times).mean()*1000))
    print(stats)
   