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
import PIL

this_path = osp.split(osp.abspath(__file__))[0]
utilspath = osp.join(this_path, '..')
sys.path.append(utilspath)
sys.path.append(this_path)

from coord_utils import homogeneous_coords, nonhomogeneous_coords, get_valid_transformations
from gen_utils import (read_json, read_image, cv2_warp, show_figures, print_shapes,
                        collect_all_files_recurse, split_train_val)


def get_template_homogenous_coords():
    dest_coords = [0, 200, 200, 0, 200, 200]
    dest_coords = np.array(dest_coords, dtype=np.int32).astype(np.int32)
    dest_coords = dest_coords.reshape(3, 2)    
    return homogeneous_coords(dest_coords)


def get_train_val_split(root, val_ratio=0.15):
    """to get train/val split from the data available in root folder

    Args:
        root (str): root folder path (containing 'train_folders' folder)
        val_ratio (float, optional): validation ratio. Defaults to 0.15.

    Returns:
        train list, val list
    """
    trainframes_path = osp.join(root, 'train_frames')
    template_triangles_path = osp.join(root, 'triangle_templates')
    templates_json_path = osp.join(root, 'template_labels.json')

    all_files = collect_all_files_recurse(trainframes_path)
    print(f"total image files inside {root}: {len(all_files)}")

    template_coords = get_template_homogenous_coords()

    template_triangles = read_template_triangles(template_triangles_path, templates_json_path, template_coords)
    print(f"number of template triangles: {len(template_triangles)}")

    valid_template_tforms = get_valid_transformations(anchor_coords=template_coords, output_shape=(200, 200))
    print(f"number of random transformations: {len(valid_template_tforms)}")

    train_bgfiles, val_bgfiles = split_train_val(all_files, val_ratio=val_ratio)
    print(f"train bgfiles # {len(train_bgfiles)}, val bgfiles # {len(val_bgfiles)}")

    train_triangles, val_triangles = split_train_val(template_triangles, val_ratio=0.20)
    print(f"train triangles # {len(train_triangles)}, val triangles # {len(val_triangles)}")
    
    train_tforms, val_tforms = split_train_val(valid_template_tforms, val_ratio=val_ratio)
    print(f"train transforms # {len(train_tforms)}, val transforms # {len(val_tforms)}")

    return (train_bgfiles, train_triangles, train_tforms), (val_bgfiles, val_triangles, val_tforms)


def read_template_triangles(images_path, templates_json, template_coords):
    template_contents = read_json(templates_json)
    all_triangle_clips = []

    for filename, triangles in template_contents.items():
        # if filename != 'template0008.jpg': continue
        filepath = osp.join(images_path, filename)
        img = read_image(filepath)
        img = np.array(img)
        h, w, c = img.shape

        for triangle in triangles:
            src_coords = np.array(
                triangle, dtype=np.int32).astype(np.int32)
            src_coords = src_coords.reshape(3, 2)

            # make actual coords into homogenous coords
            src_coords = homogeneous_coords(src_coords)  # x, y, 1 vertical

            # crop the triangle
            norm = np.eye(3)
            norm[0, 0] = 1 / w
            norm[1, 1] = 1 / h
            invnorm = np.linalg.inv(norm)

            # normalize co-ords for numerical stability
            # norm_x, norm_y, 1 in vertical
            norm_src_coords = np.matmul(norm, src_coords)
            # norm_x, norm_y, 1 in vertical
            norm_dest_coords = np.matmul(norm, template_coords)

            xx = np.concatenate([norm_src_coords.T, np.zeros((3, 3))], axis=1)  # 3x6
            yy = np.concatenate([np.zeros((3, 3)), norm_src_coords.T], axis=1)  # 3x6
            AA = np.concatenate([xx, yy], axis=0)  # 6x6
            y = norm_dest_coords[:2].reshape(6, 1)
            A = np.matmul(np.linalg.inv(AA), y)
            A = np.concatenate([A.reshape(2, 3), np.array([0, 0, 1]).reshape(1, 3)], axis=0)

            # find A for unnormalized coords
            unnorm_A = np.matmul(invnorm, np.matmul(A, norm))
            warped_img = cv2_warp(img, unnorm_A)
            triangle_clip = warped_img[:202, :202]

            triangle_clip_coords = np.array([[0, 0], [0, 199], [199, 0]]).astype(np.int32)
            cv2.fillPoly(triangle_clip, pts=[triangle_clip_coords], color=(0, 0, 0))
            all_triangle_clips.append(triangle_clip)

    return all_triangle_clips


class SynthTriangleDataset(data.Dataset):
    def __init__(self, bgfiles, template_triangles, random_triangle_transforms,
                        stride, num_wtiles, num_htiles, 
                        return_strided_outputs, p_triangle=1.0):
        """Constructor to create a SynthTriangle dataset

        Args:
            root (str): root folder of the dataset (typically ../data/train)
            stride (int): Downssampling size for labels (to be the ground truth for the loss function)
            num_wtiles (int): number of splits of the image in width direction
            num_htiles (_type_): number of splits of the image in height direction
            return_strided_outputs (bool): whether to output strided labels or not
            p_triangle (float): likelihood of drawing a triangle as opposed to a rectangle
        """
        super().__init__()
        self.stride = stride
        self.num_wtiles = num_wtiles
        self.num_htiles = num_htiles
        self.return_strided_outputs = return_strided_outputs
        self.p_triangle = p_triangle

        # save the file details in instance variables
        self.bgfiles = bgfiles
        self.template_triangles = template_triangles
        self.random_triangle_transforms = random_triangle_transforms

        # the anchor coords of template triangle
        self.dest_coords = get_template_homogenous_coords() # x, y, 1 vertical
        # self.show_warp_triangles()

    def show_warp_triangles(self):
        print('showing warped triangles')

        for tform in self.valid_template_tforms:
            # choose a random triangle
            triangle = random.choice(self.all_triangle_clips)
            warped_triangle = cv2_warp(triangle, tform['params'])
            warped_triangle = (warped_triangle * 255).astype(np.int32)
            coords = nonhomogeneous_coords(tform['coords']).astype(np.int32)
            # cv2.fillPoly(warped_triangle, pts=[coords], color=(255, 0, 0))

            print(warped_triangle.min(), warped_triangle.max())
            plt.imshow(warped_triangle)
            plt.pause(30)

    def __len__(self):
        return len(self.bgfiles)

    def __getitem__(self, index):
        bgfile = self.bgfiles[index]
        bgimg = read_image(bgfile)

        # resize to standard size
        bgimg = bgimg.resize((900,900), PIL.Image.BILINEAR)
        bgimg = np.array(bgimg)

        # select random number of horizontal and vertical tiles
        num_w_tiles = self.num_wtiles # np.random.randint(2, 5)
        num_h_tiles = self.num_htiles # np.random.randint(2, 5)
        h, w, c = bgimg.shape

        # make sure h, w are multiples of num patches
        assert w % num_w_tiles == 0
        assert h % num_h_tiles == 0

        patch_size_w = (w // num_w_tiles)
        patch_size_h = (h // num_h_tiles)

        mask = np.zeros((h,w))  # mask to merge the pasted triangles into bgimg
        fgimg = np.zeros_like(bgimg)
        triangle_coords = []

        for w_index in range(num_w_tiles):
            for h_index in range(num_h_tiles):
                patch_start_h = h_index * patch_size_h
                patch_start_w = w_index * patch_size_w
                if np.random.rand() < self.p_triangle:
                    # select random triangle template
                    triangle = random.choice(self.template_triangles)
                    triangle_h, triangle_w, _ = triangle.shape

                    # select a random transformation from the list of transformations
                    tform = random.choice(self.random_triangle_transforms)

                    # perform random transformation
                    warped_triangle = cv2_warp(triangle, tform['params'])
                    warped_anchor = tform['coords']

                    # resize the warped triangle to standard patch size
                    resize_tform = transform.AffineTransform(scale=(patch_size_w/triangle_w, patch_size_h/triangle_h))
                    resize_tform_params = resize_tform.params
                    # resize_tform_invparams = resize_tform._inv_matrix
                    warped_triangle = cv2_warp(warped_triangle, resize_tform_params, size=(patch_size_w, patch_size_h))
                    warped_anchor = nonhomogeneous_coords(np.matmul(resize_tform_params, warped_anchor))  # 3 x 2

                    warped_mask = np.zeros((patch_size_h, patch_size_w))
                    cv2.fillPoly(warped_mask, pts=[warped_anchor.astype(np.int32)], color=(1,))

                    # paste in the orignal images
                    fgimg[patch_start_h:patch_start_h+patch_size_h, patch_start_w:patch_start_w+patch_size_w] = warped_triangle
                    mask[patch_start_h:patch_start_h+patch_size_h, patch_start_w:patch_start_w+patch_size_w] = warped_mask
                    warped_anchor = np.array([patch_start_w, patch_start_h]).reshape(1, 2) + warped_anchor
                    triangle_coords.append(warped_anchor.astype(np.int32))
                else:
                    color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                    scalex = np.random.uniform(0.10, 0.75)
                    scaley = np.random.uniform(0.10, 0.75)
                    x0 = int(round(patch_start_w + 0.5 * (1 - scalex) * patch_size_w))
                    x1 = int(round(patch_start_w + 0.5 * (1 + scalex) * patch_size_w))
                    y0 = int(round(patch_start_h + 0.5 * (1 - scaley) * patch_size_h))
                    y1 = int(round(patch_start_h + 0.5 * (1 + scaley) * patch_size_h))
                    cv2.rectangle(fgimg, (x0, y0), (x1, y1), color, -1)
                    mask[y0:y1, x0:x1] = 1

        rgb_mask = mask[:, :, None]
        synth_img = bgimg * (1 - rgb_mask) + fgimg * (rgb_mask)
        synth_img = synth_img.astype(np.uint8)
        # print(bgimg.min(), bgimg.max(), fgimg.min(), fgimg.max(), synth_img.min(), synth_img.max())
        # show_figures([fgimg, bgimg, synth_img])      
        # plt.pause(30)
        # input()     

        synth_img = np.transpose(synth_img, axes=(2,0,1))
        triangle_coords = np.array(triangle_coords)
   
        # return the strided labels only if necessary
        if self.return_strided_outputs:
            # strided mask calculation
            strided_corner_map = np.zeros((h // self.stride, w // self.stride))
            strided_coords = []
            for triangle in triangle_coords:
                coords = []
                for x, y in triangle:
                    x_i = x // self.stride
                    y_i = y // self.stride
                    strided_corner_map[y_i, x_i] = 1
                    coords.append([x_i, y_i])

                strided_coords.append(np.array(coords).astype(np.int32))

            strided_mask = np.zeros_like(strided_corner_map)
            cv2.fillPoly(strided_mask, pts=strided_coords, color=(1,))
            # arr = [synth_img, triangle_coords, mask, strided_corner_map, strided_mask, strided_coords]
            # print([type(x) for x in arr])

            strided_coords = np.array(strided_coords)

            return synth_img, triangle_coords, mask, strided_corner_map, strided_mask, strided_coords
        else:
            return synth_img, triangle_coords, mask


class SynthTrianglePatchDataset(SynthTriangleDataset):
    def __init__(self, triangle_patch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.triangle_patch_size = triangle_patch_size

    def get_bbox_coords(self, triangle_coords):
        # triangle_coords = 3 x 2 np array
        top_left = triangle_coords.min(axis=0)
        bottom_left = triangle_coords.max(axis=0)
        return np.vstack([top_left, bottom_left])

    def extract_random_patch(self, img_patch, mask, triangle_coords, bbox):
        # find the relaxations possible for all the sides of the bbox (left, top, right, bottom)
        h, w, c = img_patch.shape
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        rint = random.randint

        # find random start point and subtract it from triangle, bbox coords
        start_point = np.array([rint(0, x1), rint(0, y1)])
        triangle_coords = triangle_coords - start_point
        bbox = bbox - start_point

        # find random end point
        end_point = np.array([rint(x2, w), rint(y2, h)])
        img_crop = img_patch[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        mask_crop = mask[start_point[1]:end_point[1], start_point[0]:end_point[0]]
        
        # perform resize transform
        triangle_h, triangle_w, _ = img_crop.shape
        resize_tform = transform.AffineTransform(scale=(self.triangle_patch_size/triangle_w, self.triangle_patch_size/triangle_h))
        resize_tform_params = resize_tform.params
        resize_tform_invparams = resize_tform._inv_matrix
        resized_img = cv2_warp(img_crop, resize_tform_params, size=(self.triangle_patch_size, self.triangle_patch_size))
        resized_mask = cv2_warp(mask_crop, resize_tform_params, size=(self.triangle_patch_size, self.triangle_patch_size))
        resized_mask = (resized_mask > 0).astype(np.float32)

        resized_triangle_coords = nonhomogeneous_coords(np.matmul(resize_tform_params, homogeneous_coords(triangle_coords)))  # 3 x 2
        resized_triangle_coords = resized_triangle_coords.astype(np.int32)
        resized_bbox = nonhomogeneous_coords(np.matmul(resize_tform_params, homogeneous_coords(bbox)))
        resized_bbox = resized_bbox.astype(np.int32)

        # create corner map according to resized triangle_coords
        
        corner_map = np.zeros_like(resized_mask)
        resized_triangle_coords[resized_triangle_coords>0] -= 1
        for x, y in resized_triangle_coords:
            if x >= self.triangle_patch_size or y >= self.triangle_patch_size:
                print(self.triangle_patch_size, triangle_w, self.triangle_patch_size, triangle_h)
                print(triangle_coords)
                print(resized_triangle_coords)
                print(nonhomogeneous_coords(np.matmul(resize_tform_params, homogeneous_coords(triangle_coords))))                
                print(resize_tform_params)
                input()

            corner_map[y, x] = 1.

        return resized_img, resized_mask, corner_map, resized_triangle_coords, resized_bbox

    def __getitem__(self, index):
        synth_img, triangle_coords, mask = super().__getitem__(index)

        rects = []
        for triangle in triangle_coords:
            rects.append(self.get_bbox_coords(triangle))

        _, h, w = synth_img.shape
        patch_size_w = (w // self.num_wtiles)
        patch_size_h = (h // self.num_htiles)

        res_triangles, res_triangle_coords, res_rects, res_corner_map, res_seg_mask = [], [], [], [], []

        # for each tile/patch, try to perturb the triangle patch
        for w_index in range(self.num_wtiles):
            for h_index in range(self.num_htiles):
                # find the start coords of the patch
                patch_start_h = h_index * patch_size_h
                patch_start_w = w_index * patch_size_w
                patch_index = w_index * self.num_htiles + h_index

                # extract out the particular patch and its detail s(mask, coords, bbox)
                start_coord = np.array([patch_start_w, patch_start_h])
                currimg_patch = synth_img[:, patch_start_h:patch_start_h+patch_size_h, patch_start_w:patch_start_w+patch_size_w]
                currimg_patch = np.transpose(currimg_patch, axes=(1,2,0))
                curr_mask = mask[patch_start_h:patch_start_h+patch_size_h, patch_start_w:patch_start_w+patch_size_w]

                # print(start_coord, triangle_coords[patch_index], rects[patch_index])
                # show_figures([currimg_patch, curr_mask])
                # plt.pause(50)

                # subtract the start coords to make it independent of the start coords
                curr_triangle_coords = triangle_coords[patch_index] - start_coord
                curr_bbox = rects[patch_index] - start_coord
                # print(currimg_patch.shape, curr_mask.shape, curr_triangle_coords.shape, curr_bbox.shape)

                # perturb the patch along with the coords
                triangle, seg_mask, corner_map, triangle_anchors, rect = self.extract_random_patch(currimg_patch, curr_mask, 
                                                                                                   curr_triangle_coords, curr_bbox)

                triangle = np.transpose(triangle, axes=(2,0,1))
                res_triangles.append(triangle)
                res_triangle_coords.append(triangle_anchors)
                res_rects.append(rect)
                res_corner_map.append(corner_map)
                res_seg_mask.append(seg_mask)

        res_triangles = np.array(res_triangles)
        res_triangle_coords = np.array(res_triangle_coords)
        res_rects = np.array(res_rects)
        res_corner_map = np.array(res_corner_map)
        res_seg_mask = np.array(res_seg_mask)      
        return res_triangles, res_seg_mask, res_corner_map, res_triangle_coords, res_rects


if __name__ == '__main__':
    train_particulars, val_particulars = get_train_val_split(root='../../data/train')
    ds = SynthTriangleDataset(*train_particulars, stride=4, num_htiles=4, num_wtiles=3, return_strided_outputs=False)

    count = 0
    for img in tqdm(ds):
        img, coords, mask = img

        img = np.transpose(img, (1,2,0))
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(mask)

        outmask = np.zeros_like(img)
        cv2.fillPoly(outmask, pts=coords, color=(255, 0, 0))
        plt.figure()
        plt.imshow(outmask)

        plt.pause(30)
        if count >= 1: break
        count += 1

    print('val')
    ds = SynthTriangleDataset(*val_particulars, stride=4, num_htiles=3, num_wtiles=4, return_strided_outputs=False)
    count = 0
    for img in tqdm(ds):
        img, coords, mask = img

        img = np.transpose(img, (1,2,0))
        plt.figure()
        plt.imshow(img)
        plt.figure()
        plt.imshow(mask)

        outmask = np.zeros_like(img)
        cv2.fillPoly(outmask, pts=coords, color=(255, 0, 0))
        plt.figure()
        plt.imshow(outmask)

        plt.pause(30)        
        if count >= 2: break
        count += 1

    # input()
    # all_indices = list(range(len(ds)))
    # while True:
    #     img, coords, mask, strided_corner_map, strided_mask, strided_coords = ds[random.choice(all_indices)]

    #     img = np.transpose(img, (1,2,0))
    #     plt.figure()
    #     plt.imshow(img)
    #     plt.figure()
    #     plt.imshow(mask)

    #     outmask = np.zeros_like(img)
    #     cv2.fillPoly(outmask, pts=coords, color=(255, 0, 0))
    #     plt.figure()
    #     plt.imshow(outmask)

    #     plt.figure()
    #     plt.imshow(strided_corner_map)

    #     # plt.figure()
    #     # plt.imshow(strided_mask)

    #     plt.pause(30)
    #     plt.close('all')

    # ds = SynthTrianglePatchDataset(root='../../data/train', stride=5, num_htiles=3, num_wtiles=2, 
    #                                return_strided_outputs=False, triangle_patch_size=128)
    # all_indices = list(range(len(ds)))
    # # for x in tqdm(ds): pass

    # while True:
    #     res_triangles, res_seg_mask, res_corner_map, res_triangle_coords, res_rects  = ds[random.choice(all_indices)]
    #     # print(res_triangles.shape, res_triangle_coords.shape, res_rects.shape, res_corner_map.shape, res_seg_mask.shape)
    #     # input()

    #     for triangle, seg_mask, corner_map, triangle_coords, bbox in zip(res_triangles, res_seg_mask, res_corner_map, res_triangle_coords, res_rects):
    #         # print(triangle.shape, seg_mask.shape, corner_map.shape, triangle_coords.shape, bbox.shape)
    #         # input()

    #         triangle = np.transpose(triangle, (1,2,0)).copy() 
    #         plt.figure()
    #         plt.imshow(triangle)

    #         plt.figure()
    #         plt.imshow(seg_mask)

    #         plt.figure()
    #         plt.imshow(corner_map)       

    #         print(triangle.shape, triangle_coords.shape)
    #         # input()     

    #         cv2.fillPoly(triangle, pts=[triangle_coords], color=(255,0,0))
    #         cv2.rectangle(triangle, pt1=bbox[0], pt2=bbox[1], color=(0,255,0), thickness=2)
    #         plt.figure()
    #         plt.imshow(triangle)

    #         plt.pause(30)
    #         plt.close('all')    
