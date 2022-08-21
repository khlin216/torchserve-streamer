
from tqdm import tqdm
import cv2
from collections import defaultdict, Counter
from natsort import natsorted
import numpy as np
import kornia as kn
import imageio
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
import torch
from einops import rearrange, repeat, reduce
from time import sleep
from re import L
import os
import sys
import os.path as osp

this_path = osp.split(osp.abspath(__file__))[0]
utilspath = osp.join(this_path, '..')
sys.path.append(utilspath)

from gen_utils import (get_immediate_filefolder_paths, read_json, read_image, show_image)

class ValTrackDataset(data.Dataset):
    def __init__(self, root, giveout_mask=True, giveout_coords=True,
                 maxfiles=150, transform=None):
        self.root = root
        self.giveout_mask = giveout_mask
        self.giveout_coords = giveout_coords
        self.maxfiles = maxfiles
        self.transform = transform

        # collect filenames and parse coords
        self.trackpaths, self.tracknames = get_immediate_filefolder_paths(root)
        self.track2index = {trackname: i for i,
                            trackname in enumerate(self.tracknames)}

        # read annotations for all tracks
        # contains mapping from trackname to frame details
        self.track_details = defaultdict(list)
        # imgname2id - mapping from image name to ID
        # id2details - mapping from ID to frame details (height, width, trackname, file_name,
        #                                               all_seg_bbox, valid_seg_bbox, all_file_names, valid_file_names)
        #

        # contains all valid frames in the dataset (list of {'trackname':..., 'file_name':...})
        self.all_valid_frames = []
        self.read_annotations()

    def __len__(self):
        return len(self.all_valid_frames)

    def read_annotations(self):
        for trackpath, trackname in zip(self.trackpaths, self.tracknames):
            print(f"processing {trackpath}")
            annotation_path = osp.join(trackpath, 'annotations', 'instances_default.json')
            content = read_json(annotation_path)
            details = []

            imgname2id = {}
            id2details = {}
            self.track_details[trackname] = {}

            # first collect filenames and ids
            for image_detail in content['images']:
                file_name = image_detail['file_name']
                image_id = image_detail['id']

                # make sure the id and file_name doe4sn't exist already
                assert file_name not in imgname2id
                assert image_id not in id2details

                imgname2id[file_name] = image_id

                id2details[image_id] = {}
                id2details[image_id]['height'] = image_detail['height']
                id2details[image_id]['width'] = image_detail['width']
                id2details[image_id]['trackname'] = trackname
                id2details[image_id]['file_name'] = file_name
                id2details[image_id]['all_seg_bbox'] = []
                id2details[image_id]['valid_seg_bbox'] = []

            # collect the annotations
            for annotation in content['annotations']:
                image_id = annotation['image_id']
                assert image_id in id2details, f"{image_id} not present in id2details"

                is_occluded = annotation['attributes']['occluded']
                file_name = id2details[image_id]['file_name']
                segmentation = annotation['segmentation'][0]
                bbox = annotation['bbox']

                id2details[image_id]['all_seg_bbox'].append({"seg": segmentation, "bbox": bbox, 'occluded': is_occluded})

            # insert the details for the current track in the order of file names
            self.track_details[trackname]['imgname2id'] = imgname2id
            self.track_details[trackname]['id2details'] = id2details

            file_names = natsorted(list(imgname2id.keys()))
            self.track_details[trackname]['all_file_names'] = file_names

            valid_file_names = []

            # here check for the frames that contain valid triangle annotations
            for image_id, details in id2details.items():
                valid_seg_bbox = []
                for seg_bbox in details['all_seg_bbox']:
                    # only consider valid triangles
                    if seg_bbox['occluded'] or len(seg_bbox['seg']) != 6: continue
                    valid_seg_bbox.append(seg_bbox)

                id2details[image_id]['valid_seg_bbox'] = valid_seg_bbox
                if len(valid_seg_bbox) > 0:  # if there are proper triangles found in this frame
                    valid_file_names.append(details['file_name'])

            self.track_details[trackname]['valid_file_names'] = natsorted(valid_file_names)

        # finally collect all track, filenames that contain valid annotations
        for trackname in self.tracknames:
            all_file_names = self.track_details[trackname]['all_file_names']
            valid_file_names = self.track_details[trackname]['valid_file_names']
            print(f"{trackname:60s}: {len(valid_file_names)} out of {len(all_file_names)}")
            for file_name in valid_file_names:
                self.all_valid_frames.append({'trackname': trackname, 'file_name': file_name})

        print(f"total valid frames: {len(self.all_valid_frames)}")

    def __getitem__(self, index):
        currframe = self.all_valid_frames[index]
        trackname = currframe['trackname']
        file_name = currframe['file_name']

        # read image
        img = self.read_track_image(trackname, file_name)

        # collect coords and prepare mask
        currtrack_details = self.track_details[trackname]
        image_id = currtrack_details['imgname2id'][file_name]
        image_details = currtrack_details['id2details'][image_id]

        # image_details --> valid_seg_bbox contains all the valid triangles
        valid_triangles = image_details['valid_seg_bbox']
        all_seg_coords = []
        all_bbox_coords = []
        for triangle in valid_triangles:
            seg_coords = np.array(triangle['seg'], dtype=np.int32).astype(np.int32)
            seg_coords = seg_coords.reshape(3, 2)
            all_seg_coords.append(seg_coords)

            bbox_coords = np.array(triangle['bbox'], dtype=np.int32).astype(np.int32)
            all_bbox_coords.append(bbox_coords)

        h, w, c = img.shape
        mask = np.zeros((h,w))
        cv2.fillPoly(mask, pts=all_seg_coords, color=(1,))

        all_seg_coords = np.array(all_seg_coords)
        all_bbox_coords = np.array(all_bbox_coords)
        return img, mask, all_seg_coords, all_bbox_coords

    def read_track_image(self, trackname, file_name):
        imgpath = osp.join(self.root, trackname, 'images', file_name)
        img = read_image(imgpath)
        return np.array(img)

    def play_annotated_video(self, trackname):
        assert trackname in self.track_details, f'track {trackname} is not found in {self.root}'

        currtrack_details = self.track_details[trackname]
        valid_files_map = Counter(currtrack_details['valid_file_names'])

        allimgs = []
        # collect triangle-drawn images
        for imgname in tqdm(currtrack_details['all_file_names']):

            # print(img.shape)
            if imgname in valid_files_map:
                # print(f"plotting {imgname}")
                image_id = currtrack_details['imgname2id'][imgname]
                image_details = currtrack_details['id2details'][image_id]

                # image_details --> valid_seg_bbox contains all the valid triangles
                valid_triangles = image_details['valid_seg_bbox']
                all_coords = []
                for triangle in valid_triangles:
                    coords = np.array(triangle['seg'], dtype=np.int32).astype(np.int32)
                    coords = coords.reshape(3, 2)
                    all_coords.append(coords)

                cv2.fillPoly(img, pts=all_coords, color=(255, 0, 0))

            allimgs.append(img)

        # show the video
        fig = plt.figure()
        viewer = fig.add_subplot(111)
        for img in allimgs:
            viewer.clear()      # Clears the previous image
            viewer.set_axis_off()
            viewer.imshow(img)  # Loads the new image
            fig.canvas.draw()   # Draws the image to the screen
            plt.pause(0.003)       # Delay in seconds


if __name__ == '__main__':
    ds = ValTrackDataset(root='../../data/val')
    # ds.play_annotated_video('val_1')

    fig = plt.figure()
    img_viewer = fig.add_subplot(121)
    mask_viewer = fig.add_subplot(122)

    for img, mask, _, _ in ds:
        # plot image and mask
        img_viewer.clear()      # Clears the previous image
        mask_viewer.clear()      # Clears the previous mask
        img_viewer.set_axis_off()
        mask_viewer.set_axis_off()

        img_viewer.imshow(img)  # Loads the new image
        mask_viewer.imshow(mask)  # Loads the new mask

        fig.canvas.draw()   # Draws the image to the screen
        plt.pause(0.003)       # Delay in seconds
