from __future__ import division, print_function, absolute_import

import os
import sys
import json
import time
import errno
import numpy as np
import random
import os.path as osp
import warnings
import PIL
import torch
from PIL import Image
import shutil
import matplotlib.pyplot as plt
import zipfile
from glob import glob
from einops import rearrange, repeat, reduce
from natsort import natsorted
import torch, torch.nn.functional as F
import torch.utils.data as data
import kornia as kn
import cv2

__all__ = [
    'mkdir_if_missing', 'check_isfile', 'read_json', 'write_json',
    'download_url', 'read_image', 'collect_env_info',
    'parse_path', 'show_image', 'unzip_file', 'load_image_in_PIL',
    'save_scripts', 'get_current_time', 'setup_log_folder', 'get_all_files'
]


def add_general_argparse_args(parser):
    parser.add_argument('--backbone', type=str,
                        default="tv_resnet34", help='model backbone')
    parser.add_argument('--lr', type=float,
                        default=3e-4, help='learning rate')
    parser.add_argument('--momentum', type=float,
                        default=3e-4, help='momentum')
    parser.add_argument('--wd', type=float,
                        default=3e-4, help='weight decay')
                            
    return parser


def split_train_val(all_instances, val_ratio=0.15, shuffle=True):
    assert isinstance(all_instances, list)

    total_instances = len(all_instances)
    total_val = int(total_instances * val_ratio)
    total_train = total_instances - total_val

    # shuffle if needed
    if shuffle: random.shuffle(all_instances)

    train_instances = all_instances[:total_train]
    val_instances = all_instances[total_train:]
    return train_instances, val_instances


def cv2_warp(img, M, size=None):
    h, w = img.shape[:2]
    if size is None: size = (w, h)
    img = cv2.warpPerspective(img, M, size)
    return img


def show_figures(figs):
    for f in figs:
        plt.figure()
        plt.imshow(f)


def print_shapes(arr):
    print([a.shape for a in arr])


def show_image(image):
    print(image.shape)
    dpi = 80
    figsize = (image.shape[1] / float(dpi), image.shape[0] / float(dpi))
    fig = plt.figure()
    plot = plt.imshow(image)
    fig.show()
    return plot


def parse_path(path):
    path = osp.abspath(path)
    parent, fullfilename = osp.split(path)
    filename, ext = osp.splitext(fullfilename)
    return parent, filename, ext

def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib
    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    with urllib.request.urlopen(url) as response, open(dst, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    sys.stdout.write('\n')


def unzip_file(filepath, destdir):
    """unzip the given zip file to destination directory.

    Args:
        filepath (str): zip file path to be extracted
        destpath (str): source path to extract the zip file contents to
    """
    print('unzipping {} to {}'.format(filepath, destdir))
    with zipfile.ZipFile(filepath,"r") as zip_ref:
        zip_ref.extractall(destdir)


def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


def collect_env_info():
    """Returns env info as a string.

    Code source: github.com/facebookresearch/maskrcnn-benchmark
    """
    from torch.utils.collect_env import get_pretty_env_info
    env_str = get_pretty_env_info()
    env_str += '\n        Pillow ({})'.format(PIL.__version__)
    return env_str


def get_current_time(f='l'):
    """get current time
    :param f: 'l' for log, 'f' for file name
    :return: formatted time
    """
    if f == 'l':
        return time.strftime('%m/%d %H:%M:%S', time.localtime(time.time()))
    elif f == 'f':
        return time.strftime('%d-%b-%y-%H:%M', time.localtime(time.time()))


def save_scripts(path, scripts_to_save=None):
    """To backup files (typically, before starting an experiment)

     usage:
        myutils.save_scripts(log_dir, scripts_to_save=glob('*.*'))
        myutils.save_scripts(log_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('model/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('myutils/*.py', recursive=True))
    """
    if not os.path.exists(os.path.join(path, 'scripts')):
        os.makedirs(os.path.join(path, 'scripts'))

    if scripts_to_save is not None:
        for script in scripts_to_save:
            dst_path = os.path.join(path, 'scripts', script)
            try:
                shutil.copy(script, dst_path)
            except IOError:
                os.makedirs(os.path.dirname(dst_path))
                shutil.copy(script, dst_path)


def load_image_in_PIL(path, mode='RGB'):
    """Read image as PIL """
    img = Image.open(path)
    img.load()  # Very important for loading large image
    return img.convert(mode)


def setup_log_folder(name, prefix_time=True, log_root='logs', copy_src_files=None):
    """setup log folder

    Args:
        name (str): name of the log folder
        prefix_time (bool, optional): whether to prepend time as prefix. Defaults to True.
        log_root (str, optional): log root folder. Defaults to 'logs'.
        copy_src_files (list or tuple, optional): list of dirs with .py files to be copied. Defaults to None.

    Returns:
        str: log folder path
    """    
    folder_name = name
    # prepend time if needed
    if prefix_time: folder_name = get_current_time(f='f') + '-' + folder_name

    log_folder = osp.join(log_root, folder_name)
    assert not osp.exists(log_folder), f"{log_folder} already exists"
    mkdir_if_missing(log_folder)

    # copy all the source files
    if copy_src_files is not None:
        assert isinstance(copy_src_files, (list, tuple))
        for src_folder in copy_src_files:
            save_scripts(log_folder, scripts_to_save=glob(f'{src_folder}/*.py', recursive=True))

    return log_folder


def collect_all_files_recurse(root):
    """collect paths of all internal files in immediate / subfolders

    Args:
        root (str): dir path

    Returns:
        list of file paths
    """
    files = []
    for filename in os.listdir(root):
        currpath = osp.join(root, filename)
        if osp.isdir(currpath):
            files += collect_all_files_recurse(currpath)
        else:
            files.append(currpath)

    return files


def get_immediate_filefolder_paths(directory : str):
    """to get paths of all files/folders inside directory

    Args:
        directory (str): directory path

    Raises:
        Exception: if directory doesnt exist

    Returns:
        filepaths, filenames
    """

    if not osp.isdir(directory):
        raise Exception(f"{directory} does not exist or not a directory")

    filenames = natsorted(os.listdir(directory))
    filepaths = [osp.join(directory, file) for file in filenames]
    return filepaths, filenames


def coco_to_crop(dir_coco, dir_dest, prefix_yolo=None, annotation_file="instances_default.json", every_n_frame=1,
                 exclude_occluded=True, p_valid=0.10, w_dest=64, h_dest=64, scale_bump=0.20, p_move_bbox_around=0.50):
    """Crop out triangles from coco.

    Usage:
    coco_to_crop(
      dir_coco="path/to/coco/root/dir",
      dir_dest="path/to/destination/root/dir",
    )

    :param dir_coco:
    :param dir_yolo:
    :param annotation_file:
    :return:
    """
    from pathlib import Path
    dir_coco = Path(dir_coco)
    dir_dest = Path(dir_dest)
    ann = json.load(open(dir_coco / "annotations" / annotation_file))
    (dir_dest / "train" / "imgs").mkdir(exist_ok=True, parents=True)
    (dir_dest / "train" / "coords").mkdir(exist_ok=True, parents=True)
    (dir_dest / "val" / "imgs").mkdir(exist_ok=True, parents=True)
    (dir_dest / "val" / "coords").mkdir(exist_ok=True, parents=True)
    if prefix_yolo is None:
        prefix_yolo = dir_coco.parts[-1] + "_"
    imgs_to_include = [img["file_name"] for i, img in enumerate(ann["images"]) if i % every_n_frame == 0]
    tri_id = [x["id"] for x in ann["categories"] if x["name"] == "triangle"][0]
    for annot in ann["annotations"]:
        # get image info
        img_info = [img for img in ann["images"] if img["id"] == annot["image_id"]]
        assert len(img_info) == 1
        img_info = img_info[0]
        fname_img_src = img_info["file_name"]
        if fname_img_src not in imgs_to_include:
            # skip this image
            continue
        if exclude_occluded and annot["attributes"].get("occluded", False):
            # skip occluded
            continue
        if len(annot["segmentation"]) != 1:
            continue
        pts = annot["segmentation"][0]
        if len(pts) != 6:
            # expected triangle
            continue

        triangle_id = annot["id"]
        fname_root, fname_ext = os.path.splitext(fname_img_src)
        fname_img_dst = prefix_yolo + fname_root + f"_{triangle_id}" + fname_ext  # ensure no overlap when merging together
        fname_lbl_dst = str(os.path.splitext(fname_img_dst)[0]) + ".json"

        # copy image over
        fpath_coco = dir_coco / "images" / fname_img_src

        if random.random() < p_valid:
            data_type = "val"
        else:
            data_type = "train"
        fpath_dest_image = dir_dest / data_type / "imgs" / fname_img_dst
        fpath_dest_label = dir_dest / data_type / "coords" / fname_lbl_dst

        x0, y0, w, h = annot["bbox"]
        x1, y1 = x0 + w, y0 + h
        img = Image.open(fpath_coco)
        w_img, h_img = img.size
        if random.random() < p_move_bbox_around:
            x0 = max(0, x0 - scale_bump * random.random() * (x1 - x0))
            y0 = max(0, y0 - scale_bump * random.random() * (y1 - y0))
            x1 = min(w_img, x1 + scale_bump * random.random() * (x1 - x0))
            y1 = min(h_img, y1 + scale_bump * random.random() * (y1 - y0))

        img = img.crop((x0, y0, x1, y1)).resize((w_dest, h_dest)).save(fpath_dest_image)

        coords_dest = []
        for x, y in np.array(pts).reshape(3, 2):
            # need to transform original coords
            # first based on crop
            x -= x0
            y -= y0
            # and then based on resize
            x *= w_dest / (x1 - x0)
            y *= h_dest / (y1 - y0)
            coords_dest.extend([x, y])
        with open(str(fpath_dest_label), "w") as fh:
            fh.write(json.dumps({"coords": [int(round(c)) for c in coords_dest]}))

