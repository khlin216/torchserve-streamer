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

this_path = osp.split(osp.abspath(__file__))[0]
datautils_path = osp.join(this_path, 'data_utils')
sys.path.append(datautils_path)

from gen_utils import (show_figures, print_shapes, add_general_argparse_args)


def main():
    parser = argparse.ArgumentParser(description="Triangle detection")
    parser = pl.Trainer.add_argparse_args(parser)

    parser = add_general_argparse_args(parser)
    parser = SimplePatchDataModule.add_argparse_args(parser)
    parser = SimplePatchCornerModule.add_argparse_args(parser)
    args = parser.parse_args()

    print(args)

    m = SimplePatchCornerModule(args)
    data_module = SimplePatchDataModule(args)

    # trainer creation
    model_checkpoint = ModelCheckpoint(save_top_k=-1, every_n_train_steps=1000)
    lr_logger = LearningRateMonitor(logging_interval='step')
    callbacks = [model_checkpoint, lr_logger]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(m, data_module)    


if __name__ == '__main__':
    main()

