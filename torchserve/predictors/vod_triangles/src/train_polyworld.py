import torch, os, sys, os.path as osp
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as tv_models
import timm
import pytorch_lightning as pl
import torch.optim as optim
from models import LightningTriangleModule
from data_utils import TriangleDataModule, TrianglePatchesDataModule
import argparse

this_path = osp.split(osp.abspath(__file__))[0]
datautils_path = osp.join(this_path, 'data_utils')
sys.path.append(datautils_path)

def main():
    parser = argparse.ArgumentParser(description="Triangle detection")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TriangleDataModule.add_argparse_args(parser)
    parser = LightningTriangleModule.add_argparse_args(parser)
    args = parser.parse_args()

    m = LightningTriangleModule(args)
    data_module = TriangleDataModule(args)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(m, data_module)


if __name__ == '__main__':
    main()
