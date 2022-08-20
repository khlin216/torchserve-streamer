import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from .train_dataset import SynthTriangleDataset, SynthTrianglePatchDataset
from .val_dataset import ValTrackDataset
from .simple_patch_dataset import SimplePatchDataModule
import coord_utils

class TriangleDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        train_transform = None
        test_transform = None

        self.train = SynthTriangleDataset(root=self.args.trainroot, stride=self.args.data_stride,
                                            num_wtiles=self.args.num_wtiles, num_htiles=self.args.num_htiles)
        self.val = ValTrackDataset(root=self.args.valroot)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.train_batch, drop_last=True,
                          shuffle=True, num_workers=self.args.workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.args.val_batch,
                          num_workers=self.args.workers)

    def test_dataloader(self):
        pass

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--trainroot", type=str,
                            default=None, help="training data root folder")
        parser.add_argument("--valroot", type=str, default=None,
                            help="validation data root folder")
        parser.add_argument("--height", type=int,
                            default=900, help="image height")
        parser.add_argument("--width", type=int,
                            default=900, help="image width")
        parser.add_argument("--data_stride", type=int,
                            default=8, help="label downsampling ratio") 
        parser.add_argument("--num_htiles", type=int,
                            default=2, help="num patches in height dimension")                                                        
        parser.add_argument("--num_wtiles", type=int,
                            default=2, help="num patches in width dimension")                                                        
        parser.add_argument("--workers", type=int, default=4,
                            help="num workers to load the data")
        parser.add_argument("--train_batch", type=int,
                            default=16, help="train batch size")
        parser.add_argument("--val_batch", type=int,
                            default=16, help="val/test batch size")
        parser.add_argument("--return_strided_outputs", action="store_true",
                            help="whether to return strided output")
        return parser


class TrianglePatchesDataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        train_transform = None
        test_transform = None

        self.train = SynthTrianglePatchDataset(root=self.args.trainroot, stride=self.args.data_stride,
                                            num_wtiles=self.args.num_wtiles, num_htiles=self.args.num_htiles,
                                            return_strided_outputs=self.args.return_strided_outputs, 
                                            triangle_patch_size=self.args.triangle_patch_size)
        self.val = ValTrackDataset(root=self.args.valroot)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.args.train_batch, 
                            drop_last=True, shuffle=True, num_workers=self.args.workers)

    def val_dataloader(self):
        return self.val


    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--triangle_patch_size", type=int, default=64,
                            help="size of triangle patch")
        return parser