import torch, os, sys, os.path as osp
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as tv_models
import timm
import pytorch_lightning as pl
import torch.optim as optim
import torchvision.transforms as tv_transform
import kornia as kn
import numpy as np
import matplotlib.pyplot as plt
import cv2, io

this_path = osp.split(osp.abspath(__file__))[0]
sys.path.append(this_path)
utilspath = osp.join(this_path, '..')
sys.path.append(utilspath)

from corner_matching import OptimalMatching
from einops import rearrange, repeat, reduce
from gen_utils import (cv2_warp, show_figures, print_shapes)
from tiny_selfattention_model import SelfAttentionModel
from losses import BipartiteCornerMatchingLoss

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

class ConvMask(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class TimmModel(nn.Module):
    def __init__(self, backbone, set_stride_to1=True, outlevel=2):
        super().__init__()
        self.backbone = backbone

        # instantiate the backbone
        self.feature_extractor = timm.create_model(self.backbone, pretrained=True, num_classes=0,
                                                   features_only=True, global_pool='', out_indices=(outlevel,))    
        self.last_channel_num = self.feature_extractor.feature_info.channels()[-1]

        # make stride 1
        if set_stride_to1:
            for name, mod in self.feature_extractor.named_modules():
                if isinstance(mod, nn.Conv2d):
                    mod.stride = 1

                if isinstance(mod, nn.MaxPool2d):
                    self.feature_extractor[name] = nn.Identity()        

    def forward(self, x):
        return self.feature_extractor(x)


backbone_models_hash = {
    'tiny_self_att': SelfAttentionModel,
}


class TrianglePatchSegment(nn.Module):
    def __init__(self, backbone, set_stride_to1=True, outlevel=2):
        super().__init__()
        self.backbone = backbone
        if backbone in backbone_models_hash:
            self.backbone_net = backbone_models_hash.get(backbone)()
        else:
            self.backbone_net = TimmModel(backbone=backbone, set_stride_to1=set_stride_to1, outlevel=outlevel)

        last_channel_num = self.backbone_net.last_channel_num

        # segmentation head
        self.seghead = ConvMask(last_channel_num)

        # corner map head
        self. corner_map_head = ConvMask(last_channel_num)

        # regression head
        self.coord_predictor = nn.Linear(last_channel_num, 6)
        
    def forward(self, x):
        feats = self.backbone_net(x)

        # predict the model outputs
        last_feat = feats[-1]
        seg_mask_logits = self.seghead(last_feat)
        corner_map_logits = self. corner_map_head(last_feat)

        avg_pool_feats = reduce(last_feat, 'b c h w -> b c', reduction='mean')
        coords = self.coord_predictor(avg_pool_feats)

        return seg_mask_logits, corner_map_logits, coords


class LightningTrianglePatchModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()        
        self.save_hyperparameters()
        self.args = args
        self.model = TrianglePatchSegment(backbone=self.args.backbone, set_stride_to1=not self.args.dont_set_stride, 
                                            outlevel=self.args.outlevel)
        self.seg_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.]))
        self.cornermap_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.]))
        self.regress_loss = BipartiteCornerMatchingLoss()
        self.normalize = kn.enhance.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                                                std=torch.tensor([0.229, 0.224, 0.225]))  
        self.denormalize = kn.enhance.Denormalize(mean=torch.tensor([0.485, 0.456, 0.406]), 
                                                std=torch.tensor([0.229, 0.224, 0.225]))                                                        
        self.train_step_count = 0
        self.val_step_count = 0

    def configure_optimizers(self):
        params = self.parameters()
        opimizer = optim.Adam(
            params=params, lr=self.args.lr, weight_decay=self.args.wd)
        return opimizer

    def plot_train_figure(self, imgs, tgt_seg_masks, tgt_corner_maps, tgt_triangle_coords,
                                    seg_mask_logits, corner_map_logits, coords, label='training_progress', step_index=None):

        # plot every 100 iterations 
        assert step_index is not None
        if step_index % 100 != 0: return

        b, c, h, w = imgs.shape
        imgs = imgs.cpu().detach().numpy()
        tgt_seg_masks = tgt_seg_masks.cpu().detach().numpy()
        tgt_corner_maps = tgt_corner_maps.cpu().detach().numpy()
        tgt_triangle_coords = tgt_triangle_coords.cpu().detach().numpy()
        tgt_triangle_coords = rearrange(tgt_triangle_coords, 'b (n d) -> b n d', n=3)
        tgt_triangle_coords[:,:,0] = tgt_triangle_coords[:,:,0] * w
        tgt_triangle_coords[:,:,1] = tgt_triangle_coords[:,:,1] * h
        tgt_triangle_coords = tgt_triangle_coords.astype(np.int32)

        seg_mask_logits = seg_mask_logits.sigmoid().cpu().detach().numpy()
        corner_map_logits = corner_map_logits.sigmoid().cpu().detach().numpy()
        coords = coords.cpu().detach().numpy()
        coords = rearrange(coords, 'b (n d) -> b n d', n=3)
        coords[:,:,0] = coords[:,:,0] * w
        coords[:,:,1] = coords[:,:,1] * h
        coords = coords.astype(np.int32)

        def plot_points(points):
            img = np.zeros((h, w))
            cv2.circle(img, points[0], radius=1, color=(1,), thickness=-1)
            cv2.circle(img, points[1], radius=1, color=(1,), thickness=-1)
            cv2.circle(img, points[2], radius=1, color=(1,), thickness=-1)
            return img

        fig, axes = plt.subplots(nrows=4, ncols=7, figsize=(12, 8))
        count = 0
        for i, (img, tgtseg, tgtcorner, tgtcoord, predseg, predcorner, predcoord) in enumerate(zip(imgs, tgt_seg_masks, tgt_corner_maps, tgt_triangle_coords,
                                                                                    seg_mask_logits, corner_map_logits, coords)):
            img = np.transpose(img, (1,2,0)) # h, w, 3
            predseg = predseg[0]       # s, s
            predcorner = predcorner[0] # s, s

            axes[i,0].imshow(img / 255.)
            axes[i,0].set_title("RGB")

            axes[i,1].imshow(tgtseg)
            axes[i,1].set_title("tgt_seg")

            axes[i,2].imshow(tgtcorner)
            axes[i,2].set_title("tgt_corner")

            axes[i,3].imshow(plot_points(tgtcoord))
            axes[i,3].set_title("tgt_coords")

            axes[i,4].imshow(predseg)
            axes[i,4].set_title("pred_seg")

            axes[i,5].imshow(predcorner)
            axes[i,5].set_title("pred_corner")

            axes[i,6].imshow(plot_points(predcoord))
            axes[i,6].set_title("pred_coords")

            count += 1
            if count >= 4: break

        # get np frame from fig
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        data = get_img_from_fig(fig)
        data = np.transpose(data, (2,0,1))
        plt.close('all')

        self.logger.experiment.add_image(label, data, step_index)

    def parse_train_batch(self, batch_data):
        res_triangles, res_seg_mask, res_corner_map, res_triangle_coords, res_rects = batch_data
        imgs = rearrange(res_triangles, 'b sb c h w -> (b sb) c h w').float()
        tgt_seg_masks = rearrange(res_seg_mask, 'b sb h w -> (b sb) h w').float()
        tgt_corner_maps = rearrange(res_corner_map, 'b sb h w -> (b sb) h w').float()
        tgt_triangle_coords = rearrange(res_triangle_coords, 'b sb x y -> (b sb) x y').float()
        tgt_rects = rearrange(res_rects, 'b sb x y -> (b sb) x y').float()
        return imgs, tgt_seg_masks, tgt_corner_maps, tgt_triangle_coords, tgt_rects

    def training_step(self, train_batch, batch_idx):
        imgs, tgt_seg_masks, tgt_corner_maps, tgt_triangle_coords, tgt_rects = self.parse_train_batch(train_batch)

        # normalize data
        bsb, c, h, w = imgs.shape
        tgt_triangle_coords[:, :, 0] /= w
        tgt_triangle_coords[:, :, 1] /= h
        tgt_rects[:, :, 0] /= w
        tgt_rects[:, :, 1] /= h

        normalized_imgs = self.normalize(imgs / 255.)
        seg_mask_logits, corner_map_logits, coords = self.forward(normalized_imgs)

        b = imgs.shape[0]
        seg_loss = self.args.lambda_seg_loss * self.seg_bce_loss(seg_mask_logits.squeeze(1), tgt_seg_masks)
        cornermap_loss = self.args.lambda_cornermap_loss * self.cornermap_bce_loss(corner_map_logits.squeeze(1), tgt_corner_maps)
        regression_loss = self.args.lambda_regress_loss * self.regress_loss(coords.view(b, -1), tgt_triangle_coords.view(b, -1))
        loss = seg_loss + cornermap_loss + regression_loss

        # prepare figure to add to tensorboard
        self.plot_train_figure(imgs, tgt_seg_masks, tgt_corner_maps, tgt_triangle_coords.view(b, -1),
                                    seg_mask_logits, corner_map_logits, coords.view(b, -1), step_index=self.train_step_count)
        self.train_step_count += 1

        # log all the losses
        self.log('train_seg_loss', seg_loss)
        self.log('train_cornermap_loss', cornermap_loss)
        self.log('train_regression_loss', regression_loss)
        self.log('train_total_loss', loss)

        return loss

    def forward(self, x):
        seg_mask_logits, corner_map_logits, coords = self.model(x)
        return seg_mask_logits, corner_map_logits, coords

    def validation_step(self, val_batch, batch_idx):
        pass

    @staticmethod
    def add_argparse_args(parser):  
        parser.add_argument('--dont_set_stride', action='store_true', help='by default, stride will be set to 1. use this flag to disable it')                                      
        parser.add_argument('--outlevel', type=int, default=2, help='out level of feature map from backbone')                                      
        parser.add_argument('--lambda_seg_loss', type=float, default=1.0, help='ratio for segmentation loss')                                      
        parser.add_argument('--lambda_cornermap_loss', type=float, default=1.0, help='ratio for corner map loss')                                      
        parser.add_argument('--lambda_regress_loss', type=float, default=0.0, help='ratio for regression loss')                                      
        return parser


class SimplePatchCornerModule(LightningTrianglePatchModule):

    # only change the 
    def parse_batch(self, batch_data):
        return self.parse_train_batch(batch_data)

    def parse_train_batch(self, batch_data):        
        imgs, tgt_seg_masks, tgt_corner_maps, tgt_triangle_coords, tgt_rects = batch_data
        imgs = imgs.float()
        tgt_seg_masks = tgt_seg_masks.float()
        tgt_corner_maps = tgt_corner_maps.float()
        tgt_triangle_coords = tgt_triangle_coords.float()
        tgt_rects  = tgt_rects.float()
        return imgs, tgt_seg_masks, tgt_corner_maps, tgt_triangle_coords, tgt_rects

    def validation_step(self, val_batch, batch_idx):
        imgs, tgt_seg_masks, tgt_corner_maps, tgt_triangle_coords, tgt_rects = self.parse_batch(val_batch)

        # normalize data
        bsb, c, h, w = imgs.shape
        tgt_triangle_coords[:, :, 0] /= w
        tgt_triangle_coords[:, :, 1] /= h
        tgt_rects[:, :, 0] /= w
        tgt_rects[:, :, 1] /= h

        normalized_imgs = self.normalize(imgs / 255.)
        seg_mask_logits, corner_map_logits, coords = self.forward(normalized_imgs)

        b = imgs.shape[0]
        seg_loss = self.args.lambda_seg_loss * self.seg_bce_loss(seg_mask_logits.squeeze(1), tgt_seg_masks)
        cornermap_loss = self.args.lambda_cornermap_loss * self.cornermap_bce_loss(corner_map_logits.squeeze(1), tgt_corner_maps)
        regression_loss = self.args.lambda_regress_loss * self.regress_loss(coords.view(b, -1), tgt_triangle_coords.view(b, -1))
        loss = seg_loss + cornermap_loss + regression_loss

        # prepare figure to add to tensorboard
        self.plot_train_figure(imgs, tgt_seg_masks, tgt_corner_maps, tgt_triangle_coords.view(b, -1),
                                    seg_mask_logits, corner_map_logits, coords.view(b, -1), label='validation_progress', step_index=self.val_step_count)

        # log all the losses
        self.log('val_seg_loss', seg_loss)
        self.log('val_cornermap_loss', cornermap_loss)
        self.log('val_regression_loss', regression_loss)
        self.log('val_total_loss', loss)
        self.val_step_count += 1
        return loss


if __name__ == "__main__":
    m = TrianglePatchSegment(backbone='tv_resnet34').cuda()
    data = torch.randn(2,3,128,128).cuda()
    seg_mask_logits, corner_map_logits, coords = m(data)
    print([x.shape for x in seg_mask_logits])
    print(seg_mask_logits.shape, corner_map_logits.shape, coords.shape)
