import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as tv_models
import timm
import pytorch_lightning as pl
import torch.optim as optim
from .corner_matching import OptimalMatching
from einops import rearrange, repeat, reduce

# from https://github.com/zorzi-s/PolyWorldPretrainedNetwork/blob/19c847848bb7a4b2221e6ff3cfb8da627849141f/models/backbone.py


class DetectionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class NonMaxSuppression(nn.Module):
    def __init__(self, n_peaks=256):
        super().__init__()
        self.k = 3  # kernel
        self.p = 1  # padding
        self.s = 1  # stride
        self.center_idx = self.k**2//2
        self.sigmoid = nn.Sigmoid()
        self.unfold = nn.Unfold(kernel_size=self.k, padding=self.p, stride=self.s)
        self.n_peaks = n_peaks

    def sample_peaks(self, x):
        B, _, H, W = x.shape
        for b in range(B):
            x_b = x[b, 0]
            idx = torch.topk(x_b.flatten(), self.n_peaks).indices
            idx_i = torch.div(idx, W, rounding_mode='floor')
            idx_j = idx % W
            idx = torch.cat((idx_j.unsqueeze(1), idx_i.unsqueeze(1)), dim=1)
            idx = idx.unsqueeze(0)

            if b == 0:
                graph = idx
            else:
                graph = torch.cat((graph, idx), dim=0)

        return graph

    def forward(self, feat):
        B, C, H, W = feat.shape

        x = self.sigmoid(feat)

        # Prepare filter
        f = self.unfold(x).view(B, self.k**2, H, W)
        f = torch.argmax(f, dim=1).unsqueeze(1)
        f = (f == self.center_idx).float()

        # Apply filter
        x = x * f

        # Sample top peaks
        graph = self.sample_peaks(x)
        return x, graph


class TrianglePolyWorld(nn.Module):
    def __init__(self, backbone, backbone_stride):
        super().__init__()
        self.backbone = backbone
        self.backbone_stride = backbone_stride

        # instantiate the backbone
        self.feature_extractor = timm.create_model(self.backbone, pretrained=True, num_classes=0,
                                                   features_only=True, output_stride=backbone_stride, global_pool='')
        print(
            f'Feature channels: {self.feature_extractor.feature_info.channels()}')
        print(
            f'Feature reduction: {self.feature_extractor.feature_info.reduction()}')
        
        self.bottleneck = self.conv = nn.Sequential(
                                        nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True))

        # define the triangle corner predictor
        self.corner_detector = DetectionBranch()
        self.non_max_suppress = NonMaxSuppression()

        # GNN / attention module + polygon prediction
        self.polygon_match = OptimalMatching()

    def forward(self, x):
        b, _, h, w = x.shape
        feats = self.feature_extractor(x)
        last_feat = feats[-1]

        last_feat = F.interpolate(last_feat, size=(h//self.backbone_stride, w//self.backbone_stride), align_corners=True)

        print([f.shape for f in feats])
        feats = self.bottleneck(last_feat)  # consider last level feature
        corners_map_logits = self.corner_detector(feats)
        nonmaxsupp_corners_map, corner_coords = self.non_max_suppress(corners_map_logits)
        poly = self.polygon_match.predict(x, feats, corner_coords)

        return corners_map_logits, nonmaxsupp_corners_map, corner_coords, poly

class LightningTriangleModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = TrianglePolyWorld(backbone=args.backbone, backbone_stride=self.args.data_stride)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([100.]))

    def configure_optimizers(self):
        params = self.parameters()
        opimizer = optim.Adam(
            params=params, lr=self.args.lr, weight_decay=self.args.wd)
        return opimizer

    def training_step(self, train_batch, batch_idx):
        synth_img, triangle_coords, mask, strided_corner_map, strided_mask, strided_coords = train_batch

        synth_img = synth_img.float()
        triangle_coords = triangle_coords.float()
        mask = mask.float()
        strided_corner_map = strided_corner_map.float()
        strided_mask = strided_mask.float()
        strided_coords = strided_coords.float()

        corners_map_logits, nonmaxsupp_corners_map, corner_coords, poly = self.forward(synth_img)
        print(corners_map_logits.shape)
        corners_map_logits = rearrange(corners_map_logits, 'b 1 h w -> b (h w)')
        corner_map_target = rearrange(strided_corner_map, 'b h w -> b (h w)')
        loss = self.bce_loss(corners_map_logits, corner_map_target)
        self.log('train_loss', loss)
        return loss

    def forward(self, x):
        corners_map_logits, nonmaxsupp_corners_map, corner_coords, poly = self.model(x)
        return corners_map_logits, nonmaxsupp_corners_map, corner_coords, poly

    def validation_step(self, val_batch, batch_idx):
        loss = None
        img, mask, all_seg_coords, all_bbox_coords = val_batch
        self.log('val_loss', 0.)
        return loss

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()

    @staticmethod
    def add_argparse_args(parser):
        return parser


if __name__ == '__main__':
    m = TrianglePolyWorld('tv_resnet34', backbone_stride=4)
    m.cuda()
    data = torch.randn(3,3,900,900).cuda()
    m(data)