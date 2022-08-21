import torch, torch.nn as nn
import torch.nn.functional as F
import timm
import torch.autograd.profiler as profiler
from einops import reduce, rearrange, repeat
import numpy as np, math
import torch.nn.init as init

def split(x, num_heads=4):
    x = rearrange(x, 'b hw (nhead d) -> b nhead hw d', nhead=num_heads)
    return x

def desplit(x):
    x = rearrange(x, 'b nhead hw d -> b hw (nhead d)')
    return x
    
def self_attention(q,k,v):
    # q = nxd, k = nxd, v = nxd
    q = split(q)
    k = split(k)
    v = split(v)
    d = q.shape[-1]
    
    qk = torch.einsum('bhnd,bhmd->bhnm', q, k) / np.sqrt(d)
    qk = qk.softmax(dim=-1)
    val = torch.einsum('bhnm,bhmd->bhnd', qk, v)
    val = desplit(val)
    return val


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),            
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),            
        )

    def forward(self, x):
        x = self.conv(x)
        return x        

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


class SelfAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_encodings = nn.Parameter(torch.empty(1,64,8,8))
        init.kaiming_uniform_(self.pos_encodings, a=math.sqrt(5))

        self.init_block = ConvBlock(3, 64, stride=4)
        self.qkv_transform = nn.Sequential(
            nn.Conv2d(64, 64*3, kernel_size=1)
        )
        self.post_attention_mlp = Conv1x1(in_ch=64, out_ch=64)
        self.seg_head = ConvMask(in_ch=64)
        self.corner_map = ConvMask(in_ch=64)

    def arrange_conv_to_attention(self, q, k, v):
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b (h w) c')
        v = rearrange(v, 'b c h w -> b (h w) c')
        return q, k, v

    def arrange_attention_to_conv(self, val, h, w):
        val = rearrange(val, 'b (h w) c -> b c h w', h=h, w=w)
        return val

    def forward(self, x):
        b, _, h, w = x.shape
        out = self.init_block(x)
        # print(out.shape, self.pos_encodings.shape)
        out = out + self.pos_encodings

        b, c, dh, dw = out.shape

        # out = x # layer norm
        out = self.qkv_transform(out)
        q = out[:,:64]
        k = out[:,64:128]
        v = out[:,128:]

        q, k, v = self.arrange_conv_to_attention(q, k, v)
        val = self_attention(q, k, v)

        out = self.arrange_attention_to_conv(val, dh, dw)
        out = self.post_attention_mlp(out)

        # upsample
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        seg = self.seg_head(out)
        cornermap = self.corner_map(out)
        return seg, cornermap

if __name__ == '__main__':
    d = torch.randn(1,3,64,64)
    m = SelfAttentionModel()
    out = m(d)