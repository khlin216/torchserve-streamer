import torch, torch.nn as nn
import torch.nn.functional as F
import timm
import torch.autograd.profiler as profiler
from einops import reduce, rearrange, repeat
import numpy as np, math
import torch.nn.init as init
from itertools import permutations

class BipartiteCornerMatchingLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.perm_indices = list(permutations([0,1,2]))

    def forward(self, pred, tgt):
        # calculate L1 dist
        pred = rearrange(pred, 'b (n d) -> b n d', n=3)
        tgt = rearrange(tgt, 'b (n d) -> b n d', n=3)

        # normalize target
        dist = torch.cdist(pred, tgt, p=1)

        # perform all permutations
        all_permutations = dist[:, range(3), self.perm_indices]

        # determine total cost
        all_cost = all_permutations.sum(dim=-1)

        # use the min cost as final cost
        min_cost, _ = all_cost.min(dim=-1)
        return min_cost.mean()
