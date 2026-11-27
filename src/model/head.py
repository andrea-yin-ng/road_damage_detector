"""Anchor-free FCOS-style head.  (OWNER: model role)  -- OURS, written from scratch

Shared across pyramid levels: a small cls tower and reg tower (cfg["model"]["head_convs"]
3x3 convs each, GroupNorm + ReLU).

Per level outputs:
    cls   [B, num_classes, H, W]  class logits (sigmoid applied in the loss / decode)
    reg   [B, 4, H, W]            ltrb distances in IMAGE pixels (>0): left, top, right, bottom
                                  from the cell centre to the box sides
    ctr   [B, 1, H, W]            centerness logit

Distances are exp(scale_l * raw) * stride_l, with a learnable scale per level, so every level
starts predicting "about one cell" and stays positive.
"""
import math

import torch
import torch.nn as nn

from .neck import _gn_groups


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return x * self.scale


class FCOSHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=4, num_convs=2,
                 strides=(8, 16, 32, 64, 128)):
        super().__init__()
        self.strides = strides
        g = _gn_groups(in_channels)

        def tower():
            layers = []
            for _ in range(num_convs):
                layers += [nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                           nn.GroupNorm(g, in_channels), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.reg_tower = tower()
        self.cls_out = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.reg_out = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.ctr_out = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList(Scale(1.0) for _ in strides)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Start every cell at p = 0.01 "damage" so focal loss does not collapse on the
        # huge number of background cells at step 0.
        nn.init.constant_(self.cls_out.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, feats):
        cls_l, reg_l, ctr_l = [], [], []
        for feat, scale, stride in zip(feats, self.scales, self.strides):
            c = self.cls_tower(feat)
            r = self.reg_tower(feat)
            cls_l.append(self.cls_out(c))
            # clamp the exponent to avoid inf under AMP
            reg_l.append(torch.exp(scale(self.reg_out(r)).clamp(-8, 8)) * stride)
            ctr_l.append(self.ctr_out(r))
        return cls_l, reg_l, ctr_l
