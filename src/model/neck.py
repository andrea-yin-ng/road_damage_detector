"""Feature pyramid + strip block.  (OWNER: model role)  -- OURS, written from scratch

FPN: 1x1 lateral convs on C3-C5, top-down add, 3x3 smoothing -> P3, P4, P5, plus P6/P7
(stride 64/128, via stride-2 convs) so large damage has a level to live on.

StripBlock (Run 2): parallel 1xk and kx1 convs added residually to each pyramid level.
A thin crack is long in one direction and 1-2 pixels wide in the other, so a long
thin kernel can "see" it while a 3x3 kernel mostly sees background.
Switch with cfg["model"]["strip_block"].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StripBlock(nn.Module):
    """x + GN(conv_1xk(x)) + GN(conv_kx1(x)). Zero-init last norm scale so it starts as identity."""

    def __init__(self, channels, k=11, groups=32):
        super().__init__()
        g = _gn_groups(channels, groups)
        p = k // 2
        self.horiz = nn.Conv2d(channels, channels, (1, k), padding=(0, p), bias=False)
        self.vert = nn.Conv2d(channels, channels, (k, 1), padding=(p, 0), bias=False)
        self.norm_h = nn.GroupNorm(g, channels)
        self.norm_v = nn.GroupNorm(g, channels)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.norm_h.weight)
        nn.init.zeros_(self.norm_v.weight)

    def forward(self, x):
        out = self.norm_h(self.horiz(x)) + self.norm_v(self.vert(x))
        return x + self.act(out)


def _gn_groups(channels, want=32):
    g = min(want, channels)
    while channels % g:
        g -= 1
    return g


class FPN(nn.Module):
    def __init__(self, in_channels=(128, 256, 512), out_channels=128,
                 strip_block=False, strip_k=11, extra_levels=2):
        super().__init__()
        self.lateral = nn.ModuleList(nn.Conv2d(c, out_channels, 1) for c in in_channels)
        self.smooth = nn.ModuleList(nn.Conv2d(out_channels, out_channels, 3, padding=1)
                                    for _ in in_channels)
        self.extra = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            for _ in range(extra_levels))
        n_levels = len(in_channels) + extra_levels
        self.strip = nn.ModuleList(
            StripBlock(out_channels, strip_k) if strip_block else nn.Identity()
            for _ in range(n_levels))
        self.out_channels = out_channels
        self.num_levels = n_levels

    def forward(self, feats):
        c3, c4, c5 = feats
        lat = [l(c) for l, c in zip(self.lateral, (c3, c4, c5))]
        # top-down: upsample the coarser map and add it to the finer one
        p5 = lat[2]
        p4 = lat[1] + F.interpolate(p5, size=lat[1].shape[-2:], mode="nearest")
        p3 = lat[0] + F.interpolate(p4, size=lat[0].shape[-2:], mode="nearest")
        outs = [s(p) for s, p in zip(self.smooth, (p3, p4, p5))]
        x = outs[-1]
        for conv in self.extra:
            x = conv(F.relu(x))
            outs.append(x)
        return [s(o) for s, o in zip(self.strip, outs)]
