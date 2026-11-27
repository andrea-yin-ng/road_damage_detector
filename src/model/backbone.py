"""ResNet-34 (ImageNet weights) returning C3, C4, C5.  (OWNER: model role)

The ONLY pretrained piece (approved). Returns the stride-8 / 16 / 32 feature maps
(layer2, layer3, layer4) with 128 / 256 / 512 channels.

Images arrive as 0..1 floats; ImageNet normalisation is done HERE (see dataset.py contract).
BatchNorm is frozen (eval mode, no grad on its affine params) because batches are small.
"""
import torch
import torch.nn as nn
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet34Backbone(nn.Module):
    out_channels = (128, 256, 512)  # C3, C4, C5

    def __init__(self, pretrained=True, freeze_bn=True):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet34(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)  # stride 4
        self.layer1 = r.layer1  # stride 4
        self.layer2 = r.layer2  # stride 8  -> C3
        self.layer3 = r.layer3  # stride 16 -> C4
        self.layer4 = r.layer4  # stride 32 -> C5
        self.freeze_bn = freeze_bn
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def train(self, mode=True):
        # model.train() would switch BN back to train mode, so re-freeze it each time.
        super().train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.layer1(self.stem(x))
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5
