"""ResNet-34 (ImageNet weights) returning C3, C4, C5.  (OWNER: model role)

TODO
- torchvision.models.resnet34(weights=...) is the ONLY pretrained piece (approved).
- Return the stride-8 / 16 / 32 feature maps (layer2, layer3, layer4).
- Small batches: freeze BatchNorm statistics (put BN layers in eval mode) or the backbone gets noisy.
"""
