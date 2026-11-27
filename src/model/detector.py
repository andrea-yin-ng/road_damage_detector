"""Glue: backbone -> neck -> head, plus inference decode (boxes, scores, labels) + NMS.  (OWNER: model role)

forward(images) -> dict with per-level lists (index 0 = stride 8 ... index 4 = stride 128):
    "cls":     [B, 4, H, W] logits
    "reg":     [B, 4, H, W] ltrb distances in RESIZED-image pixels (already positive)
    "ctr":     [B, 1, H, W] centerness logits
    "strides": tuple of ints
    "points":  list of [H*W, 2] cell-centre (x, y) in resized-image pixels, (x+0.5)*stride style
The loss module should use "points" + "strides" for target assignment so it and the decode agree.

predict(images, scales, orig_sizes, ...) -> list of {"boxes","scores","labels"} in ORIGINAL image
coordinates (resize undone with the `scale` from the dataset target), so evaluation does not
depend on the input size.
"""
import torch
import torch.nn as nn
import torchvision

from .backbone import ResNet34Backbone
from .head import FCOSHead
from .neck import FPN

STRIDES = (8, 16, 32, 64, 128)


def make_points(feat_shapes, strides, device):
    """Cell centres for each level, in resized-image pixel coords. Returns list of [H*W, 2]."""
    pts = []
    for (h, w), s in zip(feat_shapes, strides):
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * s
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * s
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pts.append(torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1))
    return pts


class RoadDamageDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.num_classes = 4
        self.backbone = ResNet34Backbone(pretrained=m.get("pretrained", True))
        self.neck = FPN(self.backbone.out_channels, m.get("fpn_channels", 128),
                        strip_block=m.get("strip_block", False), strip_k=m.get("strip_k", 11))
        self.head = FCOSHead(m.get("fpn_channels", 128), self.num_classes,
                             m.get("head_convs", 2), STRIDES)
        ev = cfg.get("eval", {})
        self.nms_iou = ev.get("nms_iou", 0.6)

    def forward(self, images):
        feats = self.neck(self.backbone(images))
        cls, reg, ctr = self.head(feats)
        shapes = [f.shape[-2:] for f in feats]
        return {"cls": cls, "reg": reg, "ctr": ctr, "strides": STRIDES,
                "points": make_points(shapes, STRIDES, images.device)}

    @torch.no_grad()
    def predict(self, images, scales, orig_sizes, score_thresh=0.05, nms_iou=None,
                pre_nms_topk=1000, max_det=100):
        """scales: list of floats (target["scale"]); orig_sizes: list of (h, w)."""
        nms_iou = self.nms_iou if nms_iou is None else nms_iou
        out = self.forward(images)
        B = images.shape[0]
        results = []
        for b in range(B):
            all_boxes, all_scores, all_labels = [], [], []
            for lvl in range(len(out["cls"])):
                cls = out["cls"][lvl][b].flatten(1).t().float().sigmoid()      # [HW, C]
                ctr = out["ctr"][lvl][b].flatten(1).t().float().sigmoid()      # [HW, 1]
                reg = out["reg"][lvl][b].flatten(1).t().float()                # [HW, 4]
                pts = out["points"][lvl]
                scores = (cls * ctr).sqrt()                               # geometric mean
                flat = scores.flatten()
                keep = flat > score_thresh
                idx = keep.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                if idx.numel() > pre_nms_topk:
                    idx = idx[flat[idx].topk(pre_nms_topk).indices]
                cell, label = idx // self.num_classes, idx % self.num_classes
                p, d = pts[cell], reg[cell]
                boxes = torch.stack([p[:, 0] - d[:, 0], p[:, 1] - d[:, 1],
                                     p[:, 0] + d[:, 2], p[:, 1] + d[:, 3]], dim=1)
                all_boxes.append(boxes); all_scores.append(flat[idx]); all_labels.append(label)

            if not all_boxes:
                results.append({"boxes": images.new_zeros((0, 4)),
                                "scores": images.new_zeros((0,)),
                                "labels": torch.zeros((0,), dtype=torch.long, device=images.device)})
                continue
            boxes = torch.cat(all_boxes); scores = torch.cat(all_scores); labels = torch.cat(all_labels)
            # NMS per class. torchvision.ops.batched_nms is a utility, not a detector
            # (confirm with instructor, as noted in the original TODO).
            keep = torchvision.ops.batched_nms(boxes, scores, labels, nms_iou)[:max_det]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            # undo the resize, then clip to the original image
            h, w = orig_sizes[b]
            boxes = boxes / float(scales[b])
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
        return results
