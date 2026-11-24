"""Glue: backbone -> neck -> head, plus inference decode (boxes, scores, labels) + NMS.  (OWNER: model role)

TODO
- forward(images) -> raw per-level outputs for training.
- predict(images, score_thresh, nms_iou) -> list of {"boxes","scores","labels"} in ORIGINAL image
  coordinates (undo the resize) so evaluation is not affected by the input size.
- NMS: torchvision.ops.batched_nms is a utility, not a detector (confirm with your instructor).
"""
