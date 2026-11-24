"""Training losses.  (OWNER: loss role)

TODO
- Classification: sigmoid focal loss (alpha, gamma from cfg["loss"]); normalise by number of POSITIVE cells.
- Box: GIoU (or IoU) loss on positive cells.
- Optional centerness: BCE on positive cells.
- torchvision.ops.sigmoid_focal_loss / generalized_box_iou_loss are utilities; confirm with your instructor.
"""
