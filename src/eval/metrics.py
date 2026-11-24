"""Evaluation.  (OWNER: eval role)

All metrics use predictions in ORIGINAL image coordinates and the same held-out test split for every run.

TODO
- coco_map(preds, gts): mAP@[.5:.95], AP50, per-class AP, and APs/APm/APl via pycocotools or torchmetrics.
- f1_at_iou(preds, gts, iou=0.5, score_thresh): TP/FP/FN with class match -> precision, recall, F1
  (the official challenge definition).
- false_alarms_per_image(preds, empty_image_ids, score_thresh): boxes on images with NO labelled damage,
  divided by the number of those images.
- Choose score_thresh on VAL (max F1), freeze it, apply to TEST. Say so in the report.
- Sanity tests: feed ground truth back in as predictions (expect F1 = mAP = 1.0); feed nothing (expect 0).
"""
