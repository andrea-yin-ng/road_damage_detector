"""Evaluate a checkpoint on the TEST split:  python evaluate.py --config configs/baseline.yaml --ckpt runs/run1_baseline/best.pt

TODO
- Run predict() on test images, then call src/eval/metrics.py: mAP, per-class AP, APs/APm/APl, F1@0.5, false alarms/image.
- Save runs/<name>/test_metrics.json and a figure with ~12 qualitative examples.
"""
