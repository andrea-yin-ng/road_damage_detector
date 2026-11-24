"""Train one run:  python train.py --config configs/baseline.yaml  (OWNER: everyone; keep it small)

TODO
- Load YAML, set seeds (python, numpy, torch), build data / sampler / model / loss.
- SGD + warmup, AMP, gradient clipping, per-epoch validation (mAP on val), save best + last checkpoint to runs/<name>/.
- Write runs/<name>/results.json including `git rev-parse HEAD`, the config, and the seed.
- First test: overfit 16 images. Loss should go near zero and drawn predictions should match the boxes.
"""
