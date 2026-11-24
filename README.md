# Road Damage Detection (DATA 255, Group 4)

An object detector we design ourselves to find cracks and potholes in everyday road photos.
Given a dashcam or phone image, the model draws a box around every piece of damage and names its type.

**Team:** Andrea Ng, Caner Akoglu, Tanya Shah, Uday Patel, Vedant Vilas Vartak

## What we are building

| Part | Choice |
|---|---|
| Data | RDD2022 (six countries, four classes: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole) |
| Backbone | ResNet-34 with ImageNet weights (the only pretrained piece) |
| Neck | Feature pyramid plus our strip block (written by us) |
| Head | Anchor-free, FCOS-style, focal loss (written by us) |
| Rule | No YOLO and no ready-made detector (no `torchvision.models.detection`) |

The three problems and our fix for each:

| Problem | Fix | How we prove it |
|---|---|---|
| Thin cracks are hard to see | Strip convolutions (1xk and kx1) in the neck | Crack AP (D00, D10) goes up |
| Look-alikes (shadows, manholes, markings) cause false alarms | 25% empty-road images in every batch | False alarms per image drop |
| Rare damage is under-learned | Repeat-factor sampling | Pothole and transverse-crack AP go up |

## Experiments (one change per run, same data and seed)

| Run | Config | Strip block | Empty roads | Rare sampling |
|---|---|---|---|---|
| 1 | `configs/baseline.yaml` | no | no | no |
| 2 | `configs/strip.yaml` | **yes** | no | no |
| 3 | `configs/empty.yaml` | no | **25%** | no |
| 4 | `configs/rare.yaml` | no | no | **yes** |
| 5 | `configs/all.yaml` | yes | 25% | yes |

Run 1 trains on damage-containing images only, so Run 3 isolates the effect of adding empty roads.

## Setup

```bash
git clone https://github.com/andrea-yin-ng/road-damage-detector.git
cd road-damage-detector
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Colab: `!git clone ...` then `!pip install -r requirements.txt`. Keep the dataset on the GPU machine, not in git.

## Data

1. Download RDD2022 from [Figshare](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547) or the [GitHub repo](https://github.com/sekilab/RoadDamageDetector) and unzip it, e.g. to `/content/RDD2022`. Licence: CC BY-SA 4.0.
2. Build the manifest (everyone uses the same seed, so everyone gets the same slice and split):

```bash
python scripts/prepare_manifest.py --data-root /content/RDD2022 --out data/manifest.csv --seed 42
```

This samples a fixed slice of about 10,000 labeled images, drops `China_Drone` (its photos look straight down), splits 80/10/10 into train/val/test stratified by country, and prints:
- images per country and split
- damage boxes per class
- the fraction of empty images, which decides how Run 3 can work
- image frequency per class on the train split, needed to set the repeat-factor threshold

The official RDD2022 test labels are not public, so our test split comes from the labeled training data.
Consecutive dashcam frames can look alike, so a random split may leak slightly. This affects all runs equally and is noted as a limitation.

## Train and evaluate

```bash
python train.py --config configs/baseline.yaml
python evaluate.py --config configs/baseline.yaml --ckpt runs/run1_baseline/best.pt
```

Before any long run: overfit 16 images and check that loss goes near zero and the drawn predictions match the boxes.

## Metrics

All computed on the same held-out test split, in original image coordinates.

- **mAP@[.5:.95], AP50, per-class AP, APs/APm/APl** (COCO-style)
- **F1 at IoU 0.5** (the official challenge score). The score threshold is chosen on validation, frozen, then applied to test.
- **False alarms per image**: boxes on images with no labeled damage divided by the number of those images, at the same frozen threshold.

Note: "empty" means no labeled damage, so some of those images may contain unlabeled damage.
We also manually tag about 100 false-alarm crops (shadow, manhole, marking, tire mark, other) to show what actually causes them.

## Repo layout

```
configs/        one YAML per run (the only thing that differs between runs)
scripts/        prepare_manifest.py
src/data/       dataset, samplers (empty-road mixing, repeat-factor sampling)
src/model/      backbone, neck (FPN + strip block), head, detector
src/loss/       FCOS target assignment, focal / box losses
src/eval/       mAP, F1, false alarms per image
train.py        train one run from a config
evaluate.py     evaluate a checkpoint on the test split
data/  runs/    NOT in git (datasets, checkpoints, logs)
```

## Team workflow

- Do not push straight to `main`. Make a branch (`git checkout -b eval-script`), push it, open a pull request, and have one teammate glance at it.
- `main` must always run. `git pull` before you start each session.
- Never commit data, checkpoints, or `runs/` (they are in `.gitignore`; GitHub rejects files over 100 MB).
- Every run writes `runs/<name>/results.json` with the config, the seed, and `git rev-parse HEAD`, so any number can be traced to the exact code.

## Results

*(fill in as runs finish; same split and seed for every row)*

| Run | mAP@[.5:.95] | AP50 | F1@0.5 | D00 AP | D10 AP | D20 AP | D40 AP | False alarms / image |
|---|---|---|---|---|---|---|---|---|
| 1 Baseline | | | | | | | | |
| 2 + Strip | | | | | | | | |
| 3 + Empty roads | | | | | | | | |
| 4 + Rare sampling | | | | | | | | |
| 5 All three | | | | | | | | |

Baseline noise: run Run 1 with seeds 42, 43, 44 and report the spread. Only differences larger than that spread are claimed as real.

## Limitations

- Numbers are on our own split, so they are not directly comparable to official challenge F1 scores.
- Single slice of about 10,000 images; per-country per-class counts are too small to report (e.g. India has 68 transverse cracks in the whole training set).
- Possible leakage between neighbouring frames in a random split.
- Small compute budget, so short training schedules.

## References

- Arya et al., *RDD2022: A multi-national image dataset for automatic Road Damage Detection*, 2022. https://arxiv.org/abs/2209.08538
- Arya et al., *Crowdsensing-based Road Damage Detection Challenge (CRDDC'2022)*, 2022. https://arxiv.org/abs/2211.11362
- Maeda et al., *Road damage detection and classification using deep neural networks with smartphone images*, 2018. https://arxiv.org/abs/1801.09454
- Lin et al., *Focal Loss for Dense Object Detection*, 2017. https://arxiv.org/abs/1708.02002
- Tian et al., *FCOS: Fully Convolutional One-Stage Object Detection*, ICCV 2019. *(add link and verify)*
- Gupta et al., *LVIS: A Dataset for Large Vocabulary Instance Segmentation*, CVPR 2019 (repeat-factor sampling). *(add link and verify)*
- StripRFNet (2025), strip convolutions for road damage. *(add full citation from the report)*
