import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class RoadDamageDataset(Dataset):
    def __init__(self, manifest, data_root, split, img_size=640, train=None,
                 only_damage=False, hflip_p=0.5, color_jitter=True):
        """
        manifest   : path to data/manifest.csv (or a DataFrame)
        data_root  : folder that the manifest's `image` column is relative to
                     (the same folder you passed as --data-root to prepare_manifest.py)
        split      : "train" | "val" | "test"
        train      : turn augmentation on; defaults to (split == "train")
        only_damage: keep only images that contain at least one box (baseline run)
        """
        df = pd.read_csv(manifest) if isinstance(manifest, str) else manifest
        df = df[df["split"] == split]
        if only_damage:
            df = df[df["has_damage"].astype(bool)]
        self.df = df.reset_index(drop=True)
        self.orig_index = df.index.to_numpy()  # row ids in the full manifest
        self.data_root = data_root
        self.img_size = img_size
        self.train = (split == "train") if train is None else train
        self.hflip_p = hflip_p
        self.color_jitter = color_jitter

    def __len__(self):
        return len(self.df)

    def _load_boxes(self, row):
        items = json.loads(row["boxes"]) if isinstance(row["boxes"], str) else []
        if not items:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        boxes = np.array([b["xyxy"] for b in items], np.float32).reshape(-1, 4)
        labels = np.array([CLASS_TO_IDX[b["cls"]] for b in items], np.int64)
        return boxes, labels

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.data_root, row["image"])).convert("RGB")
        w0, h0 = img.size
        boxes, labels = self._load_boxes(row)

        # resize long side to img_size, scale boxes with the image
        scale = self.img_size / max(w0, h0)
        nw, nh = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        boxes = boxes * scale

        # light colour jitter (training only)
        if self.train and self.color_jitter:
            arr = np.asarray(img, np.float32)
            arr = arr * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        # pad bottom/right to a square canvas (grey fill)
        canvas = Image.new("RGB", (self.img_size, self.img_size), (114, 114, 114))
        canvas.paste(img, (0, 0))
        x = torch.from_numpy(np.asarray(canvas, np.uint8).copy()).permute(2, 0, 1).float() / 255.0

        # horizontal flip of the whole padded canvas (keeps boxes consistent)
        if self.train and np.random.rand() < self.hflip_p:
            x = x.flip(-1)
            if len(boxes):
                boxes = np.stack([self.img_size - boxes[:, 2], boxes[:, 1],
                                  self.img_size - boxes[:, 0], boxes[:, 3]], 1)

        # clip to the image area and drop degenerate boxes
        if len(boxes):
            boxes = np.clip(boxes, 0, self.img_size)
            keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            boxes, labels = boxes[keep], labels[keep]

        target = {
            "boxes": torch.from_numpy(boxes.astype(np.float32)).reshape(-1, 4),
            "labels": torch.from_numpy(labels.astype(np.int64)),
            "image_id": int(self.orig_index[i]),
            "scale": float(scale),
            "orig_size": (h0, w0),
        }
        return x, target


def collate_fn(batch):
    """Stack images; keep targets as a list (box counts differ per image)."""
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)


def build_dataset(cfg, split, only_damage=False):
    d = cfg["data"]
    return RoadDamageDataset(d["manifest"], d.get("root", "data/RDD2022"), split,
                             img_size=d["img_size"], only_damage=only_damage)


def draw_samples(ds, out_dir="runs/debug_boxes", n=20):
    """Debug helper: save n images with boxes drawn. Look at these before training."""
    from PIL import ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    colors = ["red", "yellow", "cyan", "magenta"]
    for k in range(min(n, len(ds))):
        x, t = ds[k]
        im = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        for b, l in zip(t["boxes"].tolist(), t["labels"].tolist()):
            dr.rectangle(b, outline=colors[l], width=2)
            dr.text((b[0] + 2, b[1] + 2), CLASS_NAMES[l], fill=colors[l])
        im.save(os.path.join(out_dir, f"{k:02d}_id{t['image_id']}.jpg"))
    print(f"saved {min(n, len(ds))} images to {out_dir}")


if __name__ == "__main__":
    import argparse
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--only-damage", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ds = build_dataset(cfg, a.split, a.only_damage)
    print(f"{a.split}: {len(ds)} images")
    draw_samples(ds, n=a.n)
