"""Build data/manifest.csv from a downloaded RDD2022 folder.

Usage:
    python scripts/prepare_manifest.py --data-root /path/to/RDD2022 --out data/manifest.csv

What it does
- Finds every Pascal-VOC .xml annotation under --data-root (the "train" folders; test images have no labels).
- Matches each .xml to its .jpg by file name, and takes the country from the top-level folder name.
- Drops excluded countries (default: China_Drone, whose photos look straight down).
- Samples a fixed slice (default 10,000 images), proportional to each country's size, with a fixed seed.
- Splits the slice 80/10/10 into train/val/test, stratified by country, with the same seed.
- Prints counts per country/split, per-class box counts, and the fraction of empty images
  (an image with an annotation file but none of the four damage classes).

Everyone running this with the same seed and the same download gets the identical slice and split.
Commit data/manifest.csv ONLY if your team agrees to (it is small); otherwise each person regenerates it.
"""
import argparse
import json
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

CLASSES = ["D00", "D10", "D20", "D40"]


def index_jpgs(root):
    """(country, stem) -> jpg path. Country = first folder under root."""
    idx = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full = os.path.join(dirpath, f)
                country = os.path.relpath(full, root).split(os.sep)[0]
                idx[(country, os.path.splitext(f)[0])] = full
    return idx


def parse_xml(path):
    tree = ET.parse(path).getroot()
    size = tree.find("size")
    w = int(float(size.findtext("width", "0"))) if size is not None else 0
    h = int(float(size.findtext("height", "0"))) if size is not None else 0
    boxes = []
    for obj in tree.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASSES:
            continue
        bb = obj.find("bndbox")
        xyxy = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            continue  # skip degenerate boxes
        boxes.append({"cls": name, "xyxy": xyxy})
    return w, h, boxes


def build_rows(root, exclude):
    jpgs = index_jpgs(root)
    rows, missing_jpg = [], 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(".xml"):
                continue
            xml_path = os.path.join(dirpath, f)
            rel = os.path.relpath(xml_path, root)
            country = rel.split(os.sep)[0]
            if country in exclude:
                continue
            stem = os.path.splitext(f)[0]
            jpg = jpgs.get((country, stem))
            if jpg is None:
                missing_jpg += 1
                continue
            try:
                w, h, boxes = parse_xml(xml_path)
            except ET.ParseError:
                continue
            row = {
                "image": os.path.relpath(jpg, root),
                "country": country,
                "width": w,
                "height": h,
                "has_damage": int(len(boxes) > 0),
                "boxes": json.dumps(boxes),
            }
            for c in CLASSES:
                row[f"n_{c}"] = sum(b["cls"] == c for b in boxes)
            rows.append(row)
    return pd.DataFrame(rows), missing_jpg


def stratified_slice(df, slice_size, rng):
    if slice_size >= len(df):
        return df.copy()
    frac = slice_size / len(df)
    parts = []
    for _, g in df.groupby("country"):
        n = max(1, round(len(g) * frac))
        idx = list(g.index)
        rng.shuffle(idx)
        parts.append(df.loc[idx[:n]])
    return pd.concat(parts)


def assign_splits(df, rng, val_frac, test_frac):
    df = df.copy()
    df["split"] = "train"
    for _, g in df.groupby("country"):
        idx = list(g.index)
        rng.shuffle(idx)
        n_val = round(len(idx) * val_frac)
        n_test = round(len(idx) * test_frac)
        df.loc[idx[:n_val], "split"] = "val"
        df.loc[idx[n_val:n_val + n_test], "split"] = "test"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--slice-size", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--exclude", nargs="*", default=["China_Drone"])
    args = ap.parse_args()

    df, missing = build_rows(args.data_root, set(args.exclude))
    if df.empty:
        raise SystemExit("No annotations found. Check --data-root (expects country folders with .xml + .jpg).")
    print(f"Labeled images found (after excluding {args.exclude}): {len(df)}   [xml without jpg: {missing}]")

    rng = random.Random(args.seed)
    df = df.sort_values("image").reset_index(drop=True)  # deterministic order before shuffling
    sl = stratified_slice(df, args.slice_size, rng)
    sl = assign_splits(sl, rng, args.val_frac, args.test_frac)
    sl = sl.sort_values("image").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sl.to_csv(args.out, index=False)

    print(f"\nSlice: {len(sl)} images -> {args.out}")
    print("\nImages per country/split:")
    print(pd.crosstab(sl["country"], sl["split"], margins=True))
    print("\nDamage boxes per class/split:")
    print(sl.groupby("split")[[f"n_{c}" for c in CLASSES]].sum())
    print("\nFraction of EMPTY images (no labeled damage) per split:")
    print((1 - sl.groupby("split")["has_damage"].mean()).round(3))
    print("\nImage frequency per class, TRAIN split (needed for repeat-factor sampling):")
    tr = sl[sl["split"] == "train"]
    for c in CLASSES:
        print(f"  {c}: {(tr[f'n_{c}'] > 0).mean():.3f}")


if __name__ == "__main__":
    main()
