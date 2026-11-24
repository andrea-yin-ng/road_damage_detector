"""Samplers for Run 3 (empty-road mixing), Run 4 (repeat-factor sampling) and Run 5 (both).
(OWNER: data role)

Pools
    damage pool : train images that contain at least one box
    empty pool  : train images with no boxes (unlabeled roads, shadows, manholes ...)

Run 1 (baseline) / Run 2 (strip) : dataset built with only_damage=True, plain shuffle.
Run 3 (empty)   : dataset built with only_damage=False. Every batch = 75% damage + 25% empty.
Run 4 (rare)    : only_damage=True. Damage images are repeated by repeat-factor sampling.
Run 5 (all)     : only_damage=False. Damage pool uses repeat-factor sampling AND every batch
                  gets 25% empty images.

Repeat-factor sampling (Gupta et al. 2019, LVIS)
    f_c  = fraction of DAMAGE images that contain class c
    r_c  = max(1, sqrt(t / f_c))
    r_img = max r_c over the classes present in the image
    Each epoch an image is used floor(r_img) times, plus one more time with probability
    frac(r_img) (stochastic rounding, as in LVIS).
Frequencies are computed from the pool the sampler is given, so the same code is correct for
whichever pool you use; only the threshold t changes (see configs).

Use build_train_loader(cfg) to get the right DataLoader for any of the five configs.
Call loader.batch_sampler.set_epoch(e) / sampler.set_epoch(e) each epoch (optional; the
samplers also advance their own epoch counter every time they are iterated).
"""
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from src.data.dataset import CLASS_NAMES, build_dataset, collate_fn

COUNT_COLS = [f"n_{c}" for c in CLASS_NAMES]


def class_image_frequencies(df):
    """Fraction of the images in df that contain each class -> np.array[4]."""
    present = (df[COUNT_COLS].to_numpy() > 0)
    return present.mean(axis=0)


def repeat_factors(df, threshold):
    """Per-image repeat factor for every row of df (df = the damage pool).
    Returns (image_factors np.array[len(df)], class_factors np.array[4], f_c np.array[4])."""
    f = class_image_frequencies(df)
    r_c = np.maximum(1.0, np.sqrt(threshold / np.maximum(f, 1e-12)))
    present = (df[COUNT_COLS].to_numpy() > 0)
    r_img = np.where(present, r_c[None, :], 1.0).max(axis=1)
    return r_img, r_c, f


def _pools(dataset):
    has = dataset.df["has_damage"].astype(bool).to_numpy()
    return np.flatnonzero(has), np.flatnonzero(~has)


class _DamagePool:
    """Yields damage-image indices for one epoch, with optional repeat-factor repetition."""

    def __init__(self, dataset, rfs_threshold=None):
        self.idx, _ = _pools(dataset)
        if len(self.idx) == 0:
            raise ValueError("no damage images in this dataset")
        self.rfs_threshold = rfs_threshold
        if rfs_threshold is not None:
            self.r_img, self.r_c, self.f_c = repeat_factors(dataset.df.iloc[self.idx], rfs_threshold)
        else:
            self.r_img = np.ones(len(self.idx))

    def expected_len(self):
        return int(round(self.r_img.sum()))

    def epoch_indices(self, rng):
        base = np.floor(self.r_img).astype(int)
        extra = (rng.random(len(self.idx)) < (self.r_img - base)).astype(int)
        reps = base + extra
        out = np.repeat(self.idx, reps)
        rng.shuffle(out)
        return out


class RFSSampler(Sampler):
    """Plain (non-batch) sampler for Run 4: damage images only, repeat-factor sampled."""

    def __init__(self, dataset, threshold, seed=42):
        self.pool = _DamagePool(dataset, threshold)
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.pool.expected_len()

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        return iter(self.pool.epoch_indices(rng).tolist())


class EmptyMixBatchSampler:
    """Batch sampler for Run 3 / Run 5: each batch holds round(batch_size * empty_fraction)
    empty-road images and the rest damage images. One epoch = one pass over the damage pool
    (repeat-factor sampled if rfs_threshold is given); empty images are drawn by cycling
    through shuffled passes of the empty pool."""

    def __init__(self, dataset, batch_size, empty_fraction=0.25, rfs_threshold=None,
                 seed=42, drop_last=True):
        self.pool = _DamagePool(dataset, rfs_threshold)
        _, self.empty_idx = _pools(dataset)
        if len(self.empty_idx) == 0:
            raise ValueError("empty_fraction > 0 but the dataset has no empty images "
                             "(build it with only_damage=False)")
        self.bs = batch_size
        self.n_empty = int(round(batch_size * empty_fraction))
        self.n_damage = batch_size - self.n_empty
        if self.n_damage < 1:
            raise ValueError("empty_fraction too large for this batch size")
        self.drop_last, self.seed, self.epoch = drop_last, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        n = self.pool.expected_len()
        return n // self.n_damage if self.drop_last else math.ceil(n / self.n_damage)

    def _empty_stream(self, rng):
        while True:
            yield from rng.permutation(self.empty_idx)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        dmg = self.pool.epoch_indices(rng)
        empties = self._empty_stream(rng)
        n_batches = len(dmg) // self.n_damage if self.drop_last else math.ceil(len(dmg) / self.n_damage)
        for b in range(n_batches):
            d = dmg[b * self.n_damage:(b + 1) * self.n_damage].tolist()
            e = [int(next(empties)) for _ in range(self.n_empty)]
            batch = d + e
            rng.shuffle(batch)
            yield batch


def build_train_loader(cfg, split="train"):
    """Returns (loader, dataset). Chooses the dataset pool and sampler from cfg['sampling']."""
    s = cfg["sampling"]
    empty_fraction = float(s.get("empty_fraction", 0.0) or 0.0)
    use_rfs = bool(s.get("repeat_factor", False))
    thr = s.get("rfs_threshold")
    if use_rfs and thr is None:
        raise ValueError("repeat_factor is true but rfs_threshold is null; set it in the config")
    thr = float(thr) if use_rfs else None

    ds = build_dataset(cfg, split, only_damage=(empty_fraction == 0.0))
    t, d = cfg["train"], cfg["data"]
    common = dict(collate_fn=collate_fn, num_workers=d.get("num_workers", 0), pin_memory=True)
    seed = cfg.get("seed", 42)

    if empty_fraction > 0:
        bs = EmptyMixBatchSampler(ds, t["batch_size"], empty_fraction, thr, seed)
        loader = DataLoader(ds, batch_sampler=bs, **common)
    elif use_rfs:
        loader = DataLoader(ds, batch_size=t["batch_size"], sampler=RFSSampler(ds, thr, seed),
                            drop_last=True, **common)
    else:
        g = torch.Generator().manual_seed(seed)
        loader = DataLoader(ds, batch_size=t["batch_size"], shuffle=True, generator=g,
                            drop_last=True, **common)
    return loader, ds


def describe(cfg):
    """Print what the sampler will do (run this once per config and eyeball it)."""
    loader, ds = build_train_loader(cfg)
    n_dmg = int(ds.df["has_damage"].astype(bool).sum())
    print(f"images in dataset: {len(ds)}  (damage {n_dmg}, empty {len(ds) - n_dmg})")
    print(f"batches per epoch: {len(loader)}")
    thr = cfg["sampling"].get("rfs_threshold")
    if cfg["sampling"].get("repeat_factor"):
        dmg = ds.df[ds.df["has_damage"].astype(bool)]
        r_img, r_c, f = repeat_factors(dmg, float(thr))
        for c, fc, rc in zip(CLASS_NAMES, f, r_c):
            print(f"  {c}: f={fc:.3f}  r={rc:.2f}")
        print(f"damage images per epoch after repeats: {r_img.sum():.0f} (was {len(dmg)})")
    x, tg = next(iter(loader))
    n_empty = sum(len(t["boxes"]) == 0 for t in tg)
    print(f"first batch: images {tuple(x.shape)}, empty images in batch: {n_empty}/{len(tg)}")


if __name__ == "__main__":
    import argparse
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    a = ap.parse_args()
    describe(yaml.safe_load(open(a.config)))
