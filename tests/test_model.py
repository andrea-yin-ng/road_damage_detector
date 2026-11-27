"""Shape / sanity tests for the model.  Run: python -m pytest tests/test_model.py  (or python tests/test_model.py)"""
import math
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model.detector import RoadDamageDetector


def load(name):
    with open(f"configs/{name}.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["pretrained"] = False  # no download in tests
    return cfg


def test_shapes_and_init():
    for name in ("baseline", "strip"):
        model = RoadDamageDetector(load(name)).eval()
        out = model(torch.rand(2, 3, 640, 640))
        sizes = [80, 40, 20, 10, 5]
        for lvl, s in enumerate(sizes):
            assert out["cls"][lvl].shape == (2, 4, s, s)
            assert out["reg"][lvl].shape == (2, 4, s, s)
            assert out["ctr"][lvl].shape == (2, 1, s, s)
            assert out["points"][lvl].shape == (s * s, 2)
        assert (out["reg"][0] > 0).all()
        p = torch.sigmoid(out["cls"][0]).mean().item()
        assert abs(p - 0.01) < 0.005, p  # focal-loss bias init


def test_strip_switch():
    has = lambda m: any(type(x).__name__ == "StripBlock" for x in m.modules())
    assert not has(RoadDamageDetector(load("baseline")))
    assert has(RoadDamageDetector(load("strip")))


def test_predict_maps_back():
    model = RoadDamageDetector(load("baseline")).eval()
    res = model.predict(torch.rand(1, 3, 640, 640), scales=[0.5], orig_sizes=[(1280, 1280)],
                        score_thresh=0.0, max_det=20)
    r = res[0]
    assert r["boxes"].shape[1] == 4 and len(r["boxes"]) == len(r["scores"]) == len(r["labels"])
    assert r["boxes"].max() <= 1280 and r["boxes"].min() >= 0


def test_backward():
    model = RoadDamageDetector(load("all")).train()
    out = model(torch.rand(2, 3, 320, 320))
    loss = sum(o.mean() for o in out["cls"] + out["reg"] + out["ctr"])
    loss.backward()
    bn = [m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert all(not m.training for m in bn)  # BN stayed frozen after .train()


if __name__ == "__main__":
    for t in (test_shapes_and_init, test_strip_switch, test_predict_maps_back, test_backward):
        t(); print("ok", t.__name__)
