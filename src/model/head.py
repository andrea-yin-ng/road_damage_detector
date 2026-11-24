"""Anchor-free FCOS-style head.  (OWNER: model role)  -- OURS, written from scratch

TODO
- Shared across levels: a small cls tower and reg tower (cfg["model"]["head_convs"] convs each; GroupNorm).
- Outputs per level: class logits [B,4,H,W], ltrb distances [B,4,H,W] (+ optional centerness [B,1,H,W]).
- Init the classification bias to -log((1-0.01)/0.01) so focal loss does not stall at the start.
- Learnable per-level scale on the regression output; exp()/relu to keep distances positive.
"""
