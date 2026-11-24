"""Feature pyramid + strip block.  (OWNER: model role)  -- OURS, written from scratch

TODO
- FPN: 1x1 lateral convs on C3-C5, top-down add, 3x3 smoothing -> P3, P4, P5 (optionally P6).
- StripBlock (Run 2): parallel 1xk and kx1 convs (k = cfg["model"]["strip_k"]) added residually
  to each pyramid level. Must be switchable with cfg["model"]["strip_block"].
- Unit test: output shapes for a 640x640 input.
"""
