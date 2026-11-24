"""FCOS target assignment.  (OWNER: loss role)  -- the riskiest piece; write unit tests

TODO
- For each pyramid level, grid-cell centres in image coords.
- A cell is positive for a GT box if the centre lies inside the box (optionally a centre-sampling radius),
  and max(l,t,r,b) falls in that level's size range (e.g. P3 [0,64], P4 [64,128], P5 [128,inf]).
- If a cell matches several boxes, take the smallest-area box.
- Return per-cell class target (or background), ltrb target, positive mask.
- Test on 3 hand-made boxes: draw the positive cells and check they sit inside the boxes.
"""
