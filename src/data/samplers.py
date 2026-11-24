"""Batch sampling for the Run 3 / Run 4 fixes.  (OWNER: data role)

TODO
- Empty-road mixing: each batch = (1 - empty_fraction) damage images + empty_fraction empty images.
  Baseline uses empty_fraction = 0 (damage images only).
- Repeat-factor sampling (Gupta et al., LVIS 2019):
    f(c)  = fraction of TRAIN images containing class c
    r(c)  = max(1, sqrt(t / f(c)))
    r(img)= max over the classes present in the image
  NOTE: LVIS's t (0.001) would make this a no-op here, since all four classes appear in many images.
  Compute f(c) from the manifest, then choose t so potholes (D40) get r about 2-3. Log f(c), t, r(c).
- Apply repeat factors among damage images only, THEN mix in empty images.
"""
