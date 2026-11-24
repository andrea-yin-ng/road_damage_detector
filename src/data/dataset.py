"""PyTorch Dataset over data/manifest.csv.  (OWNER: data role)

TODO
- __getitem__ returns (image_tensor, target) where target = {"boxes": FloatTensor[N,4] xyxy in
  RESIZED pixel coords, "labels": LongTensor[N] in {0..3}, "image_id": int}.
- Resize long side to cfg["data"]["img_size"], pad to a square, and RESIZE THE BOXES TOO.
- Augmentation: horizontal flip + light colour jitter only (no flips that change meaning).
- Use only rows where split == the requested split.
- Add a collate_fn (images differ in box count).
- Debug helper: draw boxes on 20 images and eyeball them BEFORE any training.
"""

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
