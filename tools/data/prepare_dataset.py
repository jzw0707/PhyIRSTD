#!/usr/bin/env python3
"""
Prepare TSIRMT, NUDT-MIRSTD or PhyIRSTD sequences for DTA-SAM.

Expected source layout (under --src_root):

    images/<seq_id>/<frame>.{jpg,png,...}
    masks/<seq_id>/<frame>.{png,jpg,...}   # binary or grayscale, same stem as image
Output (--out_root), compatible with datasets/infrared.py:

    train/JPEGImages/<seq_id>/<frame>.jpg
    train/Annotations/<seq_id>/<frame>.png   # palette-style: fg pixel value == obj_id (default 1)
    train/meta.json
    meta_expressions/train/meta_expressions.json

Split policy (first match wins):
  1) ImageSets/val.txt: valid/ contains those ids and train/ contains the remainder.
  2) ImageSets/train.txt and ImageSets/test.txt: use the two lists directly.
  3) --val_ratio > 0 (only if neither 1 nor 2 applies): random disjoint valid/train.
  4) Otherwise: only train/ (all sequences).

The generated expression records contain only an object id. No text is required.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _read_imageset_lines(txt_path: Path) -> list[str]:
    """Ordered sequence ids; empty lines and # comments skipped."""
    names: list[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            names.append(s)
    return names


def load_val_sequences_from_imagesets(src_root: Path) -> list[str] | None:
    """Read ImageSets/val.txt if present; return ordered sequence names, else None."""
    val_file = src_root / "ImageSets" / "val.txt"
    if not val_file.is_file():
        return None
    return _read_imageset_lines(val_file)


def load_train_test_split_from_imagesets(
    src_root: Path, all_seqs: list[str], seq_set: set[str]
) -> tuple[list[str], list[str]] | None:
    """Use ImageSets/train.txt and test.txt as disjoint train/validation splits."""
    train_file = src_root / "ImageSets" / "train.txt"
    test_file = src_root / "ImageSets" / "test.txt"
    if not train_file.is_file() or not test_file.is_file():
        return None

    train_lines = _read_imageset_lines(train_file)
    test_lines = _read_imageset_lines(test_file)

    train_seqs = [s for s in train_lines if s in seq_set]
    val_seqs = [s for s in test_lines if s in seq_set]

    missing_test = [s for s in test_lines if s not in seq_set]
    if missing_test:
        print(
            f"Warning: {len(missing_test)} sequence(s) in ImageSets/test.txt not under images/: "
            f"{missing_test[:5]}{'...' if len(missing_test) > 5 else ''}"
        )

    missing_train = [s for s in train_lines if s not in seq_set]
    if missing_train:
        print(
            f"Warning: {len(missing_train)} sequence(s) in ImageSets/train.txt not under images/: "
            f"{missing_train[:5]}{'...' if len(missing_train) > 5 else ''}"
        )

    overlap = set(train_seqs) & set(val_seqs)
    if overlap:
        raise ValueError(f"ImageSets train/test overlap: {sorted(overlap)[:10]}")
    return train_seqs, val_seqs


def list_frames(images_dir: Path) -> list[tuple[str, Path]]:
    """Return sorted list of (stem, path) for image files."""
    pairs = []
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            pairs.append((p.stem, p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def find_mask_path(masks_dir: Path, stem: str) -> Path | None:
    for ext in MASK_EXTS:
        cand = masks_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def mask_to_instance_png(mask_path: Path, out_path: Path, obj_id: int = 1, fg_threshold: int = 127):
    """Load user mask, binarize, save uint8 PNG with fg == obj_id (YTVOS convention).

    Pixel values are 0 (bg) and obj_id (default 1). Viewers show this as almost all black
    because 1/255 is imperceptible — that is expected; YTVOS uses mask == obj_id, not 0/255.

    If source is already 0/1, we use fg = (arr > 0). If max > 1, we use arr > fg_threshold
    so 0/255 masks still work. (Using only >127 on 0/1 masks would wipe all foreground.)
    """
    m = Image.open(mask_path)
    if m.mode not in ("L", "1", "P", "RGB"):
        m = m.convert("L")
    arr = np.array(m)
    if arr.ndim == 3:
        arr = arr[..., 0]
    mx = int(arr.max()) if arr.size else 0
    if mx <= 1:
        fg = arr > 0
    else:
        fg = arr > fg_threshold
    out = np.zeros_like(arr, dtype=np.uint8)
    out[fg] = obj_id
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(out_path)


def ensure_jpeg(src: Path, dst: Path, quality: int = 95):
    im = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, "JPEG", quality=quality)


def write_meta_json(video_ids: list[str], out_train_dir: Path, category: str, obj_id: int):
    """meta.json: category name must exist in datasets/categories.py ytvos_category_dict (e.g. 'others' or 'targets')."""
    oid = str(obj_id)
    videos = {}
    for vid in video_ids:
        videos[vid] = {"objects": {oid: {"category": category}}}
    meta = {"videos": videos}
    out_train_dir.mkdir(parents=True, exist_ok=True)
    with open(out_train_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def write_meta_expressions(
    video_ids: list[str],
    frames_map: dict[str, list[str]],
    out_path: Path,
    obj_id: int,
):
    """obj_id must be string in JSON so vid_meta['objects'][obj_id] matches meta.json keys (see ytvos.prepare_metas)."""
    oid = str(obj_id)
    videos = {}
    for vid in video_ids:
        videos[vid] = {
            "frames": frames_map[vid],
            "expressions": {"0": {"obj_id": oid}},
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"videos": videos}, f, indent=2)


def convert_split(
    seq_ids: list[str],
    src_root: Path,
    out_split_dir: Path,
    meta_exp_path: Path,
    obj_id: int,
    fg_threshold: int,
    category: str,
):
    jpeg_root = out_split_dir / "JPEGImages"
    anno_root = out_split_dir / "Annotations"
    images_src = src_root / "images"
    masks_src = src_root / "masks"

    frames_map: dict[str, list[str]] = {}

    for seq in seq_ids:
        idir = images_src / seq
        mdir = masks_src / seq
        if not idir.is_dir():
            raise FileNotFoundError(f"Missing images dir: {idir}")
        if not mdir.is_dir():
            raise FileNotFoundError(f"Missing masks dir: {mdir}")

        frames = list_frames(idir)
        if not frames:
            raise RuntimeError(f"No images in {idir}")
        stems = [s for s, _ in frames]

        frames_map[seq] = stems

        for stem, ipath in frames:
            mpath = find_mask_path(mdir, stem)
            if mpath is None:
                raise FileNotFoundError(f"No mask for sequence {seq} stem {stem} under {mdir}")

            ensure_jpeg(ipath, jpeg_root / seq / f"{stem}.jpg")
            mask_to_instance_png(mpath, anno_root / seq / f"{stem}.png", obj_id=obj_id, fg_threshold=fg_threshold)

    write_meta_json(seq_ids, out_split_dir, category, obj_id)
    meta_exp_path.parent.mkdir(parents=True, exist_ok=True)
    write_meta_expressions(seq_ids, frames_map, meta_exp_path, obj_id)


def main():
    ap = argparse.ArgumentParser(description="TSIRMT -> Ref-Youtube-VOS layout for DTASAM")
    ap.add_argument("--src_root", type=str, required=True, help="Dataset root containing images/ and masks/")
    ap.add_argument("--out_root", type=str, required=True, help="Output root")
    ap.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="If ImageSets/val.txt and train+test.txt are absent: fraction for valid/. Ignored when val.txt or train+test.txt exists.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--obj_id", type=int, default=1, help="Instance id encoded in Annotations PNG (YTVOS uses int)")
    ap.add_argument("--fg_threshold", type=int, default=127, help="Mask binarization threshold for uint8 inputs")
    ap.add_argument(
        "--category",
        type=str,
        default="targets",
        help="DAVIS/YTVOS category name; must be in ytvos_category_dict (default targets)",
    )
    args = ap.parse_args()

    src = Path(args.src_root).resolve()
    out = Path(args.out_root).resolve()
    all_seqs = sorted([p.name for p in (src / "images").iterdir() if p.is_dir()])
    if not all_seqs:
        raise RuntimeError(f"No sequence folders under {src / 'images'}")

    seq_set = set(all_seqs)
    val_from_imagesets = load_val_sequences_from_imagesets(src)
    train_test_split = None
    if val_from_imagesets is None:
        train_test_split = load_train_test_split_from_imagesets(
            src, all_seqs, seq_set
        )

    if val_from_imagesets is not None:
        val_seqs = [s for s in val_from_imagesets if s in seq_set]
        val_set = set(val_seqs)
        train_seqs = [s for s in all_seqs if s not in val_set]
        missing = [s for s in val_from_imagesets if s not in seq_set]
        if missing:
            print(
                f"Warning: {len(missing)} sequence(s) in ImageSets/val.txt not found under images/: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        print(f"Split from ImageSets/val.txt: train={len(train_seqs)}, valid={len(val_seqs)}")
    elif train_test_split is not None:
        train_seqs, val_seqs = train_test_split
        print(
            f"Split from ImageSets/train.txt + test.txt: train={len(train_seqs)}, "
            f"valid={len(val_seqs)}"
        )
    elif args.val_ratio > 0:
        rng = random.Random(args.seed)
        seqs = all_seqs.copy()
        rng.shuffle(seqs)
        n_val = max(1, int(round(len(seqs) * args.val_ratio)))
        n_val = min(n_val, len(seqs) - 1) if len(seqs) > 1 else 0
        val_set = set(seqs[:n_val])
        train_seqs = [s for s in all_seqs if s not in val_set]
        val_seqs = [s for s in all_seqs if s in val_set]
    else:
        train_seqs = all_seqs
        val_seqs = []

    if len(train_seqs) != len(set(train_seqs)):
        raise ValueError("Training split contains duplicate sequence ids")
    if len(val_seqs) != len(set(val_seqs)):
        raise ValueError("Validation split contains duplicate sequence ids")
    overlap = set(train_seqs) & set(val_seqs)
    if overlap:
        raise ValueError(f"Training/validation split overlap: {sorted(overlap)[:10]}")

    out.mkdir(parents=True, exist_ok=True)

    convert_split(
        train_seqs,
        src,
        out / "train",
        out / "meta_expressions" / "train" / "meta_expressions.json",
        args.obj_id,
        args.fg_threshold,
        args.category,
    )
    print(f"Train: {len(train_seqs)} sequences -> {out / 'train'}")

    if val_seqs:
        convert_split(
            val_seqs,
            src,
            out / "valid",
            out / "meta_expressions" / "val" / "meta_expressions.json",
            args.obj_id,
            args.fg_threshold,
            args.category,
        )
        print(f"Valid: {len(val_seqs)} sequences -> {out / 'valid'}")


if __name__ == "__main__":
    main()
