"""Validate prepared infrared datasets before training or evaluation."""

import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np


def validate(root):
    root = Path(root)
    report = {}
    sequences = {}
    for split, meta_split in [("train", "train"), ("valid", "val")]:
        meta = root / "meta_expressions" / meta_split / "meta_expressions.json"
        if split == "valid" and not meta.exists():
            meta = root / "meta_expressions" / "valid" / "meta_expressions.json"
        data = json.loads(meta.read_text())["videos"]
        if not data:
            raise ValueError(f"{split} contains no sequences")
        sequences[split] = set(data)
        frames = targets = 0
        for video, record in data.items():
            if not record["frames"] or len(set(record["frames"])) != len(
                record["frames"]
            ):
                raise ValueError(f"{split}/{video}: empty or duplicate frames")
            ids = {int(item["obj_id"]) for item in record["expressions"].values()}
            if ids != {1}:
                raise ValueError(
                    f"{split}/{video}: DTA-SAM expects binary target masks with obj_id=1"
                )
            for frame in record["frames"]:
                with Image.open(
                    root / split / "JPEGImages" / video / f"{frame}.jpg"
                ) as image:
                    size = image.size
                with Image.open(
                    root / split / "Annotations" / video / f"{frame}.png"
                ) as mask:
                    if mask.size != size:
                        raise ValueError(
                            f"{split}/{video}/{frame}: image/mask sizes differ"
                        )
                    arr = np.asarray(mask)
                    if not set(np.unique(arr)).issubset({0, 1}):
                        raise ValueError(
                            f"{split}/{video}/{frame}: annotation values must be 0/1"
                        )
                    targets += int((arr == 1).any())
                frames += 1
        report[split] = dict(videos=len(data), frames=frames, positive_frames=targets)
    overlap = sequences["train"] & sequences["valid"]
    if overlap:
        raise ValueError(f"Train/validation overlap: {sorted(overlap)[:10]}")
    train_meta = json.loads((root / "train" / "meta.json").read_text())["videos"]
    for video in sequences["train"]:
        if train_meta[video]["objects"]["1"]["category"] != "targets":
            raise ValueError(f"{video}: expected category targets")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    print(json.dumps(validate(parser.parse_args().data_root), indent=2))
