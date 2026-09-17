#!/usr/bin/env python3
"""Inference and metric evaluation for YTVOS-layout IRSTD datasets."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure
from tqdm import tqdm

import opts

METRIC_GEOMETRY = "pixel_masks"


def remove_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    """Remove 8-connected foreground components smaller than the given area."""
    mask = np.asarray(mask, dtype=bool)
    if min_component_area <= 1 or not mask.any():
        return mask
    labels = measure.label(mask, connectivity=2)
    component_sizes = np.bincount(labels.ravel())
    keep = component_sizes >= min_component_area
    keep[0] = False
    return keep[labels]


class PD_FA:
    """Pixel-centroid detection probability and unmatched-component false alarms."""

    def __init__(self, distance_thr: float = 3.0):
        self.distance_thr = distance_thr
        self.reset()

    def update(self, pred_bin: np.ndarray, gt_bin: np.ndarray):
        pred_bin = np.asarray(pred_bin, dtype=bool)
        gt_bin = np.asarray(gt_bin, dtype=bool)
        if pred_bin.ndim != 2 or gt_bin.ndim != 2:
            raise ValueError("Expected two-dimensional binary masks")
        if pred_bin.shape != gt_bin.shape:
            raise ValueError("Prediction and ground-truth masks must have the same shape")
        pred_regions = list(measure.regionprops(measure.label(pred_bin, connectivity=2)))
        gt_regions = measure.regionprops(measure.label(gt_bin, connectivity=2))
        self.all_pixel += pred_bin.size
        self.target += len(gt_regions)

        # Preserve one-to-one, first-within-distance matching with actual centroids.
        for gt_region in gt_regions:
            gt_center = np.asarray(gt_region.centroid)
            for index, pred_region in enumerate(pred_regions):
                distance = np.linalg.norm(np.asarray(pred_region.centroid) - gt_center)
                if float(distance) < self.distance_thr:
                    self.pd_count += 1
                    pred_regions.pop(index)
                    break

        # Labeled components do not overlap; count their actual foreground pixels.
        self.dismatch_pixel += sum(int(region.area) for region in pred_regions)

    def get(self):
        eps = 1e-8
        false_alarm = self.dismatch_pixel / (self.all_pixel + eps)
        probability_detection = (
            self.pd_count / (self.target + eps) if self.target > 0 else 0.0
        )
        return float(probability_detection), float(false_alarm)

    def reset(self):
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.pd_count = 0
        self.target = 0


def load_model(args):
    """Load the complete fixed DTA-SAM graph; incomplete inference weights fail."""
    import torch

    import util.misc as utils
    from models.dta_sam import build_dta_sam
    from util.misc import on_load_checkpoint

    if not args.resume:
        raise ValueError(
            "Inference requires --resume pointing to a complete checkpoint"
        )
    utils.init_distributed_mode(args)
    model = build_dta_sam(args)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    args.loaded_checkpoint_epoch = checkpoint.get("epoch")
    checkpoint = on_load_checkpoint(model, checkpoint, for_inference=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(torch.device(args.device)).eval()


def evaluate_from_saved_masks(
    pred_root,
    gt_ann_root,
    data,
    pd_fa_distance: float = 3.0,
    min_component_area: int = 0,
):
    """Evaluate actual mask pixels, preserving contours, holes and thin structures."""
    pred_root = Path(pred_root)
    gt_ann_root = Path(gt_ann_root)
    pd_fa = PD_FA(distance_thr=pd_fa_distance)

    total_tp = total_fp = total_fn = total_tn = 0
    total_intersection = total_union = 0

    for video, video_meta in tqdm(data.items(), desc="Evaluate"):
        expressions = video_meta["expressions"]
        frames = video_meta["frames"]
        for exp_id, exp_meta in expressions.items():
            obj_id = int(exp_meta["obj_id"])
            pred_dir = pred_root / video / exp_id
            if not pred_dir.is_dir():
                raise FileNotFoundError(f"缺少预测目录: {pred_dir}")

            for frame_name in frames:
                pred_path = pred_dir / f"{frame_name}.png"
                gt_path = gt_ann_root / video / f"{frame_name}.png"
                if not pred_path.is_file():
                    raise FileNotFoundError(f"缺少预测掩码: {pred_path}")
                if not gt_path.is_file():
                    raise FileNotFoundError(f"缺少GT掩码: {gt_path}")

                pred = np.array(Image.open(pred_path).convert("L"), dtype=np.uint8)
                gt_full = np.array(Image.open(gt_path).convert("P"), dtype=np.uint8)
                pred_bin = remove_small_components(
                    pred > 127, min_component_area=min_component_area
                )
                gt_bin = gt_full == obj_id
                if pred_bin.shape != gt_bin.shape:
                    raise ValueError(
                        f"Mask size mismatch: {pred_path} {pred_bin.shape} vs "
                        f"{gt_path} {gt_bin.shape}"
                    )
                pd_fa.update(pred_bin, gt_bin)

                tp = np.logical_and(pred_bin, gt_bin).sum(dtype=np.int64)
                fp = np.logical_and(pred_bin, np.logical_not(gt_bin)).sum(
                    dtype=np.int64
                )
                fn = np.logical_and(np.logical_not(pred_bin), gt_bin).sum(
                    dtype=np.int64
                )
                tn = np.logical_and(
                    np.logical_not(pred_bin), np.logical_not(gt_bin)
                ).sum(dtype=np.int64)
                intersection = tp
                union = np.logical_or(pred_bin, gt_bin).sum(dtype=np.int64)

                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
                total_intersection += intersection
                total_union += union

    eps = 1e-8
    iou = total_intersection / (total_union + eps)
    f1 = 2.0 * total_tp / (2.0 * total_tp + total_fp + total_fn + eps)
    pd, fa = pd_fa.get()
    return {
        "metric_geometry": METRIC_GEOMETRY,
        "IoU": float(iou),
        "F1": float(f1),
        "Pd": float(pd),
        "Fa": float(fa),
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "TN": int(total_tn),
        "Pd_FA_target_regions": int(pd_fa.target),
        "Pd_FA_matched_regions": int(pd_fa.pd_count),
        "Pd_FA_dismatch_pixels": int(pd_fa.dismatch_pixel),
        "Pd_FA_all_pixels": int(pd_fa.all_pixel),
        "pd_fa_distance_thr": float(pd_fa_distance),
    }


def main(args):
    from inference import main as run

    return run(args, compute_metrics=True)


if __name__ == "__main__":
    main(opts.get_args_parser().parse_args())
