#!/usr/bin/env python3
"""Inference and metric evaluation for YTVOS-layout IRSTD datasets."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage import measure
from tqdm import tqdm

import opts
import util.misc as utils
from models.dta_sam import build_dta_sam
from util.misc import on_load_checkpoint


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
    """Region-level probability of detection and pixel-level false alarm."""

    def __init__(self, distance_thr: float = 3.0):
        self.distance_thr = distance_thr
        self.reset()

    def update(self, pred_bin: np.ndarray, gt_bin: np.ndarray):
        predictions = np.asarray(pred_bin, dtype=np.int64)
        labels = np.asarray(gt_bin, dtype=np.int64)
        height, width = predictions.shape
        self.all_pixel += height * width

        if predictions.max() == 0 and labels.max() == 0:
            return

        pred_regions = measure.regionprops(measure.label(predictions, connectivity=2))
        gt_regions = measure.regionprops(measure.label(labels, connectivity=2))
        self.target += len(gt_regions)

        matched_prediction = np.zeros_like(predictions, dtype=np.int64)
        remaining_predictions = list(pred_regions)
        for gt_region in gt_regions:
            gt_centroid = np.asarray(gt_region.centroid, dtype=np.float64)
            matched_index = None
            for index, pred_region in enumerate(remaining_predictions):
                pred_centroid = np.asarray(pred_region.centroid, dtype=np.float64)
                distance = float(np.linalg.norm(pred_centroid - gt_centroid))
                if distance < self.distance_thr:
                    self.pd_count += 1
                    coordinates = pred_region.coords
                    matched_prediction[coordinates[:, 0], coordinates[:, 1]] = 1
                    matched_index = index
                    break
            if matched_index is not None:
                remaining_predictions.pop(matched_index)

        self.dismatch_pixel += int((predictions - matched_prediction).sum())

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
    """Evaluate saved binary predictions against indexed annotation masks."""
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
                pd_fa.update(pred_bin.astype(np.int64), gt_bin.astype(np.int64))

    eps = 1e-8
    iou = total_intersection / (total_union + eps)
    f1 = 2.0 * total_tp / (2.0 * total_tp + total_fp + total_fn + eps)
    pd, fa = pd_fa.get()
    return {
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
