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

METRIC_GEOMETRY = "component_bounding_boxes"


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


def mask_to_bounding_boxes(mask: np.ndarray):
    """Return one axis-aligned box per 8-connected foreground component.

    Coordinates are (top, left, bottom, right), with exclusive bottom/right.
    Components stay distinct even if their bounding rectangles overlap.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Expected a two-dimensional binary mask")
    return [
        tuple(int(value) for value in region.bbox)
        for region in measure.regionprops(measure.label(mask, connectivity=2))
    ]


def bounding_boxes_to_mask(boxes, shape):
    """Rasterize the union of rectangles, counting overlapping pixels once."""
    result = np.zeros(shape, dtype=bool)
    for top, left, bottom, right in boxes:
        result[top:bottom, left:right] = True
    return result


class PD_FA:
    """Box-center detection probability and unmatched-box-area false alarms."""

    def __init__(self, distance_thr: float = 3.0):
        self.distance_thr = distance_thr
        self.reset()

    def update(self, pred_bin: np.ndarray, gt_bin: np.ndarray):
        pred_bin = np.asarray(pred_bin, dtype=bool)
        gt_bin = np.asarray(gt_bin, dtype=bool)
        if pred_bin.shape != gt_bin.shape:
            raise ValueError("Prediction and ground-truth masks must have the same shape")
        self.update_boxes(
            mask_to_bounding_boxes(pred_bin),
            mask_to_bounding_boxes(gt_bin),
            pred_bin.shape,
        )

    def update_boxes(self, pred_boxes, gt_boxes, shape):
        height, width = shape
        self.all_pixel += height * width
        self.target += len(gt_boxes)
        remaining_predictions = list(pred_boxes)

        def center(box):
            top, left, bottom, right = box
            return np.array(
                [(top + bottom - 1) / 2.0, (left + right - 1) / 2.0]
            )

        # Keep the existing one-to-one, first-within-distance matching rule.
        for gt_box in gt_boxes:
            gt_center = center(gt_box)
            for index, pred_box in enumerate(remaining_predictions):
                if float(np.linalg.norm(center(pred_box) - gt_center)) < self.distance_thr:
                    self.pd_count += 1
                    remaining_predictions.pop(index)
                    break

        # Union avoids double-counting overlapping unmatched rectangles.
        unmatched_mask = bounding_boxes_to_mask(remaining_predictions, shape)
        self.dismatch_pixel += int(unmatched_mask.sum())

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
    """Evaluate per-component bounding rectangles of saved prediction/GT masks."""
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
                # Filter original components before expanding their rectangles.
                pred_boxes = mask_to_bounding_boxes(pred_bin)
                gt_boxes = mask_to_bounding_boxes(gt_bin)
                pd_fa.update_boxes(pred_boxes, gt_boxes, pred_bin.shape)
                pred_bin = bounding_boxes_to_mask(pred_boxes, pred_bin.shape)
                gt_bin = bounding_boxes_to_mask(gt_boxes, gt_bin.shape)

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
