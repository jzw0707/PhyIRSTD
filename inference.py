"""DTA-SAM inference for prepared infrared video datasets."""

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
import util.misc as utils
from datasets.transform_utils import VideoEvalDataset, vis_add_mask


def resolve_eval_paths(args):
    root = Path(args.data_root)
    split = args.image_split
    candidates = (
        [args.meta_split]
        if args.meta_split
        else (["val", "valid"] if split == "valid" else [split])
    )
    meta_file = next(
        (
            root / "meta_expressions" / item / "meta_expressions.json"
            for item in candidates
            if (root / "meta_expressions" / item / "meta_expressions.json").is_file()
        ),
        None,
    )
    images = root / split / "JPEGImages"
    if meta_file is None:
        raise FileNotFoundError(
            f"Missing metadata under {root}/meta_expressions; set --meta_split"
        )
    if not images.is_dir():
        raise FileNotFoundError(f"Missing image directory: {images}")
    return images, meta_file


@torch.inference_mode()
def run_inference(
    args,
    model,
    data,
    save_path_prefix,
    save_visualize_path_prefix,
    img_folder,
    video_list,
    progress,
):
    """Write binary PNG masks at the original image resolution, one video at a time."""
    model.eval()
    for video in video_list:
        record = data[video]
        frames = record["frames"]
        if not frames:
            raise ValueError(f"Video {video} has no frames")
        for exp_id in record["expressions"]:
            # Each record is an independent sequence; memory persists between clips only.
            model.memory_bank = {}
            model.last_frame_rme_applied = 0
            dataset = VideoEvalDataset(
                str(Path(img_folder) / video), frames, max_size=args.max_size
            )
            loader = DataLoader(
                dataset,
                batch_size=args.eval_clip_window,
                num_workers=args.num_workers,
                shuffle=False,
            )
            output = Path(save_path_prefix) / video / str(exp_id)
            output.mkdir(parents=True, exist_ok=True)
            for images, frame_ids in loader:
                ids = frame_ids.tolist()
                predictions = model([images.to(args.device)], [{"frame_ids": ids}])[
                    "pred_masks"
                ]
                probabilities = F.interpolate(
                    predictions.unsqueeze(0),
                    size=(dataset.origin_h, dataset.origin_w),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()[0]
                masks = (probabilities > args.threshold).cpu().numpy()
                for frame_id, mask in zip(ids, masks):
                    name = frames[frame_id]
                    Image.fromarray(mask.astype(np.uint8) * 255).save(
                        output / f"{name}.png"
                    )
                    if args.visualize:
                        destination = (
                            Path(save_visualize_path_prefix) / video / str(exp_id)
                        )
                        destination.mkdir(parents=True, exist_ok=True)
                        with Image.open(
                            Path(img_folder) / video / f"{name}.jpg"
                        ) as image:
                            vis_add_mask(image, mask, [255, 80, 40]).save(
                                destination / f"{name}.png"
                            )
        progress.update(1)


def run_split(args, model, output_dir, compute_metrics=False, keep_predictions=True):
    images, meta_file = resolve_eval_paths(args)
    data = json.loads(meta_file.read_text())["videos"]
    if not data:
        raise ValueError(f"No videos in {meta_file}")
    if compute_metrics:
        annotations = Path(args.data_root) / args.image_split / "Annotations"
        if not annotations.is_dir():
            raise FileNotFoundError(
                f"Evaluation requires ground-truth masks: {annotations}"
            )
    output_dir = Path(output_dir)
    predictions = output_dir / "Annotations"
    videos = sorted(data)[utils.get_rank() :: utils.get_world_size()]
    with tqdm(total=len(videos), desc=f"Inference rank {utils.get_rank()}") as progress:
        run_inference(
            args,
            model,
            data,
            predictions,
            output_dir / "visualizations",
            images,
            videos,
            progress,
        )
    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
    metrics = {}
    if utils.is_main_process():
        if compute_metrics:
            from evaluate import evaluate_from_saved_masks

            metrics = evaluate_from_saved_masks(
                predictions, annotations, data, args.pd_fa_distance
            )
            metrics.update(
                data_root=args.data_root,
                threshold=args.threshold,
                checkpoint=args.resume,
                image_split=args.image_split,
                checkpoint_epoch=getattr(args, "loaded_checkpoint_epoch", None),
            )
            (output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            print(json.dumps(metrics, indent=2))
        if not keep_predictions:
            shutil.rmtree(predictions)
    if utils.is_dist_avail_and_initialized():
        shared = [metrics]
        torch.distributed.broadcast_object_list(shared, src=0)
        metrics = shared[0]
    return metrics


def evaluate_epoch(args, model, output_dir):
    return run_split(
        args,
        model,
        Path(output_dir) / "evaluation",
        compute_metrics=True,
        keep_predictions=args.keep_epoch_predictions,
    )


def main(args, compute_metrics=False):
    from evaluate import load_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = load_model(args)
    output = Path(args.output_dir) / args.name
    result = run_split(args, model, output, compute_metrics=compute_metrics)
    if utils.is_main_process():
        print(f"Results: {output}")
    if utils.is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()
    return result


if __name__ == "__main__":
    main(opts.get_args_parser().parse_args())
