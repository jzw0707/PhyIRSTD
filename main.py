"""
Training script of DTASAM
Modified from DETR (https://github.com/facebookresearch/detr)
"""

import datetime
import json
import math
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
from util.misc import on_load_checkpoint
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch
from models.dta_sam import build_dta_sam
from os.path import join
import sys
import opts
from evaluate import METRIC_GEOMETRY
from util.lr_scheduler import build_lr_scheduler


NEW_MODULE_PREFIXES = (
    "thermal_prompt_generator.",
    "multiscale_temporal_selection_unit.",
    "dynamic_temporal_aggregator.",
)

ST_ADAPTER_NEW_COMPONENTS = frozenset(
    {
        "proj_token_down",
        "proj_token_up",
        "spatiotemporal_attention",
        "spatial_pos_embed",
        "pre_temporal_norm",
        "visual_scale",
        "token_scale",
        "hsa_scale",
    }
)


def is_new_module_parameter(name):
    if name.startswith(NEW_MODULE_PREFIXES):
        return True
    if not name.startswith("st_adapters."):
        return False
    parts = name.split(".")
    return len(parts) >= 3 and parts[2] in ST_ADAPTER_NEW_COMPONENTS


def build_optimizer_param_groups(model, args):
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    new_module_lr = getattr(args, "new_module_lr", None)
    has_decoder_parameters = any(
        name.startswith("sam.sam_mask_decoder.") for name, _ in trainable
    )
    if new_module_lr is None and not has_decoder_parameters:
        return [
            {
                "params": [parameter for _, parameter in trainable],
                "initial_lr": args.lr,
            }
        ]
    if new_module_lr is not None and new_module_lr <= 0.0:
        raise ValueError("--new_module_lr must be positive")

    pretrained_parameters = []
    new_module_parameters = []
    decoder_parameters = []
    for name, parameter in trainable:
        if name.startswith("sam.sam_mask_decoder."):
            decoder_parameters.append(parameter)
        elif new_module_lr is not None and is_new_module_parameter(name):
            new_module_parameters.append(parameter)
        else:
            pretrained_parameters.append(parameter)

    groups = []
    if pretrained_parameters:
        groups.append(
            {
                "params": pretrained_parameters,
                "lr": args.lr,
                "initial_lr": args.lr,
                "group_name": "pretrained",
            }
        )
    if new_module_parameters:
        groups.append(
            {
                "params": new_module_parameters,
                "lr": new_module_lr,
                "initial_lr": new_module_lr,
                "group_name": "new_modules",
            }
        )
    if decoder_parameters:
        decoder_lr = getattr(args, "decoder_lr", 1e-6)
        if not math.isfinite(decoder_lr) or decoder_lr <= 0:
            raise ValueError("--decoder_lr must be finite and positive")
        groups.append(
            {
                "params": decoder_parameters,
                "lr": decoder_lr,
                "initial_lr": decoder_lr,
                "group_name": "decoder",
            }
        )
    return groups


METRIC_DIRECTIONS = {
    "IoU": "max",
    "F1": "max",
    "Pd": "max",
    "Fa": "min",
}


def initial_best_metrics():
    return {
        metric_name: float("-inf") if direction == "max" else float("inf")
        for metric_name, direction in METRIC_DIRECTIONS.items()
    }


def update_best_metrics(best_metrics, eval_metrics):
    """Update validation extrema and return the metric names that improved."""
    improved_metrics = []
    for metric_name, direction in METRIC_DIRECTIONS.items():
        if metric_name not in eval_metrics:
            continue
        metric_value = float(eval_metrics[metric_name])
        is_improvement = (
            metric_value > best_metrics[metric_name]
            if direction == "max"
            else metric_value < best_metrics[metric_name]
        )
        if math.isfinite(metric_value) and is_improvement:
            best_metrics[metric_name] = metric_value
            improved_metrics.append(metric_name)
    return improved_metrics


def main(args):
    opts.validate_training_settings(args)
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation_steps must be positive")
    if (
        Path(args.output_dir) / "checkpoint.pth"
    ).exists() and not args.resume_optimizer:
        raise FileExistsError(
            "Run already contains a checkpoint; use a new --name or --resume_optimizer"
        )
    utils.init_distributed_mode(args)
    if args.output_dir and utils.get_rank() == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        args.log_file = join(args.log_dir, f"{args.name}.log")
        log_mode = "a" if args.resume else "w"
        with open(args.log_file, log_mode) as fp:
            fp.writelines(" ".join(sys.argv) + "\n")
            fp.writelines(str(args.__dict__) + "\n\n")

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_dta_sam(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model_without_ddp = model
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )

    n_parameters_tot = sum(p.numel() for p in model.parameters())
    print("number of params:", n_parameters_tot)

    head = []
    fix = []
    for _, parameter in model_without_ddp.named_parameters():
        if parameter.requires_grad:
            head.append(parameter)
        else:
            fix.append(parameter)

    print("Trainable parameters: ", sum(p.numel() for p in head))
    print("Parameters fixed: ", sum(p.numel() for p in fix))

    param_list = build_optimizer_param_groups(model_without_ddp, args)
    for group in param_list:
        print(
            "Optimizer group: "
            f"{group.get('group_name', 'trainable')}, "
            f"parameters={sum(parameter.numel() for parameter in group['params'])}, "
            f"lr={group.get('lr', args.lr):.8g}"
        )

    optimizer = torch.optim.AdamW(
        param_list, lr=args.lr, weight_decay=args.weight_decay
    )
    dataset_train = build_dataset(image_set="train", args=args)

    if args.batch_size < args.ngpu or args.batch_size % args.ngpu != 0:
        raise ValueError(
            f"--batch_size is the global batch size and must be a positive multiple "
            f"of world size {args.ngpu}; got {args.batch_size}"
        )
    batch_size_per_gpu = args.batch_size // args.ngpu
    args.batch_size_per_gpu = batch_size_per_gpu
    print(
        f"Batch size: global={args.batch_size}, per_gpu={batch_size_per_gpu}, "
        f"world_size={args.ngpu}"
    )
    if args.distributed:
        sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size_per_gpu, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(data_loader_train) / args.accumulation_steps
    )
    print(
        f"Effective batch size: {args.batch_size * args.accumulation_steps}; "
        f"optimizer updates per epoch: {optimizer_steps_per_epoch}"
    )
    lr_scheduler = build_lr_scheduler(optimizer, args, optimizer_steps_per_epoch)

    output_dir = Path(args.output_dir)
    best_metrics = initial_best_metrics()
    args.start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume_optimizer:
            if checkpoint.get("model_config") != model_without_ddp.model_config:
                raise ValueError(
                    "Optimizer resume requires a checkpoint from the current fixed model; use weights-only initialization for older checkpoints"
                )
            saved_args = checkpoint.get("args")
            if getattr(saved_args, "dataset", None) != args.dataset:
                raise ValueError("Optimizer resume requires the same --dataset profile")
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint["model"], strict=False
        )
        unexpected_keys = [
            k
            for k in unexpected_keys
            if not (k.endswith("total_params") or k.endswith("total_ops"))
        ]
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))
        saved_best_metrics = (
            checkpoint.get("best_metrics", {})
            if args.resume_optimizer and checkpoint.get("metric_geometry") == METRIC_GEOMETRY
            else {}
        )
        if args.resume_optimizer and checkpoint.get("metric_geometry") != METRIC_GEOMETRY:
            print("Reset best metrics: checkpoint uses a different evaluation geometry")
        for metric_name in best_metrics:
            if metric_name in saved_best_metrics:
                best_metrics[metric_name] = float(saved_best_metrics[metric_name])
        if args.resume_optimizer:
            if not all(k in checkpoint for k in ("optimizer", "lr_scheduler", "epoch")):
                raise ValueError("Checkpoint has no complete optimizer/scheduler state")
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.start_epoch >= args.epochs:
        raise ValueError(
            f"Checkpoint has already completed the fixed {opts.TRAINING_EPOCHS} "
            "training epochs; no training remains"
        )
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            lr_scheduler=lr_scheduler,
            args=args,
        )

        # Persist the completed training epoch before validation. Validation is
        # comparatively long and should never make the epoch weights unrecoverable.
        if args.output_dir:
            recovery_state = {
                "model": model_without_ddp.state_dict(),
                "model_config": model_without_ddp.model_config,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "eval_metrics": {},
                "evaluation_status": "pending",
                "metric_geometry": METRIC_GEOMETRY,
                "best_metrics": best_metrics.copy(),
            }
            if utils.is_main_process():
                print("Save recovery checkpoint before validation")
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            n = max(1, args.checkpoint_every)
            if n == 1 or (epoch + 1) % n == 0 or (epoch + 1) == args.epochs:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(recovery_state, checkpoint_path)

        from inference import evaluate_epoch

        out_dir = join(args.output_dir, f"valid_epoch{epoch:02d}")
        eval_metrics = evaluate_epoch(args, model_without_ddp, out_dir)

        improved_metrics = update_best_metrics(best_metrics, eval_metrics)
        checkpoint_state = {
            "model": model_without_ddp.state_dict(),
            "model_config": model_without_ddp.model_config,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "eval_metrics": eval_metrics,
            "evaluation_status": "complete",
            "metric_geometry": METRIC_GEOMETRY,
            "best_metrics": best_metrics.copy(),
        }
        if args.output_dir:
            print("Save Model")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(checkpoint_state, checkpoint_path)
            for metric_name in improved_metrics:
                best_path = output_dir / f"checkpoint_best_{metric_name.lower()}.pth"
                utils.save_on_master(checkpoint_state, best_path)
                if utils.is_main_process():
                    print(
                        f"New best {metric_name}: {best_metrics[metric_name]:.6f} "
                        f"-> {best_path}"
                    )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in eval_metrics.items()},
            **{f"best_{k}": v for k, v in best_metrics.items()},
            "epoch": epoch,
            "n_parameters": n_parameters_tot,
        }

        if utils.is_main_process():
            with Path(args.log_file).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = opts.get_args_parser().parse_args()
    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.output_dir = os.path.join(args.output_dir, args.name)
    main(args)
