#!/usr/bin/env python3
"""Run one real FP32 DTASAM optimizer step to measure a candidate batch size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import opts
import datasets.samplers as samplers
import util.misc as utils
from datasets import build_dataset
from models.dta_sam import build_dta_sam
from models.segmentation import loss_masks, object_presence_loss
from main import build_optimizer_param_groups


def run(args) -> dict:
    if args.device != "cuda":
        raise ValueError("Batch probing is intended for --device cuda")
    utils.init_distributed_mode(args)
    if args.batch_size < args.ngpu or args.batch_size % args.ngpu != 0:
        raise ValueError(
            f"--batch_size must be divisible by world size {args.ngpu}; "
            f"got {args.batch_size}"
        )
    batch_size_per_gpu = args.batch_size // args.ngpu

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    dataset = build_dataset(image_set="train", args=args)
    if args.distributed:
        sampler = samplers.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )
    samples, targets = next(iter(loader))

    model = build_dta_sam(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint = utils.on_load_checkpoint(model, checkpoint)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
    model = model.cuda().train()
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model_without_ddp = model
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
        )
    optimizer = torch.optim.AdamW(
        build_optimizer_param_groups(model_without_ddp, args),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    samples = samples.to("cuda")
    model.train()
    outputs = model(samples, targets)
    loss_dict = loss_masks(
        torch.cat(outputs["masks"]),
        targets,
        num_frames=samples.tensors.shape[1],
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    loss_dict["loss_mask"] *= args.focal_loss_weight
    loss_dict["loss_dice"] *= args.dice_loss_weight
    if args.object_score_loss_weight > 0:
        loss_dict["loss_object_score"] = (
            args.object_score_loss_weight
            * object_presence_loss(torch.cat(outputs["object_score_logits"]), targets)
        )
    if "pred_rme_logits" in outputs:
        rme_loss = torch.nn.functional.cross_entropy(
            torch.cat(outputs["pred_rme_logits"]),
            torch.tensor(outputs["rme_label"], dtype=torch.long, device="cuda"),
            weight=torch.tensor([1.0, 2.0], device="cuda"),
            ignore_index=-1,
        )
        loss_dict["RME_loss"] = (
            rme_loss if not rme_loss.isnan() else rme_loss.new_zeros(())
        )
    loss = sum(loss_dict.values())
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {float(loss.detach())}")
    loss.backward()
    decoder_gradient_norms = {
        name: float(parameter.grad.norm())
        for name, parameter in model_without_ddp.named_parameters()
        if name.startswith("sam.sam_mask_decoder.") and parameter.grad is not None
    }
    mtsu_gradient_norms = {
        name: float(parameter.grad.norm())
        for name, parameter in model_without_ddp.named_parameters()
        if name.startswith("multiscale_temporal_selection_unit.")
        and parameter.grad is not None
    }
    deep_tpg_gradients = {
        name: float(parameter.grad.norm())
        for name, parameter in model_without_ddp.named_parameters()
        if name.startswith(
            (
                "thermal_prompt_generator.mask_prior_generator.first_multiply.",
                "thermal_prompt_generator.mask_prior_generator.refinement_blocks.",
                "thermal_prompt_generator.prompt_output_",
            )
        )
        and parameter.grad is not None
    }
    if not deep_tpg_gradients:
        raise RuntimeError("Deep TPG did not receive gradients")
    if any(
        not torch.isfinite(parameter.grad).all()
        for parameter in model_without_ddp.parameters()
        if parameter.grad is not None
    ):
        raise RuntimeError("Non-finite gradients in training probe")
    optimizer.step()
    torch.cuda.synchronize()
    eval_shape = None
    if getattr(args, "probe_eval", False):
        model.eval()
        eval_targets = [
            dict(target, frame_ids=list(range(args.num_frames))) for target in targets
        ]
        with torch.no_grad():
            predictions = model(samples, eval_targets)["pred_masks"]
        if not torch.isfinite(predictions).all():
            raise RuntimeError("Non-finite evaluation predictions")
        eval_shape = list(predictions.shape)

    peak_memory = torch.tensor(
        torch.cuda.max_memory_allocated(), dtype=torch.float64, device="cuda"
    )
    if args.distributed:
        torch.distributed.all_reduce(peak_memory, op=torch.distributed.ReduceOp.MAX)

    return {
        "status": "PASS",
        "dataset": args.data_root,
        "sam2_version": args.sam2_version,
        "world_size": args.ngpu,
        "global_batch_size": args.batch_size,
        "batch_size_per_gpu": batch_size_per_gpu,
        "num_frames": args.num_frames,
        "max_size": args.max_size,
        "input_shape": list(samples.tensors.shape),
        "loss": float(loss.detach()),
        "loss_components": {
            name: float(value.detach()) for name, value in loss_dict.items()
        },
        "mtsu_gradient_norms": mtsu_gradient_norms,
        "tpg_variant": model_without_ddp.model_config["tpg_variant"],
        "deep_tpg_gradient_norms": deep_tpg_gradients,
        "eval_prediction_shape": eval_shape,
        "decoder_gradient_norms": decoder_gradient_norms,
        "trainable_parameters": sum(
            p.numel() for p in model_without_ddp.parameters() if p.requires_grad
        ),
        "peak_memory_gib": round(float(peak_memory.item()) / 2**30, 3),
    }


def main() -> None:
    parser = opts.get_args_parser()
    parser.add_argument(
        "--probe_eval",
        action="store_true",
        help="Also check a full evaluation forward pass",
    )
    args = parser.parse_args()
    try:
        result = run(args)
    except torch.cuda.OutOfMemoryError as error:
        print(
            json.dumps(
                {
                    "status": "OOM",
                    "batch_size_per_gpu": args.batch_size,
                    "error": str(error),
                }
            )
        )
        sys.exit(2)
    is_main_process = utils.is_main_process()
    if utils.is_dist_avail_and_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if is_main_process:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
