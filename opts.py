"""Command-line options for DTA-SAM training, inference and evaluation."""

import argparse
import math

TRAINING_EPOCHS = 25
TRAINING_BATCH_SIZE = 8


def validate_training_settings(args):
    """Check fixed training settings before initialization or output writes."""
    for name, expected in (("epochs", TRAINING_EPOCHS), ("batch_size", TRAINING_BATCH_SIZE)):
        actual = getattr(args, name)
        if actual != expected:
            raise ValueError(
                f"Training requires --{name} {expected}; got {actual}. Training stopped."
            )


DATASETS = {
    "tsirmt": ("data/TSIRMT", False, 0.0),
    "nudt": ("data/NUDT-MIRSTD", True, 0.1),
    "phyirstd": ("data/PhyIRSTDs", False, 0.0),
}


class DTAArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args, namespace)
        root, native_loss, score_weight = DATASETS[args.dataset]
        args.data_root = args.data_root or root
        args.native_mask_loss = native_loss
        args.object_score_loss_weight = score_weight
        args.focal_alpha, args.focal_gamma = 0.75, 2.0
        args.focal_loss_weight, args.dice_loss_weight = 5.0, 1.0
        if args.resume_optimizer and not args.resume:
            self.error("--resume_optimizer requires --resume")
        for key in (
            "batch_size",
            "accumulation_steps",
            "num_frames",
            "max_size",
            "epochs",
            "checkpoint_every",
            "eval_clip_window",
        ):
            if getattr(args, key) < 1:
                self.error(f"--{key} must be positive")
        if not 8 <= args.max_size <= 1024:
            self.error("--max_size must be between 8 and 1024")
        if not 0 <= args.threshold <= 1:
            self.error("--threshold must be in [0, 1]")
        if args.num_workers < 0 or args.warmup_epochs < 0:
            self.error("Worker count and warmup epochs must be nonnegative")
        for key in ("lr", "new_module_lr", "decoder_lr"):
            if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
                self.error(f"--{key} must be finite and positive")
        if not 0 <= args.min_lr <= args.lr:
            self.error("--min_lr must be between zero and --lr")
        if args.pd_fa_distance <= 0 or not math.isfinite(args.pd_fa_distance):
            self.error("--pd_fa_distance must be finite and positive")
        return args


def get_args_parser():
    parser = DTAArgumentParser(description="DTA-SAM infrared video segmentation")
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset", choices=DATASETS, default="tsirmt", help="Dataset training profile"
    )
    data.add_argument(
        "--data_root", help="Prepared dataset root; defaults to the selected profile"
    )
    data.add_argument(
        "--num_frames", type=int, default=8, help="Frames in each training clip"
    )
    data.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="Resize longest image edge (at most 1024)",
    )
    data.add_argument("--num_workers", type=int, default=4)
    data.add_argument(
        "--augm_hflip",
        action="store_true",
        help="Random horizontal flip, consistent across each clip",
    )
    model = parser.add_argument_group("Weights")
    model.add_argument(
        "--sam2_version", choices=["tiny", "small", "base"], default="base"
    )
    model.add_argument(
        "--sam2_checkpoint", help="Override the local SAM2 initialization checkpoint"
    )
    model.add_argument(
        "--resume",
        default="",
        help="DTA-SAM checkpoint; weights-only initialization by default",
    )
    model.add_argument(
        "--resume_optimizer",
        action="store_true",
        help="Resume the complete training state",
    )
    train = parser.add_argument_group("Training")
    train.add_argument(
        "--epochs", type=int, default=TRAINING_EPOCHS,
        help="Training total is fixed at 25 epochs",
    )
    train.add_argument(
        "--batch_size",
        type=int,
        default=TRAINING_BATCH_SIZE,
        help="Fixed global batch of 8 clips per microbatch, divided across GPUs",
    )
    train.add_argument(
        "--accumulation_steps",
        type=int,
        default=2,
        help="Effective batch = global batch × accumulation steps",
    )
    train.add_argument("--lr", type=float, default=1e-6)
    train.add_argument(
        "--new_module_lr",
        type=float,
        default=1e-5,
        help="TPG/MTSU/DTA and adapter adaptation learning rate",
    )
    train.add_argument(
        "--decoder_lr",
        type=float,
        default=1e-6,
        help="NUDT object-presence head learning rate",
    )
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument(
        "--warmup_epochs", type=int, default=1, help="Warmup before cosine decay"
    )
    train.add_argument("--min_lr", type=float, default=1e-7)
    train.add_argument("--clip_max_norm", type=float, default=1.0)
    train.add_argument("--new_module_clip_max_norm", type=float, default=1.0)
    train.add_argument("--checkpoint_every", type=int, default=1)
    run = parser.add_argument_group("Runtime")
    run.add_argument("--output_dir", default="output")
    run.add_argument("--log_dir", default="logs")
    run.add_argument("--name", default="dta_sam", help="Run subdirectory name")
    run.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument(
        "--no_distributed", action="store_true", help="Run a single process"
    )
    evaluation = parser.add_argument_group("Inference and evaluation")
    evaluation.add_argument(
        "--image_split", choices=["valid", "test", "train"], default="valid"
    )
    evaluation.add_argument(
        "--meta_split",
        default="",
        help="Metadata folder; automatically resolves val/valid for validation",
    )
    evaluation.add_argument("--threshold", type=float, default=0.5)
    evaluation.add_argument("--eval_clip_window", type=int, default=8)
    evaluation.add_argument("--pd_fa_distance", type=float, default=3.0)
    evaluation.add_argument("--visualize", action="store_true")
    evaluation.add_argument("--keep_epoch_predictions", action="store_true")
    return parser
