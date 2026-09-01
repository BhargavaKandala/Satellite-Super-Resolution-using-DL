#!/usr/bin/env python
"""Phase 4 + 5: train the super-resolution model.

    python scripts/train.py
    python scripts/train.py --epochs 5 --batch-size 8
    python scripts/train.py --model bicubic     # sanity-check the plumbing

Requires a prepared dataset (``scripts/prepare_dataset.py``). Writes
``checkpoints/best.pth``, ``checkpoints/last.pth`` and a training history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import load_config, set_seed
from src.data.patch_dataset import build_dataloaders, read_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--patches", default=None, help="prepared patch directory")
    parser.add_argument("--checkpoints", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--model", default=None, help="override model.name")
    parser.add_argument("--device", default=None, help="cuda | cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    return parser.parse_args(argv)


def apply_overrides(cfg, args):
    overrides: dict = {"training": {}, "model": {}, "project": {}}
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["training"]["lr"] = args.lr
    if args.workers is not None:
        overrides["training"]["num_workers"] = args.workers
    if args.no_amp:
        overrides["training"]["amp"] = False
    if args.model is not None:
        overrides["model"]["name"] = args.model
    if args.seed is not None:
        overrides["project"]["seed"] = args.seed
    return cfg.merge({k: v for k, v in overrides.items() if v})


def main(argv: list[str] | None = None) -> int:
    import torch

    from src.inference.predict import check_overlap, resolve_device
    from src.training.train import Trainer

    args = parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
    set_seed(int(cfg.project.seed))

    patch_dir = Path(args.patches) if args.patches else cfg.get_path("data.patch_dir")
    try:
        manifest = read_manifest(patch_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if manifest["bands"] != list(cfg.data.bands):
        print(
            f"error: the prepared dataset has bands {manifest['bands']} but the "
            f"config expects {list(cfg.data.bands)}. Re-run prepare_dataset.py.",
            file=sys.stderr,
        )
        return 2
    if int(manifest["scale"]) != int(cfg.patches.scale):
        print(
            f"error: dataset was prepared at scale {manifest['scale']} but the "
            f"config asks for {cfg.patches.scale}. Re-run prepare_dataset.py.",
            file=sys.stderr,
        )
        return 2

    print("=" * 70)
    print("  TRAIN")
    print("=" * 70)

    train_loader, val_loader = build_dataloaders(patch_dir, cfg)
    if val_loader is None:
        print("warning: no validation split — metrics will track training loss only")

    for warning in check_overlap(
        int(cfg.inference.tile_overlap), int(cfg.model.num_blocks)
    ):
        print(f"warning: {warning}")

    device = resolve_device(args.device)
    checkpoint_dir = (
        Path(args.checkpoints) if args.checkpoints else cfg.get_path("training.checkpoint_dir")
    )

    trainer = Trainer(cfg, train_loader, val_loader, device=device, checkpoint_dir=checkpoint_dir)
    try:
        trainer.fit()
    except torch.cuda.OutOfMemoryError:
        print(
            "\nerror: CUDA out of memory. Lower training.batch_size or "
            "patches.hr_patch_size, or run with --device cpu.",
            file=sys.stderr,
        )
        return 3
    except KeyboardInterrupt:
        print("\ninterrupted — saving current state")
        if trainer.history:
            trainer.save_checkpoint(trainer.checkpoint_dir / "interrupted.pth", trainer.history[-1])
            trainer.write_history()
        return 130

    print("\nnext: python scripts/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
