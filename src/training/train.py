"""Phase 4 + 5: the training loop.

Performance choices, and why they are safe here
-----------------------------------------------
*Mixed precision* (CUDA only). Activations run in fp16/bf16 while the master
weights stay fp32. Roughly 2x throughput and half the memory on the RTX-class
GPUs this is aimed at. Reflectance in ``[0, 1]`` sits comfortably inside fp16's
range, so the usual overflow risk does not apply; loss scaling still guards the
gradients.

*channels_last memory format*. Convolutions on tensor cores want NHWC. For a
model that is almost entirely 3x3 convolutions this is a large win and changes
nothing numerically.

*cuDNN autotuning*. Every training tile has identical shape, so the one-off
algorithm search amortises immediately.

*Cosine annealing*. Reaches a better minimum than step decay for a fixed epoch
budget, with one fewer hyperparameter to tune.

Validation metrics are computed in torch on the GPU rather than round-tripping
through NumPy — at 40 epochs the difference is minutes, not seconds.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Config
from ..models import build_model
from ..models.losses import SSIM, CombinedLoss


# ---------------------------------------------------------------------------
# Metrics (torch, on-device)
# ---------------------------------------------------------------------------
@torch.no_grad()
def batch_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Mean PSNR over a batch, computed per-sample.

    Per-sample then averaged — pooling the MSE across the batch first would let
    one easy tile mask a badly reconstructed one.
    """
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return float((10.0 * torch.log10(data_range**2 / mse)).mean())


@torch.no_grad()
def batch_sam(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean spectral angle in degrees — the spectral-consistency tracker."""
    cosine = torch.nn.functional.cosine_similarity(pred, target, dim=1, eps=1e-8)
    return float(torch.rad2deg(torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))).mean())


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float | None = None
    psnr: float | None = None
    ssim: float | None = None
    sam: float | None = None
    lr: float = 0.0
    seconds: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row = {
            "epoch": self.epoch,
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6) if self.val_loss is not None else None,
            "psnr": round(self.psnr, 4) if self.psnr is not None else None,
            "ssim": round(self.ssim, 5) if self.ssim is not None else None,
            "sam_deg": round(self.sam, 4) if self.sam is not None else None,
            "lr": self.lr,
            "seconds": round(self.seconds, 2),
        }
        row.update({f"loss_{k}": round(v, 6) for k, v in self.parts.items()})
        return row


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Owns the model, optimiser and loop state for one training run."""

    def __init__(
        self,
        cfg: Config,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: torch.device | None = None,
        checkpoint_dir: Path | None = None,
    ):
        from ..inference.predict import describe_device, resolve_device

        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or resolve_device()
        self.device_name = describe_device(self.device)

        self.model = build_model(cfg).to(self.device)
        if bool(cfg.training.channels_last) and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        if bool(cfg.training.get("compile", False)):
            self.model = torch.compile(self.model)

        self.criterion = CombinedLoss.from_config(cfg).to(self.device)
        self.ssim = SSIM(data_range=float(cfg.evaluation.data_range)).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.training.lr),
            weight_decay=float(cfg.training.weight_decay),
            betas=tuple(cfg.training.betas),
        )
        self.epochs = int(cfg.training.epochs)
        self.scheduler = self._build_scheduler()

        self.amp = bool(cfg.training.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.grad_clip = float(cfg.training.grad_clip)

        self.checkpoint_dir = Path(
            checkpoint_dir or Path(cfg.training.checkpoint_dir)
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: list[EpochResult] = []
        self.best_score = -float("inf")
        self.best_epoch = -1

    def _build_scheduler(self):
        kind = str(self.cfg.training.scheduler).lower()
        if kind == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=float(self.cfg.training.min_lr)
            )
        if kind == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, self.epochs // 3), gamma=0.5
            )
        if kind == "none":
            return None
        raise ValueError(f"unknown scheduler {kind!r}; expected cosine | step | none")

    # -- loops ------------------------------------------------------------
    def train_epoch(self, epoch: int) -> tuple[float, dict[str, float]]:
        self.model.train()
        total = 0.0
        seen = 0
        accumulated: dict[str, float] = {}
        log_every = int(self.cfg.training.log_every)

        for step, batch in enumerate(self.train_loader, start=1):
            lr = batch["lr"].to(self.device, non_blocking=True)
            hr = batch["hr"].to(self.device, non_blocking=True)
            if bool(self.cfg.training.channels_last) and self.device.type == "cuda":
                lr = lr.to(memory_format=torch.channels_last)
                hr = hr.to(memory_format=torch.channels_last)

            # set_to_none frees the gradient buffers instead of zeroing them:
            # slightly faster and lowers peak memory.
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                pred = self.model(lr)
                loss, parts = self.criterion(pred, hr)

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = lr.shape[0]
            total += float(loss.detach()) * batch_size
            seen += batch_size
            for key, value in parts.items():
                accumulated[key] = accumulated.get(key, 0.0) + value * batch_size

            if log_every and step % log_every == 0:
                print(
                    f"  epoch {epoch:3d} step {step:5d}/{len(self.train_loader)}  "
                    f"loss {total / seen:.5f}",
                    flush=True,
                )

        if seen == 0:
            raise RuntimeError(
                "training loader produced no batches — the prepared dataset is "
                "empty or smaller than one batch. Lower training.batch_size or "
                "prepare more patches."
            )
        return total / seen, {k: v / seen for k, v in accumulated.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        sums = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0, "sam": 0.0}
        seen = 0

        for batch in self.val_loader:
            lr = batch["lr"].to(self.device, non_blocking=True)
            hr = batch["hr"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                pred = self.model(lr)
            pred = pred.float()
            loss, _ = self.criterion(pred, hr)

            n = lr.shape[0]
            sums["loss"] += float(loss) * n
            sums["psnr"] += batch_psnr(pred, hr, float(self.cfg.evaluation.data_range)) * n
            sums["ssim"] += float(self.ssim(pred, hr)) * n
            sums["sam"] += batch_sam(pred, hr) * n
            seen += n

        return {k: v / seen for k, v in sums.items()} if seen else {}

    def fit(self) -> list[EpochResult]:
        print(f"device: {self.device_name}")
        print(f"model:  {type(self.model).__name__}", end="")
        if hasattr(self.model, "num_parameters"):
            print(f" ({self.model.num_parameters():,} parameters)", end="")
        print(f"\nloss:   {self.criterion.extra_repr()}")
        print(f"train:  {len(self.train_loader.dataset)} patches", end="")
        if self.val_loader:
            print(f" | val: {len(self.val_loader.dataset)} patches", end="")
        print(f"\nepochs: {self.epochs}\n", flush=True)

        for epoch in range(1, self.epochs + 1):
            started = time.perf_counter()
            train_loss, parts = self.train_epoch(epoch)
            val = self.validate()
            if self.scheduler:
                self.scheduler.step()

            result = EpochResult(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val.get("loss"),
                psnr=val.get("psnr"),
                ssim=val.get("ssim"),
                sam=val.get("sam"),
                lr=self.optimizer.param_groups[0]["lr"],
                seconds=time.perf_counter() - started,
                parts=parts,
            )
            self.history.append(result)
            self._report(result)
            self._maybe_save_best(result)

        self.save_checkpoint(self.checkpoint_dir / "last.pth", self.history[-1])
        self.write_history()
        if self.best_epoch > 0:
            print(
                f"\nbest epoch {self.best_epoch} "
                f"({self.cfg.training.save_best_on} = {self.best_score:.4f}) "
                f"-> {self.checkpoint_dir / 'best.pth'}"
            )
        return self.history

    def _report(self, r: EpochResult) -> None:
        line = f"epoch {r.epoch:3d}/{self.epochs}  loss {r.train_loss:.5f}"
        if r.val_loss is not None:
            line += (
                f"  val {r.val_loss:.5f}  PSNR {r.psnr:.2f} dB"
                f"  SSIM {r.ssim:.4f}  SAM {r.sam:.3f} deg"
            )
        line += f"  lr {r.lr:.2e}  {r.seconds:.1f}s"
        print(line, flush=True)

    def _maybe_save_best(self, result: EpochResult) -> None:
        criterion = str(self.cfg.training.save_best_on).lower()
        if criterion == "psnr":
            score = result.psnr
        elif criterion == "loss":
            score = -result.val_loss if result.val_loss is not None else -result.train_loss
        else:
            raise ValueError(f"unknown save_best_on {criterion!r}; expected psnr | loss")
        if score is None:
            score = -result.train_loss

        if score > self.best_score:
            self.best_score = score
            self.best_epoch = result.epoch
            self.save_checkpoint(self.checkpoint_dir / "best.pth", result)

    # -- persistence ------------------------------------------------------
    def save_checkpoint(self, path: Path, result: EpochResult) -> Path:
        """Persist weights *with* the config that produced them.

        Inference reconstructs the architecture from this embedded config, so a
        later edit to ``config.yaml`` can never silently load weights into the
        wrong model.
        """
        model = getattr(self.model, "_orig_mod", self.model)  # unwrap torch.compile
        payload = {
            "model_state": model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.cfg.to_dict(),
            "epoch": result.epoch,
            "metrics": result.as_row(),
            "best_score": self.best_score,
            "device": self.device_name,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    def write_history(self) -> tuple[Path, Path]:
        rows = [r.as_row() for r in self.history]
        csv_path = self.checkpoint_dir / "history.csv"
        json_path = self.checkpoint_dir / "history.json"

        if rows:
            fieldnames = sorted({key for row in rows for key in row})
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return csv_path, json_path
