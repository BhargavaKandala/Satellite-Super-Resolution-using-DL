"""Phase 3: interpolation baselines.

The bicubic baseline is the honest control for this project. It resamples the
observation without adding information, so any metric gain a learned model
shows over it is the actual contribution of the model — and any *spectral*
degradation relative to it is a red flag. Every evaluation report compares
against this baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


class InterpolationSR(nn.Module):
    """Parameter-free super-resolution by interpolation.

    Implemented as an ``nn.Module`` so it drops into the same training,
    inference and evaluation code paths as the learned models — the comparison
    is then guaranteed to run through identical tiling, normalisation and
    metric code, which is what makes it a fair control.
    """

    def __init__(
        self,
        scale: int = 4,
        mode: str = "bicubic",
        in_channels: int = 4,
        out_channels: int | None = None,
        **_ignored,
    ):
        super().__init__()
        if mode not in ("bicubic", "bilinear", "nearest"):
            raise ValueError(f"unsupported interpolation mode {mode!r}")
        if out_channels not in (None, in_channels):
            raise ValueError(
                "InterpolationSR cannot change the channel count "
                f"({in_channels} -> {out_channels}); it only resamples."
            )
        self.scale = int(scale)
        self.mode = mode
        self.in_channels = int(in_channels)
        self.out_channels = self.in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kwargs = {"align_corners": False} if self.mode != "nearest" else {}
        out = F.interpolate(x, scale_factor=self.scale, mode=self.mode, **kwargs)
        # Bicubic overshoots at strong edges; reflectance cannot leave [0, 1].
        return out.clamp_(0.0, 1.0) if self.mode == "bicubic" else out

    def extra_repr(self) -> str:
        return f"scale={self.scale}, mode={self.mode}, channels={self.in_channels}"


@register_model("bicubic")
def _bicubic(**kwargs) -> InterpolationSR:
    return InterpolationSR(mode="bicubic", **_strip(kwargs))


@register_model("bilinear")
def _bilinear(**kwargs) -> InterpolationSR:
    return InterpolationSR(mode="bilinear", **_strip(kwargs))


def _strip(kwargs: dict) -> dict:
    """Drop learned-model-only config keys so one config drives both."""
    ignored = {
        "num_features",
        "num_blocks",
        "res_scale",
        "dropout",
        "upsampler",
        "global_residual",
        "out_channels",
    }
    return {k: v for k, v in kwargs.items() if k not in ignored}
