"""Phase 4: the CNN super-resolution generator.

``EDSRLite`` is a residual CNN in the EDSR family, sized to train on a single
consumer GPU (or a CPU, slowly) within a hackathon timeline. Three design
choices are specific to *satellite* super-resolution rather than copied from
natural-image SR:

**No batch normalisation.** BN rescales activations using batch statistics,
which destroys the absolute radiometry that makes reflectance physically
meaningful. EDSR removed BN for image quality; here it is also a correctness
requirement — a spectrally consistent product cannot pass through a layer that
shifts band means per batch.

**Global residual over a bicubic upsample.** The network predicts only the
high-frequency *residual* that bicubic cannot produce. The low-frequency
content — which carries the spectral signature — is passed through unchanged
by construction, so band ratios such as NDVI are preserved even before the
spectral loss term acts. It also converges far faster, because the model never
has to learn the identity mapping.

**Sub-pixel (PixelShuffle) upsampling.** All convolutions run at LR resolution
and the channel dimension is folded into space once at the end. For scale 4
that is ~16x less convolution work than upsampling first, and it avoids the
checkerboard artefacts of transposed convolutions — artefacts that would be
indistinguishable from real fine structure in an SR product.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_model


class ResidualBlock(nn.Module):
    """Conv-ReLU-(Dropout)-Conv with a scaled identity shortcut.

    ``res_scale`` keeps deep stacks of unnormalised residual blocks stable:
    without normalisation the activation magnitude would otherwise grow with
    depth. Dropout sits between the convolutions so it can be re-enabled at
    inference for Monte-Carlo uncertainty estimation (Phase 8).
    """

    def __init__(self, features: int, res_scale: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(self.dropout(out))
        return x + out * self.res_scale


class PixelShuffleUpsampler(nn.Sequential):
    """Sub-pixel upsampling by ``scale``, built from factors of 2 and 3."""

    def __init__(self, features: int, scale: int):
        layers: list[nn.Module] = []
        remaining = scale
        for factor in (2, 3):
            while remaining % factor == 0:
                layers += [
                    nn.Conv2d(features, features * factor * factor, 3, padding=1),
                    nn.PixelShuffle(factor),
                ]
                remaining //= factor
        if remaining != 1:
            raise ValueError(
                f"scale {scale} must factor into 2s and 3s (got remainder {remaining})"
            )
        super().__init__(*layers)


class NearestConvUpsampler(nn.Sequential):
    """Nearest-neighbour upsample followed by a conv — the artefact-free fallback."""

    def __init__(self, features: int, scale: int):
        super().__init__(
            nn.Upsample(scale_factor=scale, mode="nearest"),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
        )


class EDSRLite(nn.Module):
    """Residual CNN super-resolution generator for multispectral imagery."""

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 4,
        out_channels: int = 4,
        num_features: int = 64,
        num_blocks: int = 12,
        res_scale: float = 0.1,
        dropout: float = 0.1,
        upsampler: str = "pixelshuffle",
        global_residual: bool = True,
        **_ignored,
    ):
        super().__init__()
        self.scale = int(scale)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.global_residual = bool(global_residual)

        if self.global_residual and self.in_channels != self.out_channels:
            raise ValueError(
                "global_residual requires in_channels == out_channels "
                f"(got {self.in_channels} != {self.out_channels}); the residual "
                "is added to an upsample of the input"
            )

        self.head = nn.Conv2d(self.in_channels, num_features, 3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(num_features, res_scale, dropout) for _ in range(num_blocks)]
        )
        self.body_tail = nn.Conv2d(num_features, num_features, 3, padding=1)

        if upsampler == "pixelshuffle":
            self.upsampler = PixelShuffleUpsampler(num_features, self.scale)
        elif upsampler == "nearest_conv":
            self.upsampler = NearestConvUpsampler(num_features, self.scale)
        else:
            raise ValueError(f"unknown upsampler {upsampler!r}")

        self.tail = nn.Conv2d(num_features, self.out_channels, 3, padding=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Start the residual branch at zero: the model's first output is exactly
        # the bicubic baseline, so training can only improve on it from step 0.
        if self.global_residual:
            nn.init.zeros_(self.tail.weight)
            nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.head(x)
        features = self.body_tail(self.body(features)) + features
        residual = self.tail(self.upsampler(features))

        if not self.global_residual:
            return residual.clamp(0.0, 1.0)

        base = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return (base + residual).clamp(0.0, 1.0)

    # -- introspection ----------------------------------------------------
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_mc_dropout(self) -> int:
        """Put dropout layers into training mode while everything else stays eval.

        This is what makes Monte-Carlo dropout possible at inference time.
        Returns the number of layers switched, so callers can fail loudly if
        the checkpoint was trained with ``dropout: 0``.
        """
        count = 0
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train()
                count += 1
        return count

    def extra_repr(self) -> str:
        return (
            f"scale={self.scale}, in={self.in_channels}, out={self.out_channels}, "
            f"global_residual={self.global_residual}"
        )


@register_model("edsr_lite")
def _edsr_lite(**kwargs) -> EDSRLite:
    return EDSRLite(**kwargs)


# ---------------------------------------------------------------------------
# Extension point
# ---------------------------------------------------------------------------
# A Transformer (SwinIR) or diffusion model plugs in here:
#
#   @register_model("swinir")
#   def _swinir(**kwargs) -> nn.Module:
#       return SwinIR(**kwargs)
#
# and becomes selectable with `model.name: swinir` in config.yaml. The only
# contract is the forward signature at the top of src/models/__init__.py.
# `enable_mc_dropout` is optional; the uncertainty module falls back to
# deep-ensemble inference when a model does not provide it.
def receptive_field(num_blocks: int, kernel: int = 3) -> int:
    """Approximate LR-space receptive field, in pixels.

    Reported in training logs because it bounds how much spatial context the
    model can use — a useful sanity check when choosing the inference tile
    overlap (which must exceed half the receptive field to avoid seams).
    """
    return 1 + (kernel - 1) * (2 * num_blocks + 3)
