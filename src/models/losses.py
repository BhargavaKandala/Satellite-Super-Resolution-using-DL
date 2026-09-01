"""Phase 5: the combined training objective.

Why each term exists
--------------------
``pixel`` (L1 / Charbonnier)
    Radiometric fidelity — the reconstruction must land on the right
    reflectance value, not merely look plausible. L1 is preferred over L2
    because L2's quadratic penalty is minimised by the conditional *mean* of
    plausible textures, which produces blurred output. This is the dominant
    term; everything else is a corrective.

``structural`` (1 - SSIM)
    Pixel losses are indifferent to whether an edge is sharp, as long as the
    average is right. SSIM compares local means, variances and covariance, so
    it rewards recovering *structure* — the building outlines and road edges
    that make an SR product useful for urban mapping.

``spectral`` (1 - cosine similarity between band vectors)
    The term that makes this satellite super-resolution rather than image
    upscaling. It penalises rotation of the per-pixel spectral vector,
    independently of its magnitude, so band *ratios* — and therefore indices
    like NDVI and NDWI, and any downstream classifier trained on them — survive
    the reconstruction. Without it a model can lower pixel loss by trading
    error between bands, which is exactly the failure that makes SR output
    scientifically unusable.

``gradient``
    A small penalty on the difference of spatial gradients. It specifically
    targets over-smoothing, the characteristic regression-to-the-mean failure
    of pixel-loss-trained SR, without the hallucination risk of an adversarial
    term.

Note on SAM
-----------
The spectral term uses ``1 - cos(theta)`` rather than the Spectral Angle
Mapper's ``arccos(cos(theta))``. The two share the same minimum and the same
ordering, but ``arccos`` has an unbounded derivative as the angle approaches
zero — precisely where a well-trained model operates — which destabilises
training. True SAM in degrees is reported by the evaluation module
(:mod:`src.evaluation.metrics`); it is a metric there, not an objective here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ---------------------------------------------------------------------------
# Pixel losses
# ---------------------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    """Smooth L1 variant, ``sqrt((x - y)^2 + eps^2)``.

    Behaves like L1 for large errors (robust, sharp) and like L2 near zero
    (differentiable at the origin, so gradients do not chatter at convergence).
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = float(eps) ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


def _pixel_loss(kind: str, charbonnier_eps: float) -> nn.Module:
    kinds = {
        "l1": nn.L1Loss,
        "l2": nn.MSELoss,
        "charbonnier": lambda: CharbonnierLoss(charbonnier_eps),
    }
    if kind not in kinds:
        raise ValueError(f"unknown pixel_type {kind!r}; expected one of {sorted(kinds)}")
    return kinds[kind]()


# ---------------------------------------------------------------------------
# Structural loss
# ---------------------------------------------------------------------------
def _gaussian_window(size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = g[:, None] @ g[None, :]
    return window.expand(channels, 1, size, size).contiguous()


class SSIM(nn.Module):
    """Differentiable SSIM over an 11x11 Gaussian window, computed per band.

    Per-band rather than pooled: a structural score averaged across bands would
    let a model hide a badly reconstructed NIR band behind three good visible
    bands, which is the opposite of what a multispectral product needs.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2
        self._window: torch.Tensor | None = None

    def _get_window(self, channels: int, device, dtype) -> torch.Tensor:
        if (
            self._window is None
            or self._window.shape[0] != channels
            or self._window.device != device
            or self._window.dtype != dtype
        ):
            self._window = _gaussian_window(
                self.window_size, self.sigma, channels, device, dtype
            )
        return self._window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.shape[1]
        window = self._get_window(channels, pred.device, pred.dtype)
        pad = self.window_size // 2

        mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
        mu2 = F.conv2d(target, window, padding=pad, groups=channels)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu1_mu2

        numerator = (2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)
        denominator = (mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2)
        return (numerator / (denominator + EPS)).mean()


class SSIMLoss(nn.Module):
    """``1 - SSIM``, so that lower is better and the term composes additively."""

    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.ssim = SSIM(data_range=data_range)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.ssim(pred, target)


# ---------------------------------------------------------------------------
# Spectral loss
# ---------------------------------------------------------------------------
class SpectralAngleLoss(nn.Module):
    """``1 - cos(theta)`` between the predicted and reference band vectors.

    Scale-invariant by construction: only the *direction* of the per-pixel
    spectrum is penalised, so this term constrains band ratios while leaving
    overall brightness to the pixel loss. See the module docstring for why
    ``arccos`` is deliberately not applied here.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = F.cosine_similarity(pred, target, dim=1, eps=EPS)
        return (1.0 - cosine).mean()


# ---------------------------------------------------------------------------
# Gradient loss
# ---------------------------------------------------------------------------
class GradientLoss(nn.Module):
    """L1 between the spatial gradients of prediction and reference.

    Finite differences rather than Sobel: cheaper, and the extra smoothing a
    Sobel kernel applies is counterproductive for a term whose whole purpose is
    to preserve the highest spatial frequencies.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dx_p, dy_p = pred[..., :, 1:] - pred[..., :, :-1], pred[..., 1:, :] - pred[..., :-1, :]
        dx_t, dy_t = (
            target[..., :, 1:] - target[..., :, :-1],
            target[..., 1:, :] - target[..., :-1, :],
        )
        return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)


# ---------------------------------------------------------------------------
# Combined objective
# ---------------------------------------------------------------------------
class CombinedLoss(nn.Module):
    """Weighted sum of the terms above, with per-term values exposed for logging.

    Terms whose weight is zero are skipped entirely rather than multiplied by
    zero, so disabling a term in ``config.yaml`` also removes its compute cost.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        pixel_type: str = "l1",
        charbonnier_eps: float = 1e-3,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.weights = {
            "pixel": 1.0,
            "structural": 0.0,
            "spectral": 0.0,
            "gradient": 0.0,
            **(weights or {}),
        }
        unknown = set(self.weights) - {"pixel", "structural", "spectral", "gradient"}
        if unknown:
            raise ValueError(f"unknown loss weight(s): {sorted(unknown)}")
        if all(w <= 0 for w in self.weights.values()):
            raise ValueError("at least one loss weight must be positive")

        self.pixel = _pixel_loss(pixel_type, charbonnier_eps)
        self.structural = SSIMLoss(data_range=data_range)
        self.spectral = SpectralAngleLoss()
        self.gradient = GradientLoss()

    @classmethod
    def from_config(cls, cfg) -> "CombinedLoss":
        return cls(
            weights=dict(cfg.loss.weights),
            pixel_type=cfg.loss.pixel_type,
            charbonnier_eps=float(cfg.loss.charbonnier_eps),
            data_range=float(cfg.evaluation.data_range),
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if pred.shape != target.shape:
            raise ValueError(
                f"prediction {tuple(pred.shape)} and target {tuple(target.shape)} "
                "must have the same shape"
            )

        total = pred.new_zeros(())
        parts: dict[str, float] = {}
        for name in ("pixel", "structural", "spectral", "gradient"):
            weight = self.weights.get(name, 0.0)
            if weight <= 0:
                continue
            value = getattr(self, name)(pred, target)
            total = total + weight * value
            parts[name] = float(value.detach())

        parts["total"] = float(total.detach())
        return total, parts

    def extra_repr(self) -> str:
        active = {k: v for k, v in self.weights.items() if v > 0}
        return f"weights={active}"
