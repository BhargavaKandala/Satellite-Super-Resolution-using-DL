"""Phase 8: uncertainty and confidence estimation.

What these numbers are — and are not
------------------------------------
A super-resolution model invents plausible detail. The problem statement
requires that the product say *where* it has done so. The maps produced here
are **relative indicators of model instability**, expressed in reflectance
units (or normalised to ``[0, 1]`` for display). They are explicitly **not**
calibrated probabilities, and nothing in this codebase claims they are: no
calibration data exists for the inferred sub-pixel detail, because by
definition it was never observed.

Read them as: *"the model's answer here is not robust, so more of what you see
is reconstruction than observation."* High uncertainty is a reliable warning.
Low uncertainty is **not** a guarantee of correctness — a model can be
confidently wrong, and these methods cannot detect that.

Methods
-------
``mc_dropout``
    Keeps dropout active at inference and samples the predictive distribution.
    Approximates *epistemic* uncertainty: disagreement among the sub-networks
    the training procedure found equally acceptable. Requires the checkpoint to
    have been trained with ``model.dropout > 0``.

``ensemble``
    Geometric self-ensembling: predicts under each of the eight D4 symmetries
    and maps the results back. Spread across orientations measures how much the
    output depends on an arbitrary framing choice rather than on the data.
    Needs no special training, works with any architecture, and the mean is a
    genuinely better prediction than a single pass — so this is the default
    fallback when a checkpoint has no dropout.

``reprojection``
    Degrades the SR result back to the input resolution with the same forward
    model used in training and measures the residual against the *observed* LR
    pixels. Unlike the other two this is grounded in real measurements: it
    detects reconstructions that are inconsistent with what the sensor actually
    recorded. It is blind to sub-pixel invention that happens to average back
    correctly, so it complements rather than replaces the sampling methods.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..data.preprocessing import DegradationConfig, degrade
from ..inference.predict import resolve_device, super_resolve_array

METHODS = ("mc_dropout", "ensemble", "reprojection", "none")

# Below this spread (in reflectance units) an uncertainty map carries no usable
# signal — roughly a tenth of the uint16 output quantisation step.
DEGENERATE_THRESHOLD = 1e-5


class UncertaintyResult(dict):
    """Prediction plus its uncertainty map, with the caveats attached.

    Subclasses ``dict`` so it serialises straight into the dashboard and JSON
    reports while still allowing attribute access at call sites.
    """

    @property
    def prediction(self) -> np.ndarray:
        return self["prediction"]

    @property
    def uncertainty(self) -> np.ndarray:
        return self["uncertainty"]

    @property
    def confidence(self) -> np.ndarray:
        return self["confidence"]


DISCLAIMER = (
    "Relative model-disagreement indicator, not a calibrated probability. "
    "High values reliably flag unstable reconstruction; low values do not "
    "guarantee the detail is real."
)


# ---------------------------------------------------------------------------
# Normalisation for display
# ---------------------------------------------------------------------------
def normalise_map(
    array: np.ndarray,
    method: str = "percentile",
    percentiles: Sequence[float] = (1.0, 99.0),
) -> np.ndarray:
    """Scale an uncertainty map to ``[0, 1]`` for rendering.

    Percentile clipping is the default because uncertainty maps are heavily
    right-skewed — a handful of extreme pixels at scene edges would otherwise
    compress everything else into the bottom of the colour ramp and hide the
    structure that matters.
    """
    array = np.asarray(array, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array)

    if method == "none":
        return array
    if method == "percentile":
        lo, hi = np.percentile(finite, percentiles)
    elif method == "minmax":
        lo, hi = float(finite.min()), float(finite.max())
    else:
        raise ValueError(f"unknown normalisation {method!r}; expected {METHODS}")

    if hi <= lo:
        return np.zeros_like(array)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Monte-Carlo dropout
# ---------------------------------------------------------------------------
@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    lr: np.ndarray,
    scale: int,
    passes: int = 8,
    *,
    device: torch.device | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, std)`` over ``passes`` stochastic forward passes.

    The model is put in ``eval`` mode and *only* the dropout layers are
    reactivated, so any normalisation layers keep their inference behaviour.
    """
    if passes < 2:
        raise ValueError(f"mc_dropout needs at least 2 passes, got {passes}")

    device = device or resolve_device()
    model = model.to(device).eval()

    enable = getattr(model, "enable_mc_dropout", None)
    active = enable() if callable(enable) else _enable_dropout_layers(model)
    if active == 0:
        raise RuntimeError(
            "this checkpoint has no dropout layers, so MC-dropout would return "
            "zero variance. Retrain with model.dropout > 0, or set "
            "uncertainty.method to 'ensemble' or 'reprojection'."
        )

    samples = np.stack(
        [
            super_resolve_array(model, lr, scale, device=device, **kwargs)
            for _ in range(passes)
        ]
    )
    model.eval()  # restore deterministic behaviour for subsequent callers
    return samples.mean(axis=0), samples.std(axis=0)


def _enable_dropout_layers(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()
            count += 1
    return count


# ---------------------------------------------------------------------------
# Geometric self-ensemble
# ---------------------------------------------------------------------------
_D4 = [(f, r) for f in (False, True) for r in range(4)]


def _apply_d4(array: np.ndarray, flip: bool, rot: int) -> np.ndarray:
    if flip:
        array = array[..., ::-1, :]
    return np.rot90(array, k=rot, axes=(-2, -1)) if rot else array


def _undo_d4(array: np.ndarray, flip: bool, rot: int) -> np.ndarray:
    if rot:
        array = np.rot90(array, k=-rot, axes=(-2, -1))
    return array[..., ::-1, :] if flip else array


@torch.no_grad()
def ensemble_predict(
    model: nn.Module,
    lr: np.ndarray,
    scale: int,
    passes: int = 8,
    *,
    device: torch.device | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, std)`` over the D4 symmetry group of the input.

    The mean is a self-ensembled prediction and is typically 0.1-0.3 dB better
    than a single pass at no training cost; the spread is the uncertainty.
    """
    device = device or resolve_device()
    model = model.to(device).eval()

    transforms = _D4[: max(2, min(passes, len(_D4)))]
    samples = np.stack(
        [
            _undo_d4(
                super_resolve_array(
                    model,
                    np.ascontiguousarray(_apply_d4(lr, flip, rot)),
                    scale,
                    device=device,
                    **kwargs,
                ),
                flip,
                rot,
            )
            for flip, rot in transforms
        ]
    )
    return samples.mean(axis=0), samples.std(axis=0)


# ---------------------------------------------------------------------------
# Reprojection consistency
# ---------------------------------------------------------------------------
def reprojection_residual(
    sr: np.ndarray,
    lr: np.ndarray,
    scale: int,
    degradation: DegradationConfig | None = None,
) -> np.ndarray:
    """Per-pixel ``|degrade(SR) - LR|``, upsampled back to SR resolution.

    This is the one uncertainty signal anchored to real measurements: it asks
    whether the reconstruction is still consistent with the observation the
    sensor actually made. Large residuals mean the model has drifted away from
    the data, not merely that it is unsure.
    """
    from ..data.preprocessing import resize

    reprojected = degrade(np.ascontiguousarray(sr, dtype=np.float32), scale, degradation)
    residual = np.abs(reprojected - np.asarray(lr, dtype=np.float32))
    return resize(residual, sr.shape[-2:], "bilinear")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def estimate(
    model: nn.Module,
    lr: np.ndarray,
    scale: int,
    cfg,
    *,
    device: torch.device | None = None,
    **kwargs,
) -> UncertaintyResult:
    """Run the configured uncertainty method and package the result.

    Falls back from ``mc_dropout`` to ``ensemble`` when the checkpoint carries
    no dropout layers, so a model trained with ``dropout: 0`` still produces an
    uncertainty map instead of failing at demo time.
    """
    method = str(cfg.uncertainty.method).lower()
    if method not in METHODS:
        raise ValueError(f"unknown uncertainty method {method!r}; expected {METHODS}")

    passes = int(cfg.uncertainty.passes)
    degradation = DegradationConfig.from_config(cfg.patches.degradation)
    notes: list[str] = []

    if method == "none":
        prediction = super_resolve_array(model, lr, scale, device=device, **kwargs)
        spread = np.zeros_like(prediction)
    elif method == "mc_dropout":
        try:
            prediction, spread = mc_dropout_predict(
                model, lr, scale, passes, device=device, **kwargs
            )
        except RuntimeError as exc:
            notes.append(f"mc_dropout unavailable ({exc}); used geometric ensemble")
            method = "ensemble"
            prediction, spread = ensemble_predict(
                model, lr, scale, passes, device=device, **kwargs
            )
    elif method == "ensemble":
        prediction, spread = ensemble_predict(
            model, lr, scale, passes, device=device, **kwargs
        )
    else:  # reprojection
        prediction = super_resolve_array(model, lr, scale, device=device, **kwargs)
        spread = reprojection_residual(prediction, lr, scale, degradation)

    # Collapse the per-band spread into one map: the max, not the mean, so a
    # single badly reconstructed band is surfaced rather than averaged away.
    raw = spread.max(axis=0) if spread.ndim == 3 else spread

    # A near-flat map is not "high confidence" — it usually means the sampling
    # method found nothing to disagree about, which for a lightly-trained model
    # is because the output barely departs from the bicubic base. Say so rather
    # than let a reader read zeros as certainty.
    if method != "none" and float(raw.max()) < DEGENERATE_THRESHOLD:
        notes.append(
            f"{method} spread is near zero (max {raw.max():.2e}). The model's "
            "output is very close to its bicubic base, so this map carries "
            "little information — it is NOT evidence of high confidence. Train "
            "longer, raise model.dropout, or use the reprojection method."
        )
    display = normalise_map(
        raw,
        method=str(cfg.uncertainty.normalise),
        percentiles=tuple(cfg.uncertainty.percentile_clip),
    )

    consistency = reprojection_residual(prediction, lr, scale, degradation)

    return UncertaintyResult(
        prediction=prediction,
        uncertainty=raw,
        uncertainty_normalised=display,
        confidence=1.0 - display,
        reprojection_residual=consistency.max(axis=0),
        method=method,
        passes=passes,
        statistics=summarise(raw, consistency),
        notes=notes,
        disclaimer=DISCLAIMER,
    )


def summarise(uncertainty: np.ndarray, consistency: np.ndarray | None = None) -> dict[str, Any]:
    """Scalar summary of an uncertainty map, for the metrics report."""
    finite = uncertainty[np.isfinite(uncertainty)]
    out: dict[str, Any] = {
        "mean": float(finite.mean()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "p95": float(np.percentile(finite, 95)) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "units": "reflectance (0-1 scale)",
        "calibrated": False,
        "interpretation": DISCLAIMER,
    }
    if consistency is not None:
        valid = consistency[np.isfinite(consistency)]
        out["reprojection_mae"] = float(valid.mean()) if valid.size else None
    return out
