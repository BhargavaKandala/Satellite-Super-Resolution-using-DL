"""Phase 3 + 6: reconstruction quality and spectral consistency metrics.

All functions take channel-first ``(C, H, W)`` float arrays of reflectance in
``[0, 1]`` and an optional ``valid_mask`` of shape ``(H, W)``. Masked pixels are
excluded from every statistic — nodata regions and cloud fill would otherwise
dominate a scene-level average and make the numbers meaningless.

Two families of metric are reported, and they answer different questions:

*Reconstruction quality* (PSNR, SSIM, RMSE) — "how close is the output to the
reference image?" These are the standard super-resolution metrics.

*Spectral consistency* (SAM, ERGAS, per-band RMSE) — "does the output still
represent the same physical surface?" A model can score well on PSNR while
shifting band ratios enough to break every downstream index and classifier.
For a remote-sensing product these are the metrics that decide usability.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from skimage.metrics import structural_similarity

EPS = 1e-10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _prepare(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    border_crop: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate shapes, apply the border crop and build a boolean mask."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim == 2:
        pred, target = pred[None], target[None]
    if pred.shape != target.shape:
        raise ValueError(
            f"prediction {pred.shape} and reference {target.shape} must match; "
            "resample or crop them to a common grid before evaluating"
        )

    if border_crop > 0:
        b = border_crop
        if min(pred.shape[-2:]) <= 2 * b:
            raise ValueError(
                f"border_crop={b} removes the whole image of size {pred.shape[-2:]}"
            )
        pred, target = pred[:, b:-b, b:-b], target[:, b:-b, b:-b]
        if valid_mask is not None:
            valid_mask = valid_mask[b:-b, b:-b]

    if valid_mask is None:
        mask = np.ones(pred.shape[-2:], dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != pred.shape[-2:]:
            raise ValueError(
                f"valid_mask {mask.shape} does not match image {pred.shape[-2:]}"
            )

    mask &= np.isfinite(pred).all(axis=0) & np.isfinite(target).all(axis=0)
    if not mask.any():
        raise ValueError("no valid pixels remain after masking; cannot compute metrics")
    return pred, target, mask


# ---------------------------------------------------------------------------
# Reconstruction quality
# ---------------------------------------------------------------------------
def rmse(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    border_crop: int = 0,
) -> float:
    """Root mean squared error over valid pixels, in reflectance units."""
    pred, target, mask = _prepare(pred, target, valid_mask, border_crop)
    diff = (pred - target)[:, mask]
    return float(np.sqrt(np.mean(diff**2)))


def per_band_rmse(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    border_crop: int = 0,
    band_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """RMSE for each band separately.

    A pooled RMSE can hide a single badly reconstructed band — typically NIR,
    which has the widest dynamic range and the fewest natural-image analogues
    in the model's inductive bias.
    """
    pred, target, mask = _prepare(pred, target, valid_mask, border_crop)
    names = list(band_names) if band_names else [f"band_{i}" for i in range(pred.shape[0])]
    if len(names) != pred.shape[0]:
        raise ValueError(f"got {len(names)} band names for {pred.shape[0]} bands")
    return {
        name: float(np.sqrt(np.mean((pred[i][mask] - target[i][mask]) ** 2)))
        for i, name in enumerate(names)
    }


def psnr(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    valid_mask: np.ndarray | None = None,
    border_crop: int = 0,
) -> float:
    """Peak signal-to-noise ratio in dB. Returns ``inf`` for an exact match."""
    error = rmse(pred, target, valid_mask, border_crop)
    if error < EPS:
        return float("inf")
    return float(20.0 * math.log10(data_range / error))


def ssim(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    valid_mask: np.ndarray | None = None,
    border_crop: int = 0,
) -> float:
    """Mean structural similarity across bands.

    SSIM is windowed, so it cannot simply skip masked pixels. Invalid pixels
    are set to the reference value in *both* images, which makes those windows
    contribute a neutral score instead of a spurious error.
    """
    pred, target, mask = _prepare(pred, target, valid_mask, border_crop)
    if not mask.all():
        pred = np.where(mask[None], pred, target)

    win = min(7, min(pred.shape[-2:]))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        raise ValueError(f"image {pred.shape[-2:]} is too small for SSIM")

    return float(
        structural_similarity(
            np.moveaxis(target, 0, -1),
            np.moveaxis(pred, 0, -1),
            data_range=data_range,
            channel_axis=-1,
            win_size=win,
        )
    )


# ---------------------------------------------------------------------------
# Spectral consistency
# ---------------------------------------------------------------------------
def sam(
    pred: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray | None = None,
    border_crop: int = 0,
    degrees: bool = True,
) -> float:
    """Spectral Angle Mapper: mean angle between per-pixel band vectors.

    Zero means every pixel's spectrum points in the same direction as the
    reference, i.e. band ratios are perfectly preserved. SAM is invariant to
    per-pixel brightness scaling, so it isolates *spectral* distortion from
    radiometric error — which is exactly the property that makes it the primary
    spectral-consistency metric for this project.
    """
    pred, target, mask = _prepare(pred, target, valid_mask, border_crop)
    if pred.shape[0] < 2:
        raise ValueError("SAM requires at least 2 spectral bands")

    p = pred[:, mask]
    t = target[:, mask]
    dot = np.sum(p * t, axis=0)
    norms = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0)

    # Pixels where either spectrum is all-zero have no defined angle.
    usable = norms > EPS
    if not usable.any():
        return 0.0

    cosine = np.clip(dot[usable] / norms[usable], -1.0, 1.0)
    angle = float(np.mean(np.arccos(cosine)))
    return math.degrees(angle) if degrees else angle


def sam_map(
    pred: np.ndarray,
    target: np.ndarray,
    degrees: bool = True,
) -> np.ndarray:
    """Per-pixel SAM as an ``(H, W)`` array, for spatial display in the dashboard."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    dot = np.sum(p * t, axis=0)
    norms = np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0)
    cosine = np.clip(np.divide(dot, norms, out=np.ones_like(dot), where=norms > EPS), -1.0, 1.0)
    angle = np.arccos(cosine)
    return np.degrees(angle) if degrees else angle


def ergas(
    pred: np.ndarray,
    target: np.ndarray,
    ratio: float = 4.0,
    valid_mask: np.ndarray | None = None,
    border_crop: int = 0,
) -> float:
    """Erreur Relative Globale Adimensionnelle de Synthese (Wald, 2000).

    ``ERGAS = (100 / ratio) * sqrt( mean_k (RMSE_k / mean_k)^2 )``

    where ``ratio`` is the resolution ratio between the coarse and fine grids
    (equal to the super-resolution scale factor). Normalising each band's RMSE
    by that band's mean makes the error dimensionless and comparable across
    bands with very different reflectance levels — the reason ERGAS, and not a
    raw RMSE, is the standard global quality index for resolution-enhanced
    multispectral products. Lower is better; values below ~3 are conventionally
    considered good.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    pred, target, mask = _prepare(pred, target, valid_mask, border_crop)

    terms = []
    for band in range(pred.shape[0]):
        ref = target[band][mask]
        mean = float(np.mean(ref))
        if abs(mean) < EPS:
            continue  # an all-zero reference band has no defined relative error
        band_rmse = float(np.sqrt(np.mean((pred[band][mask] - ref) ** 2)))
        terms.append((band_rmse / mean) ** 2)

    if not terms:
        return float("nan")
    return float((100.0 / ratio) * math.sqrt(np.mean(terms)))


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    data_range: float = 1.0,
    ratio: float = 4.0,
    valid_mask: np.ndarray | None = None,
    border_crop: int = 0,
    band_names: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute the full metric set in one pass, returning a JSON-safe dict."""
    requested = set(metrics) if metrics else {"psnr", "ssim", "rmse", "sam", "ergas"}
    unknown = requested - {"psnr", "ssim", "rmse", "sam", "ergas"}
    if unknown:
        raise ValueError(f"unknown metric(s): {sorted(unknown)}")

    common = dict(valid_mask=valid_mask, border_crop=border_crop)
    out: dict[str, Any] = {}

    if "psnr" in requested:
        out["psnr"] = psnr(pred, target, data_range=data_range, **common)
    if "ssim" in requested:
        out["ssim"] = ssim(pred, target, data_range=data_range, **common)
    if "rmse" in requested:
        out["rmse"] = rmse(pred, target, **common)
    if "sam" in requested:
        out["sam"] = sam(pred, target, **common)
    if "ergas" in requested:
        out["ergas"] = ergas(pred, target, ratio=ratio, **common)

    out["per_band_rmse"] = per_band_rmse(pred, target, band_names=band_names, **common)
    return out


def compare(
    results: dict[str, dict[str, Any]],
    reference_key: str = "bicubic",
) -> dict[str, Any]:
    """Tabulate several methods' metrics against a reference method.

    ``higher_is_better`` is encoded per metric so the reported deltas always
    carry the correct sign convention — a positive ``delta`` always means the
    method improved on the reference.
    """
    higher_is_better = {"psnr": True, "ssim": True, "rmse": False, "sam": False, "ergas": False}

    if reference_key not in results:
        raise KeyError(f"reference method {reference_key!r} not in {sorted(results)}")
    reference = results[reference_key]

    table: dict[str, Any] = {"reference": reference_key, "methods": {}}
    for method, values in results.items():
        row: dict[str, Any] = {}
        for metric, better in higher_is_better.items():
            if metric not in values or metric not in reference:
                continue
            value, base = values[metric], reference[metric]
            if not (np.isfinite(value) and np.isfinite(base)):
                row[metric] = {"value": value, "delta": None, "improved": None}
                continue
            raw = value - base
            row[metric] = {
                "value": float(value),
                "delta": float(raw if better else -raw),
                "improved": bool(raw > 0) if better else bool(raw < 0),
            }
        table["methods"][method] = row
    return table
