"""Preprocessing: normalisation, nodata handling, degradation and patching.

All array arguments and returns use channel-first ``(C, H, W)`` layout with
``float32`` reflectance in ``[0, 1]``, which is the single internal convention
of this project. Conversion to/from sensor DN happens only at the I/O edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .geotiff import RasterInfo, iter_windows, read_mask, read_raster

# cv2.resize handles at most 4 channels in a single call; wider stacks are
# processed in chunks rather than band-by-band to keep the call count low.
_CV2_MAX_CHANNELS = 4

_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def normalize_reflectance(
    array: np.ndarray,
    dn_scale: float = 10000.0,
    clip: Sequence[float] | None = (0.0, 1.0),
) -> np.ndarray:
    """Convert sensor digital numbers to float32 reflectance in ``[0, 1]``.

    Sentinel-2 L2A stores reflectance as ``uint16`` with a 10000 scale factor.
    A fixed, physically meaningful scale is used rather than per-scene min/max
    normalisation: per-scene statistics would make the same ground target map
    to different network inputs in different scenes, which destroys the
    spectral consistency the problem statement requires.
    """
    if dn_scale <= 0:
        raise ValueError(f"dn_scale must be positive, got {dn_scale}")
    out = np.asarray(array, dtype=np.float32) / np.float32(dn_scale)
    if clip is not None:
        out = np.clip(out, clip[0], clip[1], out=out)
    return out


def denormalize_reflectance(
    array: np.ndarray,
    dn_scale: float = 10000.0,
    dtype: str = "uint16",
) -> np.ndarray:
    """Inverse of :func:`normalize_reflectance`, with saturation-safe rounding."""
    scaled = np.asarray(array, dtype=np.float32) * np.float32(dn_scale)
    if np.dtype(dtype).kind in "iu":
        info = np.iinfo(dtype)
        scaled = np.clip(np.rint(scaled), info.min, info.max)
    return scaled.astype(dtype)


def apply_nodata(
    array: np.ndarray,
    valid_mask: np.ndarray | None,
    fill: float = 0.0,
) -> np.ndarray:
    """Set invalid pixels to ``fill``. ``valid_mask`` is ``(H, W)``, True = keep."""
    if valid_mask is None:
        return array
    out = array.copy()
    out[:, ~valid_mask] = fill
    return out


def nodata_mask_from_value(array: np.ndarray, nodata: float | None) -> np.ndarray:
    """Valid-pixel mask derived from a nodata sentinel, ANDed across bands."""
    if nodata is None:
        return np.ones(array.shape[-2:], dtype=bool)
    if isinstance(nodata, float) and np.isnan(nodata):
        return ~np.isnan(array).any(axis=0)
    return ~(array == nodata).all(axis=0)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def resize(array: np.ndarray, out_hw: tuple[int, int], interpolation: str) -> np.ndarray:
    """Resize a ``(C, H, W)`` float32 array to ``out_hw``."""
    if interpolation not in _INTERP:
        raise ValueError(
            f"unknown interpolation {interpolation!r}; expected one of {sorted(_INTERP)}"
        )
    flag = _INTERP[interpolation]
    height, width = out_hw
    array = np.ascontiguousarray(array, dtype=np.float32)
    channels = array.shape[0]

    out = np.empty((channels, height, width), dtype=np.float32)
    for start in range(0, channels, _CV2_MAX_CHANNELS):
        stop = min(start + _CV2_MAX_CHANNELS, channels)
        chunk = np.moveaxis(array[start:stop], 0, -1)  # -> (H, W, c)
        resized = cv2.resize(chunk, (width, height), interpolation=flag)
        if resized.ndim == 2:  # cv2 drops a trailing singleton axis
            resized = resized[..., None]
        out[start:stop] = np.moveaxis(resized, -1, 0)
    return out


def bicubic_upsample(array: np.ndarray, scale: int) -> np.ndarray:
    """Phase 3 baseline: plain bicubic interpolation to ``scale`` times the size.

    This is the reference every learned model must beat. It adds no
    information — it only resamples what the sensor already observed.
    """
    _, height, width = array.shape
    return resize(array, (height * scale, width * scale), "bicubic")


# ---------------------------------------------------------------------------
# Degradation (Phase 2: how LR inputs are synthesised from HR references)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DegradationConfig:
    """Sensor-like degradation applied to HR references to make LR inputs."""

    kernel: str = "gaussian"       # gaussian | bicubic | box
    gaussian_sigma: float = 0.8    # approximates the sensor MTF / PSF
    downsample: str = "area"       # area | bicubic
    noise_std: float = 0.0

    @classmethod
    def from_config(cls, cfg) -> "DegradationConfig":
        return cls(
            kernel=cfg.get("kernel", "gaussian"),
            gaussian_sigma=float(cfg.get("gaussian_sigma", 0.8)),
            downsample=cfg.get("downsample", "area"),
            noise_std=float(cfg.get("noise_std", 0.0)),
        )


def degrade(
    hr: np.ndarray,
    scale: int,
    config: DegradationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesise the LR observation corresponding to an HR reference.

    Blur-then-decimate mirrors how a real sensor forms a coarser pixel: the
    optics/detector integrate over the footprint (blur) before sampling
    (decimation). Decimating without blurring aliases high frequencies into
    the LR image and teaches the network to undo an artefact that real
    Sentinel-2 data does not contain.
    """
    config = config or DegradationConfig()
    _, height, width = hr.shape
    if height % scale or width % scale:
        raise ValueError(
            f"HR size {height}x{width} is not divisible by scale {scale}"
        )
    out_hw = (height // scale, width // scale)

    work = np.asarray(hr, dtype=np.float32)
    if config.kernel == "gaussian" and config.gaussian_sigma > 0:
        # sigma is expressed in LR pixels; scale it into HR pixel units.
        sigma = config.gaussian_sigma * scale
        work = gaussian_filter(work, sigma=(0.0, sigma, sigma), mode="reflect")
    elif config.kernel == "box":
        work = resize(work, out_hw, "area")
        work = resize(work, (height, width), "nearest")
    elif config.kernel not in ("gaussian", "bicubic", "box"):
        raise ValueError(f"unknown degradation kernel {config.kernel!r}")

    lr = resize(work, out_hw, config.downsample)

    if config.noise_std > 0:
        rng = rng or np.random.default_rng()
        lr = lr + rng.normal(0.0, config.noise_std, size=lr.shape).astype(np.float32)

    return np.clip(lr, 0.0, 1.0, out=lr)


def make_pair(
    hr: np.ndarray,
    scale: int,
    config: DegradationConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lr, hr)`` — the training pair for one HR reference patch."""
    return degrade(hr, scale, config, rng), hr


# ---------------------------------------------------------------------------
# Patch extraction (Phase 1 + 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatchStats:
    """Bookkeeping for a patch-extraction pass over one scene."""

    examined: int = 0
    accepted: int = 0
    rejected_nodata: int = 0
    rejected_flat: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "examined": self.examined,
            "accepted": self.accepted,
            "rejected_nodata": self.rejected_nodata,
            "rejected_flat": self.rejected_flat,
        }


def extract_patches(
    path,
    band_indices: Sequence[int],
    patch_size: int,
    stride: int,
    *,
    dn_scale: float = 10000.0,
    max_nodata_fraction: float = 0.10,
    min_std: float = 0.002,
    max_patches: int | None = None,
    nodata_fill: float = 0.0,
) -> Iterator[tuple[np.ndarray, RasterInfo]]:
    """Stream normalised HR patches out of a scene, one window at a time.

    Memory stays bounded at roughly ``patch_size**2 * bands * 4`` bytes
    regardless of scene size, which is what makes full Sentinel-2 tiles
    tractable. Patches that are mostly nodata, or near-constant (cloud fill,
    open water, image margins), are discarded: they contribute no
    high-frequency signal and bias the loss towards trivial solutions.
    """
    from .geotiff import read_info  # local import keeps the module import light

    info = read_info(path)
    overlap = patch_size - stride if stride < patch_size else 0
    yielded = 0

    for window in iter_windows(info, patch_size, overlap=overlap):
        if window.width != patch_size or window.height != patch_size:
            continue  # drop ragged edge tiles; the model needs a fixed size

        array, win_info = read_raster(path, band_indices, window=window)
        valid = read_mask(path, window=window)
        if valid.size and (~valid).mean() > max_nodata_fraction:
            continue

        patch = normalize_reflectance(array, dn_scale=dn_scale)
        patch = apply_nodata(patch, valid, fill=nodata_fill)

        if float(patch.std()) < min_std:
            continue

        yield patch, win_info
        yielded += 1
        if max_patches is not None and yielded >= max_patches:
            return


def count_patch_grid(info: RasterInfo, patch_size: int, stride: int) -> int:
    """Number of full patches a scene can yield — used to size preallocations."""
    overlap = patch_size - stride if stride < patch_size else 0
    return sum(
        1
        for w in iter_windows(info, patch_size, overlap=overlap)
        if w.width == patch_size and w.height == patch_size
    )


# ---------------------------------------------------------------------------
# Display helpers (dashboard only — never feed these to the model)
# ---------------------------------------------------------------------------
def percentile_stretch(
    array: np.ndarray,
    percentiles: Sequence[float] = (2.0, 98.0),
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Contrast-stretch to ``[0, 1]`` for human viewing.

    Applied per band and *only* for rendering. Metrics and model inputs always
    use the physical reflectance values, so a display stretch can never leak
    into a reported number.
    """
    out = np.empty_like(array, dtype=np.float32)
    for band in range(array.shape[0]):
        plane = array[band]
        sample = plane[valid_mask] if valid_mask is not None else plane
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            out[band] = 0.0
            continue
        lo, hi = np.percentile(sample, percentiles)
        out[band] = 0.0 if hi <= lo else np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
    return out


def to_rgb(
    array: np.ndarray,
    band_names: Sequence[str],
    rgb_names: Sequence[str],
    stretch: bool = True,
    percentiles: Sequence[float] = (2.0, 98.0),
) -> np.ndarray:
    """Select and stack RGB bands into an ``(H, W, 3)`` display array."""
    lookup = {name: idx for idx, name in enumerate(band_names)}
    missing = [n for n in rgb_names if n not in lookup]
    if missing:
        raise ValueError(f"RGB band(s) {missing} not present in {list(band_names)}")
    rgb = array[[lookup[n] for n in rgb_names]]
    if stretch:
        rgb = percentile_stretch(rgb, percentiles)
    return np.moveaxis(np.clip(rgb, 0.0, 1.0), 0, -1)
