"""Phase 7: super-resolution inference with correct georeferencing.

Tiling strategy
---------------
A Sentinel-2 tile is 10980x10980 per 10 m band. At scale 4 the output is
43920x43920 x 4 bands, which is ~30 TB as float32 — so the SR product can
never be held in memory, and neither can a weight accumulator over it.

This module therefore uses **context-padded, non-overlapping writes** rather
than the usual overlap-and-blend:

    read  [-------- padded input tile --------]
    keep          [--- output block ---]

Each output block is written exactly once, from a prediction that saw
``overlap`` pixels of real context on every side. Because blocks never overlap
there is nothing to blend, no accumulator, and no seam — provided the padding
exceeds half the model's receptive field, which
:func:`~src.models.generator.receptive_field` reports and
:func:`check_overlap` verifies. Peak memory is one padded tile, independent of
scene size.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import rasterio
import torch
import torch.nn as nn
from rasterio.windows import Window

from ..data.geotiff import RasterInfo, read_info, read_mask, superres_info, write_raster
from ..data.preprocessing import denormalize_reflectance, normalize_reflectance


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick CUDA when available, otherwise CPU. ``preferred`` overrides."""
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index or 0
        name = torch.cuda.get_device_name(index)
        total = torch.cuda.get_device_properties(index).total_memory / 1e9
        return f"cuda:{index} ({name}, {total:.1f} GB)"
    return "cpu"


def check_overlap(overlap: int, num_blocks: int) -> list[str]:
    """Warn when the tile padding is too small to hide tiling seams."""
    from ..models.generator import receptive_field

    needed = receptive_field(num_blocks) // 2
    if overlap < needed:
        return [
            f"inference.tile_overlap={overlap} is below half the model receptive "
            f"field ({needed} px); tile boundaries may be visible. "
            f"Set tile_overlap >= {needed}."
        ]
    return []


# ---------------------------------------------------------------------------
# Core tiled prediction
# ---------------------------------------------------------------------------
@dataclass
class InferenceStats:
    """Timing and shape bookkeeping, surfaced by the CLI and the dashboard."""

    tiles: int = 0
    seconds: float = 0.0
    device: str = "cpu"
    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        megapixels = (
            float(np.prod(self.output_shape[-2:])) / 1e6 if self.output_shape else 0.0
        )
        return {
            "tiles": self.tiles,
            "seconds": round(self.seconds, 3),
            "device": self.device,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "megapixels_per_second": round(megapixels / self.seconds, 2)
            if self.seconds > 0
            else None,
        }


@torch.no_grad()
def _forward(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    amp: bool,
    channels_last: bool,
) -> torch.Tensor:
    if channels_last and device.type == "cuda":
        batch = batch.to(memory_format=torch.channels_last)
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        out = model(batch)
    return out.float()


@torch.no_grad()
def super_resolve_array(
    model: nn.Module,
    lr: np.ndarray,
    scale: int,
    *,
    tile_size: int = 256,
    overlap: int = 32,
    batch_size: int = 4,
    device: torch.device | None = None,
    amp: bool = True,
    channels_last: bool = True,
) -> np.ndarray:
    """Super-resolve an in-memory ``(C, h, w)`` array to ``(C, h*s, w*s)``.

    Used by the dashboard and by evaluation, where scenes are small enough to
    hold in RAM. Full scenes go through :func:`super_resolve_file`.
    """
    device = device or resolve_device()
    model = model.to(device).eval()

    lr = np.ascontiguousarray(lr, dtype=np.float32)
    channels, height, width = lr.shape
    out = np.empty((channels, height * scale, width * scale), dtype=np.float32)

    jobs = list(_plan_blocks(height, width, tile_size, overlap))
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        arrays = [lr[:, b.r0 : b.r1, b.c0 : b.c1] for b in chunk]

        # Blocks at the raster edge lose part of their padding, so a batch can
        # contain more than one tile shape; group before stacking.
        for shaped, group in _group_by_shape(arrays, chunk):
            tiles = torch.from_numpy(np.stack(shaped)).to(device, non_blocking=True)
            preds = _forward(model, tiles, device, amp, channels_last).cpu().numpy()
            for block, pred in zip(group, preds):
                out[
                    :,
                    block.out_r0 * scale : block.out_r1 * scale,
                    block.out_c0 * scale : block.out_c1 * scale,
                ] = pred[
                    :,
                    (block.out_r0 - block.r0) * scale : (block.out_r1 - block.r0) * scale,
                    (block.out_c0 - block.c0) * scale : (block.out_c1 - block.c0) * scale,
                ]
    return out


@dataclass(frozen=True)
class _Block:
    """A padded read region ``[r0:r1, c0:c1]`` and the sub-region kept from it."""

    r0: int
    r1: int
    c0: int
    c1: int
    out_r0: int
    out_r1: int
    out_c0: int
    out_c1: int


def _plan_blocks(height: int, width: int, tile: int, pad: int):
    """Enumerate non-overlapping output blocks with their padded read windows."""
    if tile <= 0:
        raise ValueError(f"tile_size must be positive, got {tile}")
    if pad < 0:
        raise ValueError(f"tile_overlap must be non-negative, got {pad}")

    for out_r0 in range(0, height, tile):
        out_r1 = min(out_r0 + tile, height)
        r0, r1 = max(0, out_r0 - pad), min(height, out_r1 + pad)
        for out_c0 in range(0, width, tile):
            out_c1 = min(out_c0 + tile, width)
            c0, c1 = max(0, out_c0 - pad), min(width, out_c1 + pad)
            yield _Block(r0, r1, c0, c1, out_r0, out_r1, out_c0, out_c1)


# ---------------------------------------------------------------------------
# Whole-file inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def super_resolve_file(
    model: nn.Module,
    input_path: str | Path,
    output_path: str | Path,
    *,
    scale: int,
    band_indices: Sequence[int],
    tile_size: int = 256,
    overlap: int = 32,
    batch_size: int = 4,
    dn_scale: float = 10000.0,
    output_dtype: str = "uint16",
    output_dn_scale: float = 10000.0,
    compress: str = "deflate",
    device: torch.device | None = None,
    amp: bool = True,
    channels_last: bool = True,
    band_names: Sequence[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, RasterInfo, InferenceStats]:
    """Super-resolve a GeoTIFF, streaming tiles to disk in bounded memory.

    The output carries the source CRS, a transform rescaled by ``scale`` (so
    the ground footprint is identical), band descriptions and provenance tags
    marking it as AI-generated.
    """
    device = device or resolve_device()
    model = model.to(device).eval()
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    input_path, output_path = Path(input_path), Path(output_path)
    src_info = read_info(input_path)
    out_info = superres_info(
        src_info, scale, count=len(band_indices), dtype=output_dtype, nodata=None
    )

    profile = {
        "driver": "GTiff",
        "width": out_info.width,
        "height": out_info.height,
        "count": out_info.count,
        "dtype": output_dtype,
        "crs": out_info.crs,
        "transform": out_info.transform,
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    blocks = list(_plan_blocks(src_info.height, src_info.width, tile_size, overlap))
    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src, rasterio.open(output_path, "w", **profile) as dst:
        for done, start in enumerate(range(0, len(blocks), batch_size), start=1):
            chunk = blocks[start : start + batch_size]
            arrays = [
                normalize_reflectance(
                    src.read(
                        list(band_indices),
                        window=Window(b.c0, b.r0, b.c1 - b.c0, b.r1 - b.r0),
                        out_dtype="float32",
                    ),
                    dn_scale=dn_scale,
                )
                for b in chunk
            ]

            # Ragged edge tiles cannot be stacked; run them one at a time.
            groups = _group_by_shape(arrays, chunk)
            for shaped, group in groups:
                tiles = torch.from_numpy(np.stack(shaped)).to(device, non_blocking=True)
                preds = _forward(model, tiles, device, amp, channels_last).cpu().numpy()
                for block, pred in zip(group, preds):
                    keep = pred[
                        :,
                        (block.out_r0 - block.r0) * scale : (block.out_r1 - block.r0) * scale,
                        (block.out_c0 - block.c0) * scale : (block.out_c1 - block.c0) * scale,
                    ]
                    dst.write(
                        denormalize_reflectance(keep, output_dn_scale, output_dtype)
                        if output_dtype != "float32"
                        else keep.astype("float32"),
                        window=Window(
                            block.out_c0 * scale,
                            block.out_r0 * scale,
                            (block.out_c1 - block.out_c0) * scale,
                            (block.out_r1 - block.out_r0) * scale,
                        ),
                        indexes=list(range(1, out_info.count + 1)),
                    )
            if progress:
                progress(min(start + batch_size, len(blocks)), len(blocks))

        names = list(band_names) if band_names else list(src_info.band_descriptions)
        for idx, name in enumerate(names[: out_info.count], start=1):
            if name:
                dst.set_band_description(idx, name)
        dst.update_tags(
            SR_SCALE=str(scale),
            SR_SOURCE=input_path.name,
            SR_SOURCE_RES_X=f"{src_info.resolution[0]:.6f}",
            SR_OUTPUT_RES_X=f"{out_info.resolution[0]:.6f}",
            SR_MODEL=type(model).__name__,
            SR_DISCLAIMER=(
                "AI-generated super-resolved imagery containing reconstructed "
                "fine-scale information. Not a direct high-resolution observation; "
                "validate before analytical use."
            ),
        )

    stats = InferenceStats(
        tiles=len(blocks),
        seconds=time.perf_counter() - started,
        device=describe_device(device),
        input_shape=(len(band_indices), src_info.height, src_info.width),
        output_shape=(out_info.count, out_info.height, out_info.width),
    )
    return output_path, out_info, stats


def _group_by_shape(arrays: list[np.ndarray], blocks: list[_Block]):
    """Group same-shaped tiles so ragged edge tiles do not break batching."""
    buckets: dict[tuple[int, ...], tuple[list[np.ndarray], list[_Block]]] = {}
    for array, block in zip(arrays, blocks):
        buckets.setdefault(array.shape, ([], []))
        buckets[array.shape][0].append(array)
        buckets[array.shape][1].append(block)
    return list(buckets.values())


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def load_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model from a checkpoint written by :mod:`src.training.train`.

    The checkpoint stores the model config alongside the weights, so inference
    never has to guess the architecture — a config edit after training cannot
    silently produce a mismatched model.
    """
    from ..config import Config
    from ..models import build_model

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path} — run scripts/train.py first"
        )

    device = device or resolve_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    if "config" not in payload or "model_state" not in payload:
        raise ValueError(f"{path} is not a valid training checkpoint")

    cfg = Config(payload["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def read_scene(
    path: str | Path,
    band_indices: Sequence[int],
    dn_scale: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray, RasterInfo]:
    """Read a whole (small) scene as normalised reflectance plus its valid mask."""
    from ..data.geotiff import read_raster

    array, info = read_raster(path, band_indices)
    return normalize_reflectance(array, dn_scale=dn_scale), read_mask(path), info


def write_uncertainty(
    path: str | Path,
    uncertainty: np.ndarray,
    src_info: RasterInfo,
    scale: int,
    tags: dict[str, str] | None = None,
) -> Path:
    """Write an uncertainty/confidence map as a single-band float32 GeoTIFF."""
    array = np.asarray(uncertainty, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=0)
    out_info = superres_info(src_info, scale, count=1, dtype="float32", nodata=None)
    base = {
        "SR_LAYER": "uncertainty",
        "SR_UNCERTAINTY_NOTE": (
            "Relative model-disagreement indicator, not a calibrated probability. "
            "Higher values mark pixels where the model's reconstruction is less "
            "stable and more of the detail is inferred rather than observed."
        ),
    }
    if tags:
        base.update(tags)
    return write_raster(
        path, array[None], out_info, nodata=None, band_descriptions=["uncertainty"], tags=base
    )
