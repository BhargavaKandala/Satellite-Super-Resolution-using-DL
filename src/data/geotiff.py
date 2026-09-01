"""GeoTIFF I/O with strict preservation of spatial metadata.

Everything the pipeline knows about a scene's georeferencing flows through
:class:`RasterInfo`. Two rules are enforced throughout:

1. **Never load a whole scene into memory.** Sentinel-2 tiles are 10980x10980
   per 10 m band; four bands at float32 is ~1.9 GB before any processing.
   Readers here are window-based and callers iterate with :func:`iter_windows`.
2. **The affine transform is derived, never guessed.** Super-resolving by a
   factor ``s`` keeps the geographic extent identical and divides the pixel
   size by ``s`` — see :func:`scaled_transform`.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.windows import Window


@dataclass(frozen=True)
class RasterInfo:
    """Spatial metadata for a raster, independent of its pixel data."""

    width: int
    height: int
    count: int
    dtype: str
    crs: CRS | None
    transform: Affine
    nodata: float | None
    band_descriptions: tuple[str | None, ...] = ()
    tags: dict[str, str] = field(default_factory=dict)
    path: str | None = None

    # -- derived properties ----------------------------------------------
    @property
    def bounds(self) -> BoundingBox:
        """Geographic extent as (left, bottom, right, top)."""
        left, top = self.transform @ (0, 0)
        right, bottom = self.transform @ (self.width, self.height)
        return BoundingBox(left, bottom, right, top)

    @property
    def resolution(self) -> tuple[float, float]:
        """Pixel size (x, y) in CRS units, always positive."""
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def is_georeferenced(self) -> bool:
        """True when the raster carries a usable CRS and a non-identity transform."""
        return self.crs is not None and self.transform != Affine.identity()

    def summary(self) -> dict[str, object]:
        """JSON-serialisable description, used by the dashboard and reports."""
        b = self.bounds
        rx, ry = self.resolution
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "bands": self.count,
            "dtype": self.dtype,
            "crs": self.crs.to_string() if self.crs else None,
            "epsg": self.crs.to_epsg() if self.crs else None,
            "transform": list(self.transform)[:6],
            "resolution_x": rx,
            "resolution_y": ry,
            "bounds": {"left": b.left, "bottom": b.bottom, "right": b.right, "top": b.top},
            "nodata": self.nodata,
            "band_descriptions": list(self.band_descriptions),
        }


PathLike = str | os.PathLike[str]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_info(path: PathLike) -> RasterInfo:
    """Read spatial metadata only — no pixel data is touched."""
    with rasterio.open(path) as src:
        return _info_from_dataset(src, path)


def _info_from_dataset(src: rasterio.DatasetReader, path: PathLike | None = None) -> RasterInfo:
    return RasterInfo(
        width=src.width,
        height=src.height,
        count=src.count,
        dtype=src.dtypes[0],
        crs=src.crs,
        transform=src.transform,
        nodata=src.nodata,
        band_descriptions=tuple(src.descriptions),
        tags=dict(src.tags()),
        path=str(path) if path is not None else None,
    )


def read_raster(
    path: PathLike,
    band_indices: Sequence[int] | None = None,
    window: Window | None = None,
    dtype: str = "float32",
) -> tuple[np.ndarray, RasterInfo]:
    """Read a raster (or a window of it) as a ``(C, H, W)`` array.

    ``band_indices`` are 1-indexed rasterio band numbers. When ``window`` is
    given the returned :class:`RasterInfo` describes *the window*, with its
    transform offset to the window's origin, so the result stays georeferenced.
    """
    with rasterio.open(path) as src:
        indices = list(band_indices) if band_indices else list(range(1, src.count + 1))
        _check_band_indices(indices, src.count, path)
        array = src.read(indices, window=window, out_dtype=dtype)

        # The returned info must describe *what was read*, not the file on disk:
        # band count, order and descriptions all follow the requested subset.
        info = RasterInfo(
            width=src.width if window is None else int(window.width),
            height=src.height if window is None else int(window.height),
            count=len(indices),
            dtype=dtype,
            crs=src.crs,
            transform=src.transform if window is None else src.window_transform(window),
            nodata=src.nodata,
            band_descriptions=tuple(src.descriptions[i - 1] for i in indices),
            tags=dict(src.tags()),
            path=str(path),
        )
    return array, info


def _check_band_indices(indices: Sequence[int], count: int, path: PathLike) -> None:
    bad = [i for i in indices if i < 1 or i > count]
    if bad:
        raise ValueError(
            f"{Path(path).name} has {count} band(s); requested out-of-range "
            f"band index/indices {bad}. Check data.band_indices in config.yaml."
        )


def read_mask(path: PathLike, window: Window | None = None) -> np.ndarray:
    """Boolean array, ``True`` where the pixel is valid in *every* band."""
    with rasterio.open(path) as src:
        # dataset_mask() folds nodata, alpha and internal masks into one plane.
        return src.dataset_mask(window=window).astype(bool)


def iter_windows(
    info: RasterInfo,
    tile_size: int,
    overlap: int = 0,
) -> Iterator[Window]:
    """Yield tiling windows covering the raster, clipped at the edges.

    Windows step by ``tile_size - overlap``. The final row/column is snapped
    back so it stays inside the raster rather than being padded, which keeps
    every read a genuine on-disk block read.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0 <= overlap < tile_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < tile_size, got {overlap}")

    step = tile_size - overlap
    for row in _axis_offsets(info.height, tile_size, step):
        for col in _axis_offsets(info.width, tile_size, step):
            yield Window(
                col_off=col,
                row_off=row,
                width=min(tile_size, info.width - col),
                height=min(tile_size, info.height - row),
            )


def _axis_offsets(extent: int, tile: int, step: int) -> list[int]:
    if extent <= tile:
        return [0]
    offsets = list(range(0, extent - tile + 1, step))
    if offsets[-1] + tile < extent:
        offsets.append(extent - tile)
    return offsets


# ---------------------------------------------------------------------------
# Transform arithmetic
# ---------------------------------------------------------------------------
def scaled_transform(transform: Affine, scale: int | float) -> Affine:
    """Affine transform for an image super-resolved by ``scale``.

    The upper-left corner is unchanged and the pixel size shrinks by ``scale``,
    so a ``(H, W)`` raster becomes ``(H*scale, W*scale)`` covering *exactly*
    the same ground extent. This is the only correct way to georeference an SR
    product; copying the source transform would silently expand the footprint
    by a factor of ``scale``.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return transform @ Affine.scale(1.0 / scale, 1.0 / scale)


def superres_info(
    src: RasterInfo,
    scale: int,
    count: int | None = None,
    dtype: str | None = None,
    nodata: float | None = ...,  # type: ignore[assignment]
) -> RasterInfo:
    """Derive the :class:`RasterInfo` of the SR product from its source."""
    return RasterInfo(
        width=src.width * scale,
        height=src.height * scale,
        count=count if count is not None else src.count,
        dtype=dtype if dtype is not None else src.dtype,
        crs=src.crs,
        transform=scaled_transform(src.transform, scale),
        nodata=src.nodata if nodata is ... else nodata,
        band_descriptions=src.band_descriptions,
        tags=dict(src.tags),
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_raster(
    path: PathLike,
    array: np.ndarray,
    info: RasterInfo,
    *,
    nodata: float | None = ...,  # type: ignore[assignment]
    band_descriptions: Sequence[str | None] | None = None,
    tags: dict[str, str] | None = None,
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 256,
) -> Path:
    """Write a ``(C, H, W)`` (or ``(H, W)``) array as a georeferenced GeoTIFF.

    ``info`` supplies CRS and transform; width/height/count/dtype are taken
    from ``array`` so the file always matches the pixels actually written.
    Output is tiled + compressed, which makes the dashboard's windowed reads
    fast and keeps SR products (16x the source pixel count) manageable on disk.
    """
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"expected a 2D or 3D array, got shape {array.shape}")

    count, height, width = array.shape
    out_nodata = info.nodata if nodata is ... else nodata

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": array.dtype.name,
        "crs": info.crs,
        "transform": info.transform,
        "compress": compress,
        "predictor": 2 if array.dtype.kind in "iu" else 3,
        "BIGTIFF": "IF_SAFER",
    }
    if out_nodata is not None:
        profile["nodata"] = out_nodata
    if tiled and width >= blocksize and height >= blocksize:
        profile.update(tiled=True, blockxsize=blocksize, blockysize=blocksize)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
        descriptions = band_descriptions if band_descriptions is not None else info.band_descriptions
        for idx, name in enumerate(list(descriptions)[:count], start=1):
            if name:
                dst.set_band_description(idx, name)
        merged_tags = dict(info.tags)
        if tags:
            merged_tags.update(tags)
        if merged_tags:
            dst.update_tags(**{k: str(v) for k, v in merged_tags.items()})
    return path


def write_superres(
    path: PathLike,
    array: np.ndarray,
    src_info: RasterInfo,
    scale: int,
    *,
    nodata: float | None = ...,  # type: ignore[assignment]
    tags: dict[str, str] | None = None,
    compress: str = "deflate",
) -> Path:
    """Write an SR product, deriving its georeferencing from the source scene.

    Validates that ``array`` really is ``scale`` times the source raster before
    writing — a mismatch here would produce a file that looks valid but is
    misregistered on the ground.
    """
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, ...]
    _, height, width = array.shape
    expected = (src_info.height * scale, src_info.width * scale)
    if (height, width) != expected:
        raise ValueError(
            f"SR array is {height}x{width} but source {src_info.height}x"
            f"{src_info.width} at scale {scale} implies {expected[0]}x{expected[1]}"
        )

    out_info = superres_info(src_info, scale, count=array.shape[0], dtype=array.dtype.name)
    provenance = {
        "SR_SCALE": scale,
        "SR_SOURCE": Path(src_info.path).name if src_info.path else "unknown",
        "SR_SOURCE_RES_X": f"{src_info.resolution[0]:.6f}",
        "SR_OUTPUT_RES_X": f"{out_info.resolution[0]:.6f}",
        "SR_DISCLAIMER": (
            "AI-generated super-resolved imagery containing reconstructed "
            "fine-scale information. Not a direct high-resolution observation."
        ),
    }
    if tags:
        provenance.update(tags)
    return write_raster(
        path, array, out_info, nodata=nodata, tags=provenance, compress=compress
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_geospatial(
    sr_path: PathLike,
    src_info: RasterInfo,
    scale: int,
    tolerance: float = 1e-6,
) -> dict[str, object]:
    """Check that an SR GeoTIFF is spatially consistent with its source.

    Returns a report dict with a boolean ``valid`` plus the individual checks,
    so both the test-suite and the dashboard can display *why* something failed.
    """
    sr = read_info(sr_path)
    expected_res = tuple(r / scale for r in src_info.resolution)
    src_b, sr_b = src_info.bounds, sr.bounds

    checks = {
        "has_crs": sr.crs is not None,
        "crs_matches_source": (
            src_info.crs is not None and sr.crs is not None and sr.crs == src_info.crs
        ),
        "dimensions_scaled": (
            sr.width == src_info.width * scale and sr.height == src_info.height * scale
        ),
        "resolution_scaled": all(
            math.isclose(a, b, rel_tol=1e-9, abs_tol=tolerance)
            for a, b in zip(sr.resolution, expected_res)
        ),
        "bounds_preserved": all(
            math.isclose(a, b, rel_tol=1e-9, abs_tol=max(tolerance, 1e-6))
            for a, b in zip(tuple(src_b), tuple(sr_b))
        ),
        "transform_not_identity": sr.transform != Affine.identity(),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "source": src_info.summary(),
        "superresolved": sr.summary(),
        "expected_resolution": list(expected_res),
    }


def check_pair_alignment(
    a: RasterInfo,
    b: RasterInfo,
    scale: int = 1,
    tolerance: float = 1e-3,
) -> dict[str, object]:
    """Validate that two rasters form a usable LR/HR training pair.

    ``b`` is expected to be ``scale`` times finer than ``a`` over the same
    footprint. User-supplied datasets are frequently *not* co-registered, so
    this returns a report with warnings rather than raising — the caller
    decides whether to skip the pair or abort.
    """
    warnings: list[str] = []

    if a.crs != b.crs:
        warnings.append(f"CRS mismatch: {a.crs} vs {b.crs}; reproject before pairing")

    if (b.width, b.height) != (a.width * scale, a.height * scale):
        warnings.append(
            f"dimension mismatch: {b.width}x{b.height} is not {scale}x "
            f"{a.width}x{a.height}"
        )

    exp_res = tuple(r / scale for r in a.resolution)
    if not all(math.isclose(x, y, rel_tol=1e-3) for x, y in zip(b.resolution, exp_res)):
        warnings.append(
            f"resolution mismatch: expected ~{exp_res}, got {b.resolution}"
        )

    # Sub-pixel origin offset is the usual co-registration failure mode.
    px = max(a.resolution)
    dx = abs(a.bounds.left - b.bounds.left)
    dy = abs(a.bounds.top - b.bounds.top)
    offset_px = max(dx, dy) / px if px else float("inf")
    if offset_px > tolerance:
        warnings.append(
            f"origin offset of {offset_px:.3f} source-pixels "
            f"(dx={dx:.3f}, dy={dy:.3f} CRS units); co-registration required"
        )

    return {
        "aligned": not warnings,
        "warnings": warnings,
        "origin_offset_pixels": offset_px,
    }
