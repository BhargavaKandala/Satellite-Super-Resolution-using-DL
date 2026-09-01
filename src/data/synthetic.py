"""Synthetic Sentinel-2-like scenes for tests and offline demos.

These scenes exist so the pipeline is runnable and testable end-to-end without
shipping gigabytes of imagery. They are **not** a substitute for real data:
any metric computed on synthetic scenes measures the pipeline's plumbing, not
its scientific performance. Scripts that use them label their outputs
accordingly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from .geotiff import RasterInfo, write_raster

# Approximate L2A surface reflectance for common covers, ordered
# [B04 red, B03 green, B02 blue, B08 NIR].
_SPECTRA = {
    "water":      (0.030, 0.055, 0.070, 0.015),
    "vegetation": (0.045, 0.080, 0.035, 0.360),
    "bare_soil":  (0.230, 0.190, 0.150, 0.290),
    "built_up":   (0.190, 0.185, 0.180, 0.210),
    "road":       (0.120, 0.120, 0.125, 0.135),
    "roof":       (0.280, 0.230, 0.210, 0.240),
}


def make_scene(
    height: int = 512,
    width: int = 512,
    bands: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """Build a ``(bands, H, W)`` float32 reflectance scene with urban structure.

    The scene deliberately contains hard edges (roads, roof outlines) and
    fine texture, because those are exactly the frequencies a super-resolution
    model has to reconstruct and a bicubic baseline cannot.
    """
    rng = np.random.default_rng(seed)
    labels = np.full((height, width), 1, dtype=np.int8)  # start as vegetation
    names = list(_SPECTRA)

    # A river across the lower third.
    yy, xx = np.mgrid[0:height, 0:width]
    river = np.abs(yy - (0.72 * height + 0.05 * height * np.sin(xx / (width / 6)))) < height * 0.035
    labels[river] = names.index("water")

    # Bare soil blobs.
    for _ in range(6):
        cy, cx = rng.integers(0, height), rng.integers(0, width)
        r = rng.integers(height // 20, height // 8)
        labels[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = names.index("bare_soil")

    # Urban grid: blocks of built-up separated by roads, with roof patches.
    block = max(16, height // 12)
    urban = (yy < height * 0.65) & (xx > width * 0.08)
    grid_road = ((yy % block) < max(2, block // 10)) | ((xx % block) < max(2, block // 10))
    labels[urban & ~grid_road] = names.index("built_up")
    labels[urban & grid_road] = names.index("road")

    for _ in range(120):
        cy = int(rng.integers(0, int(height * 0.62)))
        cx = int(rng.integers(int(width * 0.08), width))
        h = int(rng.integers(3, max(4, block // 2)))
        w = int(rng.integers(3, max(4, block // 2)))
        labels[cy : cy + h, cx : cx + w] = names.index("roof")

    scene = np.zeros((bands, height, width), dtype=np.float32)
    for idx, name in enumerate(names):
        spectrum = _SPECTRA[name]
        mask = labels == idx
        if not mask.any():
            continue
        for band in range(bands):
            base = spectrum[band % len(spectrum)]
            scene[band][mask] = base

    # Within-class texture and sensor noise, so the scene is not piecewise flat.
    texture = rng.normal(1.0, 0.06, size=(bands, height, width)).astype(np.float32)
    scene *= texture
    scene += rng.normal(0.0, 0.002, size=scene.shape).astype(np.float32)
    return np.clip(scene, 0.0, 1.0, out=scene)


def scene_info(
    height: int,
    width: int,
    bands: int,
    resolution: float = 10.0,
    epsg: int = 32643,          # UTM zone 43N — covers much of central India
    origin: tuple[float, float] = (300000.0, 2000000.0),
    nodata: float | None = 0,
    dtype: str = "uint16",
) -> RasterInfo:
    """RasterInfo for a synthetic scene, georeferenced in a real projected CRS."""
    transform = Affine.translation(origin[0], origin[1]) @ Affine.scale(
        resolution, -resolution
    )
    return RasterInfo(
        width=width,
        height=height,
        count=bands,
        dtype=dtype,
        crs=CRS.from_epsg(epsg),
        transform=transform,
        nodata=nodata,
        band_descriptions=tuple(["B04", "B03", "B02", "B08"][:bands]),
        tags={"SYNTHETIC": "true"},
    )


def write_scene(
    path: str | Path,
    height: int = 512,
    width: int = 512,
    bands: int = 4,
    resolution: float = 10.0,
    seed: int = 0,
    dn_scale: float = 10000.0,
) -> Path:
    """Write a synthetic scene as a uint16 GeoTIFF, mimicking Sentinel-2 L2A."""
    from .preprocessing import denormalize_reflectance

    scene = make_scene(height, width, bands, seed=seed)
    info = scene_info(height, width, bands, resolution=resolution)
    dn = denormalize_reflectance(scene, dn_scale=dn_scale, dtype="uint16")
    # 0 is the nodata sentinel; keep valid pixels off it.
    dn = np.maximum(dn, 1).astype("uint16")
    return write_raster(path, dn, info, tags={"SYNTHETIC": "true"})


def write_pair(
    directory: str | Path,
    stem: str = "synthetic",
    hr_size: int = 512,
    scale: int = 4,
    bands: int = 4,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Write a co-registered ``(lr_path, hr_path)`` pair for smoke tests.

    The HR scene is treated as the "reference" at ``10/scale`` m and the LR
    scene is its degraded 10 m counterpart, matching the Wald-protocol setup
    used for training.
    """
    from .preprocessing import DegradationConfig, degrade, denormalize_reflectance

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    hr = make_scene(hr_size, hr_size, bands, seed=seed)
    lr = degrade(hr, scale, DegradationConfig())

    hr_res = 10.0 / scale
    hr_info = scene_info(hr_size, hr_size, bands, resolution=hr_res)
    lr_info = scene_info(hr_size // scale, hr_size // scale, bands, resolution=10.0)

    hr_path = directory / f"{stem}_hr.tif"
    lr_path = directory / f"{stem}_lr.tif"
    write_raster(hr_path, np.maximum(denormalize_reflectance(hr), 1), hr_info)
    write_raster(lr_path, np.maximum(denormalize_reflectance(lr), 1), lr_info)
    return lr_path, hr_path
