#!/usr/bin/env python
"""Phase 1 + 2: turn Sentinel-2 GeoTIFFs into a training-ready patch dataset.

    python scripts/prepare_dataset.py                  # use data/raw/*.tif
    python scripts/prepare_dataset.py --synthetic      # generate demo scenes
    python scripts/prepare_dataset.py --input path/to/scene.tif

Writes ``data/patches/patches.npy`` (a memory-mapped uint16 array of HR
reference patches) plus ``manifest.json``. Low-resolution inputs are not stored
— they are synthesised from the manifest's degradation settings at load time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from src.config import REPO_ROOT, load_config, set_seed
from src.data.geotiff import read_info
from src.data.patch_dataset import (
    ARRAY_NAME,
    STORE_DTYPE,
    STORE_SCALE,
    PatchRecord,
    spatial_split,
    write_manifest,
)
from src.data.preprocessing import count_patch_grid, extract_patches

COPY_CHUNK = 256  # patches per chunk when trimming the preallocated array


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument(
        "--profile",
        default=None,
        help="hardware overlay, e.g. cpu | dgx_b200 (changes patch geometry)",
    )
    parser.add_argument("--input", nargs="*", default=None, help="scene GeoTIFF(s)")
    parser.add_argument("--output", default=None, help="patch output directory")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="generate synthetic demo scenes when no real data is available",
    )
    parser.add_argument("--synthetic-scenes", type=int, default=2)
    parser.add_argument("--synthetic-size", type=int, default=1024)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None, help="HR patch edge in pixels")
    return parser.parse_args(argv)


def discover_scenes(cfg, explicit: list[str] | None) -> list[Path]:
    if explicit:
        paths = [Path(p) for p in explicit]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"input scene(s) not found: {missing}")
        return paths

    raw_dir = cfg.get_path("data.raw_dir")
    scenes = sorted(
        p for p in raw_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff", ".jp2")
    )
    return scenes


def make_synthetic(cfg, count: int, size: int) -> list[Path]:
    """Generate demo scenes at the SR target resolution.

    Patches are cut from these and treated as the HR *reference*; the LR input
    is derived by degradation at load time. The scenes are written at
    ``target_resolution_m`` so the georeferencing of the training data matches
    what the model is being asked to produce.
    """
    from src.data.synthetic import write_scene

    raw_dir = cfg.get_path("data.raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    resolution = float(cfg.data.target_resolution_m)

    paths = []
    for index in range(count):
        path = raw_dir / f"synthetic_{index:02d}.tif"
        write_scene(
            path,
            height=size,
            width=size,
            bands=len(cfg.data.bands),
            resolution=resolution,
            seed=1000 + index,
        )
        paths.append(path)
        print(f"  wrote {path.name}  ({size}x{size} @ {resolution} m)")
    return paths


def make_sample_input(cfg, reference: Path, scale: int) -> Path:
    """Write a 10 m demo scene for inference and the dashboard.

    The synthetic scenes in ``data/raw`` are *references* at the target
    resolution — feeding one to ``inference.py`` would super-resolve an
    already-fine image. This writes the matching coarse observation so the
    demo represents the real task: 10 m in, 2.5 m out.
    """
    from src.data.geotiff import read_raster, scaled_transform, write_raster
    from src.data.preprocessing import (
        DegradationConfig,
        degrade,
        denormalize_reflectance,
        normalize_reflectance,
    )

    array, info = read_raster(reference, list(cfg.data.band_indices))
    fine = normalize_reflectance(array, dn_scale=float(cfg.data.dn_scale))
    coarse = degrade(fine, scale, DegradationConfig.from_config(cfg.patches.degradation))

    from dataclasses import replace

    coarse_info = replace(
        info,
        width=coarse.shape[2],
        height=coarse.shape[1],
        dtype="uint16",
        transform=scaled_transform(info.transform, 1.0 / scale),
    )
    path = REPO_ROOT / "sample.tif"
    write_raster(
        path,
        np.maximum(denormalize_reflectance(coarse, float(cfg.data.dn_scale)), 1),
        coarse_info,
        band_descriptions=list(cfg.data.bands),
        tags={"SYNTHETIC": "true", "NOTE": "Demo 10 m observation for inference"},
    )
    print(
        f"  wrote {path.name}  ({coarse.shape[2]}x{coarse.shape[1]} @ "
        f"{cfg.data.source_resolution_m} m)  <- demo input for inference"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, args.profile)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    set_seed(int(cfg.project.seed))

    patch_size = args.patch_size or int(cfg.patches.hr_patch_size)
    stride = int(cfg.patches.stride)
    max_patches = args.max_patches or int(cfg.patches.max_patches)
    band_indices = list(cfg.data.band_indices)
    bands = list(cfg.data.bands)

    if patch_size % int(cfg.patches.scale):
        print(
            f"error: patch size {patch_size} is not divisible by scale "
            f"{cfg.patches.scale}",
            file=sys.stderr,
        )
        return 2

    print("=" * 70)
    print("  PREPARE DATASET")
    print("=" * 70)

    sample_path = None
    scenes = discover_scenes(cfg, args.input)
    if not scenes and args.synthetic:
        print(f"\ngenerating {args.synthetic_scenes} synthetic scene(s):")
        scenes = make_synthetic(cfg, args.synthetic_scenes, args.synthetic_size)
        sample_path = make_sample_input(cfg, scenes[0], int(cfg.patches.scale))
    elif not scenes:
        raw_dir = cfg.get_path("data.raw_dir")
        print(
            f"\nNo scenes found in {raw_dir}.\n"
            "  Place Sentinel-2 GeoTIFFs there (bands ordered as data.band_indices),\n"
            "  or re-run with --synthetic to generate demo data.",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.output) if args.output else cfg.get_path("data.patch_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- size the store ---------------------------------------------------
    print(f"\nscenes: {len(scenes)}")
    upper_bound = 0
    usable: list[Path] = []
    for path in scenes:
        info = read_info(path)
        if info.count < max(band_indices):
            print(
                f"  skip {path.name}: has {info.count} band(s), config needs "
                f"band index {max(band_indices)}"
            )
            continue
        grid = count_patch_grid(info, patch_size, stride)
        upper_bound += grid
        usable.append(path)
        print(
            f"  {path.name}: {info.width}x{info.height}, {info.count} bands, "
            f"{info.resolution[0]:g} m -> up to {grid} patches"
        )

    if not usable:
        print("\nerror: no usable scenes", file=sys.stderr)
        return 1

    capacity = min(upper_bound, max_patches)
    if capacity == 0:
        print(
            f"\nerror: patch size {patch_size} exceeds every scene's dimensions",
            file=sys.stderr,
        )
        return 1

    # -- extract ----------------------------------------------------------
    temp_path = out_dir / "_patches_tmp.npy"
    store = np.lib.format.open_memmap(
        temp_path,
        mode="w+",
        dtype=STORE_DTYPE,
        shape=(capacity, len(band_indices), patch_size, patch_size),
    )
    print(f"\nallocated {_size_str(store.nbytes)} for up to {capacity} patches")

    records: list[PatchRecord] = []
    written = 0
    for path in usable:
        info = read_info(path)
        before = written
        for patch, patch_info in extract_patches(
            path,
            band_indices,
            patch_size,
            stride,
            dn_scale=float(cfg.data.dn_scale),
            max_nodata_fraction=float(cfg.patches.max_nodata_fraction),
            min_std=float(cfg.patches.min_std),
            max_patches=capacity - written,
            nodata_fill=float(cfg.data.nodata_fill),
        ):
            store[written] = np.clip(patch * STORE_SCALE, 0, 65535).astype(STORE_DTYPE)
            col = round((patch_info.bounds.left - info.bounds.left) / info.resolution[0])
            row = round((info.bounds.top - patch_info.bounds.top) / info.resolution[1])
            records.append(PatchRecord(written, path.name, int(row), int(col)))
            written += 1
            if written >= capacity:
                break
        print(f"  {path.name}: {written - before} patches accepted")
        if written >= capacity:
            if capacity == max_patches:
                print(f"  reached the max_patches cap ({max_patches}); "
                      "remaining scenes were not sampled")
            break

    store.flush()
    del store

    if written == 0:
        temp_path.unlink(missing_ok=True)
        print(
            "\nerror: every candidate patch was rejected. Lower "
            "patches.min_std or raise patches.max_nodata_fraction.",
            file=sys.stderr,
        )
        return 1

    array_path = _finalise(temp_path, out_dir / ARRAY_NAME, written)

    # -- manifest ---------------------------------------------------------
    train_idx, val_idx = spatial_split(
        records, float(cfg.patches.val_fraction), patch_size, seed=int(cfg.project.seed)
    )
    manifest = {
        "array": ARRAY_NAME,
        "count": written,
        "patch_size": patch_size,
        "stride": stride,
        "scale": int(cfg.patches.scale),
        "bands": bands,
        "band_indices": band_indices,
        "store_dtype": STORE_DTYPE,
        "store_scale": STORE_SCALE,
        "dn_scale": float(cfg.data.dn_scale),
        "target_resolution_m": float(cfg.data.target_resolution_m),
        "degradation": dict(cfg.patches.degradation),
        "seed": int(cfg.project.seed),
        "scenes": [str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p) for p in usable],
        "records": [r.as_dict() for r in records],
        "split": {"train": len(train_idx), "val": len(val_idx), "method": "spatial_block"},
    }
    manifest_path = write_manifest(out_dir, manifest)

    print("\n" + "-" * 70)
    print(f"  patches:  {written}")
    print(f"  split:    {len(train_idx)} train / {len(val_idx)} val (spatial blocks)")
    print(f"  HR patch: {patch_size}x{patch_size} @ {cfg.data.target_resolution_m} m")
    print(
        f"  LR patch: {patch_size // int(cfg.patches.scale)}"
        f"x{patch_size // int(cfg.patches.scale)} @ {cfg.data.source_resolution_m} m "
        "(synthesised on load)"
    )
    print(f"  array:    {array_path}  ({_size_str(array_path.stat().st_size)})")
    print(f"  manifest: {manifest_path}")
    if sample_path is not None:
        print(f"  sample:   {sample_path}  (10 m demo input)")
    print("-" * 70)
    print("\nnext: python scripts/train.py")
    return 0


def _finalise(temp_path: Path, final_path: Path, count: int) -> Path:
    """Trim the preallocated store to the number of patches actually accepted.

    Copied in chunks through a memmap so peak memory stays at one chunk rather
    than the whole dataset.
    """
    source = np.load(temp_path, mmap_mode="r")
    if count == source.shape[0]:
        del source
        final_path.unlink(missing_ok=True)
        temp_path.replace(final_path)
        return final_path

    target = np.lib.format.open_memmap(
        final_path, mode="w+", dtype=source.dtype, shape=(count, *source.shape[1:])
    )
    for start in range(0, count, COPY_CHUNK):
        stop = min(start + COPY_CHUNK, count)
        target[start:stop] = source[start:stop]
    target.flush()
    del target, source
    temp_path.unlink(missing_ok=True)
    return final_path


def _size_str(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} GB"


if __name__ == "__main__":
    raise SystemExit(main())
