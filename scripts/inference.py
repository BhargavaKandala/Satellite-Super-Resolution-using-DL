#!/usr/bin/env python
"""Phase 7 + 8: super-resolve a Sentinel-2 GeoTIFF and quantify uncertainty.

    python scripts/inference.py --input sample.tif
    python scripts/inference.py --input scene.tif --output-dir outputs/run1
    python scripts/inference.py --input scene.tif --no-uncertainty --stream

Produces:
    <stem>_sr.tif            super-resolved GeoTIFF (below 4 m)
    <stem>_uncertainty.tif   per-pixel model-disagreement map
    <stem>_inference.json    geospatial validation + timings + caveats

The output carries the source CRS and a transform rescaled for the finer pixel
size, so it opens in QGIS/ArcGIS aligned with the input.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from src.config import load_config, set_seed

DISCLAIMER = (
    "Super-resolved imagery contains AI-inferred information and should not be "
    "interpreted as direct high-resolution observation without validation."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Sentinel-2 GeoTIFF")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help="cuda | cpu")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-uncertainty", action="store_true")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="write tiles straight to disk (required for full Sentinel-2 scenes)",
    )
    parser.add_argument("--baseline", action="store_true", help="use bicubic instead of the model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.data.geotiff import read_info, validate_geospatial, write_superres
    from src.inference.predict import (
        InferenceStats,
        check_overlap,
        describe_device,
        load_checkpoint,
        read_scene,
        resolve_device,
        super_resolve_array,
        super_resolve_file,
        write_uncertainty,
    )
    from src.models import build_model

    args = parse_args(argv)
    cfg = load_config(args.config)
    set_seed(int(cfg.project.seed))

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1

    scale = int(cfg.patches.scale)
    band_indices = list(cfg.data.band_indices)
    tile_size = args.tile_size or int(cfg.inference.tile_size)
    overlap = args.overlap if args.overlap is not None else int(cfg.inference.tile_overlap)
    batch_size = args.batch_size or int(cfg.inference.batch_size)
    device = resolve_device(args.device)

    print("=" * 70)
    print("  INFERENCE")
    print("=" * 70)

    src_info = read_info(input_path)
    if src_info.count < max(band_indices):
        print(
            f"error: {input_path.name} has {src_info.count} band(s) but the config "
            f"requires band index {max(band_indices)}. Check data.band_indices.",
            file=sys.stderr,
        )
        return 2
    if not src_info.is_georeferenced:
        print(
            f"warning: {input_path.name} has no CRS/transform — the output will "
            "not be geospatially usable."
        )

    print(f"\ninput:   {input_path.name}")
    print(f"  size:      {src_info.width} x {src_info.height}, {len(band_indices)} bands")
    print(f"  CRS:       {src_info.crs}")
    print(f"  pixel:     {src_info.resolution[0]:g} m")
    print(f"  bounds:    {tuple(round(v, 2) for v in src_info.bounds)}")
    print(f"  device:    {describe_device(device)}")

    # -- model ------------------------------------------------------------
    if args.baseline:
        model = build_model(cfg.merge({"model": {"name": "bicubic"}}))
        model_name = "bicubic (baseline)"
    else:
        checkpoint = Path(args.checkpoint) if args.checkpoint else cfg.get_path(
            "training.checkpoint_dir"
        ) / "best.pth"
        if not checkpoint.exists():
            print(
                f"\nerror: no checkpoint at {checkpoint}.\n"
                "  Run scripts/train.py first, or pass --baseline for the bicubic control.",
                file=sys.stderr,
            )
            return 1
        model, payload = load_checkpoint(checkpoint, device)
        model_name = f"{type(model).__name__} @ {checkpoint.name} (epoch {payload.get('epoch')})"

    print(f"  model:     {model_name}")
    for warning in check_overlap(overlap, int(cfg.model.num_blocks)):
        print(f"  warning:   {warning}")

    out_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    sr_path = out_dir / f"{input_path.stem}_sr.tif"

    # -- super-resolve ----------------------------------------------------
    uncertainty_result = None
    if args.stream:
        print("\nstreaming tiles to disk...")
        sr_path, out_info, stats = super_resolve_file(
            model,
            input_path,
            sr_path,
            scale=scale,
            band_indices=band_indices,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            dn_scale=float(cfg.data.dn_scale),
            output_dtype=str(cfg.inference.output_dtype),
            output_dn_scale=float(cfg.inference.output_dn_scale),
            compress=str(cfg.inference.compress),
            device=device,
            amp=bool(cfg.training.amp),
            channels_last=bool(cfg.training.channels_last),
            band_names=list(cfg.data.bands),
            progress=lambda done, total: print(f"  {done}/{total} tiles", end="\r", flush=True),
        )
        print(" " * 40, end="\r")
        if not args.no_uncertainty:
            print(
                "note: uncertainty estimation is skipped in --stream mode "
                "(it needs the scene in memory). Re-run without --stream on a subset."
            )
    else:
        lr, _, _ = read_scene(input_path, band_indices, float(cfg.data.dn_scale))
        started = time.perf_counter()
        tiling = dict(
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            amp=bool(cfg.training.amp),
            channels_last=bool(cfg.training.channels_last),
        )

        if args.no_uncertainty:
            print("\nsuper-resolving...")
            sr = super_resolve_array(model, lr, scale, device=device, **tiling)
        else:
            from src.uncertainty.uncertainty import estimate

            print(f"\nsuper-resolving with {cfg.uncertainty.method} uncertainty "
                  f"({cfg.uncertainty.passes} passes)...")
            uncertainty_result = estimate(model, lr, scale, cfg, device=device, **tiling)
            sr = uncertainty_result.prediction

        elapsed = time.perf_counter() - started
        stats = InferenceStats(
            tiles=-1,
            seconds=elapsed,
            device=describe_device(device),
            input_shape=lr.shape,
            output_shape=sr.shape,
        )

        out_dtype = str(cfg.inference.output_dtype)
        payload_array = (
            np.clip(sr * float(cfg.inference.output_dn_scale), 0, 65535).astype(out_dtype)
            if out_dtype != "float32"
            else sr.astype("float32")
        )
        write_superres(
            sr_path,
            payload_array,
            src_info,
            scale,
            nodata=None,
            tags={"SR_MODEL": type(model).__name__},
            compress=str(cfg.inference.compress),
        )
        out_info = read_info(sr_path)

    print(f"\noutput:  {sr_path.name}")
    print(f"  size:      {out_info.width} x {out_info.height}")
    print(f"  pixel:     {out_info.resolution[0]:g} m")
    print(f"  time:      {stats.seconds:.2f} s")

    # -- uncertainty ------------------------------------------------------
    uncertainty_path = None
    if uncertainty_result is not None:
        uncertainty_path = out_dir / f"{input_path.stem}_uncertainty.tif"
        write_uncertainty(
            uncertainty_path,
            uncertainty_result["uncertainty"],
            src_info,
            scale,
            tags={"SR_UNCERTAINTY_METHOD": uncertainty_result["method"]},
        )
        summary = uncertainty_result["statistics"]
        print(f"\nuncertainty: {uncertainty_path.name}")
        print(f"  method:    {uncertainty_result['method']} "
              f"({uncertainty_result['passes']} passes)")
        # Scientific notation: a well-behaved model's spread is often ~1e-4,
        # which fixed-point formatting would render as a misleading "0.00000".
        print(f"  mean:      {summary['mean']:.3e}   p95: {summary['p95']:.3e}")
        print(f"  reproj MAE:{summary['reprojection_mae']:.3e}")
        for note in uncertainty_result["notes"]:
            print(f"  note:      {note}")

    # -- geospatial validation --------------------------------------------
    validation = validate_geospatial(sr_path, src_info, scale)
    print("\ngeospatial validation:")
    for check, passed in validation["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check}")
    if not validation["valid"]:
        print("  ERROR: the output is not geospatially consistent with the input",
              file=sys.stderr)

    # -- report -----------------------------------------------------------
    report = {
        "input": str(input_path),
        "output": str(sr_path),
        "uncertainty": str(uncertainty_path) if uncertainty_path else None,
        "model": model_name,
        "scale": scale,
        "source_resolution_m": src_info.resolution[0],
        "output_resolution_m": out_info.resolution[0],
        "performance": stats.as_dict(),
        "geospatial_validation": validation,
        "uncertainty_summary": uncertainty_result["statistics"] if uncertainty_result else None,
        "disclaimer": DISCLAIMER,
    }
    report_path = out_dir / f"{input_path.stem}_inference.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\nreport:  {report_path}")
    print("\n" + "!" * 70)
    print(f"  {DISCLAIMER}")
    print("!" * 70)
    return 0 if validation["valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
