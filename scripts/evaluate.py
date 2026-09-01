#!/usr/bin/env python
"""Phase 3 + 6 + 9: evaluate the model against the bicubic baseline.

    python scripts/evaluate.py
    python scripts/evaluate.py --input scene.tif --reference hires.tif
    python scripts/evaluate.py --protocol reduced_resolution --downstream

Writes per-scene JSON reports and a combined CSV to ``outputs/evaluation``.
Every report records which evaluation protocol produced its numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from src.config import load_config, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", nargs="*", default=None, help="scene GeoTIFF(s)")
    parser.add_argument("--reference", default=None, help="co-registered high-res reference")
    parser.add_argument("--checkpoint", default=None, help="model checkpoint (default: checkpoints/best.pth)")
    parser.add_argument(
        "--protocol",
        default="auto",
        choices=["auto", "reduced_resolution", "full_resolution", "reference_free"],
    )
    parser.add_argument("--output", default=None, help="results directory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--downstream", action="store_true", help="also run the land-cover experiment")
    parser.add_argument("--baseline-only", action="store_true", help="skip the learned model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from src.evaluation.evaluate import evaluate_scene, format_report
    from src.inference.predict import load_checkpoint, resolve_device

    args = parse_args(argv)
    cfg = load_config(args.config)
    set_seed(int(cfg.project.seed))
    device = resolve_device(args.device)

    scenes = _discover(cfg, args.input)
    if not scenes:
        print(
            f"error: no scenes found. Pass --input, or place GeoTIFFs in "
            f"{cfg.get_path('data.raw_dir')} (see scripts/prepare_dataset.py --synthetic).",
            file=sys.stderr,
        )
        return 1

    model = None
    if not args.baseline_only:
        checkpoint = Path(args.checkpoint) if args.checkpoint else cfg.get_path(
            "training.checkpoint_dir"
        ) / "best.pth"
        if checkpoint.exists():
            model, payload = load_checkpoint(checkpoint, device)
            print(f"model: {checkpoint} (epoch {payload.get('epoch')})")
        else:
            print(
                f"warning: no checkpoint at {checkpoint} — evaluating the bicubic "
                "baseline only. Run scripts/train.py for an AI-SR comparison."
            )

    out_dir = Path(args.output) if args.output else cfg.get_path("evaluation.results_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for scene in scenes:
        report = evaluate_scene(
            scene,
            cfg,
            model,
            reference_path=args.reference,
            protocol=args.protocol,
            device=device,
        )
        print(format_report(report))
        report.save(out_dir / f"{Path(scene).stem}_metrics.json")
        rows = report.to_rows()
        for row in rows:
            row["scene"] = Path(scene).name
        all_rows.extend(rows)

        if args.downstream:
            _run_downstream(scene, cfg, model, device, out_dir)

    if all_rows:
        csv_path = out_dir / "metrics.csv"
        fields = sorted({k for row in all_rows for k in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"results: {out_dir}")
        print(f"  {csv_path}")
    return 0


def _discover(cfg, explicit: list[str] | None) -> list[Path]:
    if explicit:
        paths = [Path(p) for p in explicit]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"scene(s) not found: {missing}")
        return paths
    raw_dir = cfg.get_path("data.raw_dir")
    return sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff", ".jp2"))


def _run_downstream(scene: Path, cfg, model, device, out_dir: Path) -> None:
    """Phase 9: land-cover classification on bicubic vs AI-SR imagery."""
    import numpy as np

    from src.applications.urban_mapping import run_experiment
    from src.data.preprocessing import DegradationConfig, bicubic_upsample, degrade
    from src.inference.predict import read_scene, super_resolve_array

    scale = int(cfg.patches.scale)
    reference, _, _ = read_scene(scene, list(cfg.data.band_indices), float(cfg.data.dn_scale))

    # Reduced-resolution setup: the observed scene is the ground truth and the
    # products are reconstructions of it, so the comparison uses real data.
    _, height, width = reference.shape
    height, width = height - height % scale, width - width % scale
    reference = reference[:, :height, :width]
    coarse = degrade(reference, scale, DegradationConfig.from_config(cfg.patches.degradation))

    products = {"bicubic": bicubic_upsample(coarse, scale)}
    if model is not None:
        products["ai_sr"] = super_resolve_array(
            model,
            coarse,
            scale,
            tile_size=int(cfg.inference.tile_size),
            overlap=int(cfg.inference.tile_overlap),
            batch_size=int(cfg.inference.batch_size),
            device=device,
        )

    labels = None
    labels_path = cfg.application.get("labels_path")
    if labels_path:
        from src.data.geotiff import read_raster

        labels = read_raster(labels_path)[0][0][:height, :width].astype(np.int16)

    result = run_experiment(products, cfg, reference=reference, labels=labels)

    print("  DOWNSTREAM — land-cover classification")
    for method, values in result.metrics.items():
        accuracy = values.get("overall_accuracy")
        miou = values.get("mean_iou")
        if accuracy is not None:
            print(f"    {method:<10} accuracy {accuracy:.4f}   mIoU {miou:.4f}")
        else:
            print(f"    {method:<10} {values['structural']}")
    print(f"    verdict: {result.verdict}")
    for caveat in result.caveats:
        print(f"    caveat:  {caveat}")

    app_dir = Path(cfg.get_path("application.results_dir"))
    result.save(app_dir / f"{scene.stem}_landcover.json")
    print(f"    saved:   {app_dir / f'{scene.stem}_landcover.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
