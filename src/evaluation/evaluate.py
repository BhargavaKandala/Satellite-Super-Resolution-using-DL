"""Phase 3 + 6: evaluation orchestration.

The reference problem
---------------------
Quantitative super-resolution metrics need a ground-truth image at the target
resolution. For a 10 m Sentinel-2 scene super-resolved to 2.5 m, that reference
usually does not exist. Two protocols are supported, and every report states
which one produced its numbers — reporting a metric without saying what it was
measured against is the most common way SR results become meaningless.

**Reduced-resolution (Wald's protocol)** — the default, and fully quantitative.
The observed 10 m scene is degraded to 40 m, super-resolved back to 10 m, and
compared against the original 10 m observation, which serves as real ground
truth. The assumption is that model behaviour transfers across the scale step
(40->10 behaves like 10->2.5); this is the standard assessment protocol in the
pansharpening and SR remote-sensing literature, and it is an assumption, not a
proof.

**Full-resolution** — used when a genuine higher-resolution reference image is
supplied. Metrics are then computed directly against that reference, with an
alignment check first, because a misregistered reference produces confidently
wrong numbers.

When neither is available, only reference-free consistency indicators are
reported and the result is explicitly marked as not quantitatively validated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..data.geotiff import RasterInfo, check_pair_alignment, read_info
from ..data.preprocessing import DegradationConfig, bicubic_upsample, degrade
from ..inference.predict import read_scene, resolve_device, super_resolve_array
from .metrics import compare, compute_metrics, sam_map

PROTOCOLS = ("reduced_resolution", "full_resolution", "reference_free")


@dataclass
class EvaluationReport:
    """Everything needed to reproduce and interpret a set of metrics."""

    protocol: str
    scale: int
    methods: dict[str, dict[str, Any]]
    comparison: dict[str, Any]
    geospatial: dict[str, Any]
    caveats: list[str]
    scene: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "protocol_description": _PROTOCOL_NOTES[self.protocol],
            "scale": self.scale,
            "scene": self.scene,
            "methods": self.methods,
            "comparison": self.comparison,
            "geospatial": self.geospatial,
            "caveats": self.caveats,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return path

    def to_rows(self) -> list[dict[str, Any]]:
        """Flatten to CSV-friendly rows, one per method."""
        rows = []
        for method, values in self.methods.items():
            row: dict[str, Any] = {"protocol": self.protocol, "method": method}
            for key, value in values.items():
                if key == "per_band_rmse":
                    row.update({f"rmse_{band}": v for band, v in value.items()})
                elif isinstance(value, (int, float, str)) or value is None:
                    row[key] = value
            rows.append(row)
        return rows


_PROTOCOL_NOTES = {
    "reduced_resolution": (
        "Wald's protocol: the scene was degraded by the scale factor, "
        "super-resolved back, and compared against the original observation as "
        "ground truth. Metrics are quantitative but assume model behaviour "
        "transfers across the scale step."
    ),
    "full_resolution": (
        "Compared directly against a supplied higher-resolution reference image "
        "covering the same footprint."
    ),
    "reference_free": (
        "No ground truth was available. Only reference-free consistency "
        "indicators are reported; these do NOT quantify reconstruction accuracy."
    ),
}


# ---------------------------------------------------------------------------
# Method predictions
# ---------------------------------------------------------------------------
def _predictions(
    lr: np.ndarray,
    scale: int,
    model: nn.Module | None,
    cfg,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Bicubic baseline plus, when supplied, the learned model."""
    out = {"bicubic": bicubic_upsample(lr, scale)}
    if model is not None:
        out["ai_sr"] = super_resolve_array(
            model,
            lr,
            scale,
            tile_size=int(cfg.inference.tile_size),
            overlap=int(cfg.inference.tile_overlap),
            batch_size=int(cfg.inference.batch_size),
            device=device or resolve_device(),
            amp=bool(cfg.training.amp),
            channels_last=bool(cfg.training.channels_last),
        )
    return out


def _score(
    predictions: dict[str, np.ndarray],
    reference: np.ndarray,
    cfg,
    valid_mask: np.ndarray | None,
    band_names: Sequence[str],
) -> dict[str, dict[str, Any]]:
    kwargs = dict(
        data_range=float(cfg.evaluation.data_range),
        ratio=float(cfg.evaluation.ergas_ratio),
        valid_mask=valid_mask,
        border_crop=int(cfg.evaluation.border_crop),
        band_names=band_names,
        metrics=list(cfg.evaluation.metrics),
    )
    return {
        name: compute_metrics(pred, reference, **kwargs)
        for name, pred in predictions.items()
    }


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
def evaluate_reduced_resolution(
    scene: np.ndarray,
    cfg,
    model: nn.Module | None = None,
    *,
    valid_mask: np.ndarray | None = None,
    device: torch.device | None = None,
    geospatial: dict[str, Any] | None = None,
    scene_name: str | None = None,
) -> EvaluationReport:
    """Wald's protocol: degrade, super-resolve back, compare to the observation."""
    scale = int(cfg.patches.scale)
    degradation = DegradationConfig.from_config(cfg.patches.degradation)

    # Crop to a multiple of the scale so degradation is exact.
    _, height, width = scene.shape
    height, width = height - height % scale, width - width % scale
    reference = scene[:, :height, :width]
    mask = valid_mask[:height, :width] if valid_mask is not None else None

    coarse = degrade(reference, scale, degradation)
    predictions = _predictions(coarse, scale, model, cfg, device)
    methods = _score(predictions, reference, cfg, mask, list(cfg.data.bands))

    caveats = [
        "Metrics come from Wald's reduced-resolution protocol: the model was "
        f"evaluated on a {scale}x step at coarser scale, not on the actual "
        "10 m -> "
        f"{cfg.data.target_resolution_m} m product.",
        "Performance at the operational scale is assumed, not measured.",
    ]
    if model is None:
        caveats.append("No trained model supplied; only the bicubic baseline was scored.")

    return EvaluationReport(
        protocol="reduced_resolution",
        scale=scale,
        methods=methods,
        comparison=compare(methods, reference_key="bicubic"),
        geospatial=geospatial or {},
        caveats=caveats,
        scene=scene_name,
    )


def evaluate_full_resolution(
    lr: np.ndarray,
    reference: np.ndarray,
    cfg,
    model: nn.Module | None = None,
    *,
    valid_mask: np.ndarray | None = None,
    device: torch.device | None = None,
    alignment: dict[str, Any] | None = None,
    geospatial: dict[str, Any] | None = None,
    scene_name: str | None = None,
) -> EvaluationReport:
    """Compare against a real higher-resolution reference image."""
    scale = int(cfg.patches.scale)
    expected = (lr.shape[1] * scale, lr.shape[2] * scale)
    if reference.shape[-2:] != expected:
        raise ValueError(
            f"reference is {reference.shape[-2:]} but the input {lr.shape[-2:]} "
            f"at scale {scale} implies {expected}; crop or resample first"
        )

    predictions = _predictions(lr, scale, model, cfg, device)
    methods = _score(predictions, reference, cfg, valid_mask, list(cfg.data.bands))

    caveats = []
    if alignment and not alignment.get("aligned", True):
        caveats.append(
            "Reference alignment check FAILED — metrics are unreliable: "
            + "; ".join(alignment.get("warnings", []))
        )
    if model is None:
        caveats.append("No trained model supplied; only the bicubic baseline was scored.")

    return EvaluationReport(
        protocol="full_resolution",
        scale=scale,
        methods=methods,
        comparison=compare(methods, reference_key="bicubic"),
        geospatial=geospatial or {},
        caveats=caveats,
        scene=scene_name,
    )


def evaluate_reference_free(
    lr: np.ndarray,
    cfg,
    model: nn.Module | None = None,
    *,
    device: torch.device | None = None,
    geospatial: dict[str, Any] | None = None,
    scene_name: str | None = None,
) -> EvaluationReport:
    """Reference-free consistency indicators when no ground truth exists.

    Reports how well each product reprojects back onto the observed pixels.
    This is a necessary condition for a valid reconstruction, not a sufficient
    one — it cannot score the sub-pixel detail, which is the whole point of the
    product.
    """
    scale = int(cfg.patches.scale)
    degradation = DegradationConfig.from_config(cfg.patches.degradation)
    predictions = _predictions(lr, scale, model, cfg, device)

    methods: dict[str, dict[str, Any]] = {}
    for name, pred in predictions.items():
        reprojected = degrade(pred, scale, degradation)
        methods[name] = {
            "reprojection_rmse": float(np.sqrt(np.mean((reprojected - lr) ** 2))),
            "reprojection_mae": float(np.mean(np.abs(reprojected - lr))),
            "reprojection_sam_deg": float(np.mean(sam_map(reprojected, lr))),
        }

    return EvaluationReport(
        protocol="reference_free",
        scale=scale,
        methods=methods,
        comparison={"reference": "bicubic", "methods": {}},
        geospatial=geospatial or {},
        caveats=[
            "NO GROUND TRUTH AVAILABLE. These are consistency indicators only.",
            "They cannot measure the accuracy of reconstructed fine-scale detail.",
            "Supply a co-registered high-resolution reference, or use the "
            "reduced-resolution protocol, for quantitative results.",
        ],
        scene=scene_name,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def evaluate_scene(
    scene_path: str | Path,
    cfg,
    model: nn.Module | None = None,
    *,
    reference_path: str | Path | None = None,
    protocol: str = "auto",
    device: torch.device | None = None,
) -> EvaluationReport:
    """Evaluate one scene, choosing the protocol from what is available."""
    if protocol not in PROTOCOLS + ("auto",):
        raise ValueError(f"unknown protocol {protocol!r}; expected auto or {PROTOCOLS}")

    band_indices = list(cfg.data.band_indices)
    dn_scale = float(cfg.data.dn_scale)
    scene, mask, info = read_scene(scene_path, band_indices, dn_scale)
    geospatial = {"input": info.summary()}
    scene_name = Path(scene_path).name

    if reference_path is not None and protocol in ("auto", "full_resolution"):
        ref_info = read_info(reference_path)
        alignment = check_pair_alignment(info, ref_info, scale=int(cfg.patches.scale))
        reference, ref_mask, _ = read_scene(reference_path, band_indices, dn_scale)
        geospatial["reference"] = ref_info.summary()
        geospatial["alignment"] = alignment
        return evaluate_full_resolution(
            scene,
            reference,
            cfg,
            model,
            valid_mask=ref_mask,
            device=device,
            alignment=alignment,
            geospatial=geospatial,
            scene_name=scene_name,
        )

    if protocol == "reference_free":
        return evaluate_reference_free(
            scene, cfg, model, device=device, geospatial=geospatial, scene_name=scene_name
        )

    return evaluate_reduced_resolution(
        scene,
        cfg,
        model,
        valid_mask=mask,
        device=device,
        geospatial=geospatial,
        scene_name=scene_name,
    )


def format_report(report: EvaluationReport) -> str:
    """Human-readable summary for the CLI."""
    lines = [
        "",
        "=" * 74,
        f"  EVALUATION — {report.protocol.replace('_', ' ').upper()}",
        "=" * 74,
    ]
    if report.scene:
        lines.append(f"  scene: {report.scene}   scale: x{report.scale}")
    lines.append("")

    keys: list[str] = []
    for values in report.methods.values():
        for key in values:
            if key != "per_band_rmse" and key not in keys:
                keys.append(key)

    header = f"  {'method':<12}" + "".join(f"{k:>13}" for k in keys)
    lines += [header, "  " + "-" * (len(header) - 2)]
    for method, values in report.methods.items():
        row = f"  {method:<12}"
        for key in keys:
            value = values.get(key)
            row += f"{value:>13.4f}" if isinstance(value, (int, float)) else f"{'-':>13}"
        lines.append(row)

    deltas = report.comparison.get("methods", {}).get("ai_sr", {})
    if deltas:
        lines += ["", "  vs bicubic baseline:"]
        for metric, entry in deltas.items():
            delta = entry.get("delta")
            if delta is None:
                continue
            mark = "improved" if entry.get("improved") else "worse"
            lines.append(f"    {metric:<8} {delta:+.4f}  ({mark})")

    if report.caveats:
        lines += ["", "  CAVEATS:"]
        lines += [f"    - {c}" for c in report.caveats]
    lines += ["=" * 74, ""]
    return "\n".join(lines)
