"""Phase 3 + 6: evaluation protocols and the caveats attached to each."""

from __future__ import annotations

import json

import pytest
import torch

from src.evaluation.evaluate import (
    evaluate_reduced_resolution,
    evaluate_reference_free,
    evaluate_scene,
    format_report,
)
from src.models.baseline import InterpolationSR


@pytest.fixture
def scene(scene_path, cfg):
    from src.inference.predict import read_scene

    array, _, _ = read_scene(scene_path, list(cfg.data.band_indices))
    return array[:, :128, :128]


# ---------------------------------------------------------------------------
# Reduced-resolution (Wald's protocol)
# ---------------------------------------------------------------------------
def test_reduced_resolution_scores_the_baseline(scene, cfg):
    report = evaluate_reduced_resolution(scene, cfg)
    assert report.protocol == "reduced_resolution"
    assert "bicubic" in report.methods
    for metric in ("psnr", "ssim", "rmse", "sam", "ergas"):
        assert metric in report.methods["bicubic"]


def test_reduced_resolution_compares_a_model_to_the_baseline(scene, cfg):
    model = InterpolationSR(scale=4, in_channels=4)
    report = evaluate_reduced_resolution(scene, cfg, model, device=torch.device("cpu"))
    assert set(report.methods) == {"bicubic", "ai_sr"}
    assert report.comparison["reference"] == "bicubic"
    assert "psnr" in report.comparison["methods"]["ai_sr"]


def test_reduced_resolution_states_its_assumption(scene, cfg):
    report = evaluate_reduced_resolution(scene, cfg)
    assert any("Wald" in c or "coarser scale" in c for c in report.caveats)
    assert "assumed, not measured" in " ".join(report.caveats)


def test_metrics_are_plausible_for_a_real_scene(scene, cfg):
    """A bicubic reconstruction of a real scene should be decent, not perfect."""
    report = evaluate_reduced_resolution(scene, cfg)
    metrics = report.methods["bicubic"]
    assert 10.0 < metrics["psnr"] < 60.0
    assert 0.0 < metrics["ssim"] <= 1.0
    assert metrics["rmse"] > 0.0
    assert metrics["sam"] >= 0.0


def test_a_model_reporting_no_checkpoint_is_flagged(scene, cfg):
    report = evaluate_reduced_resolution(scene, cfg, model=None)
    assert any("No trained model" in c for c in report.caveats)


# ---------------------------------------------------------------------------
# Reference-free
# ---------------------------------------------------------------------------
def test_reference_free_reports_only_consistency(scene, cfg):
    report = evaluate_reference_free(scene, cfg)
    assert report.protocol == "reference_free"
    assert "psnr" not in report.methods["bicubic"]
    assert "reprojection_rmse" in report.methods["bicubic"]


def test_reference_free_states_that_it_is_not_quantitative(scene, cfg):
    report = evaluate_reference_free(scene, cfg)
    assert any("NO GROUND TRUTH" in c for c in report.caveats)
    assert any("cannot measure" in c for c in report.caveats)


# ---------------------------------------------------------------------------
# Full-resolution
# ---------------------------------------------------------------------------
def test_full_resolution_uses_a_supplied_reference(pair_paths, cfg):
    lr_path, hr_path = pair_paths
    report = evaluate_scene(lr_path, cfg, reference_path=hr_path)
    assert report.protocol == "full_resolution"
    assert "psnr" in report.methods["bicubic"]
    assert report.geospatial["alignment"]["aligned"]


def test_misaligned_reference_is_flagged(tmp_path, pair_paths, cfg):
    """A shifted reference must not silently produce confident numbers."""
    from dataclasses import replace

    from affine import Affine

    from src.data.geotiff import read_info, read_raster, write_raster

    lr_path, hr_path = pair_paths
    array, info = read_raster(hr_path, [1, 2, 3, 4], dtype="uint16")
    shifted = replace(info, transform=Affine.translation(200.0, 0.0) @ info.transform)
    bad_ref = write_raster(tmp_path / "shifted.tif", array.astype("uint16"), shifted)

    report = evaluate_scene(lr_path, cfg, reference_path=bad_ref)
    assert any("alignment check FAILED" in c for c in report.caveats)


# ---------------------------------------------------------------------------
# Dispatch and serialisation
# ---------------------------------------------------------------------------
def test_evaluate_scene_defaults_to_reduced_resolution(scene_path, cfg):
    assert evaluate_scene(scene_path, cfg).protocol == "reduced_resolution"


def test_evaluate_scene_rejects_an_unknown_protocol(scene_path, cfg):
    with pytest.raises(ValueError, match="unknown protocol"):
        evaluate_scene(scene_path, cfg, protocol="magic")


def test_report_serialises_to_json(scene, cfg, tmp_path):
    report = evaluate_reduced_resolution(scene, cfg)
    payload = json.loads(report.save(tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["protocol"] == "reduced_resolution"
    assert "protocol_description" in payload
    assert payload["caveats"]


def test_report_flattens_to_csv_rows(scene, cfg):
    rows = evaluate_reduced_resolution(scene, cfg).to_rows()
    assert rows and rows[0]["method"] == "bicubic"
    assert any(key.startswith("rmse_B") for key in rows[0])


def test_format_report_includes_the_caveats(scene, cfg):
    text = format_report(evaluate_reduced_resolution(scene, cfg))
    assert "CAVEATS" in text
    assert "bicubic" in text


def test_geospatial_metadata_is_carried_into_the_report(scene_path, cfg):
    report = evaluate_scene(scene_path, cfg)
    assert report.geospatial["input"]["epsg"] == 32643
    assert report.geospatial["input"]["resolution_x"] == pytest.approx(10.0)
