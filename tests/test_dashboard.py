"""Phase 10 + 11: dashboard helpers and the explainability guarantees.

The Streamlit callbacks themselves need a running server, so these tests cover
the pure helpers plus the pipeline function the UI delegates to — which is
where every actual computation happens.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


@pytest.fixture(scope="module")
def dashboard():
    return importlib.import_module("app.dashboard")


def test_dashboard_imports_cleanly(dashboard):
    """Catches syntax errors and bad imports without launching a server."""
    assert hasattr(dashboard, "main")
    assert hasattr(dashboard, "run_pipeline")


def test_disclaimer_matches_the_required_wording(dashboard):
    assert dashboard.DISCLAIMER == (
        "Super-resolved imagery contains AI-inferred information and should not be "
        "interpreted as direct high-resolution observation without validation."
    )


def test_render_rgb_produces_a_display_image(dashboard, cfg):
    array = np.random.default_rng(0).random((4, 128, 128)).astype(np.float32) * 0.4
    rgb = dashboard.render_rgb(array, cfg, max_dim=64)
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert max(rgb.shape[:2]) <= 128
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_downsampling_keeps_previews_bounded(dashboard):
    image = np.zeros((4096, 4096, 3), dtype=np.float32)
    assert max(dashboard._downsample(image, 512).shape[:2]) <= 1024


def test_colourise_returns_rgb_bytes(dashboard):
    values = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    out = dashboard.colourise(values, max_dim=64)
    assert out.dtype == np.uint8 and out.shape[2] == 3


def test_colourise_handles_a_constant_field(dashboard):
    out = dashboard.colourise(np.zeros((32, 32), dtype=np.float32))
    assert np.isfinite(out).all()


def test_crop_selects_the_same_ground_area_at_both_scales(dashboard):
    lr = np.arange(4 * 64 * 64, dtype=np.float32).reshape(4, 64, 64)
    sr = np.repeat(np.repeat(lr, 4, axis=1), 4, axis=2)
    roi = (8, 16, 32)

    lr_crop = dashboard.crop(lr, roi, 1)
    sr_crop = dashboard.crop(sr, roi, 4)
    assert lr_crop.shape == (4, 32, 32)
    assert sr_crop.shape == (4, 128, 128)
    # The SR crop must upsample exactly the same pixels the LR crop selected.
    np.testing.assert_array_equal(sr_crop[:, ::4, ::4], lr_crop)


def test_crop_without_a_region_is_a_noop(dashboard):
    array = np.zeros((4, 16, 16))
    assert dashboard.crop(array, None) is array


def test_pipeline_runs_and_reports_metrics(dashboard, scene_path, cfg, monkeypatch):
    """End-to-end exercise of what the 'Run super-resolution' button triggers."""
    monkeypatch.setattr(dashboard, "get_config", lambda *_: cfg)
    state = dashboard.run_pipeline(
        scene_path,
        cfg,
        checkpoint="bicubic baseline only",
        device_str="cpu",
        method="ensemble",
        passes=2,
        protocol="reduced_resolution",
    )
    assert state["sr"].shape[0] == 4
    assert "bicubic" in state["metrics"]
    assert state["reference"] is not None
    assert "preprocess" in state["timings"] and "inference" in state["timings"]


def test_reference_free_pipeline_reports_no_metrics(dashboard, scene_path, cfg):
    """The honesty path: no ground truth means no accuracy claims."""
    state = dashboard.run_pipeline(
        scene_path,
        cfg,
        checkpoint="bicubic baseline only",
        device_str="cpu",
        method="none",
        passes=2,
        protocol="reference_free",
    )
    assert state["metrics"] == {}
    assert state["reference"] is None
