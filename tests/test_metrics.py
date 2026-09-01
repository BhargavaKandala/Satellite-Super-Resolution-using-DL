"""Phase 3 + 6: metric correctness, including the spectral-consistency family."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    compare,
    compute_metrics,
    ergas,
    per_band_rmse,
    psnr,
    rmse,
    sam,
    sam_map,
    ssim,
)


@pytest.fixture
def image():
    return np.random.default_rng(0).random((4, 64, 64)).astype(np.float32) * 0.5 + 0.1


# ---------------------------------------------------------------------------
# Reconstruction quality
# ---------------------------------------------------------------------------
def test_rmse_is_zero_for_identical_images(image):
    assert rmse(image, image) == pytest.approx(0.0, abs=1e-9)


def test_rmse_matches_a_known_offset(image):
    assert rmse(image + 0.1, image) == pytest.approx(0.1, abs=1e-6)


def test_psnr_is_infinite_for_a_perfect_match(image):
    assert psnr(image, image) == float("inf")


def test_psnr_matches_the_analytic_value(image):
    # error of 0.1 at data_range 1.0 -> 20*log10(1/0.1) = 20 dB
    assert psnr(image + 0.1, image, data_range=1.0) == pytest.approx(20.0, abs=1e-4)


def test_psnr_decreases_as_error_grows(image):
    assert psnr(image + 0.01, image) > psnr(image + 0.1, image)


def test_ssim_is_one_for_identical_images(image):
    assert ssim(image, image) == pytest.approx(1.0, abs=1e-6)


def test_ssim_drops_for_a_noisy_image(image):
    noisy = image + np.random.default_rng(1).normal(0, 0.05, image.shape).astype(np.float32)
    assert ssim(noisy, image) < 0.95


def test_per_band_rmse_isolates_a_single_bad_band(image):
    corrupted = image.copy()
    corrupted[3] += 0.2
    values = per_band_rmse(corrupted, image, band_names=["B04", "B03", "B02", "B08"])
    assert values["B08"] == pytest.approx(0.2, abs=1e-6)
    assert values["B04"] == pytest.approx(0.0, abs=1e-9)


def test_per_band_rmse_rejects_a_wrong_name_count(image):
    with pytest.raises(ValueError, match="band names"):
        per_band_rmse(image, image, band_names=["only_one"])


# ---------------------------------------------------------------------------
# Spectral consistency
# ---------------------------------------------------------------------------
def test_sam_is_zero_for_identical_images(image):
    assert sam(image, image) == pytest.approx(0.0, abs=1e-6)


def test_sam_ignores_uniform_brightness_scaling(image):
    """The defining property: SAM measures direction, not magnitude."""
    assert sam(image * 1.7, image) == pytest.approx(0.0, abs=1e-4)


def test_sam_detects_a_band_ratio_change(image):
    shifted = image.copy()
    shifted[3] *= 1.5  # NIR only
    assert sam(shifted, image) > 1.0


def test_sam_matches_a_hand_computed_angle():
    target = np.array([[[1.0]], [[0.0]]])
    pred = np.array([[[1.0]], [[1.0]]])  # 45 degrees apart
    assert sam(pred, target) == pytest.approx(45.0, abs=1e-4)


def test_sam_requires_at_least_two_bands():
    single = np.random.default_rng(0).random((1, 16, 16))
    with pytest.raises(ValueError, match="at least 2 spectral bands"):
        sam(single, single)


def test_sam_map_has_image_shape_and_matches_the_scalar(image):
    shifted = image.copy()
    shifted[3] *= 1.2
    per_pixel = sam_map(shifted, image)
    assert per_pixel.shape == image.shape[-2:]
    assert per_pixel.mean() == pytest.approx(sam(shifted, image), abs=1e-4)


def test_ergas_is_zero_for_identical_images(image):
    assert ergas(image, image, ratio=4.0) == pytest.approx(0.0, abs=1e-9)


def test_ergas_grows_with_error(image):
    small = ergas(image + 0.01, image, ratio=4.0)
    large = ergas(image + 0.05, image, ratio=4.0)
    assert 0 < small < large


def test_ergas_scales_inversely_with_the_ratio(image):
    """ERGAS = (100/ratio) * ..., so doubling the ratio halves the value."""
    a = ergas(image + 0.05, image, ratio=2.0)
    b = ergas(image + 0.05, image, ratio=4.0)
    assert a == pytest.approx(2 * b, rel=1e-6)


def test_ergas_rejects_a_nonpositive_ratio(image):
    with pytest.raises(ValueError, match="ratio must be positive"):
        ergas(image, image, ratio=0.0)


# ---------------------------------------------------------------------------
# Masking and cropping
# ---------------------------------------------------------------------------
def test_masked_pixels_are_excluded(image):
    corrupted = image.copy()
    corrupted[:, :10, :10] = 0.0
    mask = np.ones(image.shape[-2:], dtype=bool)
    mask[:10, :10] = False
    assert rmse(corrupted, image, valid_mask=mask) == pytest.approx(0.0, abs=1e-9)
    assert rmse(corrupted, image) > 0


def test_border_crop_excludes_the_edges(image):
    corrupted = image.copy()
    corrupted[:, :2, :] = 0.0
    assert rmse(corrupted, image, border_crop=4) == pytest.approx(0.0, abs=1e-9)


def test_shape_mismatch_is_reported(image):
    with pytest.raises(ValueError, match="must match"):
        rmse(image, image[:, :32, :32])


def test_a_fully_masked_image_is_rejected(image):
    mask = np.zeros(image.shape[-2:], dtype=bool)
    with pytest.raises(ValueError, match="no valid pixels"):
        rmse(image, image, valid_mask=mask)


def test_nan_pixels_are_excluded(image):
    corrupted = image.copy()
    corrupted[0, 0, 0] = np.nan
    assert np.isfinite(rmse(corrupted, image))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_compute_metrics_returns_the_full_set(image):
    out = compute_metrics(image + 0.02, image, band_names=["B04", "B03", "B02", "B08"])
    assert set(out) >= {"psnr", "ssim", "rmse", "sam", "ergas", "per_band_rmse"}
    assert all(np.isfinite(v) for k, v in out.items() if isinstance(v, float))


def test_compute_metrics_honours_the_requested_subset(image):
    out = compute_metrics(image, image, metrics=["rmse"])
    assert "psnr" not in out and "rmse" in out


def test_compute_metrics_rejects_an_unknown_metric(image):
    with pytest.raises(ValueError, match="unknown metric"):
        compute_metrics(image, image, metrics=["psnr", "bogus"])


def test_compare_signs_deltas_by_metric_direction(image):
    """A positive delta must always mean 'better', whichever way the metric runs."""
    results = {
        "bicubic": {"psnr": 30.0, "ssim": 0.90, "rmse": 0.05, "sam": 2.0, "ergas": 4.0},
        "ai_sr": {"psnr": 33.0, "ssim": 0.95, "rmse": 0.03, "sam": 1.5, "ergas": 3.0},
    }
    table = compare(results, reference_key="bicubic")["methods"]["ai_sr"]
    assert table["psnr"]["delta"] == pytest.approx(3.0) and table["psnr"]["improved"]
    assert table["rmse"]["delta"] == pytest.approx(0.02) and table["rmse"]["improved"]
    assert table["sam"]["delta"] == pytest.approx(0.5) and table["sam"]["improved"]


def test_compare_flags_a_worse_method():
    results = {
        "bicubic": {"psnr": 30.0, "rmse": 0.05},
        "ai_sr": {"psnr": 28.0, "rmse": 0.07},
    }
    table = compare(results)["methods"]["ai_sr"]
    assert not table["psnr"]["improved"]
    assert not table["rmse"]["improved"]


def test_compare_rejects_a_missing_reference():
    with pytest.raises(KeyError, match="not in"):
        compare({"ai_sr": {"psnr": 30.0}}, reference_key="bicubic")


# ---------------------------------------------------------------------------
# Baseline sanity
# ---------------------------------------------------------------------------
def test_bicubic_beats_nearest_neighbour_on_a_real_degradation(scene_path, cfg):
    """Guards the whole metric stack against sign errors."""
    from src.data.preprocessing import bicubic_upsample, degrade, resize
    from src.inference.predict import read_scene

    scene, _, _ = read_scene(scene_path, list(cfg.data.band_indices))
    scale = int(cfg.patches.scale)
    reference = scene[:, :256, :256]
    coarse = degrade(reference, scale)

    bicubic = bicubic_upsample(coarse, scale)
    nearest = resize(coarse, reference.shape[-2:], "nearest")

    assert psnr(bicubic, reference) > psnr(nearest, reference)
    assert rmse(bicubic, reference) < rmse(nearest, reference)
