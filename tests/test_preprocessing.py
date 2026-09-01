"""Phase 1 + 2: normalisation, nodata, degradation and patch generation."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.geotiff import read_info, read_raster
from src.data.preprocessing import (
    DegradationConfig,
    apply_nodata,
    bicubic_upsample,
    count_patch_grid,
    degrade,
    denormalize_reflectance,
    extract_patches,
    make_pair,
    nodata_mask_from_value,
    normalize_reflectance,
    percentile_stretch,
    resize,
    to_rgb,
)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_normalize_maps_dn_to_reflectance():
    dn = np.array([[[0, 1000, 5000, 10000]]], dtype=np.uint16)
    out = normalize_reflectance(dn, dn_scale=10000.0)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], [0.0, 0.1, 0.5, 1.0], atol=1e-6)


def test_normalize_clips_out_of_range_dn():
    dn = np.array([[[-500, 20000]]], dtype=np.int32)
    out = normalize_reflectance(dn, dn_scale=10000.0, clip=(0.0, 1.0))
    np.testing.assert_allclose(out[0, 0], [0.0, 1.0])


def test_normalize_denormalize_roundtrip():
    dn = np.random.default_rng(0).integers(0, 10000, size=(4, 16, 16)).astype("uint16")
    back = denormalize_reflectance(normalize_reflectance(dn), dtype="uint16")
    np.testing.assert_allclose(back, dn, atol=1)


def test_denormalize_saturates_instead_of_wrapping():
    out = denormalize_reflectance(np.full((1, 2, 2), 50.0, dtype=np.float32))
    assert out.dtype == np.uint16
    assert (out == np.iinfo("uint16").max).all()


def test_normalize_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="dn_scale"):
        normalize_reflectance(np.zeros((1, 2, 2)), dn_scale=0.0)


# ---------------------------------------------------------------------------
# Nodata
# ---------------------------------------------------------------------------
def test_apply_nodata_fills_invalid_pixels_in_every_band():
    array = np.ones((4, 8, 8), dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    mask[0:2, 0:3] = False
    out = apply_nodata(array, mask, fill=0.0)
    assert (out[:, 0:2, 0:3] == 0).all()
    assert (out[:, 4:, 4:] == 1).all()
    assert (array == 1).all(), "input must not be mutated"


def test_apply_nodata_is_a_noop_without_a_mask():
    array = np.ones((2, 4, 4), dtype=np.float32)
    assert apply_nodata(array, None) is array


def test_nodata_mask_requires_all_bands_to_be_sentinel():
    array = np.zeros((3, 4, 4), dtype=np.float32)
    array[0, 0, 0] = 0.5  # one band has data -> pixel is valid
    mask = nodata_mask_from_value(array, nodata=0)
    assert mask[0, 0]
    assert not mask[1, 1]


def test_nodata_mask_handles_nan_sentinel():
    array = np.zeros((2, 3, 3), dtype=np.float32)
    array[:, 1, 1] = np.nan
    mask = nodata_mask_from_value(array, nodata=float("nan"))
    assert not mask[1, 1]
    assert mask[0, 0]


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def test_resize_changes_shape_and_keeps_channels():
    array = np.random.default_rng(0).random((7, 32, 32)).astype(np.float32)
    out = resize(array, (16, 24), "bilinear")
    assert out.shape == (7, 16, 24)
    assert out.dtype == np.float32


def test_resize_preserves_a_constant_image():
    array = np.full((6, 16, 16), 0.42, dtype=np.float32)
    out = resize(array, (64, 64), "bicubic")
    np.testing.assert_allclose(out, 0.42, atol=1e-5)


def test_resize_rejects_unknown_interpolation():
    with pytest.raises(ValueError, match="unknown interpolation"):
        resize(np.zeros((1, 4, 4), dtype=np.float32), (8, 8), "magic")


def test_bicubic_upsample_scales_by_the_factor():
    array = np.random.default_rng(0).random((4, 16, 20)).astype(np.float32)
    out = bicubic_upsample(array, 4)
    assert out.shape == (4, 64, 80)


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------
def test_degrade_reduces_size_by_the_scale():
    hr = np.random.default_rng(0).random((4, 128, 128)).astype(np.float32)
    lr = degrade(hr, 4)
    assert lr.shape == (4, 32, 32)
    assert lr.dtype == np.float32


def test_degrade_keeps_values_in_reflectance_range():
    hr = np.random.default_rng(0).random((4, 64, 64)).astype(np.float32)
    lr = degrade(hr, 4, DegradationConfig(noise_std=0.05))
    assert lr.min() >= 0.0 and lr.max() <= 1.0


def test_degrade_preserves_the_scene_mean():
    """Blur + area decimation is (near) mean-preserving — no radiometric bias."""
    hr = np.random.default_rng(1).random((4, 128, 128)).astype(np.float32) * 0.5 + 0.2
    lr = degrade(hr, 4)
    assert lr.mean() == pytest.approx(hr.mean(), abs=0.01)


def test_degrade_removes_high_frequency_detail():
    hr = np.random.default_rng(2).random((4, 128, 128)).astype(np.float32)
    lr = degrade(hr, 4)
    # Low-pass filtering must reduce variance; pure decimation would not.
    assert lr.std() < hr.std()


def test_degrade_rejects_indivisible_sizes():
    with pytest.raises(ValueError, match="divisible"):
        degrade(np.zeros((4, 30, 30), dtype=np.float32), 4)


def test_degrade_is_deterministic_without_noise():
    hr = np.random.default_rng(3).random((4, 64, 64)).astype(np.float32)
    np.testing.assert_array_equal(degrade(hr, 4), degrade(hr, 4))


def test_degradation_noise_uses_the_supplied_generator():
    hr = np.random.default_rng(4).random((4, 64, 64)).astype(np.float32)
    cfg = DegradationConfig(noise_std=0.02)
    a = degrade(hr, 4, cfg, rng=np.random.default_rng(7))
    b = degrade(hr, 4, cfg, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_make_pair_returns_lr_then_hr():
    hr = np.random.default_rng(5).random((4, 64, 64)).astype(np.float32)
    lr, ref = make_pair(hr, 4)
    assert lr.shape == (4, 16, 16)
    assert ref is hr


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------
def test_extract_patches_yields_fixed_size_normalised_patches(scene_path):
    patches = list(
        extract_patches(scene_path, [1, 2, 3, 4], patch_size=64, stride=64, max_patches=8)
    )
    assert patches, "expected at least one accepted patch"
    for patch, info in patches:
        assert patch.shape == (4, 64, 64)
        assert patch.dtype == np.float32
        assert 0.0 <= patch.min() and patch.max() <= 1.0
        assert info.crs is not None
        assert info.resolution == pytest.approx((10.0, 10.0))


def test_extract_patches_respects_max_patches(scene_path):
    patches = list(
        extract_patches(scene_path, [1, 2, 3, 4], patch_size=32, stride=32, max_patches=5)
    )
    assert len(patches) == 5


def test_extract_patches_are_spatially_distinct(scene_path):
    origins = {
        (info.bounds.left, info.bounds.top)
        for _, info in extract_patches(
            scene_path, [1, 2, 3, 4], patch_size=64, stride=64, max_patches=9
        )
    }
    assert len(origins) > 1


def test_extract_patches_rejects_flat_regions(tmp_path, scene_path):
    """A constant scene carries no high-frequency signal and must be skipped."""
    from src.data.geotiff import write_raster

    info = read_info(scene_path)
    flat = np.full((4, info.height, info.width), 2000, dtype="uint16")
    path = write_raster(tmp_path / "flat.tif", flat, info)

    patches = list(extract_patches(path, [1, 2, 3, 4], 64, 64, min_std=0.002))
    assert patches == []


def test_extract_patches_rejects_mostly_nodata_windows(tmp_path, scene_path):
    from src.data.geotiff import write_raster

    array, info = read_raster(scene_path, [1, 2, 3, 4], dtype="uint16")
    array[:, :128, :] = 0  # top half becomes nodata
    path = write_raster(tmp_path / "half_nodata.tif", array.astype("uint16"), info)

    tops = [
        info_.bounds.top
        for _, info_ in extract_patches(path, [1, 2, 3, 4], 64, 64, max_nodata_fraction=0.1)
    ]
    assert tops, "the valid half should still produce patches"
    assert min(tops) < info.bounds.top - 128 * 10.0 + 1e-6


def test_extract_patches_never_loads_the_full_scene(scene_path, monkeypatch):
    """Guard the memory contract: every read must be windowed."""
    import src.data.preprocessing as pre

    real_read = pre.read_raster
    calls = []

    def spy(path, band_indices=None, window=None, dtype="float32"):
        calls.append(window)
        return real_read(path, band_indices, window, dtype)

    monkeypatch.setattr(pre, "read_raster", spy)
    list(extract_patches(scene_path, [1, 2, 3, 4], 64, 64, max_patches=3))

    assert calls, "no reads were issued"
    assert all(w is not None for w in calls)


def test_count_patch_grid_matches_extraction_geometry(scene_path):
    info = read_info(scene_path)
    # 256 / 64 = 4 across, 4 down, no overlap.
    assert count_patch_grid(info, patch_size=64, stride=64) == 16


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def test_percentile_stretch_spans_the_unit_interval():
    array = np.random.default_rng(0).random((3, 32, 32)).astype(np.float32) * 0.3
    out = percentile_stretch(array, (2.0, 98.0))
    assert out.min() == pytest.approx(0.0, abs=1e-5)
    assert out.max() == pytest.approx(1.0, abs=1e-5)


def test_percentile_stretch_handles_a_constant_band():
    array = np.zeros((2, 8, 8), dtype=np.float32)
    out = percentile_stretch(array)
    assert np.isfinite(out).all()
    assert (out == 0).all()


def test_to_rgb_reorders_bands_into_hwc():
    array = np.zeros((4, 8, 8), dtype=np.float32)
    array[0] = 0.9  # B04 / red
    rgb = to_rgb(array, ["B04", "B03", "B02", "B08"], ["B04", "B03", "B02"], stretch=False)
    assert rgb.shape == (8, 8, 3)
    assert rgb[..., 0].mean() > rgb[..., 1].mean()


def test_to_rgb_reports_missing_bands():
    array = np.zeros((2, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="not present"):
        to_rgb(array, ["B04", "B03"], ["B04", "B03", "B02"])
