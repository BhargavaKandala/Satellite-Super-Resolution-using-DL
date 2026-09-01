"""Phase 1 + 7: GeoTIFF I/O, windowing and spatial-metadata preservation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from src.data.geotiff import (
    check_pair_alignment,
    iter_windows,
    read_info,
    read_mask,
    read_raster,
    scaled_transform,
    superres_info,
    validate_geospatial,
    write_raster,
    write_superres,
)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def test_read_info_reports_geospatial_metadata(scene_path):
    info = read_info(scene_path)
    assert (info.width, info.height, info.count) == (256, 256, 4)
    assert info.crs == CRS.from_epsg(32643)
    assert info.resolution == pytest.approx((10.0, 10.0))
    assert info.is_georeferenced
    assert info.band_descriptions[:3] == ("B04", "B03", "B02")


def test_bounds_match_transform_and_size(scene_path):
    info = read_info(scene_path)
    b = info.bounds
    assert b.right - b.left == pytest.approx(info.width * 10.0)
    assert b.top - b.bottom == pytest.approx(info.height * 10.0)
    assert b.top > b.bottom and b.right > b.left


def test_read_raster_returns_channel_first_float(scene_path):
    array, info = read_raster(scene_path, band_indices=[1, 2, 3, 4])
    assert array.shape == (4, 256, 256)
    assert array.dtype == np.float32
    assert info.count == 4


def test_read_raster_band_subset_preserves_order(scene_path):
    full, _ = read_raster(scene_path, [1, 2, 3, 4])
    subset, info = read_raster(scene_path, [4, 1])
    assert subset.shape[0] == 2
    assert info.band_descriptions == ("B08", "B04")
    np.testing.assert_array_equal(subset[0], full[3])
    np.testing.assert_array_equal(subset[1], full[0])


def test_read_raster_rejects_out_of_range_band(scene_path):
    with pytest.raises(ValueError, match="out-of-range"):
        read_raster(scene_path, [1, 99])


def test_windowed_read_is_georeferenced_to_the_window(scene_path):
    from rasterio.windows import Window

    full_info = read_info(scene_path)
    window = Window(col_off=32, row_off=16, width=64, height=64)
    array, info = read_raster(scene_path, [1, 2, 3, 4], window=window)

    assert array.shape == (4, 64, 64)
    assert (info.width, info.height) == (64, 64)
    assert info.crs == full_info.crs
    # Window origin shifts the transform by exactly offset * pixel size.
    assert info.bounds.left == pytest.approx(full_info.bounds.left + 32 * 10.0)
    assert info.bounds.top == pytest.approx(full_info.bounds.top - 16 * 10.0)


def test_windowed_read_matches_the_full_read(scene_path):
    from rasterio.windows import Window

    full, _ = read_raster(scene_path, [1, 2, 3, 4])
    window = Window(col_off=10, row_off=20, width=32, height=48)
    part, _ = read_raster(scene_path, [1, 2, 3, 4], window=window)
    np.testing.assert_array_equal(part, full[:, 20:68, 10:42])


def test_read_mask_flags_nodata(tmp_path, scene_path):
    """Pixels written as the nodata sentinel must come back invalid."""
    array, info = read_raster(scene_path, [1, 2, 3, 4], dtype="uint16")
    array[:, :8, :8] = 0  # info.nodata is 0
    path = write_raster(tmp_path / "holed.tif", array.astype("uint16"), info)

    mask = read_mask(path)
    assert mask.shape == (256, 256)
    assert not mask[:8, :8].any()
    assert mask[100:, 100:].all()


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def test_iter_windows_covers_every_pixel(scene_path):
    info = read_info(scene_path)
    covered = np.zeros((info.height, info.width), dtype=bool)
    for w in iter_windows(info, tile_size=100, overlap=20):
        covered[
            int(w.row_off) : int(w.row_off + w.height),
            int(w.col_off) : int(w.col_off + w.width),
        ] = True
    assert covered.all()


def test_iter_windows_stays_inside_the_raster(scene_path):
    info = read_info(scene_path)
    for w in iter_windows(info, tile_size=100, overlap=20):
        assert w.col_off + w.width <= info.width
        assert w.row_off + w.height <= info.height


def test_iter_windows_handles_tile_larger_than_raster(scene_path):
    info = read_info(scene_path)
    windows = list(iter_windows(info, tile_size=4096))
    assert len(windows) == 1
    assert (windows[0].width, windows[0].height) == (info.width, info.height)


@pytest.mark.parametrize("overlap", [-1, 64])
def test_iter_windows_rejects_invalid_overlap(scene_path, overlap):
    info = read_info(scene_path)
    with pytest.raises(ValueError, match="overlap"):
        list(iter_windows(info, tile_size=64, overlap=overlap))


# ---------------------------------------------------------------------------
# Transform arithmetic (Phase 7)
# ---------------------------------------------------------------------------
def test_scaled_transform_shrinks_pixels_and_keeps_the_origin():
    src = Affine.translation(300000.0, 2000000.0) @ Affine.scale(10.0, -10.0)
    out = scaled_transform(src, 4)
    assert (out.a, out.e) == pytest.approx((2.5, -2.5))
    assert (out.c, out.f) == pytest.approx((src.c, src.f))


def test_superres_info_preserves_the_ground_footprint(scene_path):
    src = read_info(scene_path)
    sr = superres_info(src, scale=4)
    assert (sr.width, sr.height) == (src.width * 4, src.height * 4)
    assert sr.resolution == pytest.approx((2.5, 2.5))
    assert tuple(sr.bounds) == pytest.approx(tuple(src.bounds))
    assert sr.crs == src.crs


def test_scaled_transform_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="scale must be positive"):
        scaled_transform(Affine.identity(), 0)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def test_write_raster_roundtrips_pixels_and_metadata(tmp_path, scene_path):
    array, info = read_raster(scene_path, [1, 2, 3, 4], dtype="uint16")
    path = write_raster(tmp_path / "out.tif", array.astype("uint16"), info)

    back, back_info = read_raster(path, dtype="uint16")
    np.testing.assert_array_equal(back, array)
    assert back_info.crs == info.crs
    assert back_info.transform == info.transform
    assert back_info.nodata == info.nodata


def test_write_raster_accepts_2d_input(tmp_path, scene_path):
    info = read_info(scene_path)
    plane = np.zeros((info.height, info.width), dtype=np.float32)
    path = write_raster(tmp_path / "single.tif", plane, info, nodata=None)
    assert read_info(path).count == 1


def test_write_superres_produces_a_valid_geotiff(tmp_path, scene_path):
    src = read_info(scene_path)
    scale = 4
    sr = np.zeros((src.count, src.height * scale, src.width * scale), dtype=np.float32)
    path = write_superres(tmp_path / "sr.tif", sr, src, scale, nodata=None)

    out = read_info(path)
    assert (out.width, out.height) == (src.width * scale, src.height * scale)
    assert out.resolution == pytest.approx((2.5, 2.5))
    assert out.crs == src.crs
    assert tuple(out.bounds) == pytest.approx(tuple(src.bounds))


def test_write_superres_records_provenance_tags(tmp_path, scene_path):
    src = read_info(scene_path)
    sr = np.zeros((src.count, src.height * 4, src.width * 4), dtype=np.float32)
    path = write_superres(tmp_path / "sr_tagged.tif", sr, src, 4, nodata=None)

    tags = read_info(path).tags
    assert tags["SR_SCALE"] == "4"
    assert "AI-inferred" in tags["SR_DISCLAIMER"] or "AI-generated" in tags["SR_DISCLAIMER"]


def test_write_superres_rejects_wrongly_sized_arrays(tmp_path, scene_path):
    src = read_info(scene_path)
    wrong = np.zeros((src.count, src.height * 2, src.width * 2), dtype=np.float32)
    with pytest.raises(ValueError, match="implies"):
        write_superres(tmp_path / "bad.tif", wrong, src, 4)


def test_write_raster_preserves_band_descriptions(tmp_path, scene_path):
    array, info = read_raster(scene_path, [1, 2, 3, 4], dtype="uint16")
    path = write_raster(
        tmp_path / "named.tif",
        array.astype("uint16"),
        info,
        band_descriptions=["B04", "B03", "B02", "B08"],
    )
    assert read_info(path).band_descriptions == ("B04", "B03", "B02", "B08")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validate_geospatial_accepts_a_correct_product(tmp_path, scene_path):
    src = read_info(scene_path)
    sr = np.zeros((src.count, src.height * 4, src.width * 4), dtype=np.float32)
    path = write_superres(tmp_path / "good.tif", sr, src, 4, nodata=None)

    report = validate_geospatial(path, src, scale=4)
    assert report["valid"], report["checks"]
    assert all(report["checks"].values())


def test_validate_geospatial_rejects_a_copied_transform(tmp_path, scene_path):
    """The classic bug: writing SR pixels with the *source* transform."""
    src = read_info(scene_path)
    sr = np.zeros((src.count, src.height * 4, src.width * 4), dtype=np.float32)
    bad_info = read_info(scene_path)  # transform NOT rescaled
    path = write_raster(tmp_path / "bad_transform.tif", sr, bad_info, nodata=None)

    report = validate_geospatial(path, src, scale=4)
    assert not report["valid"]
    assert not report["checks"]["resolution_scaled"]
    assert not report["checks"]["bounds_preserved"]


def test_validate_geospatial_rejects_a_missing_crs(tmp_path, scene_path):
    src = read_info(scene_path)
    stripped = replace(superres_info(src, 4), crs=None)
    sr = np.zeros((src.count, src.height * 4, src.width * 4), dtype=np.float32)
    path = write_raster(tmp_path / "nocrs.tif", sr, stripped, nodata=None)

    report = validate_geospatial(path, src, scale=4)
    assert not report["valid"]
    assert not report["checks"]["has_crs"]


def test_check_pair_alignment_accepts_a_registered_pair(pair_paths):
    lr_path, hr_path = pair_paths
    report = check_pair_alignment(read_info(lr_path), read_info(hr_path), scale=4)
    assert report["aligned"], report["warnings"]


def test_check_pair_alignment_flags_a_shifted_reference(tmp_path, pair_paths):
    lr_path, hr_path = pair_paths
    hr = read_info(hr_path)
    shifted = replace(hr, transform=Affine.translation(37.0, 0.0) @ hr.transform)
    report = check_pair_alignment(read_info(lr_path), shifted, scale=4)
    assert not report["aligned"]
    assert any("origin offset" in w for w in report["warnings"])


def test_check_pair_alignment_flags_wrong_scale(pair_paths):
    lr_path, hr_path = pair_paths
    report = check_pair_alignment(read_info(lr_path), read_info(hr_path), scale=2)
    assert not report["aligned"]
    assert any("dimension mismatch" in w for w in report["warnings"])
