"""Phase 7: tiled inference, seam-free stitching and GeoTIFF output."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.geotiff import read_info, validate_geospatial
from src.inference.predict import (
    _plan_blocks,
    check_overlap,
    describe_device,
    load_checkpoint,
    read_scene,
    resolve_device,
    super_resolve_array,
    super_resolve_file,
    write_uncertainty,
)
from src.models.baseline import InterpolationSR


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def test_resolve_device_returns_a_usable_device():
    device = resolve_device()
    assert device.type in ("cuda", "cpu")
    assert isinstance(describe_device(device), str)


def test_explicit_device_is_honoured():
    assert resolve_device("cpu").type == "cpu"


# ---------------------------------------------------------------------------
# Tiling plan
# ---------------------------------------------------------------------------
def test_blocks_cover_every_output_pixel_exactly_once():
    covered = np.zeros((100, 130), dtype=int)
    for block in _plan_blocks(100, 130, tile=32, pad=8):
        covered[block.out_r0 : block.out_r1, block.out_c0 : block.out_c1] += 1
    assert (covered == 1).all(), "blocks must tile the output without gaps or overlap"


def test_blocks_read_padded_context_inside_the_raster():
    for block in _plan_blocks(100, 130, tile=32, pad=8):
        assert block.r0 <= block.out_r0 and block.r1 >= block.out_r1
        assert 0 <= block.r0 and block.r1 <= 100
        assert 0 <= block.c0 and block.c1 <= 130


def test_blocks_handle_a_tile_larger_than_the_raster():
    blocks = list(_plan_blocks(20, 20, tile=256, pad=32))
    assert len(blocks) == 1
    assert (blocks[0].out_r1, blocks[0].out_c1) == (20, 20)


@pytest.mark.parametrize("tile, pad", [(0, 8), (32, -1)])
def test_block_planning_rejects_invalid_geometry(tile, pad):
    with pytest.raises(ValueError):
        list(_plan_blocks(64, 64, tile, pad))


def test_check_overlap_warns_when_padding_is_too_small():
    assert check_overlap(2, num_blocks=12)
    assert check_overlap(1024, num_blocks=12) == []


# ---------------------------------------------------------------------------
# Array inference
# ---------------------------------------------------------------------------
def test_super_resolve_array_scales_the_image():
    model = InterpolationSR(scale=4, in_channels=4)
    lr = np.random.default_rng(0).random((4, 40, 56)).astype(np.float32)
    out = super_resolve_array(model, lr, 4, tile_size=16, overlap=4, device=torch.device("cpu"))
    assert out.shape == (4, 160, 224)
    assert out.dtype == np.float32


def test_tiled_inference_matches_a_single_pass():
    """The seam test: tiling must be numerically invisible.

    Bicubic interpolation is not shift-invariant at tile edges, so a small
    tolerance is used; anything larger would show as visible seams.
    """
    model = InterpolationSR(scale=4, in_channels=4)
    lr = np.random.default_rng(1).random((4, 64, 64)).astype(np.float32)

    whole = super_resolve_array(model, lr, 4, tile_size=64, overlap=0, device=torch.device("cpu"))
    tiled = super_resolve_array(model, lr, 4, tile_size=16, overlap=8, device=torch.device("cpu"))
    assert np.abs(whole - tiled).max() < 0.02


def test_tiled_inference_produces_no_edge_discontinuity():
    """A smooth input must stay smooth across tile boundaries."""
    ramp = np.linspace(0, 1, 64, dtype=np.float32)
    lr = np.tile(ramp, (4, 64, 1))
    model = InterpolationSR(scale=4, in_channels=4)
    out = super_resolve_array(model, lr, 4, tile_size=16, overlap=8, device=torch.device("cpu"))

    # A monotone ramp has a near-constant horizontal derivative; a seam would
    # show up as an isolated spike.
    steps = np.diff(out[0, 128, :])
    assert steps.max() < steps.mean() * 3 + 1e-6


def test_batching_does_not_change_the_result():
    model = InterpolationSR(scale=4, in_channels=4)
    lr = np.random.default_rng(2).random((4, 48, 48)).astype(np.float32)
    a = super_resolve_array(model, lr, 4, tile_size=16, overlap=4, batch_size=1, device=torch.device("cpu"))
    b = super_resolve_array(model, lr, 4, tile_size=16, overlap=4, batch_size=8, device=torch.device("cpu"))
    np.testing.assert_allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# File inference
# ---------------------------------------------------------------------------
def test_super_resolve_file_writes_a_georeferenced_product(tmp_path, scene_path):
    model = InterpolationSR(scale=4, in_channels=4)
    out_path, out_info, stats = super_resolve_file(
        model,
        scene_path,
        tmp_path / "sr.tif",
        scale=4,
        band_indices=[1, 2, 3, 4],
        tile_size=64,
        overlap=8,
        device=torch.device("cpu"),
        amp=False,
    )
    src = read_info(scene_path)
    assert out_info.width == src.width * 4
    assert out_info.resolution == pytest.approx((2.5, 2.5))
    assert stats.tiles > 0 and stats.seconds > 0

    report = validate_geospatial(out_path, src, scale=4)
    assert report["valid"], report["checks"]


def test_streamed_output_matches_in_memory_inference(tmp_path, scene_path):
    """The two inference paths must agree — otherwise --stream changes results."""
    model = InterpolationSR(scale=4, in_channels=4)
    lr, _, src = read_scene(scene_path, [1, 2, 3, 4])

    in_memory = super_resolve_array(
        model, lr, 4, tile_size=64, overlap=8, device=torch.device("cpu"), amp=False
    )
    out_path, _, _ = super_resolve_file(
        model,
        scene_path,
        tmp_path / "streamed.tif",
        scale=4,
        band_indices=[1, 2, 3, 4],
        tile_size=64,
        overlap=8,
        device=torch.device("cpu"),
        amp=False,
    )
    streamed, _, _ = read_scene(out_path, [1, 2, 3, 4])
    # Written as uint16 at a 1e-4 quantisation step.
    np.testing.assert_allclose(streamed, in_memory, atol=2e-4)


def test_super_resolve_file_records_provenance(tmp_path, scene_path):
    model = InterpolationSR(scale=4, in_channels=4)
    out_path, _, _ = super_resolve_file(
        model,
        scene_path,
        tmp_path / "tagged.tif",
        scale=4,
        band_indices=[1, 2, 3, 4],
        tile_size=128,
        overlap=8,
        device=torch.device("cpu"),
        amp=False,
        band_names=["B04", "B03", "B02", "B08"],
    )
    info = read_info(out_path)
    assert info.tags["SR_SCALE"] == "4"
    assert "AI-generated" in info.tags["SR_DISCLAIMER"]
    assert info.band_descriptions == ("B04", "B03", "B02", "B08")


def test_float32_output_is_supported(tmp_path, scene_path):
    model = InterpolationSR(scale=4, in_channels=4)
    out_path, _, _ = super_resolve_file(
        model,
        scene_path,
        tmp_path / "float.tif",
        scale=4,
        band_indices=[1, 2, 3, 4],
        tile_size=128,
        overlap=8,
        output_dtype="float32",
        device=torch.device("cpu"),
        amp=False,
    )
    assert read_info(out_path).dtype == "float32"


# ---------------------------------------------------------------------------
# Uncertainty output
# ---------------------------------------------------------------------------
def test_write_uncertainty_produces_a_single_band_geotiff(tmp_path, scene_path):
    src = read_info(scene_path)
    values = np.random.default_rng(0).random((src.height * 4, src.width * 4)).astype(np.float32)
    path = write_uncertainty(tmp_path / "unc.tif", values, src, 4)

    info = read_info(path)
    assert info.count == 1
    assert info.dtype == "float32"
    assert info.crs == src.crs
    assert info.resolution == pytest.approx((2.5, 2.5))
    assert "not a calibrated probability" in info.tags["SR_UNCERTAINTY_NOTE"]


def test_write_uncertainty_collapses_multiband_input(tmp_path, scene_path):
    src = read_info(scene_path)
    values = np.zeros((4, src.height * 4, src.width * 4), dtype=np.float32)
    path = write_uncertainty(tmp_path / "unc_multi.tif", values, src, 4)
    assert read_info(path).count == 1


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def test_load_checkpoint_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="run scripts/train.py"):
        load_checkpoint(tmp_path / "absent.pth")


def test_load_checkpoint_rejects_a_foreign_file(tmp_path):
    path = tmp_path / "not_a_checkpoint.pth"
    torch.save({"weights": 1}, path)
    with pytest.raises(ValueError, match="not a valid training checkpoint"):
        load_checkpoint(path)


def test_load_checkpoint_rebuilds_the_architecture(trained_checkpoint):
    model, payload = load_checkpoint(trained_checkpoint, torch.device("cpu"))
    assert "config" in payload and "epoch" in payload
    with torch.no_grad():
        out = model(torch.rand(1, 4, 16, 16))
    assert out.shape == (1, 4, 64, 64)
