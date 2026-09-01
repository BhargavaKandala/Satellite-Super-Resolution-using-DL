"""Phase 2: prepared datasets, spatial splitting and DataLoader tensors."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.data.patch_dataset import (
    ARRAY_NAME,
    MANIFEST_NAME,
    PatchDataset,
    PatchRecord,
    SceneDataset,
    build_dataloaders,
    read_manifest,
    spatial_split,
)


# ---------------------------------------------------------------------------
# Preparation output
# ---------------------------------------------------------------------------
def test_prepare_writes_an_array_and_a_manifest(patch_dir):
    assert (patch_dir / ARRAY_NAME).exists()
    assert (patch_dir / MANIFEST_NAME).exists()


def test_manifest_records_everything_needed_to_reproduce(patch_dir):
    manifest = read_manifest(patch_dir)
    assert manifest["count"] > 0
    assert manifest["patch_size"] == 64
    assert manifest["scale"] == 4
    assert manifest["bands"] == ["B04", "B03", "B02", "B08"]
    assert "degradation" in manifest and "seed" in manifest
    assert len(manifest["records"]) == manifest["count"]


def test_stored_array_matches_the_manifest(patch_dir):
    manifest = read_manifest(patch_dir)
    array = np.load(patch_dir / ARRAY_NAME, mmap_mode="r")
    assert array.shape == (manifest["count"], 4, 64, 64)
    assert array.dtype == np.dtype(manifest["store_dtype"])


def test_manifest_is_valid_json(patch_dir):
    json.loads((patch_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def test_read_manifest_reports_a_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_dataset"):
        read_manifest(tmp_path)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def _records(n=64, patch=64, per_row=8):
    return [
        PatchRecord(i, "scene.tif", (i // per_row) * patch, (i % per_row) * patch)
        for i in range(n)
    ]


def test_spatial_split_partitions_every_patch_exactly_once():
    train, val = spatial_split(_records(), 0.25, patch_size=64)
    assert sorted(train + val) == list(range(64))
    assert not set(train) & set(val)


def test_spatial_split_is_deterministic():
    a = spatial_split(_records(), 0.25, 64, seed=7)
    b = spatial_split(_records(), 0.25, 64, seed=7)
    assert a == b


def test_spatial_split_changes_with_the_seed():
    records = _records(1024, per_row=32)
    assert spatial_split(records, 0.25, 64, seed=1) != spatial_split(records, 0.25, 64, seed=2)


def test_spatial_split_is_not_degenerate_along_an_axis():
    """Regression: a CRC-32 key made the split collapse into column stripes.

    Validation blocks must scatter across the scene, otherwise the split
    systematically holds out one edge and the validation metric measures a
    different part of the image than it appears to.
    """
    records = _records(1024, per_row=32)
    _, val = spatial_split(records, 0.25, patch_size=64, seed=1, block_multiple=4)
    lookup = {r.index: r for r in records}
    blocks = {(lookup[i].row_off // 256, lookup[i].col_off // 256) for i in val}

    assert len({b[0] for b in blocks}) > 1, "validation blocks share a single row"
    assert len({b[1] for b in blocks}) > 1, "validation blocks share a single column"


def test_spatial_split_holds_out_roughly_the_requested_fraction():
    records = _records(4096, per_row=64)
    _, val = spatial_split(records, 0.25, patch_size=64, seed=1)
    assert 0.10 < len(val) / len(records) < 0.45


def test_spatial_split_keeps_whole_blocks_together():
    """The anti-leakage guarantee: overlapping patches share a split."""
    records = _records(256, per_row=16)
    train, val = spatial_split(records, 0.25, patch_size=64, seed=3, block_multiple=4)
    lookup = {r.index: r for r in records}
    block_of = lambda r: (r.scene, r.row_off // 256, r.col_off // 256)  # noqa: E731

    train_blocks = {block_of(lookup[i]) for i in train}
    val_blocks = {block_of(lookup[i]) for i in val}
    assert not train_blocks & val_blocks


def test_spatial_split_with_zero_fraction_gives_no_validation():
    train, val = spatial_split(_records(), 0.0, 64)
    assert len(train) == 64 and val == []


def test_spatial_split_always_yields_some_validation_when_requested():
    """Tiny datasets can hash into one bucket; the fallback must still split."""
    train, val = spatial_split(_records(2, per_row=1), 0.5, 64, seed=99)
    assert val and train


def test_spatial_split_rejects_an_invalid_fraction():
    with pytest.raises(ValueError, match="val_fraction"):
        spatial_split(_records(), 1.5, 64)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def test_dataset_yields_correctly_shaped_pairs(patch_dir):
    dataset = PatchDataset(patch_dir, scale=4)
    sample = dataset[0]
    assert sample["hr"].shape == (4, 64, 64)
    assert sample["lr"].shape == (4, 16, 16)
    assert sample["hr"].dtype == torch.float32
    assert sample["lr"].dtype == torch.float32


def test_dataset_values_are_normalised_reflectance(patch_dir):
    sample = PatchDataset(patch_dir, scale=4)[0]
    for key in ("lr", "hr"):
        assert float(sample[key].min()) >= 0.0
        assert float(sample[key].max()) <= 1.0


def test_dataset_lr_is_a_degraded_version_of_hr(patch_dir):
    """The LR input must actually be derived from the HR target."""
    from src.data.preprocessing import degrade

    sample = PatchDataset(patch_dir, scale=4)[0]
    expected = degrade(sample["hr"].numpy(), 4)
    np.testing.assert_allclose(sample["lr"].numpy(), expected, atol=1e-5)


def test_dataset_length_follows_the_index_subset(patch_dir):
    total = read_manifest(patch_dir)["count"]
    assert len(PatchDataset(patch_dir)) == total
    assert len(PatchDataset(patch_dir, indices=[0, 1, 2])) == 3


def test_dataset_is_deterministic_without_augmentation(patch_dir):
    dataset = PatchDataset(patch_dir, scale=4, augment=False)
    np.testing.assert_array_equal(dataset[0]["hr"].numpy(), dataset[0]["hr"].numpy())


def test_augmentation_is_reproducible_for_a_given_seed(patch_dir):
    a = PatchDataset(patch_dir, scale=4, augment=True, seed=11)[0]["hr"].numpy()
    b = PatchDataset(patch_dir, scale=4, augment=True, seed=11)[0]["hr"].numpy()
    np.testing.assert_array_equal(a, b)


def test_augmentation_preserves_shape_and_content_statistics(patch_dir):
    plain = PatchDataset(patch_dir, scale=4, augment=False)[0]["hr"].numpy()
    augmented = PatchDataset(patch_dir, scale=4, augment=True, seed=5)[0]["hr"].numpy()
    assert augmented.shape == plain.shape
    # D4 transforms only permute pixels, so the histogram is unchanged.
    np.testing.assert_allclose(np.sort(augmented.ravel()), np.sort(plain.ravel()), atol=1e-6)


def test_dataset_rejects_out_of_range_indices(patch_dir):
    with pytest.raises(IndexError, match="out of range"):
        PatchDataset(patch_dir, indices=[999999])


def test_dataset_rejects_a_scale_that_does_not_divide_the_patch(patch_dir):
    with pytest.raises(ValueError, match="divisible"):
        PatchDataset(patch_dir, scale=7)


def test_dataset_is_picklable_for_dataloader_workers(patch_dir):
    """The memmap handle must not be pickled into the worker."""
    import pickle

    dataset = PatchDataset(patch_dir, scale=4)
    _ = dataset[0]  # force the memmap open
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored[0]["hr"].shape == (4, 64, 64)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def test_build_dataloaders_produces_batched_tensors(patch_dir, cfg):
    train_loader, val_loader = build_dataloaders(patch_dir, cfg)
    batch = next(iter(train_loader))
    assert batch["lr"].ndim == 4 and batch["hr"].ndim == 4
    assert batch["lr"].shape[1] == 4
    assert batch["hr"].shape[-1] == batch["lr"].shape[-1] * 4
    assert val_loader is None or len(val_loader.dataset) > 0


def test_train_and_validation_sets_are_disjoint(patch_dir, cfg):
    train_loader, val_loader = build_dataloaders(patch_dir, cfg)
    if val_loader is None:
        pytest.skip("dataset too small to split")
    assert not set(train_loader.dataset.indices) & set(val_loader.dataset.indices)


def test_validation_loader_does_not_augment(patch_dir, cfg):
    _, val_loader = build_dataloaders(patch_dir, cfg)
    if val_loader is None:
        pytest.skip("dataset too small to split")
    assert val_loader.dataset.augment is False


# ---------------------------------------------------------------------------
# Scene tiling dataset
# ---------------------------------------------------------------------------
def test_scene_dataset_tiles_a_raster(scene_path):
    dataset = SceneDataset(scene_path, [1, 2, 3, 4], tile_size=64, overlap=8)
    assert len(dataset) > 1
    item = dataset[0]
    assert item["tile"].shape[0] == 4
    assert 0 <= float(item["tile"].min()) and float(item["tile"].max()) <= 1.0
    assert {"row_off", "col_off", "height", "width"} <= set(item)
