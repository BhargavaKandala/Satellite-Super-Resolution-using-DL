"""Config loading and the consistency checks that guard the pipeline."""

from __future__ import annotations

import pytest
import yaml

from src.config import Config, load_config, set_seed


def test_load_config_reads_the_project_defaults():
    cfg = load_config()
    assert cfg.project.name == "sih142-satellite-sr"
    assert cfg.patches.scale == cfg.model.scale
    assert cfg.model.in_channels == len(cfg.data.bands)


def test_config_supports_attribute_and_item_access():
    cfg = load_config()
    assert cfg.patches.hr_patch_size == cfg["patches"]["hr_patch_size"]
    assert cfg.get_nested("model.num_features") == cfg.model.num_features


def test_config_is_read_only():
    cfg = load_config()
    with pytest.raises(TypeError, match="read-only"):
        cfg.project = {}


def test_missing_key_raises_a_useful_error():
    cfg = load_config()
    with pytest.raises(AttributeError, match="no config key"):
        _ = cfg.does_not_exist
    with pytest.raises(KeyError, match="missing config key"):
        cfg.get_nested("model.nope")
    assert cfg.get_nested("model.nope", default=7) == 7


def test_merge_overrides_without_mutating_the_original():
    cfg = load_config()
    original = cfg.training.epochs
    merged = cfg.merge({"training": {"epochs": 1}})
    assert merged.training.epochs == 1
    assert cfg.training.epochs == original
    assert merged.training.batch_size == cfg.training.batch_size


def test_config_serialises_back_to_yaml():
    cfg = load_config()
    assert yaml.safe_load(yaml.safe_dump(cfg.to_dict()))["project"]["name"] == cfg.project.name


def test_target_resolution_is_below_the_four_metre_requirement():
    cfg = load_config()
    assert cfg.data.target_resolution_m < 4.0
    assert cfg.data.target_resolution_m == cfg.data.source_resolution_m / cfg.patches.scale


@pytest.mark.parametrize(
    "override, message",
    [
        ({"data": {"band_indices": [1, 2]}}, "same length"),
        ({"model": {"in_channels": 3}}, "in_channels"),
        ({"model": {"scale": 2}}, "must upsample"),
        ({"patches": {"hr_patch_size": 130}}, "divisible"),
        ({"data": {"target_resolution_m": 5.0}}, "target_resolution_m"),
    ],
)
def test_validation_rejects_inconsistent_configs(tmp_path, override, message):
    from src.config import _deep_merge

    data = _deep_merge(load_config().to_dict(), override)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_load_config_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_set_seed_makes_numpy_and_torch_reproducible():
    import numpy as np

    set_seed(123)
    a = np.random.rand(4)
    set_seed(123)
    np.testing.assert_array_equal(a, np.random.rand(4))
