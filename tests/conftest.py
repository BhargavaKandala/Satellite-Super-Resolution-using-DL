"""Shared pytest fixtures.

Every fixture builds its data synthetically into a tmp directory, so the test
suite runs on a clean checkout with no downloads and leaves nothing behind.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from src.config import load_config  # noqa: E402
from src.data.synthetic import write_pair, write_scene  # noqa: E402

SCALE = 4
BANDS = 4


@pytest.fixture(scope="session")
def cfg():
    """The project config, with test-scale overrides for speed."""
    return load_config().merge(
        {
            "patches": {"hr_patch_size": 64, "stride": 64, "max_patches": 64},
            "training": {"epochs": 1, "batch_size": 4, "num_workers": 0, "amp": False},
            "uncertainty": {"passes": 3},
            "inference": {"tile_size": 64, "tile_overlap": 8, "batch_size": 2},
        }
    )


@pytest.fixture(scope="session")
def scene_path(tmp_path_factory) -> Path:
    """A single 256x256 synthetic 10 m scene as a uint16 GeoTIFF."""
    directory = tmp_path_factory.mktemp("scenes")
    return write_scene(directory / "scene.tif", height=256, width=256, bands=BANDS, seed=1)


@pytest.fixture(scope="session")
def pair_paths(tmp_path_factory) -> tuple[Path, Path]:
    """A co-registered ``(lr_path, hr_path)`` pair at 10 m and 2.5 m."""
    directory = tmp_path_factory.mktemp("pairs")
    return write_pair(directory, hr_size=256, scale=SCALE, bands=BANDS, seed=2)


@pytest.fixture(scope="session")
def patch_dir(tmp_path_factory, scene_path) -> Path:
    """A prepared patch dataset, built by the real ``prepare_dataset`` script."""
    import prepare_dataset

    out = tmp_path_factory.mktemp("patches")
    code = prepare_dataset.main(
        [
            "--input",
            str(scene_path),
            "--output",
            str(out),
            "--patch-size",
            "64",
            "--max-patches",
            "32",
        ]
    )
    assert code == 0, "prepare_dataset failed"
    return out


@pytest.fixture(scope="session")
def trained_checkpoint(tmp_path_factory, patch_dir, cfg) -> Path:
    """A checkpoint from a real (tiny) training run — used by the smoke tests."""
    import train as train_script

    out = tmp_path_factory.mktemp("checkpoints")
    code = train_script.main(
        [
            "--patches",
            str(patch_dir),
            "--checkpoints",
            str(out),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--workers",
            "0",
            "--no-amp",
            "--device",
            "cpu",
        ]
    )
    assert code == 0, "training failed"
    return out / "best.pth"
