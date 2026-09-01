"""Phase 12: the end-to-end smoke test.

Runs the exact command sequence from the acceptance criteria against a small
synthetic dataset, through the real script entry points:

    prepare_dataset.py -> train.py -> evaluate.py -> inference.py

Everything is confined to tmp directories, so the suite never touches the
repository's own ``data/``, ``checkpoints/`` or ``outputs/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.data.geotiff import read_info, validate_geospatial


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Path:
    from src.data.synthetic import write_scene

    root = tmp_path_factory.mktemp("e2e")
    (root / "raw").mkdir()
    write_scene(root / "raw" / "scene.tif", height=256, width=256, bands=4, seed=99)
    return root


@pytest.fixture(scope="module")
def prepared(workspace) -> Path:
    import prepare_dataset

    out = workspace / "patches"
    code = prepare_dataset.main(
        [
            "--input", str(workspace / "raw" / "scene.tif"),
            "--output", str(out),
            "--patch-size", "64",
            "--max-patches", "48",
        ]
    )
    assert code == 0
    return out


@pytest.fixture(scope="module")
def checkpoint(workspace, prepared) -> Path:
    import train

    out = workspace / "checkpoints"
    code = train.main(
        [
            "--patches", str(prepared),
            "--checkpoints", str(out),
            "--epochs", "2",
            "--batch-size", "4",
            "--workers", "0",
            "--no-amp",
            "--device", "cpu",
        ]
    )
    assert code == 0
    return out / "best.pth"


# ---------------------------------------------------------------------------
# Step 1 — prepare
# ---------------------------------------------------------------------------
def test_prepare_produces_a_usable_dataset(prepared):
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] > 0
    assert manifest["split"]["train"] > 0
    assert (prepared / "patches.npy").exists()


def test_prepare_reports_a_missing_input(tmp_path):
    import prepare_dataset

    with pytest.raises(FileNotFoundError):
        prepare_dataset.main(["--input", str(tmp_path / "nope.tif")])


def test_prepare_can_generate_synthetic_data(tmp_path):
    """The zero-data path: the demo must run on a clean clone."""
    import prepare_dataset
    from src.config import load_config

    cfg = load_config()
    overrides = tmp_path / "config.yaml"
    data = cfg.to_dict()
    data["data"]["raw_dir"] = str(tmp_path / "raw")
    import yaml

    overrides.write_text(yaml.safe_dump(data), encoding="utf-8")

    code = prepare_dataset.main(
        [
            "--config", str(overrides),
            "--output", str(tmp_path / "patches"),
            "--synthetic",
            "--synthetic-scenes", "1",
            "--synthetic-size", "256",
            "--patch-size", "64",
            "--max-patches", "16",
        ]
    )
    assert code == 0
    assert (tmp_path / "patches" / "patches.npy").exists()

    # References are written at the *target* resolution...
    reference = read_info(tmp_path / "raw" / "synthetic_00.tif")
    assert reference.resolution[0] == pytest.approx(2.5)

    # ...and the demo input lands beside the redirected raw dir, not in the repo.
    assert (tmp_path / "sample.tif").exists()


def test_a_redirected_config_does_not_overwrite_the_repo_demo_input(tmp_path):
    """Regression: the test suite used to clobber the user's own sample.tif.

    ``make_sample_input`` wrote to a hardcoded ``REPO_ROOT / "sample.tif"``
    regardless of where the config pointed, so running pytest replaced a real
    demo input with whatever tiny scene a test happened to generate — silently
    shrinking the live demo to 1/16 of its area.
    """
    import prepare_dataset
    import yaml

    from src.config import REPO_ROOT, load_config

    repo_sample = REPO_ROOT / "sample.tif"
    before = repo_sample.read_bytes() if repo_sample.exists() else None

    data = load_config().to_dict()
    data["data"]["raw_dir"] = str(tmp_path / "raw")
    overrides = tmp_path / "config.yaml"
    overrides.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert prepare_dataset.main(
        [
            "--config", str(overrides),
            "--output", str(tmp_path / "patches"),
            "--synthetic",
            "--synthetic-scenes", "1",
            "--synthetic-size", "256",
            "--patch-size", "64",
            "--max-patches", "16",
        ]
    ) == 0

    assert (tmp_path / "sample.tif").exists(), "sample must follow the config"
    after = repo_sample.read_bytes() if repo_sample.exists() else None
    assert after == before, "a redirected run must not touch the repo's sample.tif"


def test_synthetic_mode_also_writes_a_ten_metre_demo_input():
    """`--input sample.tif` must represent the real task: 10 m in, 2.5 m out."""
    from src.config import REPO_ROOT

    sample = REPO_ROOT / "sample.tif"
    if not sample.exists():
        pytest.skip("run scripts/prepare_dataset.py --synthetic to generate sample.tif")

    info = read_info(sample)
    assert info.resolution[0] == pytest.approx(10.0)
    assert info.crs is not None
    assert info.count == 4


# ---------------------------------------------------------------------------
# Step 2 — train
# ---------------------------------------------------------------------------
def test_training_writes_checkpoints_and_history(checkpoint):
    directory = checkpoint.parent
    assert checkpoint.exists()
    assert (directory / "last.pth").exists()
    assert (directory / "history.csv").exists()

    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert history[0]["train_loss"] > 0


def test_checkpoint_embeds_the_config(checkpoint):
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["model"]["name"] == "edsr_lite"
    assert "model_state" in payload


def test_training_rejects_a_dataset_prepared_at_a_different_scale(workspace, prepared):
    import train
    from src.config import load_config
    import yaml

    data = load_config().to_dict()
    data["patches"]["scale"] = 2
    data["model"]["scale"] = 2
    data["data"]["target_resolution_m"] = 5.0
    path = workspace / "scale2.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    code = train.main(["--config", str(path), "--patches", str(prepared), "--epochs", "1"])
    assert code == 2


def test_training_reports_a_missing_dataset(tmp_path):
    import train

    assert train.main(["--patches", str(tmp_path / "absent")]) == 1


# ---------------------------------------------------------------------------
# Step 3 — evaluate
# ---------------------------------------------------------------------------
def test_evaluate_writes_metrics_for_both_methods(workspace, checkpoint):
    import evaluate

    out = workspace / "evaluation"
    code = evaluate.main(
        [
            "--input", str(workspace / "raw" / "scene.tif"),
            "--checkpoint", str(checkpoint),
            "--output", str(out),
            "--device", "cpu",
        ]
    )
    assert code == 0

    report = json.loads((out / "scene_metrics.json").read_text(encoding="utf-8"))
    assert set(report["methods"]) == {"bicubic", "ai_sr"}
    assert report["protocol"] == "reduced_resolution"
    assert (out / "metrics.csv").exists()


def test_evaluate_runs_the_downstream_experiment(workspace, checkpoint):
    import evaluate

    out = workspace / "evaluation_downstream"
    code = evaluate.main(
        [
            "--input", str(workspace / "raw" / "scene.tif"),
            "--checkpoint", str(checkpoint),
            "--output", str(out),
            "--device", "cpu",
            "--downstream",
        ]
    )
    assert code == 0


def test_evaluate_works_without_a_checkpoint(workspace):
    import evaluate

    out = workspace / "evaluation_baseline"
    code = evaluate.main(
        [
            "--input", str(workspace / "raw" / "scene.tif"),
            "--output", str(out),
            "--baseline-only",
            "--device", "cpu",
        ]
    )
    assert code == 0
    report = json.loads((out / "scene_metrics.json").read_text(encoding="utf-8"))
    assert set(report["methods"]) == {"bicubic"}


# ---------------------------------------------------------------------------
# Step 4 — inference
# ---------------------------------------------------------------------------
def test_inference_produces_a_valid_sr_geotiff(workspace, checkpoint):
    import inference

    out = workspace / "inference"
    code = inference.main(
        [
            "--input", str(workspace / "raw" / "scene.tif"),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(out),
            "--device", "cpu",
        ]
    )
    assert code == 0

    sr_path = out / "scene_sr.tif"
    source = read_info(workspace / "raw" / "scene.tif")
    report = validate_geospatial(sr_path, source, scale=4)
    assert report["valid"], report["checks"]


def test_inference_output_resolution_is_below_four_metres(workspace, checkpoint):
    """The headline requirement of the problem statement."""
    sr_info = read_info(workspace / "inference" / "scene_sr.tif")
    assert sr_info.resolution[0] < 4.0
    assert sr_info.resolution[0] == pytest.approx(2.5)


def test_inference_writes_an_uncertainty_map(workspace, checkpoint):
    path = workspace / "inference" / "scene_uncertainty.tif"
    assert path.exists()
    info = read_info(path)
    assert info.count == 1
    assert info.crs == read_info(workspace / "raw" / "scene.tif").crs


def test_inference_report_carries_the_disclaimer(workspace, checkpoint):
    report = json.loads(
        (workspace / "inference" / "scene_inference.json").read_text(encoding="utf-8")
    )
    assert "AI-inferred information" in report["disclaimer"]
    assert report["geospatial_validation"]["valid"]
    assert report["uncertainty_summary"]["calibrated"] is False


def test_streamed_inference_also_produces_a_valid_product(workspace, checkpoint):
    import inference

    out = workspace / "inference_stream"
    code = inference.main(
        [
            "--input", str(workspace / "raw" / "scene.tif"),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(out),
            "--device", "cpu",
            "--stream",
            "--no-uncertainty",
        ]
    )
    assert code == 0
    source = read_info(workspace / "raw" / "scene.tif")
    assert validate_geospatial(out / "scene_sr.tif", source, scale=4)["valid"]


def test_baseline_inference_needs_no_checkpoint(workspace):
    import inference

    out = workspace / "inference_baseline"
    code = inference.main(
        [
            "--input", str(workspace / "raw" / "scene.tif"),
            "--output-dir", str(out),
            "--baseline",
            "--device", "cpu",
            "--no-uncertainty",
        ]
    )
    assert code == 0


def test_inference_reports_a_missing_input(tmp_path):
    import inference

    assert inference.main(["--input", str(tmp_path / "absent.tif")]) == 1


# ---------------------------------------------------------------------------
# Cross-step consistency
# ---------------------------------------------------------------------------
def test_the_sr_product_is_spectrally_close_to_the_input(workspace, checkpoint):
    """A trained model must not have wrecked the spectral signature.

    Guards against the failure mode the whole project is built to avoid: an
    output that scores well on appearance but is unusable for analysis.
    """
    from src.data.preprocessing import degrade
    from src.evaluation.metrics import sam
    from src.inference.predict import read_scene

    original, _, _ = read_scene(workspace / "raw" / "scene.tif", [1, 2, 3, 4])
    sr, _, _ = read_scene(workspace / "inference" / "scene_sr.tif", [1, 2, 3, 4])

    # Bring the SR product back to the input grid before comparing.
    reprojected = degrade(sr, 4)
    assert sam(reprojected, original) < 5.0, "spectral signature drifted too far"


def test_the_full_acceptance_sequence_left_every_artefact(workspace, checkpoint):
    for relative in (
        "patches/patches.npy",
        "patches/manifest.json",
        "checkpoints/best.pth",
        "checkpoints/history.csv",
        "evaluation/scene_metrics.json",
        "evaluation/metrics.csv",
        "inference/scene_sr.tif",
        "inference/scene_uncertainty.tif",
        "inference/scene_inference.json",
    ):
        assert (workspace / relative).exists(), f"missing {relative}"
