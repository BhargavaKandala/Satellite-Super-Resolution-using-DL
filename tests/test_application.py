"""Phase 9: the downstream land-cover experiment and its honesty guarantees."""

from __future__ import annotations

import numpy as np
import pytest

from src.applications.urban_mapping import (
    CentroidClassifier,
    build_features,
    classification_metrics,
    confusion_matrix,
    normalized_index,
    run_experiment,
    structural_descriptors,
)


@pytest.fixture
def scene(scene_path, cfg):
    from src.inference.predict import read_scene

    array, _, _ = read_scene(scene_path, list(cfg.data.band_indices))
    return array[:, :128, :128]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def test_normalized_index_matches_the_formula():
    a = np.array([[0.4]], dtype=np.float32)
    b = np.array([[0.1]], dtype=np.float32)
    assert normalized_index(a, b)[0, 0] == pytest.approx(0.6)


def test_normalized_index_handles_a_zero_denominator():
    zero = np.zeros((4, 4), dtype=np.float32)
    assert np.isfinite(normalized_index(zero, zero)).all()


def test_build_features_appends_the_spectral_indices(scene):
    features, names = build_features(scene, ["B04", "B03", "B02", "B08"])
    assert names == ["B04", "B03", "B02", "B08", "NDVI", "NDWI"]
    assert features.shape == (6, 128, 128)


def test_indices_can_be_disabled(scene):
    features, names = build_features(
        scene, ["B04", "B03", "B02", "B08"], use_ndvi=False, use_ndwi=False
    )
    assert names == ["B04", "B03", "B02", "B08"]
    assert features.shape[0] == 4


def test_features_are_skipped_when_bands_are_absent(scene):
    _, names = build_features(scene[:3], ["B04", "B03", "B02"])
    assert "NDVI" not in names  # needs B08


def test_ndvi_separates_vegetation_from_water():
    """Sanity-checks the band ordering against real spectral behaviour."""
    vegetation = np.array([[[0.045]], [[0.080]], [[0.035]], [[0.360]]], dtype=np.float32)
    water = np.array([[[0.030]], [[0.055]], [[0.070]], [[0.015]]], dtype=np.float32)
    names = ["B04", "B03", "B02", "B08"]
    veg_ndvi = build_features(vegetation, names)[0][4, 0, 0]
    water_ndvi = build_features(water, names)[0][4, 0, 0]
    assert veg_ndvi > 0.5 > water_ndvi


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def test_classifier_produces_a_label_map(scene):
    features, _ = build_features(scene, ["B04", "B03", "B02", "B08"])
    classifier = CentroidClassifier.fit_unsupervised(features, 5, seed=0)
    labels = classifier.predict(features)
    assert labels.shape == (128, 128)
    assert labels.min() >= 0 and labels.max() < 5


def test_classifier_is_deterministic(scene):
    features, _ = build_features(scene, ["B04", "B03", "B02", "B08"])
    a = CentroidClassifier.fit_unsupervised(features, 5, seed=7).predict(features)
    b = CentroidClassifier.fit_unsupervised(features, 5, seed=7).predict(features)
    np.testing.assert_array_equal(a, b)


def test_the_same_centroids_transfer_across_images(scene):
    """The core of the experiment design: class k means the same thing everywhere."""
    features, _ = build_features(scene, ["B04", "B03", "B02", "B08"])
    classifier = CentroidClassifier.fit_unsupervised(features, 5, seed=0)

    perturbed, _ = build_features(scene + 0.001, ["B04", "B03", "B02", "B08"])
    labels_a = classifier.predict(features)
    labels_b = classifier.predict(perturbed)
    assert (labels_a == labels_b).mean() > 0.95


def test_supervised_fit_learns_the_supplied_classes(scene):
    features, _ = build_features(scene, ["B04", "B03", "B02", "B08"])
    labels = np.zeros((128, 128), dtype=np.int16)
    labels[64:, :] = 1
    classifier = CentroidClassifier.fit_supervised(features, labels, ["a", "b"])
    assert classifier.n_classes == 2
    assert classifier.class_names == ["a", "b"]


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------
def test_confusion_matrix_counts_correctly():
    truth = np.array([[0, 0], [1, 1]])
    predicted = np.array([[0, 1], [1, 1]])
    cm = confusion_matrix(truth, predicted, 2)
    assert cm.tolist() == [[1, 1], [0, 2]]


def test_perfect_prediction_scores_one():
    labels = np.random.default_rng(0).integers(0, 4, size=(32, 32)).astype(np.int16)
    out = classification_metrics(labels, labels, 4)
    assert out["overall_accuracy"] == pytest.approx(1.0)
    assert out["mean_iou"] == pytest.approx(1.0)
    assert out["kappa"] == pytest.approx(1.0)


def test_accuracy_degrades_with_errors():
    truth = np.zeros((32, 32), dtype=np.int16)
    predicted = truth.copy()
    predicted[:8] = 1
    out = classification_metrics(truth, predicted, 2)
    assert out["overall_accuracy"] == pytest.approx(0.75)


def test_kappa_is_near_zero_for_a_random_prediction():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 4, size=(128, 128)).astype(np.int16)
    predicted = rng.integers(0, 4, size=(128, 128)).astype(np.int16)
    assert abs(classification_metrics(truth, predicted, 4)["kappa"]) < 0.05


def test_per_class_reporting_includes_support():
    truth = np.zeros((16, 16), dtype=np.int16)
    truth[8:] = 1
    out = classification_metrics(truth, truth, 2, ["water", "land"])
    assert out["per_class"]["water"]["support"] == 128
    assert out["per_class"]["land"]["iou"] == pytest.approx(1.0)


def test_structural_descriptors_reflect_map_detail():
    flat = np.zeros((32, 32), dtype=np.int16)
    detailed = (np.indices((32, 32)).sum(axis=0) % 4).astype(np.int16)
    assert structural_descriptors(detailed)["boundary_density"] > structural_descriptors(flat)[
        "boundary_density"
    ]
    assert structural_descriptors(flat)["n_classes_present"] == 1


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def test_experiment_is_quantitative_with_a_reference(scene, cfg):
    from src.data.preprocessing import bicubic_upsample, degrade

    coarse = degrade(scene, 4)
    products = {"bicubic": bicubic_upsample(coarse, 4), "ai_sr": scene.copy()}
    result = run_experiment(products, cfg, reference=scene)

    assert result.quantitative
    assert "overall_accuracy" in result.metrics["bicubic"]
    assert "overall_accuracy" in result.metrics["ai_sr"]
    assert set(result.maps) >= {"bicubic", "ai_sr", "reference"}


def test_a_perfect_reconstruction_scores_higher_than_bicubic(scene, cfg):
    """Directional sanity check: better imagery must not score worse."""
    from src.data.preprocessing import bicubic_upsample, degrade

    coarse = degrade(scene, 4)
    products = {"bicubic": bicubic_upsample(coarse, 4), "ai_sr": scene.copy()}
    result = run_experiment(products, cfg, reference=scene)

    assert result.metrics["ai_sr"]["overall_accuracy"] == pytest.approx(1.0)
    assert result.metrics["ai_sr"]["overall_accuracy"] > result.metrics["bicubic"]["overall_accuracy"]
    assert "improved" in result.verdict


def test_experiment_without_a_reference_refuses_to_claim_a_result(scene, cfg):
    """The anti-fabrication guarantee."""
    result = run_experiment({"bicubic": scene}, cfg, reference=None)

    assert result.quantitative is False
    assert result.metrics["bicubic"].get("overall_accuracy") is None
    assert "NOT MEASURED" in result.verdict
    assert any("NO REFERENCE IMAGERY AVAILABLE" in c for c in result.caveats)


def test_clustered_reference_labels_are_disclosed(scene, cfg):
    result = run_experiment({"bicubic": scene}, cfg, reference=scene)
    assert any("not from field survey" in c for c in result.caveats)


def test_supervised_labels_remove_the_clustering_caveat(scene, cfg):
    labels = np.zeros(scene.shape[-2:], dtype=np.int16)
    labels[64:] = 1
    result = run_experiment({"bicubic": scene}, cfg, reference=scene, labels=labels)
    assert result.quantitative
    assert not any("clustering" in c for c in result.caveats)


def test_verdict_flags_a_difference_within_noise(scene, cfg):
    result = run_experiment({"bicubic": scene, "ai_sr": scene}, cfg, reference=scene)
    assert "within noise" in result.verdict


def test_result_serialises_without_the_label_maps(scene, cfg, tmp_path):
    import json

    result = run_experiment({"bicubic": scene}, cfg, reference=scene)
    payload = result.as_dict()
    assert "maps" not in payload
    json.loads(result.save(tmp_path / "app.json").read_text(encoding="utf-8"))
