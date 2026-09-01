"""Phase 8: uncertainty estimation and its honesty guarantees."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.preprocessing import DegradationConfig
from src.models.baseline import InterpolationSR
from src.models.generator import EDSRLite
from src.uncertainty.uncertainty import (
    DISCLAIMER,
    ensemble_predict,
    estimate,
    mc_dropout_predict,
    normalise_map,
    reprojection_residual,
    summarise,
)

CPU = torch.device("cpu")
TILING = dict(tile_size=32, overlap=8, device=CPU, amp=False, channels_last=False)


@pytest.fixture
def lr():
    return np.random.default_rng(0).random((4, 32, 32)).astype(np.float32) * 0.5 + 0.1


@pytest.fixture
def realistic_lr(scene_path, cfg):
    """A spatially correlated scene.

    The reprojection residual compares against a *blurred* reconstruction, so
    white noise — which is pure high frequency — can never round-trip. Only a
    realistic scene exercises the metric meaningfully.
    """
    from src.data.preprocessing import degrade
    from src.inference.predict import read_scene

    scene, _, _ = read_scene(scene_path, list(cfg.data.band_indices))
    return degrade(scene[:, :128, :128], 4)


@pytest.fixture
def dropout_model():
    return EDSRLite(scale=4, in_channels=4, num_blocks=3, num_features=16, dropout=0.3)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_normalise_map_spans_the_unit_interval():
    values = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    out = normalise_map(values, "percentile", (1.0, 99.0))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_percentile_normalisation_resists_outliers():
    values = np.full((32, 32), 0.1, dtype=np.float32)
    values[0, 0] = 1000.0
    out = normalise_map(values, "percentile", (1.0, 99.0))
    assert out[1, 1] < 1.0 and np.isfinite(out).all()


def test_minmax_normalisation_uses_the_full_range():
    values = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    out = normalise_map(values, "minmax")
    assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)


def test_normalise_map_handles_a_constant_field():
    out = normalise_map(np.full((8, 8), 0.5, dtype=np.float32), "minmax")
    assert np.isfinite(out).all() and (out == 0).all()


def test_normalise_map_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="unknown normalisation"):
        normalise_map(np.zeros((4, 4)), "bogus")


# ---------------------------------------------------------------------------
# Monte-Carlo dropout
# ---------------------------------------------------------------------------
def test_mc_dropout_returns_mean_and_spread(lr, dropout_model):
    mean, std = mc_dropout_predict(dropout_model, lr, 4, passes=4, **TILING)
    assert mean.shape == (4, 128, 128)
    assert std.shape == mean.shape
    assert std.min() >= 0.0


def test_mc_dropout_produces_nonzero_variance(lr, dropout_model):
    """With dropout live, repeated passes must actually disagree."""
    _, std = mc_dropout_predict(dropout_model, lr, 4, passes=6, **TILING)
    assert std.mean() > 0.0


def test_mc_dropout_restores_eval_mode(lr, dropout_model):
    mc_dropout_predict(dropout_model, lr, 4, passes=3, **TILING)
    assert not dropout_model.training
    assert all(not m.training for m in dropout_model.modules())


def test_mc_dropout_refuses_a_model_without_dropout(lr):
    model = EDSRLite(scale=4, in_channels=4, num_blocks=2, num_features=16, dropout=0.0)
    with pytest.raises(RuntimeError, match="no dropout layers"):
        mc_dropout_predict(model, lr, 4, passes=4, **TILING)


def test_mc_dropout_requires_multiple_passes(lr, dropout_model):
    with pytest.raises(ValueError, match="at least 2 passes"):
        mc_dropout_predict(dropout_model, lr, 4, passes=1, **TILING)


# ---------------------------------------------------------------------------
# Geometric self-ensemble
# ---------------------------------------------------------------------------
def test_ensemble_returns_correctly_shaped_output(lr):
    model = InterpolationSR(scale=4, in_channels=4)
    mean, std = ensemble_predict(model, lr, 4, passes=4, **TILING)
    assert mean.shape == (4, 128, 128) and std.shape == mean.shape


def test_ensemble_of_an_equivariant_model_has_near_zero_spread(lr):
    """Bicubic commutes with D4, so the ensemble must agree with itself.

    This is the correctness test for the transform/undo-transform pair: a bug
    there would show up as large spurious spread.
    """
    model = InterpolationSR(scale=4, in_channels=4)
    _, std = ensemble_predict(model, lr, 4, passes=8, **TILING)
    assert std.max() < 0.01


def test_ensemble_mean_stays_in_reflectance_range(lr, dropout_model):
    mean, _ = ensemble_predict(dropout_model, lr, 4, passes=4, **TILING)
    assert mean.min() >= 0.0 and mean.max() <= 1.0


# ---------------------------------------------------------------------------
# Reprojection consistency
# ---------------------------------------------------------------------------
def test_reprojection_residual_is_small_for_a_consistent_product(realistic_lr):
    """Degrading a bicubic upsample of a real scene returns close to the input."""
    from src.data.preprocessing import bicubic_upsample

    sr = bicubic_upsample(realistic_lr, 4)
    residual = reprojection_residual(sr, realistic_lr, 4, DegradationConfig())
    assert residual.mean() < 0.02


def test_reprojection_residual_detects_an_inconsistent_product(realistic_lr):
    from src.data.preprocessing import bicubic_upsample

    sr = bicubic_upsample(realistic_lr, 4)
    corrupted = np.clip(sr + 0.2, 0, 1)
    good = reprojection_residual(sr, realistic_lr, 4).mean()
    bad = reprojection_residual(corrupted, realistic_lr, 4).mean()
    assert bad > good * 2


def test_reprojection_residual_has_sr_resolution(lr):
    from src.data.preprocessing import bicubic_upsample

    residual = reprojection_residual(bicubic_upsample(lr, 4), lr, 4)
    assert residual.shape[-2:] == (128, 128)


def test_reprojection_residual_penalises_a_hallucinating_product(realistic_lr):
    """Invented high-frequency detail that averages back correctly is invisible here.

    Documents a real limitation: the consistency check is necessary, not
    sufficient. Texture added with zero local mean passes it.
    """
    from src.data.preprocessing import bicubic_upsample

    sr = bicubic_upsample(realistic_lr, 4)
    rng = np.random.default_rng(0)
    checker = np.indices(sr.shape[-2:]).sum(axis=0) % 2
    hallucinated = np.clip(sr + 0.05 * (checker * 2 - 1), 0, 1)

    baseline = reprojection_residual(sr, realistic_lr, 4).mean()
    invented = reprojection_residual(hallucinated, realistic_lr, 4).mean()
    assert invented < baseline * 2, (
        "zero-mean invented texture is expected to slip past the consistency check"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def test_estimate_returns_a_complete_result(lr, dropout_model, cfg):
    result = estimate(dropout_model, lr, 4, cfg, **TILING)
    assert result.prediction.shape == (4, 128, 128)
    assert result.uncertainty.shape == (128, 128)
    assert result.confidence.shape == (128, 128)
    assert result["reprojection_residual"].shape == (128, 128)
    assert result["method"] == cfg.uncertainty.method


def test_the_configured_default_is_not_degenerate_on_a_trained_model(lr, dropout_model, cfg):
    """The default must produce a map with real signal, not numerical zero.

    A freshly built model has a zero-initialised tail, so it *is* bicubic —
    which is D4-equivariant and therefore has genuinely zero ensemble spread.
    The tail is perturbed here to stand in for a trained residual branch.
    """
    torch.nn.init.normal_(dropout_model.tail.weight, std=0.05)
    result = estimate(dropout_model, lr, 4, cfg, **TILING)
    assert result.uncertainty.max() > 1e-5
    assert not any("near zero" in note for note in result["notes"])


def test_confidence_is_the_complement_of_normalised_uncertainty(lr, dropout_model, cfg):
    result = estimate(dropout_model, lr, 4, cfg, **TILING)
    np.testing.assert_allclose(
        result.confidence, 1.0 - result["uncertainty_normalised"], atol=1e-6
    )


def test_estimate_falls_back_when_the_model_has_no_dropout(lr, cfg):
    """A checkpoint trained with dropout: 0 must still produce a map."""
    model = EDSRLite(scale=4, in_channels=4, num_blocks=2, num_features=16, dropout=0.0)
    result = estimate(model, lr, 4, cfg.merge({"uncertainty": {"method": "mc_dropout"}}), **TILING)
    assert result["method"] == "ensemble"
    assert any("mc_dropout unavailable" in note for note in result["notes"])


@pytest.mark.parametrize("method", ["ensemble", "reprojection", "none"])
def test_every_method_produces_a_usable_map(lr, dropout_model, cfg, method):
    result = estimate(
        dropout_model, lr, 4, cfg.merge({"uncertainty": {"method": method}}), **TILING
    )
    assert result.uncertainty.shape == (128, 128)
    assert np.isfinite(result.uncertainty).all()


def test_estimate_rejects_an_unknown_method(lr, dropout_model, cfg):
    with pytest.raises(ValueError, match="unknown uncertainty method"):
        estimate(dropout_model, lr, 4, cfg.merge({"uncertainty": {"method": "magic"}}), **TILING)


def test_the_none_method_reports_zero_uncertainty(lr, dropout_model, cfg):
    result = estimate(dropout_model, lr, 4, cfg.merge({"uncertainty": {"method": "none"}}), **TILING)
    assert result.uncertainty.max() == 0.0


# ---------------------------------------------------------------------------
# Scientific honesty
# ---------------------------------------------------------------------------
def test_a_degenerate_map_is_called_out_rather_than_read_as_confidence(lr, cfg):
    """A near-zero spread means 'nothing measured', not 'total confidence'.

    Regression for the real case: MC-dropout on a global-residual model whose
    residual branch is still small produces ~1e-8 spread, which a reader would
    otherwise interpret as the model being certain.
    """
    model = EDSRLite(scale=4, in_channels=4, num_blocks=2, num_features=16, dropout=0.3)
    # Zero the residual branch so the output is exactly the bicubic base.
    torch.nn.init.zeros_(model.tail.weight)
    torch.nn.init.zeros_(model.tail.bias)

    result = estimate(
        model, lr, 4, cfg.merge({"uncertainty": {"method": "mc_dropout"}}), **TILING
    )
    assert result.uncertainty.max() < 1e-5
    assert any("NOT evidence of high confidence" in note for note in result["notes"])


def test_a_healthy_map_gets_no_degeneracy_warning(lr, dropout_model, cfg):
    result = estimate(
        dropout_model, lr, 4, cfg.merge({"uncertainty": {"method": "reprojection"}}), **TILING
    )
    assert not any("near zero" in note for note in result["notes"])


def test_summary_declares_the_map_uncalibrated():
    stats = summarise(np.random.default_rng(0).random((32, 32)).astype(np.float32))
    assert stats["calibrated"] is False
    assert "not a calibrated probability" in stats["interpretation"]


def test_result_carries_the_disclaimer(lr, dropout_model, cfg):
    result = estimate(dropout_model, lr, 4, cfg, **TILING)
    assert result["disclaimer"] == DISCLAIMER
    assert "not a calibrated probability" in result["disclaimer"]


def test_disclaimer_does_not_overclaim_low_uncertainty():
    """Guards the wording: low uncertainty must never be sold as correctness."""
    assert "do not guarantee" in DISCLAIMER or "does not guarantee" in DISCLAIMER
