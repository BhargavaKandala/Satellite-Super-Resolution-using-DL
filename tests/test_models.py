"""Phase 3 + 4 + 5: model forward passes, shapes and loss behaviour."""

from __future__ import annotations

import pytest
import torch

from src.models import available_models, build_model, register_model
from src.models.baseline import InterpolationSR
from src.models.generator import EDSRLite, receptive_field
from src.models.losses import (
    SSIM,
    CharbonnierLoss,
    CombinedLoss,
    GradientLoss,
    SpectralAngleLoss,
    SSIMLoss,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_exposes_the_built_in_architectures():
    from src.models import baseline, generator  # noqa: F401

    names = available_models()
    assert "edsr_lite" in names
    assert "bicubic" in names


def test_build_model_from_config(cfg):
    model = build_model(cfg)
    assert isinstance(model, EDSRLite)
    assert model.scale == cfg.model.scale
    assert model.in_channels == len(cfg.data.bands)


def test_build_model_rejects_an_unknown_name(cfg):
    with pytest.raises(ValueError, match="unknown model"):
        build_model(cfg.merge({"model": {"name": "not_a_model"}}))


def test_registering_a_duplicate_name_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_model("edsr_lite")(lambda **_: None)


def test_a_new_architecture_can_be_registered_and_built(cfg):
    """The extensibility contract: SwinIR/diffusion plug in without pipeline edits."""
    import torch.nn as nn

    class Stub(nn.Module):
        def __init__(self, scale=4, in_channels=4, **_):
            super().__init__()
            self.scale, self.in_channels = scale, in_channels

        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=self.scale)

    register_model("stub_arch")(lambda **kwargs: Stub(**kwargs))
    model = build_model(cfg.merge({"model": {"name": "stub_arch"}}))
    out = model(torch.rand(1, 4, 8, 8))
    assert out.shape == (1, 4, 32, 32)


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def test_bicubic_baseline_upsamples_by_the_scale():
    model = InterpolationSR(scale=4, in_channels=4)
    out = model(torch.rand(2, 4, 16, 16))
    assert out.shape == (2, 4, 64, 64)


def test_bicubic_baseline_has_no_parameters():
    model = InterpolationSR(scale=4)
    assert sum(p.numel() for p in model.parameters()) == 0


def test_bicubic_baseline_clamps_overshoot():
    """Bicubic rings at hard edges; reflectance must stay physical."""
    x = torch.zeros(1, 4, 16, 16)
    x[:, :, 8:, :] = 1.0
    out = InterpolationSR(scale=4)(x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_bicubic_baseline_rejects_a_channel_change():
    with pytest.raises(ValueError, match="cannot change the channel count"):
        InterpolationSR(scale=4, in_channels=4, out_channels=3)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [2, 3, 4])
def test_generator_output_shape_follows_the_scale(scale):
    model = EDSRLite(scale=scale, in_channels=4, out_channels=4, num_blocks=2, num_features=16)
    out = model(torch.rand(2, 4, 16, 16))
    assert out.shape == (2, 4, 16 * scale, 16 * scale)


def test_generator_handles_arbitrary_band_counts():
    """Adding Sentinel-2 bands must not require touching the architecture."""
    model = EDSRLite(scale=4, in_channels=7, out_channels=7, num_blocks=2, num_features=16)
    assert model(torch.rand(1, 7, 8, 8)).shape == (1, 7, 32, 32)


def test_generator_output_is_in_reflectance_range():
    model = EDSRLite(scale=4, in_channels=4, num_blocks=2, num_features=16)
    out = model(torch.rand(2, 4, 16, 16))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_generator_starts_as_the_bicubic_baseline():
    """The residual branch is zero-initialised, so step 0 cannot be worse."""
    model = EDSRLite(scale=4, in_channels=4, num_blocks=2, num_features=16).eval()
    x = torch.rand(1, 4, 16, 16)
    with torch.no_grad():
        out = model(x)
        base = torch.nn.functional.interpolate(
            x, scale_factor=4, mode="bicubic", align_corners=False
        ).clamp(0, 1)
    torch.testing.assert_close(out, base, atol=1e-5, rtol=1e-4)


def test_generator_is_trainable_end_to_end():
    model = EDSRLite(scale=4, in_channels=4, num_blocks=2, num_features=16)
    out = model(torch.rand(2, 4, 16, 16))
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_generator_uses_no_batch_normalisation():
    """BN would rescale by batch statistics and destroy absolute radiometry."""
    import torch.nn as nn

    model = EDSRLite(scale=4, in_channels=4, num_blocks=3, num_features=16)
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


def test_generator_rejects_an_unfactorable_scale():
    with pytest.raises(ValueError, match="must factor into"):
        EDSRLite(scale=5, in_channels=4, num_blocks=1, num_features=8)


def test_generator_rejects_residual_with_mismatched_channels():
    with pytest.raises(ValueError, match="global_residual requires"):
        EDSRLite(scale=4, in_channels=4, out_channels=3, global_residual=True)


def test_nearest_conv_upsampler_works():
    model = EDSRLite(
        scale=4, in_channels=4, num_blocks=2, num_features=16, upsampler="nearest_conv"
    )
    assert model(torch.rand(1, 4, 8, 8)).shape == (1, 4, 32, 32)


def test_enable_mc_dropout_activates_only_dropout():
    import torch.nn as nn

    model = EDSRLite(scale=4, in_channels=4, num_blocks=3, num_features=16, dropout=0.2).eval()
    count = model.enable_mc_dropout()
    assert count == 3
    assert all(
        m.training
        for m in model.modules()
        if isinstance(m, (nn.Dropout, nn.Dropout2d))
    )
    assert not model.head.training


def test_enable_mc_dropout_reports_zero_without_dropout():
    model = EDSRLite(scale=4, in_channels=4, num_blocks=2, num_features=16, dropout=0.0)
    assert model.enable_mc_dropout() == 0


def test_receptive_field_grows_with_depth():
    assert receptive_field(12) > receptive_field(4) > 1


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def test_ssim_is_one_for_identical_images():
    x = torch.rand(2, 4, 32, 32)
    assert float(SSIM()(x, x)) == pytest.approx(1.0, abs=1e-4)


def test_ssim_loss_is_zero_for_identical_images():
    x = torch.rand(2, 4, 32, 32)
    assert float(SSIMLoss()(x, x)) == pytest.approx(0.0, abs=1e-4)


def test_ssim_penalises_blurring():
    import torch.nn.functional as F

    sharp = torch.zeros(1, 4, 32, 32)
    sharp[:, :, 16:, :] = 1.0
    blurred = F.avg_pool2d(sharp, 5, stride=1, padding=2)
    assert float(SSIM()(blurred, sharp)) < float(SSIM()(sharp, sharp))


def test_spectral_loss_is_zero_when_band_ratios_match():
    x = torch.rand(2, 4, 16, 16) + 0.1
    assert float(SpectralAngleLoss()(x, x)) == pytest.approx(0.0, abs=1e-6)


def test_spectral_loss_ignores_uniform_brightness_scaling():
    """SAM is scale-invariant: doubling every band is spectrally identical."""
    x = torch.rand(2, 4, 16, 16) + 0.1
    assert float(SpectralAngleLoss()(x * 2.0, x)) == pytest.approx(0.0, abs=1e-5)


def test_spectral_loss_detects_a_band_ratio_shift():
    x = torch.rand(2, 4, 16, 16) + 0.1
    shifted = x.clone()
    shifted[:, 3] *= 2.0  # NIR only -> NDVI changes
    assert float(SpectralAngleLoss()(shifted, x)) > 1e-3


def test_spectral_loss_gradient_is_finite_near_the_optimum():
    """The reason 1-cos is used instead of arccos: arccos blows up here."""
    target = torch.rand(1, 4, 8, 8) + 0.1
    pred = (target + 1e-6).clone().requires_grad_(True)
    SpectralAngleLoss()(pred, target).backward()
    assert torch.isfinite(pred.grad).all()


def test_gradient_loss_is_zero_for_identical_images():
    x = torch.rand(2, 4, 16, 16)
    assert float(GradientLoss()(x, x)) == pytest.approx(0.0, abs=1e-7)


def test_gradient_loss_penalises_over_smoothing():
    import torch.nn.functional as F

    sharp = torch.rand(1, 4, 32, 32)
    blurred = F.avg_pool2d(sharp, 5, stride=1, padding=2)
    assert float(GradientLoss()(blurred, sharp)) > 0


def test_charbonnier_approaches_l1_for_large_errors():
    pred, target = torch.zeros(1, 1, 8, 8), torch.full((1, 1, 8, 8), 1.0)
    assert float(CharbonnierLoss(1e-3)(pred, target)) == pytest.approx(1.0, abs=1e-3)


def test_combined_loss_is_zero_for_a_perfect_reconstruction(cfg):
    x = torch.rand(2, 4, 32, 32) + 0.1
    total, parts = CombinedLoss.from_config(cfg)(x, x)
    assert float(total) == pytest.approx(0.0, abs=1e-4)
    assert set(parts) >= {"pixel", "structural", "spectral", "total"}


def test_combined_loss_is_positive_and_differentiable(cfg):
    criterion = CombinedLoss.from_config(cfg)
    pred = torch.rand(2, 4, 32, 32, requires_grad=True)
    total, _ = criterion(pred, torch.rand(2, 4, 32, 32))
    assert float(total) > 0
    total.backward()
    assert torch.isfinite(pred.grad).all()


def test_combined_loss_skips_zero_weighted_terms(cfg):
    criterion = CombinedLoss.from_config(
        cfg.merge({"loss": {"weights": {"pixel": 1.0, "structural": 0.0, "spectral": 0.0, "gradient": 0.0}}})
    )
    _, parts = criterion(torch.rand(1, 4, 16, 16), torch.rand(1, 4, 16, 16))
    assert set(parts) == {"pixel", "total"}


def test_combined_loss_rejects_all_zero_weights():
    with pytest.raises(ValueError, match="at least one loss weight"):
        CombinedLoss(weights={"pixel": 0.0, "structural": 0.0, "spectral": 0.0, "gradient": 0.0})


def test_combined_loss_rejects_unknown_weights():
    with pytest.raises(ValueError, match="unknown loss weight"):
        CombinedLoss(weights={"perceptual": 1.0})


def test_combined_loss_rejects_shape_mismatch(cfg):
    with pytest.raises(ValueError, match="same shape"):
        CombinedLoss.from_config(cfg)(torch.rand(1, 4, 16, 16), torch.rand(1, 4, 32, 32))


def test_unknown_pixel_type_is_rejected():
    with pytest.raises(ValueError, match="unknown pixel_type"):
        CombinedLoss(pixel_type="huber")
