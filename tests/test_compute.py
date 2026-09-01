"""Device selection, precision policy and hardware profiles.

The behaviour these lock down is "CPU unless you asked for something else".
A regression here is silent: the run still completes, just on the wrong device
or in the wrong precision.
"""

from __future__ import annotations

import pytest
import torch

from src.compute import (
    DEFAULTS,
    amp_enabled,
    autocast_dtype,
    check_cuda_build,
    configure_torch,
    describe_device,
    resolve_device,
    summary,
)
from src.config import available_profiles, load_config, resolve_profile

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def test_the_shipped_default_is_cpu():
    """The headline guarantee: a fresh clone runs on CPU."""
    assert DEFAULTS["device"] == "cpu"
    assert load_config().compute.device == "cpu"


def test_config_default_is_honoured_over_available_hardware(cfg):
    assert resolve_device(cfg=cfg).type == "cpu"


def test_explicit_argument_outranks_the_config(cfg):
    gpu_cfg = cfg.merge({"compute": {"device": "cuda"}})
    assert resolve_device("cpu", gpu_cfg).type == "cpu"


def test_auto_falls_back_to_cpu_without_cuda(cfg):
    device = resolve_device(None, cfg.merge({"compute": {"device": "auto"}}))
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_missing_compute_block_still_resolves(cfg):
    """Configs predating the compute section must not crash."""
    stripped = {k: v for k, v in cfg.to_dict().items() if k != "compute"}
    from src.config import Config

    assert resolve_device(cfg=Config(stripped)).type == "cpu"


def test_resolve_device_without_any_config_defaults_to_cpu():
    assert resolve_device().type == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a machine without CUDA")
def test_requesting_cuda_without_cuda_fails_loudly(cfg):
    """Better a clear error at startup than a cryptic one mid-epoch."""
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda", cfg)


def test_describe_device_reports_thread_count_on_cpu():
    assert "cpu" in describe_device(CPU)
    assert str(torch.get_num_threads()) in describe_device(CPU)


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
def test_amp_is_disabled_on_cpu(cfg):
    """CPU autocast is bf16-only and usually slower — never on by default."""
    assert amp_enabled(cfg.merge({"training": {"amp": True}}), CPU) is False


def test_explicit_dtype_requests_are_respected():
    assert autocast_dtype(CPU, "fp16") is torch.float16
    assert autocast_dtype(CPU, "bf16") is torch.bfloat16


def test_auto_prefers_bfloat16():
    assert autocast_dtype(CPU, "auto") is torch.bfloat16


def test_unknown_dtype_is_rejected():
    with pytest.raises(ValueError, match="unknown compute.amp_dtype"):
        autocast_dtype(CPU, "int4")


def test_check_cuda_build_is_silent_on_cpu():
    assert check_cuda_build(CPU) == []


# ---------------------------------------------------------------------------
# Process-wide setup
# ---------------------------------------------------------------------------
def test_configure_torch_accepts_the_shipped_config():
    assert isinstance(configure_torch(load_config(), CPU), list)


def test_thread_pinning_takes_effect(cfg):
    original = torch.get_num_threads()
    try:
        notes = configure_torch(cfg.merge({"compute": {"threads": 1}}), CPU)
        assert torch.get_num_threads() == 1
        assert any("threads pinned to 1" in note for note in notes)
    finally:
        torch.set_num_threads(original)


def test_auto_threads_leaves_the_default_alone(cfg):
    before = torch.get_num_threads()
    configure_torch(cfg.merge({"compute": {"threads": "auto"}}), CPU)
    assert torch.get_num_threads() == before


def test_bad_matmul_precision_is_rejected(cfg):
    with pytest.raises(ValueError, match="matmul_precision"):
        configure_torch(cfg.merge({"compute": {"matmul_precision": "turbo"}}), CPU)


def test_summary_is_json_friendly(cfg):
    stats = summary(cfg, CPU)
    assert stats["device_type"] == "cpu"
    assert stats["amp"] is False
    assert stats["amp_dtype"] is None
    assert isinstance(stats["torch"], str)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
def test_both_profiles_ship():
    assert {"cpu", "dgx_b200"} <= set(available_profiles())


def test_an_unknown_profile_lists_the_real_ones():
    with pytest.raises(FileNotFoundError, match="dgx_b200"):
        resolve_profile("nonexistent")


def test_profiles_are_overlays_not_replacements():
    """A profile must inherit everything it does not mention."""
    base = load_config()
    merged = load_config(profile="dgx_b200")
    assert merged.compute.device == "cuda"
    assert merged.data.bands == base.data.bands
    assert merged.model.name == base.model.name


def test_the_cpu_profile_forces_cpu():
    assert resolve_device(cfg=load_config(profile="cpu")).type == "cpu"


def test_the_dgx_profile_requests_bfloat16():
    assert load_config(profile="dgx_b200").compute.amp_dtype == "bf16"


def test_every_profile_passes_config_validation():
    """Profiles can violate the scale/patch invariants just as easily as the base."""
    for name in available_profiles():
        merged = load_config(profile=name)
        assert int(merged.patches.hr_patch_size) % int(merged.patches.scale) == 0


def test_profiles_do_not_redefine_the_sensor():
    """Hardware overlays must stay hardware-only — never touch the science."""
    import yaml

    for name in available_profiles():
        overlay = yaml.safe_load(resolve_profile(name).read_text(encoding="utf-8"))
        assert not ({"data", "loss", "evaluation"} & set(overlay)), (
            f"profile {name!r} overrides a scientific setting; a device choice "
            "must never silently change what is being measured"
        )
