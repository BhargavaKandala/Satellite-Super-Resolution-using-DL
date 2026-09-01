"""Device selection and process-wide PyTorch tuning.

CPU is the *committed default* (``compute.device: cpu`` in ``configs/config.yaml``).
That is deliberate: a fresh clone then behaves identically on every machine, and
nobody discovers halfway through a run that they were silently on a GPU with a
driver too old to actually work. Opting into an accelerator is an explicit act —
either ``--device cuda`` or, better, ``--profile dgx_b200``.

Precedence for the device, highest first:

1. an explicit ``--device`` / ``preferred=`` argument
2. ``compute.device`` in the (possibly profile-merged) config
3. ``"auto"`` — CUDA when available, else CPU

Mixed precision stays CUDA-only. CPU autocast is bf16-only and, without
AVX512-BF16, is usually *slower* than plain fp32 — so enabling it by default
would be a pessimisation dressed up as an optimisation.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

#: Used whenever the config omits a ``compute`` block, so older configs and
#: ad-hoc ``Config.merge`` overrides in tests keep working unchanged.
DEFAULTS: dict[str, Any] = {
    "device": "cpu",
    "threads": "auto",
    "amp_dtype": "auto",
    "matmul_precision": "high",
}

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _settings(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULTS)
    if cfg is not None:
        merged.update(dict(cfg.get("compute", {}) or {}))
    return merged


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def resolve_device(
    preferred: str | None = None, cfg: Mapping[str, Any] | None = None
) -> torch.device:
    """Resolve the compute device from CLI override, then config, then auto."""
    choice = str(preferred or _settings(cfg)["device"]).strip().lower()
    if choice in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. Either "
            "the machine has no NVIDIA GPU, or the installed torch build does "
            "not match the driver. Check `nvidia-smi` against "
            f"torch {torch.__version__}, or run on CPU with --device cpu."
        )
    return device


def describe_device(device: torch.device) -> str:
    """One-line human description, printed by every entry point."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return f"cpu ({torch.get_num_threads()} threads)"

    index = device.index or 0
    props = torch.cuda.get_device_properties(index)
    capability = f"sm_{props.major}{props.minor}"
    return (
        f"cuda:{index} ({props.name}, {props.total_memory / 1e9:.1f} GB, "
        f"{capability})"
    )


def check_cuda_build(device: torch.device) -> list[str]:
    """Warn when the torch build has no kernels for this GPU.

    The failure this catches — ``no kernel image is available for execution on
    the device`` — surfaces at the first convolution, long after the run has
    printed a healthy-looking banner. Newer datacentre parts are the usual
    victims: an NVIDIA B200 is ``sm_100`` (Blackwell) and needs a CUDA 12.8+
    build, which a default ``pip install torch`` will not necessarily give you.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return []

    props = torch.cuda.get_device_properties(device.index or 0)
    target = f"sm_{props.major}{props.minor}"
    supported = torch.cuda.get_arch_list()
    if supported and target not in supported:
        return [
            f"this torch build ({torch.__version__}) was compiled for "
            f"{', '.join(supported)} but the GPU is {target} ({props.name}). "
            "Expect 'no kernel image is available for execution on the device'. "
            "Install a matching build, or use the NVIDIA NGC PyTorch container."
        ]
    return []


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
def autocast_dtype(
    device: torch.device, requested: str = "auto"
) -> torch.dtype:
    """Pick the autocast dtype.

    ``auto`` prefers bf16 wherever the hardware supports it. bf16 carries the
    same exponent range as fp32, so it needs no loss scaling and cannot silently
    overflow — worth more here than fp16's extra mantissa bit, given reflectance
    already lives in ``[0, 1]``.
    """
    key = str(requested).strip().lower()
    if key in AMP_DTYPES:
        return AMP_DTYPES[key]
    if key != "auto":
        raise ValueError(
            f"unknown compute.amp_dtype {requested!r}; expected auto | bf16 | fp16"
        )
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


def amp_enabled(cfg: Mapping[str, Any] | None, device: torch.device) -> bool:
    """Mixed precision is CUDA-only here — see the module docstring."""
    if device.type != "cuda":
        return False
    training = dict(cfg.get("training", {}) or {}) if cfg is not None else {}
    return bool(training.get("amp", True))


# ---------------------------------------------------------------------------
# Process-wide setup
# ---------------------------------------------------------------------------
def configure_torch(
    cfg: Mapping[str, Any] | None = None, device: torch.device | None = None
) -> list[str]:
    """Apply the ``compute`` settings globally. Returns notes worth printing."""
    settings = _settings(cfg)
    notes: list[str] = []

    threads = settings.get("threads", "auto")
    if str(threads).strip().lower() not in ("auto", "0", "none", ""):
        count = int(threads)
        if count < 1:
            raise ValueError(f"compute.threads must be >= 1 or 'auto', got {threads!r}")
        torch.set_num_threads(count)
        notes.append(f"intra-op threads pinned to {count}")

    precision = str(settings.get("matmul_precision", "high")).strip().lower()
    if precision not in ("none", ""):
        if precision not in ("highest", "high", "medium"):
            raise ValueError(
                f"unknown compute.matmul_precision {precision!r}; "
                "expected highest | high | medium | none"
            )
        torch.set_float32_matmul_precision(precision)

    if device is not None:
        notes.extend(check_cuda_build(device))
    return notes


def summary(cfg: Mapping[str, Any] | None, device: torch.device) -> dict[str, Any]:
    """Machine-readable compute description, embedded in run manifests."""
    settings = _settings(cfg)
    enabled = amp_enabled(cfg, device)
    return {
        "device": describe_device(device),
        "device_type": device.type,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "amp": enabled,
        "amp_dtype": str(autocast_dtype(device, settings["amp_dtype"])).replace(
            "torch.", ""
        )
        if enabled
        else None,
    }
