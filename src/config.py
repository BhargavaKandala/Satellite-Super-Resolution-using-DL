"""Configuration loading.

A single YAML file drives every stage of the pipeline. Values are exposed as
plain nested dicts wrapped in a small attribute-access shim so that call sites
read as ``cfg.patches.hr_patch_size`` instead of ``cfg["patches"]["hr_patch_size"]``
while remaining trivially serialisable back to YAML/JSON for run manifests.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
PROFILE_DIR = REPO_ROOT / "configs" / "profiles"


class Config(Mapping):
    """Read-only nested mapping with attribute access.

    Implements ``Mapping`` so ``dict(cfg)``, ``**cfg`` and ``yaml.safe_dump``
    keep working — the shim never hides the underlying data.
    """

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, "_data", data)

    # -- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- attribute sugar --------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(f"no config key {key!r}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError("Config is read-only; use Config.merge() to override")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self._data!r})"

    # -- helpers ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy as plain Python containers."""
        import copy

        return copy.deepcopy(self._data)

    def get_path(self, dotted: str) -> Path:
        """Resolve a dotted config key holding a path, relative to the repo root."""
        raw = Path(str(self.get_nested(dotted)))
        return raw if raw.is_absolute() else (REPO_ROOT / raw)

    def get_nested(self, dotted: str, default: Any = ...) -> Any:
        """Look up ``"a.b.c"``. Raises KeyError unless ``default`` is given."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    raise KeyError(f"missing config key {dotted!r}")
                return default
            node = node[part]
        return Config(node) if isinstance(node, dict) else node

    def merge(self, overrides: Mapping[str, Any]) -> "Config":
        """Return a new Config with ``overrides`` deep-merged on top."""
        merged = _deep_merge(self.to_dict(), dict(overrides))
        return Config(merged)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"{path.name}: config root must be a mapping, got {type(data).__name__}"
        )
    return data


def available_profiles() -> list[str]:
    """Names of the hardware profiles shipped in ``configs/profiles/``."""
    if not PROFILE_DIR.is_dir():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.yaml"))


def resolve_profile(name: str | os.PathLike[str]) -> Path:
    """Accept either a bare profile name or a path to a YAML overlay."""
    candidate = Path(name)
    if candidate.suffix in (".yaml", ".yml"):
        path = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
        if path.exists():
            return path
    path = PROFILE_DIR / f"{Path(name).stem}.yaml"
    if not path.exists():
        known = ", ".join(available_profiles()) or "none installed"
        raise FileNotFoundError(f"unknown profile {str(name)!r}; available: {known}")
    return path


def load_config(
    path: str | os.PathLike[str] | None = None,
    profile: str | os.PathLike[str] | None = None,
) -> Config:
    """Load the YAML config, optionally deep-merging a hardware profile on top.

    Profiles are *overlays*, not replacements: ``configs/profiles/dgx_b200.yaml``
    holds only the handful of keys that differ from the CPU baseline, so the two
    cannot drift apart the way two full copies of the config would.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()

    data = _read_yaml(cfg_path)
    if profile:
        data = _deep_merge(data, _read_yaml(resolve_profile(profile)))
    _validate(data)
    return Config(data)


def _validate(data: dict[str, Any]) -> None:
    """Catch the config mistakes that would otherwise surface deep in training."""
    bands = data.get("data", {}).get("bands", [])
    band_indices = data.get("data", {}).get("band_indices", [])
    if len(bands) != len(band_indices):
        raise ValueError(
            f"data.bands ({len(bands)}) and data.band_indices "
            f"({len(band_indices)}) must have the same length"
        )

    model = data.get("model", {})
    if model.get("in_channels") not in (None, len(bands)):
        raise ValueError(
            f"model.in_channels ({model['in_channels']}) must equal "
            f"len(data.bands) ({len(bands)})"
        )

    patch_scale = data.get("patches", {}).get("scale")
    model_scale = model.get("scale")
    if patch_scale is not None and model_scale is not None and patch_scale != model_scale:
        raise ValueError(
            f"patches.scale ({patch_scale}) != model.scale ({model_scale}); "
            "the model must upsample by exactly the factor the data was degraded by"
        )

    hr = data.get("patches", {}).get("hr_patch_size")
    if hr is not None and patch_scale and hr % patch_scale != 0:
        raise ValueError(
            f"patches.hr_patch_size ({hr}) must be divisible by scale ({patch_scale})"
        )

    compute = data.get("compute", {})
    device = str(compute.get("device", "cpu")).strip().lower()
    if device not in ("cpu", "auto") and not device.startswith("cuda"):
        raise ValueError(
            f"unknown compute.device {device!r}; expected cpu | cuda | cuda:N | auto"
        )
    amp_dtype = str(compute.get("amp_dtype", "auto")).strip().lower()
    if amp_dtype not in ("auto", "bf16", "fp16"):
        raise ValueError(
            f"unknown compute.amp_dtype {amp_dtype!r}; expected auto | bf16 | fp16"
        )

    src_res = data.get("data", {}).get("source_resolution_m")
    tgt_res = data.get("data", {}).get("target_resolution_m")
    if src_res and tgt_res and patch_scale:
        expected = src_res / patch_scale
        if abs(expected - tgt_res) > 1e-6:
            raise ValueError(
                f"data.target_resolution_m ({tgt_res}) must equal "
                f"source_resolution_m / scale ({expected})"
            )


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # torch is optional for the data-only code paths
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Autotuned convolutions: materially faster for the fixed patch sizes
        # used during training, at the cost of bitwise reproducibility.
        torch.backends.cudnn.benchmark = True
