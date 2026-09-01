"""Super-resolution models and losses.

Architectures register themselves with :func:`register_model`, and every stage
of the pipeline builds models through :func:`build_model`. Swapping the CNN for
a Transformer (SwinIR) or a diffusion model is therefore a matter of adding one
file and one config line — training, inference, uncertainty and evaluation need
no changes, provided the new model honours the shared contract:

    forward(x: Tensor[B, C_in, h, w]) -> Tensor[B, C_out, h*scale, w*scale]
"""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Class/function decorator that adds an architecture to the registry."""

    def decorator(factory: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        _REGISTRY[key] = factory
        return factory

    return decorator


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(cfg) -> nn.Module:
    """Instantiate the architecture named by ``cfg.model.name``."""
    from . import baseline, generator  # noqa: F401  (populates the registry)

    spec = dict(cfg.model) if not isinstance(cfg.model, dict) else dict(cfg.model)
    name = str(spec.pop("name")).lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown model {name!r}; registered: {available_models()}. "
            "Add a new architecture with @register_model."
        )
    return _REGISTRY[name](**spec)


__all__ = ["register_model", "build_model", "available_models"]
