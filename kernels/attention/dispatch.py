# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Arch dispatch for the attention kernels.

Maps a GPU arch prefix to the module that holds that arch's knob factories and
its forward / dQ / dK-dV builders. The arch-neutral tests and
`flydsl_flash_attn_func` reach a backend only through here, so a later backend
(gfx1201, ...) is one more row in `_BACKENDS`.

Importing this module imports nothing else: the config and kernel modules are
loaded on first use, so collection and arch detection are safe on any machine,
including one without flydsl or torch.

The contract each backend satisfies:

* config module: `fwd_knobs(arch, **overrides)`, `dq_knobs(...)`, `dkdv_knobs(...)`
  return knob objects whose `.resolve(meta, hints)` fills in every derived knob; `fwd_traits(meta, knobs)` (and the dQ,
  dK/dV twins) derive the traits, and `build_cache_key(traits, knobs)` names a build;
* the forward, dQ and dK/dV builders, each `build(meta, knobs)`, named by
  `"module:function"` strings in the `Backend` row.

Each builder returns the launcher closure (`.compile`, `.traits`, `.knobs`).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

__all__ = ["Backend", "backend_for", "current_arch", "has_backend"]


@dataclass(frozen=True)
class Backend:
    """One arch's entry points. Attribute access imports lazily."""

    arch_prefix: str
    config_module: str
    fwd: str
    dq: str
    dkdv: str

    @staticmethod
    def _load(module, name):
        return getattr(importlib.import_module(module), name)

    @classmethod
    def _resolve(cls, ref):
        module, name = ref.split(":")
        return cls._load(module, name)

    def fwd_knobs(self, arch, **overrides):
        return self._load(self.config_module, "fwd_knobs")(arch, **overrides)

    def dq_knobs(self, arch, **overrides):
        return self._load(self.config_module, "dq_knobs")(arch, **overrides)

    def dkdv_knobs(self, arch, **overrides):
        return self._load(self.config_module, "dkdv_knobs")(arch, **overrides)

    def fwd_traits(self, meta, knobs):
        return self._load(self.config_module, "fwd_traits")(meta, knobs)

    def dq_traits(self, meta, knobs):
        return self._load(self.config_module, "dq_traits")(meta, knobs)

    def dkdv_traits(self, meta, knobs):
        return self._load(self.config_module, "dkdv_traits")(meta, knobs)

    def build_cache_key(self, traits, knobs):
        """The identity of a build: every trait and every knob. Two `(meta, knobs)` pairs with equal keys build the
        same kernel, whatever the metadata's real head dim says."""
        return self._load(self.config_module, "build_cache_key")(traits, knobs)

    def build_fwd(self, meta, knobs):
        return self._resolve(self.fwd)(meta, knobs)

    def build_dq(self, meta, knobs):
        return self._resolve(self.dq)(meta, knobs)

    def build_dkdv(self, meta, knobs):
        return self._resolve(self.dkdv)(meta, knobs)


_BACKENDS = (
    Backend(
        arch_prefix="gfx95",
        config_module="kernels.attention.flash_attn_gfx950_config",
        fwd="kernels.attention.flash_attn_gfx950:build_flash_attn_gfx950_fwd",
        dq="kernels.attention.flash_attn_gfx950_dq:build_flash_attn_gfx950_dq",
        dkdv="kernels.attention.flash_attn_gfx950_dkdv:build_flash_attn_gfx950_dkdv",
    ),
)


def current_arch() -> str:
    """The arch of the current device, e.g. `"gfx950"`."""
    from flydsl.runtime.device import get_rocm_arch

    return str(get_rocm_arch())


def backend_for(arch: str | None = None) -> Backend | None:
    """The backend serving `arch` (default: the current device), or None."""
    arch = current_arch() if arch is None else str(arch)
    base = arch.split(":")[0]
    for backend in _BACKENDS:
        if base.startswith(backend.arch_prefix):
            return backend
    return None


def has_backend(arch: str | None = None) -> bool:
    return backend_for(arch) is not None
