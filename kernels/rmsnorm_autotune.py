# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Autotuned RMSNorm — the first real adopter of ``flydsl.autotune``.

RMSNorm bakes its structural knob (BLOCK_THREADS) at module-build time, so the
tuner rebuilds the module per config via ``autotune_builder`` (builder mode).
Normal runs use the ``get_default`` heuristic; ``FLYDSL_AUTOTUNE=1`` sweeps
``get_all_configs``.

    rmsnorm_autotuned(input, gamma, output, M, dtype_str="bf16", stream=stream)
"""

from flydsl.autotune import autotune_builder
from kernels.rmsnorm_config import get_all_configs, get_default
from kernels.rmsnorm_kernel import build_rmsnorm_module


def _specialize(input_t, gamma, output, m_in, dtype_str="bf16", stream=None):
    # Build/lookup axes; dtype_str must be here so bf16 vs f16 keys differ.
    return {"N": int(input_t.shape[-1]), "dtype_str": dtype_str}


rmsnorm_autotuned = autotune_builder(
    name="rmsnorm",
    build=build_rmsnorm_module,
    specialize=_specialize,
    configs=get_all_configs,
    default=get_default,
    structural=("BLOCK_THREADS",),
)
