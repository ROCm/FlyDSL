# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Autotuned RMSNorm — the first real adopter of ``flydsl.autotune``.

``BLOCK_THREADS`` is a JIT Constexpr, so each structural config specializes the
normal FlyDSL JIT path. ``waves_per_eu`` remains a backend compile hint.
Normal runs use the ``get_default`` heuristic; ``FLYDSL_AUTOTUNE=1`` sweeps
``get_all_configs``.

    rmsnorm_autotuned(input, gamma, output, M, dtype_str="bf16", stream=stream)
"""

from flydsl.autotune import autotune
from kernels.norm.rmsnorm_config import _get_all_configs_for_autotune, _get_default_for_autotune
from kernels.norm.rmsnorm_kernel import rmsnorm_direct

_rmsnorm_tuner = autotune(
    configs=_get_all_configs_for_autotune,
    key=["N", "dtype_str"],
    default=_get_default_for_autotune,
)(rmsnorm_direct)


def rmsnorm_autotuned(input_t, gamma, output, m_in, dtype_str="bf16", stream=None):
    """Launch RMSNorm while deriving the compile-time row width from input."""
    return _rmsnorm_tuner(
        input_t,
        gamma,
        output,
        m_in,
        N=int(input_t.shape[-1]),
        dtype_str=dtype_str,
        stream=stream,
    )


# Preserve the small public inspection surface used by tests and tooling.
rmsnorm_autotuned.tuner = _rmsnorm_tuner
