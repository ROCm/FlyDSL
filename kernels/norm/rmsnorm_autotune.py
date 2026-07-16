# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Two-track RMSNorm autotuning through the normal direct JIT path."""

from flydsl.autotune import autotune
from kernels.norm.rmsnorm_config import _get_all_configs_for_autotune, _get_default_for_autotune
from kernels.norm.rmsnorm_kernel import rmsnorm_direct

_rmsnorm_tuner = autotune(
    configs=_get_all_configs_for_autotune,
    key=["N", "dtype_str"],
    default=_get_default_for_autotune,
)(rmsnorm_direct)


def rmsnorm_autotuned(input_t, gamma, output, m_in, dtype_str="bf16", stream=None):
    return _rmsnorm_tuner(
        input_t,
        gamma,
        output,
        m_in,
        N=int(input_t.shape[-1]),
        dtype_str=dtype_str,
        stream=stream,
    )


rmsnorm_autotuned.tuner = _rmsnorm_tuner
