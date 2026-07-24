# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Two-track RMSNorm autotuning through the normal direct JIT path."""

from flydsl.autotune import Config, autotune
from kernels.norm.rmsnorm_common import (
    BLOCK_THREADS,
    resolve_rmsnorm_weight_dtype,
)
from kernels.norm.rmsnorm_kernel import SMALL_N_THRESHOLD, rmsnorm_direct

_SEARCH_CONFIGS = (
    Config(BLOCK_THREADS=128),
    Config(BLOCK_THREADS=128, waves_per_eu=1),
    Config(BLOCK_THREADS=128, waves_per_eu=2),
    Config(BLOCK_THREADS=256),
    Config(BLOCK_THREADS=256, waves_per_eu=1),
    Config(BLOCK_THREADS=512),
    Config(BLOCK_THREADS=512, waves_per_eu=2),
)


def _default_config(*_args, **_kwargs):
    return Config(BLOCK_THREADS=BLOCK_THREADS)


def _search_configs(
    input_t,
    gamma,
    output,
    m_in,
    N,
    dtype_str="bf16",
    weight_dtype_str="bf16",
    stream=None,
):
    if N <= SMALL_N_THRESHOLD:
        return [_default_config()]
    return list(_SEARCH_CONFIGS)


_rmsnorm_tuner = autotune(
    configs=_search_configs,
    key=["m_in", "N", "dtype_str", "weight_dtype_str"],
    default=_default_config,
    artifact_name="rmsnorm",
)(rmsnorm_direct)


def rmsnorm_autotuned(
    input_t,
    gamma,
    output,
    m_in,
    dtype_str=None,
    stream=None,
    weight_dtype_str=None,
):
    import torch

    from kernels.norm.rmsnorm_common import torch_dtype_to_str

    input_dtype_str = torch_dtype_to_str(input_t.dtype)
    if dtype_str is not None and dtype_str != input_dtype_str:
        raise ValueError(f"dtype_str={dtype_str!r} does not match input dtype {input_dtype_str!r}")
    dtype_str = input_dtype_str

    gamma_dtype_str = torch_dtype_to_str(gamma.dtype)
    if weight_dtype_str is not None and weight_dtype_str != gamma_dtype_str:
        raise ValueError(f"weight_dtype_str={weight_dtype_str!r} does not match gamma dtype {gamma_dtype_str!r}")
    weight_dtype_str = resolve_rmsnorm_weight_dtype(dtype_str, gamma_dtype_str)

    with torch.cuda.device(input_t.device):
        launch_stream = torch.cuda.current_stream() if stream is None else stream
        return _rmsnorm_tuner(
            input_t,
            gamma,
            output,
            m_in,
            N=int(input_t.shape[-1]),
            dtype_str=dtype_str,
            weight_dtype_str=weight_dtype_str,
            stream=launch_stream,
        )
