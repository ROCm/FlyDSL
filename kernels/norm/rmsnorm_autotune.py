# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Two-track RMSNorm autotuning through the normal direct JIT path."""

from flydsl.autotune import Config, autotune
from kernels.norm.rmsnorm_common import BLOCK_THREADS
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


def _search_configs(input_t, gamma, output, m_in, N, dtype_str="bf16", stream=None):
    if N <= SMALL_N_THRESHOLD:
        return [_default_config()]
    return list(_SEARCH_CONFIGS)


_rmsnorm_tuner = autotune(
    configs=_search_configs,
    key=["m_in", "N", "dtype_str"],
    default=_default_config,
)(rmsnorm_direct)


def rmsnorm_autotuned(input_t, gamma, output, m_in, dtype_str="bf16", stream=None):
    import torch

    with torch.cuda.device(input_t.device):
        launch_stream = torch.cuda.current_stream() if stream is None else stream
        return _rmsnorm_tuner(
            input_t,
            gamma,
            output,
            m_in,
            N=int(input_t.shape[-1]),
            dtype_str=dtype_str,
            stream=launch_stream,
        )
