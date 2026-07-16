# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Small, explicit search space for the RMSNorm autotune adopter."""

from flydsl.autotune import Config
from kernels.norm.rmsnorm_common import BLOCK_THREADS
from kernels.norm.rmsnorm_kernel import SMALL_N_THRESHOLD

_SEARCH_CONFIGS = (
    Config(BLOCK_THREADS=128),
    Config(BLOCK_THREADS=128, waves_per_eu=1),
    Config(BLOCK_THREADS=128, waves_per_eu=2),
    Config(BLOCK_THREADS=256),
    Config(BLOCK_THREADS=256, waves_per_eu=1),
    Config(BLOCK_THREADS=512),
    Config(BLOCK_THREADS=512, waves_per_eu=2),
)


def get_default(N: int, dtype_str: str) -> Config:
    """Keep the established production block size when search is disabled."""
    return Config(BLOCK_THREADS=BLOCK_THREADS)


def get_all_configs(N: int, dtype_str: str):
    """Return direct-JIT candidates; the small-N kernel has no tunable block."""
    if N <= SMALL_N_THRESHOLD:
        return [get_default(N, dtype_str)]
    return list(_SEARCH_CONFIGS)


def _get_default_for_autotune(input_t, gamma, output, m_in, N, dtype_str="bf16", stream=None):
    """Adapt the direct launcher's call signature to the default heuristic."""
    return get_default(N, dtype_str)


def _get_all_configs_for_autotune(input_t, gamma, output, m_in, N, dtype_str="bf16", stream=None):
    """Adapt the direct launcher's call signature to the search space."""
    return get_all_configs(N, dtype_str)
