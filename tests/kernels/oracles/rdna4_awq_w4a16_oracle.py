# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Test-only torch oracles extracted from ``kernels/quant/rdna4_awq_w4a16.py``."""

import torch

from kernels.quant.rdna4_int4_codec import dequant_uint4_groupwise_awq

_DEFAULT_GROUP = 64


def reference_dequant_awq_w4a16(
    qweight: torch.Tensor, wscales: torch.Tensor, wzeros: torch.Tensor | None, group_size: int = _DEFAULT_GROUP
) -> torch.Tensor:
    """Torch reference. The arithmetic is ``dequant_uint4_groupwise_awq``."""
    return dequant_uint4_groupwise_awq(qweight, wscales, wzeros, group_size)
