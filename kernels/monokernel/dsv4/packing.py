# SPDX-License-Identifier: Apache-2.0
"""Native gfx950 FP8 x FP4 MFMA weight fragments."""

import torch


def pack_experts(q):
    """[E,N,K/2] -> [E,N/16,K/128,lane_k32,lane_row,16 bytes]."""
    q = q.view(torch.uint8)
    e, n, kh = q.shape
    if n % 16 or kh % 64:
        raise ValueError("native A8W4 packing needs N/16 and K/128")
    return q.reshape(e, n // 16, 16, kh // 64, 4, 16).permute(0, 1, 3, 4, 2, 5).contiguous().flatten()


def pack_shared_fp8(q):
    """FP8 [N,K] -> scaled MFMA K128 fragments, two K64 register halves."""
    n, k = q.shape
    if n % 16 or k % 128:
        raise ValueError("shared FP8 packing needs N/16 and K/128")
    return q.view(torch.uint8).reshape(n // 16, 16, k // 128, 2, 4, 16).permute(0, 2, 4, 1, 3, 5).contiguous().flatten()
