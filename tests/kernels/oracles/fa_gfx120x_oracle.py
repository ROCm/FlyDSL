# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Test-only FA gfx120x helpers moved out of the product host."""

import torch


def bottom_right_causal_bias(
    seq_len_q: int,
    seq_len_kv: int,
    device: torch.device,
) -> torch.Tensor:
    """Bottom-right-aligned causal additive bias ``[Sq, Skv]`` (0 / -inf).

    Matches FlashAttention: query ``i`` attends to keys ``j <= i + Skv - Sq``.
    Product gfx120x FA applies causal / causal×cross masking in-kernel; this
    helper is the test oracle only.
    """
    q_idx = torch.arange(seq_len_q, device=device, dtype=torch.int32)[:, None]
    k_idx = torch.arange(seq_len_kv, device=device, dtype=torch.int32)[None, :]
    allow = k_idx <= (q_idx + (seq_len_kv - seq_len_q))
    bias = torch.zeros(seq_len_q, seq_len_kv, dtype=torch.float32, device=device)
    bias = bias.masked_fill(~allow, float("-inf"))
    return bias
