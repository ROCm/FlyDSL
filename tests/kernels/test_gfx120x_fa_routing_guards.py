# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""CPU checks for gfx120x FA routing guards. No GPU and no kernel launch."""

import pytest
import torch

from kernels.attention.flash_attn_gfx120x_ext import (
    cu_seqlens_from_seqlens,
    lengths_uniform,
    reject_gqa,
    reject_paged_layout,
    reject_quant_extras,
    reject_sink_with_alibi,
    reject_splitk_extras,
    reject_varlen_with_paged,
)

pytestmark = [pytest.mark.l0_backend_agnostic]


def test_gqa_rejected() -> None:
    # Invalid groups still raise. Valid GQA (8 q / 2 kv) is in-kernel.
    with pytest.raises(ValueError, match="divisible"):
        reject_gqa(8, 3, None)
    with pytest.raises(ValueError, match="does not match"):
        reject_gqa(8, 2, 4)
    with pytest.raises(ValueError, match="does not match"):
        reject_gqa(8, 8, 2)
    reject_gqa(8, 2, None)
    reject_gqa(8, 2, 2)
    reject_gqa(8, 8, 8)


def test_fp8_extras_not_dropped_by_guard() -> None:
    # bias / ALiBi / sink / LSE are applied in the quant kernel; the guard must not raise.
    reject_quant_extras("fp8", bias=object(), alibi_slopes=object(), sink=object(), return_lse=True)
    reject_quant_extras("int8", attn_mask=object(), return_lse=True)
    reject_quant_extras("fp8")


def test_ragged_lengths_are_not_uniform() -> None:
    assert not lengths_uniform(torch.tensor([16, 32]), 32)
    assert lengths_uniform(torch.tensor([32, 32]), 32)
    assert lengths_uniform(torch.tensor([], dtype=torch.int32), 0)


def test_vectorized_paged_layout_rejected() -> None:
    reject_paged_layout("vectorized")
    reject_paged_layout("linear3d")
    reject_paged_layout("linear")
    reject_paged_layout(None)
    with pytest.raises(NotImplementedError, match="blocked"):
        reject_paged_layout("blocked")


def test_varlen_with_paged_allowed() -> None:
    reject_varlen_with_paged(True, True)
    reject_varlen_with_paged(True, False)
    reject_varlen_with_paged(False, True)


def test_commit_caller_out_writes_and_rejects() -> None:
    from kernels.attention.flash_attn_gfx120x_ext import commit_caller_out

    produced = torch.ones(2, 3)
    assert commit_caller_out(None, produced) is produced
    buf = torch.empty(2, 3)
    got = commit_caller_out(buf, produced)
    assert got is buf
    assert torch.equal(buf, produced)
    with pytest.raises(ValueError, match="mismatch"):
        commit_caller_out(torch.empty(1, 3), produced)


def test_sink_alibi_and_splitk_extras_allowed() -> None:
    reject_sink_with_alibi(object(), object())
    reject_sink_with_alibi(object(), None)
    reject_splitk_extras(alibi_slopes=object(), return_lse=True)
    reject_splitk_extras()


def test_cu_seqlens_from_seqlens_ragged() -> None:
    lens = torch.tensor([16, 32, 8], dtype=torch.int32)
    cu = cu_seqlens_from_seqlens(lens, device=torch.device("cpu"))
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 16, 48, 56]
    empty = cu_seqlens_from_seqlens(torch.tensor([], dtype=torch.int32), device=torch.device("cpu"))
    assert empty.tolist() == [0]
