# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest
import torch

from kernels.monokernel.config import HIDDEN, KvCacheLayout
from kernels.monokernel.glm.index_quant import pack_index_fp4, unpack_index_fp4
from kernels.monokernel.glm.op import Glm5MonoKernel


def test_fp4_exact_values_and_row_layout():
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    rows = values.repeat(8).reshape(1, 128).bfloat16()
    packed = pack_index_fp4(rows)
    assert packed.shape == (1, 68) and packed.dtype == torch.uint8
    assert packed[0, :8].tolist() == [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]
    assert packed[0, 64:].tolist() == [127] * 4
    restored = unpack_index_fp4(packed)
    assert torch.equal(restored.view(torch.int16), rows.view(torch.int16))


@pytest.mark.parametrize("power", [-20, -1, 0, 1, 20])
def test_fp4_midpoints_tie_to_even_and_block_scales(power):
    values = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    expected = torch.tensor([0, 1, 1, 2, 2, 4, 4, 6])
    rows = (values.repeat(16).reshape(1, 1, 128) * 2**power).bfloat16()
    packed = pack_index_fp4(rows)
    assert packed.shape == (1, 1, 68)
    assert packed[0, 0, 64:].tolist() == [127 + power] * 4
    assert torch.equal(unpack_index_fp4(packed).float().flatten(), expected.repeat(16) * 2**power)


def test_fp4_zero_and_independent_block_scales():
    rows = torch.zeros((2, 128), dtype=torch.bfloat16)
    rows[1] = torch.tensor([6, 12, 24, 48]).repeat_interleave(32)
    packed = pack_index_fp4(rows)
    assert packed[1, 64:].tolist() == [127, 128, 129, 130]
    assert torch.equal(unpack_index_fp4(packed), rows)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_fp4_rejects_nonfinite(value):
    with pytest.raises(ValueError, match="finite"):
        pack_index_fp4(torch.full((1, 128), value, dtype=torch.bfloat16))


@pytest.mark.parametrize("setting", ["fp4", 1, None])
def test_quant_switch_rejects_non_boolean_before_weights(setting):
    with pytest.raises(ValueError, match="must be a bool"):
        Glm5MonoKernel(object(), 1, with_indexer=True, indexer_quant=setting)


def test_quant_switch_requires_indexer_before_weights():
    with pytest.raises(ValueError, match="requires with_indexer"):
        Glm5MonoKernel(object(), 1, indexer_quant=True)


@pytest.mark.parametrize(
    "quant,dtype,width,accepted",
    [
        (True, torch.uint8, 68, True),
        (True, torch.bfloat16, 128, False),
        (True, torch.uint8, 128, False),
        (False, torch.uint8, 68, False),
        (False, torch.bfloat16, 128, True),
    ],
)
def test_forward_checks_cache_format_before_launch(quant, dtype, width, accepted):
    op = object.__new__(Glm5MonoKernel)
    op.launches_per_step, op.S = 1, 1
    op.rope_dtype = "fp32"
    op.kv_cache_layout = KvCacheLayout.ATOM
    op.with_indexer, op.indexer_quant = True, quant
    op.index_max_seq, op.index_page_size = 32, 16
    op.index_request_width, op.index_block_table_stride = 1, 2
    op.identity_block_tables = None
    h = torch.empty((1, HIDDEN), device="meta")
    pos = torch.empty((1,), dtype=torch.int32, device="meta")
    rope = torch.empty((32, 32), device="meta")
    cache = torch.empty((2, 16, width), dtype=dtype, device="meta")
    # A valid cache reaches the next validation, still before allocation/launch.
    with pytest.raises(ValueError, match="block_tables" if accepted else "index_cache"):
        op.forward(h, pos, None, None, None, rope, rope, index_cache=cache)
