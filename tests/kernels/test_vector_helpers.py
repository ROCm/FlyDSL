# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Check the bit patterns produced by shared Vector packing and store helpers."""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.attention.qk_norm_rope_quant import _store_bf16_vec_g, _store_fp8_packed
from kernels.common import buffer_ops
from kernels.common.tensor_shim import GTensor
from kernels.moe import moe_common

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@flyc.kernel
def pack_kernel(src: fx.Tensor, dst: fx.Tensor, name: fx.Constexpr, arity: fx.Constexpr):
    ptr = fx.get_iter(src)
    values = [fx.ptr_load(ptr + i) for i in range(arity)]
    fn = getattr(moe_common, name)
    result = fn(*values).bitcast(fx.Int32)
    fx.ptr_store(result, fx.get_iter(dst))


@flyc.jit
def pack_launch(src: fx.Tensor, dst: fx.Tensor, name: fx.Constexpr, arity: fx.Constexpr, stream: fx.Stream):
    pack_kernel(src, dst, name, arity).launch(grid=(1,), block=(1,), stream=stream)


@flyc.kernel
def store_kernel(src: fx.Tensor, dst: fx.Tensor, quant: fx.Constexpr, pointer_offset: fx.Constexpr):
    ptr = fx.get_iter(src)
    values = [fx.ptr_load(ptr + i) for i in range(8)]
    if fx.const_expr(quant):
        rsrc = buffer_ops.create_buffer_resource(dst, max_size=True)
        _store_fp8_packed(values, rsrc, fx.Int32(0), fx.Int32(0), 8)
    else:
        if fx.const_expr(pointer_offset):
            out = GTensor(fx.get_iter(dst), T.bf16, (8,), static_bytes_offset_i64=fx.Int64(pointer_offset))
        else:
            out = GTensor(dst, T.bf16, (8,))
        _store_bf16_vec_g(values, out, fx.Int32(0), fx.Int32(0), 8)


@flyc.jit
def store_launch(src: fx.Tensor, dst: fx.Tensor, quant: fx.Constexpr, pointer_offset: fx.Constexpr, stream: fx.Stream):
    store_kernel(src, dst, quant, pointer_offset).launch(grid=(1,), block=(1,), stream=stream)


@pytest.mark.parametrize(
    "name,arity",
    [
        ("i64_to_v4f16", 1),
        ("i64_to_v4i16", 1),
        ("i64x2_to_v8f16", 2),
        ("i64x2_to_v8bf16", 2),
        ("i64x4_to_i32x8", 4),
    ],
)
def test_moe_pack_preserves_all_bits(name, arity):
    # Include signed high bits, NaN half bit patterns, all ones and zero.
    values = [-9222949828684546049, -1, 0, 0x7FFF012380005555]
    source = torch.tensor(values[:arity], device="cuda", dtype=torch.int64)
    output = torch.empty(arity * 2, device="cuda", dtype=torch.int32)
    pack_launch(source, output, name, arity, torch.cuda.current_stream())
    torch.testing.assert_close(output, source.view(torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize(
    "quant,pointer_offset", [(False, 0), (True, 0), (False, 8)], ids=["bf16", "fp8", "pointer-offset"]
)
def test_qk_store_packs_expected_lanes(quant, pointer_offset):
    if quant and get_rocm_arch() != "gfx950":
        pytest.skip("This FP8 reference uses the gfx950 e4m3fn encoding")
    source = torch.tensor([-3.0, -1.5, -0.5, 0, 0.5, 1.5, 3, 12], device="cuda", dtype=torch.float32)
    dtype = torch.float8_e4m3fn if quant else torch.bfloat16
    start = pointer_offset // dtype.itemsize
    output = torch.zeros(8 + 2 * start, device="cuda", dtype=dtype)
    store_launch(source, output, quant, pointer_offset, torch.cuda.current_stream())
    expected = torch.zeros_like(output)
    expected[start : start + 8] = source.to(dtype)
    torch.testing.assert_close(output.float(), expected.float(), rtol=0, atol=0)
