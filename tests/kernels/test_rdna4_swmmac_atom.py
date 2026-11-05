# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device correctness tests for the RDNA4 (gfx120x) 2:4 sparse SWMMAC MMA atom.

Covers all 9 ``gfx120x.swmmac`` hardware instructions on wave32:
  * 16x16x32 floating-point (-> f32): f16, bf16, fp8_fp8, fp8_bf8, bf8_fp8, bf8_bf8
    via ``fx.make_tiled_mma`` / ``fx.make_tiled_copy_{B,C}`` / variadic ``fx.gemm``.
  * 16x16x32 iu8 and 16x16x32 / 16x16x64 iu4 (-> i32): signed, unsigned, mixed-sign,
    and saturating clamp via ``fly.mma_atom_call_ssa``.
"""

import sys

import numpy as np
import pytest

import flydsl  # noqa: F401 -- preload comgr before torch/HIP loads LLVM
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.runtime.device import get_rocm_arch
from tests.kernels.int4_pack import pack_int4

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

_ARCH = str(get_rocm_arch() or "")
if not _ARCH.startswith("gfx120"):
    pytest.skip(f"RDNA4 SWMMAC atom requires gfx120x, got {_ARCH}", allow_module_level=True)

WAVE_SIZE = 32
M = N = 16
_VALID_2TO4_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _expand_2to4_and_pack_indices(a_compressed_cpu: torch.Tensor, k: int):
    """Expand compressed (M, K//2) A into dense (M, K) and pack per-lane i32 selectors."""
    assert a_compressed_cpu.shape == (M, k // 2)
    num_groups = k // 4
    groups_per_half = num_groups // 2

    a_dense = torch.zeros((M, k), dtype=a_compressed_cpu.dtype)
    lane_indices = np.zeros((WAVE_SIZE,), dtype=np.uint32)

    for row in range(M):
        for grp in range(num_groups):
            idx0, idx1 = _VALID_2TO4_PAIRS[(row * 5 + grp * 3 + 1) % len(_VALID_2TO4_PAIRS)]
            a_dense[row, 4 * grp + idx0] = a_compressed_cpu[row, 2 * grp + 0]
            a_dense[row, 4 * grp + idx1] = a_compressed_cpu[row, 2 * grp + 1]

            half = grp // groups_per_half
            local_grp = grp % groups_per_half
            nibble = np.uint32((idx0 & 0x3) | ((idx1 & 0x3) << 2))
            lane_indices[16 * half + row] |= nibble << np.uint32(4 * local_grp)

    sparse_idx = torch.from_numpy(lane_indices.view(np.int32)).contiguous()
    return a_dense, sparse_idx


def _compile_fp_swmmac_gemm(elem_cls_a, elem_cls_b):
    """One wave, one 16x16x32 SWMMAC atom via variadic fx.gemm + tiled B/C copies."""
    k = 32
    k_comp = k // 2
    f32 = fx.Float32

    @flyc.kernel
    def swmmac_fp_kernel(A_comp: fx.Tensor, Idx: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        lane16 = tid % 16
        g = tid // 16

        bA = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A_comp)), fx.make_layout((M, k_comp), (k_comp, 1)))
        bIdx = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(Idx)), fx.make_layout((WAVE_SIZE,), (1,)))
        bB = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)), fx.make_layout((N, k), (k, 1)))
        bC = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)), fx.make_layout((M, N), (N, 1)))

        mma_atom = fx.make_mma_atom(fx.rocdl.SWMMAC(M, N, k, elem_cls_a, f32, elem_ty_b=elem_cls_b))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        frag_A = fx.make_rmem_tensor((8, 1, 1), elem_cls_a)
        frag_idx = fx.make_rmem_tensor((1, 1, 1), fx.Int32)
        frag_B = thr_mma.make_fragment_B(bB)
        frag_C = thr_mma.make_fragment_C(bC)

        for v in fx.range_constexpr(8):
            frag_A[(v, 0, 0)] = bA[lane16, 8 * g + v]
        frag_idx[(0, 0, 0)] = bIdx[tid]

        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(elem_cls_b.width), elem_cls_b)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(f32.width), f32)
        thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)

        fx.copy(copy_b, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))
        frag_C.fill(0.0)
        fx.gemm(tiled_mma, frag_C, [frag_A, frag_idx], frag_B, frag_C)
        fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    @flyc.jit
    def launch(A_comp: fx.Tensor, Idx: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        swmmac_fp_kernel(A_comp, Idx, B, C).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize(
    "elem_cls_a, elem_cls_b, torch_dtype_a, torch_dtype_b",
    [
        (fx.BFloat16, fx.BFloat16, torch.bfloat16, torch.bfloat16),
        (fx.Float16, fx.Float16, torch.float16, torch.float16),
        (fx.Float8E4M3FN, fx.Float8E4M3FN, torch.float8_e4m3fn, torch.float8_e4m3fn),
        (fx.Float8E4M3FN, fx.Float8E5M2, torch.float8_e4m3fn, torch.float8_e5m2),
        (fx.Float8E5M2, fx.Float8E4M3FN, torch.float8_e5m2, torch.float8_e4m3fn),
        (fx.Float8E5M2, fx.Float8E5M2, torch.float8_e5m2, torch.float8_e5m2),
    ],
    ids=["bf16", "f16", "fp8_fp8", "fp8_bf8", "bf8_fp8", "bf8_bf8"],
)
def test_rdna4_swmmac_fp_atom(elem_cls_a, elem_cls_b, torch_dtype_a, torch_dtype_b):
    torch.manual_seed(1201)
    k = 32
    a_comp_cpu = (torch.randn(M, k // 2) * 3).round().to(torch_dtype_a)
    b_cpu = (torch.randn(N, k) * 3).round().to(torch_dtype_b)
    a_dense_cpu, sparse_idx_cpu = _expand_2to4_and_pack_indices(a_comp_cpu, k)

    a_comp = a_comp_cpu.contiguous().to("cuda")
    sparse_idx = sparse_idx_cpu.to("cuda")
    b = b_cpu.contiguous().to("cuda")
    c = torch.full((M, N), -1.0, dtype=torch.float32, device="cuda")

    launch = _compile_fp_swmmac_gemm(elem_cls_a, elem_cls_b)
    launch(a_comp, sparse_idx, b, c, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = a_dense_cpu.float() @ b_cpu.float().T
    torch.testing.assert_close(c.cpu(), ref, atol=0, rtol=0)


def _compile_integer_swmmac(elem_cls, k: int, *, sign_a: bool, sign_b: bool, clamp: bool = False):
    k_comp = k // 2
    packed_a_words = k_comp // 8
    packed_b_words = k // 8

    @flyc.kernel
    def swmmac_int_kernel(A_comp: fx.Tensor, Idx: fx.Tensor, B: fx.Tensor, C_init: fx.Tensor, D: fx.Tensor):
        lane = fx.thread_idx.x
        lane16 = lane % 16
        g = lane // 16
        idx_view = fx.make_view(fx.get_iter(Idx), fx.make_layout((WAVE_SIZE,), (1,)))
        c2d = fx.make_view(fx.get_iter(C_init), fx.make_layout((M, N), (N, 1)))
        d2d = fx.make_view(fx.get_iter(D), fx.make_layout((M, N), (N, 1)))

        if fx.const_expr(elem_cls is fx.Int4):
            a2d = fx.make_view(fx.get_iter(A_comp), fx.make_layout((M, packed_a_words), (packed_a_words, 1)))
            b2d = fx.make_view(fx.get_iter(B), fx.make_layout((N, packed_b_words), (packed_b_words, 1)))
            if fx.const_expr(k == 32):
                a_vec = fx.Vector.from_elements([a2d[lane16, g]], fx.Int32).bitcast(fx.Int4)
                b_vec = fx.Vector.from_elements(
                    [b2d[lane16, 2 * g + w] for w in fx.range_constexpr(2)], fx.Int32
                ).bitcast(fx.Int4)
            else:
                a_vec = fx.Vector.from_elements(
                    [a2d[lane16, 2 * g + w] for w in fx.range_constexpr(2)], fx.Int32
                ).bitcast(fx.Int4)
                b_vec = fx.Vector.from_elements(
                    [b2d[lane16, 4 * g + w] for w in fx.range_constexpr(4)], fx.Int32
                ).bitcast(fx.Int4)
        else:
            a2d = fx.make_view(fx.get_iter(A_comp), fx.make_layout((M, k_comp), (k_comp, 1)))
            b2d = fx.make_view(fx.get_iter(B), fx.make_layout((N, k), (k, 1)))
            a_vec = fx.Vector.from_elements(
                [a2d[lane16, 8 * g + kk].to(elem_cls) for kk in fx.range_constexpr(8)], elem_cls
            )
            b_vec = fx.Vector.from_elements(
                [b2d[lane16, 16 * g + kk].to(elem_cls) for kk in fx.range_constexpr(16)], elem_cls
            )

        acc = fx.Vector.from_elements([c2d[8 * g + v, lane16] for v in fx.range_constexpr(8)], fx.Int32)
        idx_val = idx_view[lane]
        mma_atom = fx.make_mma_atom(
            fx.rocdl.SWMMAC(M, N, k, elem_cls, fx.Int32, sign_a=sign_a, sign_b=sign_b, clamp=clamp)
        )
        result = fx.Vector(
            fly.mma_atom_call_ssa(
                [fx.Vector.make_type(8, fx.Int32)],
                mma_atom,
                [a_vec.ir_value(), idx_val.ir_value()],
                b_vec.ir_value(),
                acc.ir_value(),
            )
        )
        for v in fx.range_constexpr(8):
            d2d[8 * g + v, lane16] = result[v]

    @flyc.jit
    def launch(
        A_comp: fx.Tensor,
        Idx: fx.Tensor,
        B: fx.Tensor,
        C_init: fx.Tensor,
        D: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        swmmac_int_kernel(A_comp, Idx, B, C_init, D).launch(grid=(1, 1, 1), block=(WAVE_SIZE, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize(
    "elem_cls, dtype_a, low_a, high_a, dtype_b, low_b, high_b, sign_a, sign_b, k",
    [
        pytest.param(fx.Int8, torch.int8, -128, 128, torch.int8, -128, 128, True, True, 32),
        pytest.param(fx.Uint8, torch.uint8, 0, 256, torch.uint8, 0, 256, False, False, 32),
        pytest.param(fx.Int8, torch.int8, -128, 128, torch.uint8, 0, 256, True, False, 32),
        pytest.param(fx.Int4, torch.int8, -8, 8, torch.int8, -8, 8, True, True, 32),
        pytest.param(fx.Int4, torch.uint8, 0, 16, torch.uint8, 0, 16, False, False, 32),
        pytest.param(fx.Int4, torch.int8, -8, 8, torch.int8, -8, 8, True, True, 64),
        pytest.param(fx.Int4, torch.uint8, 0, 16, torch.uint8, 0, 16, False, False, 64),
    ],
    ids=[
        "iu8-signed-k32",
        "iu8-unsigned-k32",
        "iu8-mixed-sign-k32",
        "iu4-signed-k32",
        "iu4-unsigned-k32",
        "iu4-signed-k64",
        "iu4-unsigned-k64",
    ],
)
def test_rdna4_swmmac_integer_atom(elem_cls, dtype_a, low_a, high_a, dtype_b, low_b, high_b, sign_a, sign_b, k):
    torch.manual_seed(2026 + k)
    a_comp_logical = torch.randint(low_a, high_a, (M, k // 2), dtype=dtype_a)
    b_logical = torch.randint(low_b, high_b, (N, k), dtype=dtype_b)
    a_comp_logical[0].fill_(low_a)
    a_comp_logical[1].fill_(high_a - 1)
    b_logical[0].fill_(high_b - 1)
    b_logical[1].fill_(low_b)

    a_dense_logical, sparse_idx_cpu = _expand_2to4_and_pack_indices(a_comp_logical, k)
    if elem_cls is fx.Int4:
        a_dev = torch.from_numpy(pack_int4(a_comp_logical.numpy())).contiguous().to("cuda")
        b_dev = torch.from_numpy(pack_int4(b_logical.numpy())).contiguous().to("cuda")
    else:
        a_dev = (
            a_comp_logical.view(torch.int8).contiguous().to("cuda")
            if elem_cls is fx.Int8
            else a_comp_logical.contiguous().to("cuda")
        )
        b_dev = (
            b_logical.view(torch.int8).contiguous().to("cuda")
            if elem_cls is fx.Int8
            else b_logical.contiguous().to("cuda")
        )

    sparse_idx = sparse_idx_cpu.to("cuda")
    c_init = torch.randint(-100, 100, (M, N), dtype=torch.int32, device="cuda")
    d = torch.full((M, N), -1, dtype=torch.int32, device="cuda")

    launch = _compile_integer_swmmac(elem_cls, k, sign_a=sign_a, sign_b=sign_b)
    launch(a_dev, sparse_idx, b_dev, c_init, d, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    ref = a_dense_logical.to(torch.int64) @ b_logical.to(torch.int64).T + c_init.cpu().to(torch.int64)
    torch.testing.assert_close(d.cpu(), ref.to(torch.int32), atol=0, rtol=0)


def test_rdna4_swmmac_integer_clamp_saturation():
    torch.manual_seed(77)
    k = 64
    a_comp_logical = torch.full((M, k // 2), 7, dtype=torch.int8)
    b_logical = torch.full((N, k), 7, dtype=torch.int8)
    _, sparse_idx_cpu = _expand_2to4_and_pack_indices(a_comp_logical, k)

    a_dev = torch.from_numpy(pack_int4(a_comp_logical.numpy())).contiguous().to("cuda")
    b_dev = torch.from_numpy(pack_int4(b_logical.numpy())).contiguous().to("cuda")
    sparse_idx = sparse_idx_cpu.to("cuda")
    i32_max = (1 << 31) - 1
    c_init = torch.full((M, N), i32_max - 10, dtype=torch.int32, device="cuda")
    d = torch.zeros((M, N), dtype=torch.int32, device="cuda")

    launch = _compile_integer_swmmac(fx.Int4, k, sign_a=True, sign_b=True, clamp=True)
    launch(a_dev, sparse_idx, b_dev, c_init, d, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert torch.all(d.cpu() == i32_max)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
