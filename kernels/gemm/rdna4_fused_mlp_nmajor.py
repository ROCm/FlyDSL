# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""N-major float GEMM building block + SwiGLU MLP host (gfx120x).

RDNA4 WMMA D is M-major in VGPRs and is a poor next-GEMM A without an LDS
transpose. Swapping A/B on the first WMMA lands D N-major so the next MMA can
consume it in-register. See GPUOpen WMMA guide part 1 (RDNA 4):
https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/

What this module provides
-------------------------
* ``build_wmma_tile_module(swap_ab=...)`` -- single-wave 16x16x16 bf16/fp16
  WMMA (Sequence ABI, matching upstream ``scaled_mm``). ``swap_ab=False`` matches
  ``A @ B.T``. ``swap_ab=True`` + standard Wave32 store yields the transpose on
  square tiles (layout probe for the A/B swap).
* ``fused_gemm_tn`` / ``build_fused_gemm_tn_module`` -- zero-LDS two-GEMM fuse:
  first WMMA swapped (N-major D0 in VGPR) -> narrow f32->bf16/fp16 -> second
  WMMA **unswapped** consumes D0 as A -> standard M-major store. Empirically
  ``max_err=0`` vs ``(A0@B0.T)@B1.T`` on gfx120x. GPUOpen's sample swaps both
  builtins; this path needs swap on GEMM0 only under the FlyDSL Sequence ABI.
* ``gemm_bf16_nmajor`` -- host-tiled ``A @ B.T`` via the unswapped zero-LDS
  16x16 tile (small-shape building block). Not a silent production layout change.
* ``gemm_bf16_nmajor_lds`` / ``build_gemm_nmajor_lds_module`` -- multi-wave
  LDS-pipelined production ``A @ B.T`` (wraps ``rdna_f16_gemm`` double-buffered
  LDS WMMA). Pads M/N/K to the picked block tile; requires ``K_pad >= 2*BK``
  for the prefetch pipeline. bf16/fp16 in; bf16/fp16/f32 out.
* ``fused_swiglu_mlp_inreg`` / ``build_fused_swiglu_mlp_module`` -- in-register
  SwiGLU fuse for one-wave 16x16 panels. Host tiles along M/N when
  ``K == FFN == 16`` and ``M, N_out`` are multiples of 16. Per tile: GEMM0_gate
  + GEMM0_up (A/B-swapped) -> SiLU(gate)*up in registers -> narrow -> GEMM1
  unswapped -> M-major store. Layout: ``silu(x @ Wgate.T) * (x @ Wup.T)`` then
  ``@ Wdown.T`` with weights ``[N, K]``.
* ``fused_swiglu_mlp_nmajor`` -- SwiGLU MLP host. ``K == FFN == 16`` with
  M/N_out multiples of 16 uses ``fused_swiglu_mlp_inreg``. Other multiples of
  16 use ``fused_swiglu_mlp_lds`` (in-kernel K/FFN loop, mid staged in LDS).
  Shapes that are not multiples of 16 still run two GEMMs with the mid in GMEM.

Known limits
------------
* Multi-wave LDS GEMM is a single-GEMM production block (M-major D store), not
  an in-kernel A/B-swap N-major fuse across two GEMMs. The zero-LDS
  ``fused_gemm_tn`` / ``fused_swiglu_mlp_inreg`` paths keep the VGPR N-major fuse.
* In-kernel fused SwiGLU with a K/FFN loop keeps the 16-wide mid in LDS
  (``fused_swiglu_mlp_lds``) when K, FFN, M, and N_out are multiples of 16.
  The fragment is spilled per lane (32×8), reloaded, then consumed as A of
  GEMM1. It is not a multi-wave LDS GEMM. Shapes that are not multiples of
  16 still use separate GEMMs and a GMEM mid.
* Host-tiled in-reg covers ``K == FFN == 16`` with ``M, N_out`` multiples of 16
  (one fused 16x16 launch per (M,N) tile; N>16 recomputes GEMM0+SiLU per N tile).
* ``rdna_f16_gemm`` LDS path needs ``K_pad >= 2 * BLOCK_K`` (prefetch); hosts
  zero-pad K when the logical K is shallower.
* Production ``scaled_mm_fp8`` / ``int8_linear`` defaults are untouched.
* vs HIP baseline: honest miss (no fused-MLP HIP baseline). Microbench vs
  unfused FlyDSL gemm+SiLU is informational only.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import PointerJitArg
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.tensor_shim import _run_compiled

WM = WN = WK = 16
WAVE = 32
_TILE = 16


def _ptr(t: torch.Tensor) -> PointerJitArg:
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _zeros8() -> list[fx.Float32]:
    return [
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
        fx.Float32(0.0),
    ]


@lru_cache(maxsize=8)
def build_wmma_tile_module(dtype_name: str = "bfloat16", *, swap_ab: bool = False) -> Callable[..., None]:
    """One wave, one 16×16×16 WMMA tile. ``swap_ab`` selects GPUOpen A/B swap."""
    if dtype_name not in ("bfloat16", "float16"):
        raise ValueError(f"supports bf16/fp16, got {dtype_name}")
    Elem = fx.BFloat16 if dtype_name == "bfloat16" else fx.Float16
    M = N = K = _TILE
    do_swap = bool(swap_ab)

    if do_swap:

        @flyc.kernel
        def wmma_tile_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor) -> None:
            tid = fx.thread_idx.x
            bA = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(A)),
                fx.make_layout((M, K), (K, 1)),
            )
            bB = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(B)),
                fx.make_layout((N, K), (K, 1)),
            )
            bC = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(C)),
                fx.make_layout((M, N), (N, 1)),
            )
            mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, Elem, fx.Float32))
            tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
            thr_mma = tiled_mma.thr_slice(tid)
            frag_A = thr_mma.make_fragment_A(bA)
            frag_B = thr_mma.make_fragment_B(bB)
            frag_C = thr_mma.make_fragment_C(bC)
            copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
            copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
            copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
            thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
            thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
            thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
            fx.copy(copy_a, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
            fx.copy(copy_b, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))
            a_rm = fx.make_rmem_tensor(8, Elem)
            b_rm = fx.make_rmem_tensor(8, Elem)
            c_rm = fx.make_rmem_tensor(8, fx.Float32)
            a_rm.store(Vec(frag_A.load()))
            b_rm.store(Vec(frag_B.load()))
            c_rm.store(Vec.from_elements(_zeros8(), fx.Float32))
            # GPUOpen: wmma(B, A, C) → D N-major; std M-major store → transpose.
            fx.gemm(mma_atom, c_rm, [b_rm], [a_rm], c_rm)
            frag_C.store(Vec(c_rm.load()))
            fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    else:

        @flyc.kernel
        def wmma_tile_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor) -> None:
            tid = fx.thread_idx.x
            bA = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(A)),
                fx.make_layout((M, K), (K, 1)),
            )
            bB = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(B)),
                fx.make_layout((N, K), (K, 1)),
            )
            bC = fx.make_view(
                fx.get_iter(fx.rocdl.make_buffer_tensor(C)),
                fx.make_layout((M, N), (N, 1)),
            )
            mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, Elem, fx.Float32))
            tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
            thr_mma = tiled_mma.thr_slice(tid)
            frag_A = thr_mma.make_fragment_A(bA)
            frag_B = thr_mma.make_fragment_B(bB)
            frag_C = thr_mma.make_fragment_C(bC)
            copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
            copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
            copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
            thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
            thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
            thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
            fx.copy(copy_a, thr_copy_A.partition_S(bA), thr_copy_A.retile(frag_A))
            fx.copy(copy_b, thr_copy_B.partition_S(bB), thr_copy_B.retile(frag_B))
            a_rm = fx.make_rmem_tensor(8, Elem)
            b_rm = fx.make_rmem_tensor(8, Elem)
            c_rm = fx.make_rmem_tensor(8, fx.Float32)
            a_rm.store(Vec(frag_A.load()))
            b_rm.store(Vec(frag_B.load()))
            c_rm.store(Vec.from_elements(_zeros8(), fx.Float32))
            fx.gemm(mma_atom, c_rm, [a_rm], [b_rm], c_rm)
            frag_C.store(Vec(c_rm.load()))
            fx.copy(copy_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_S(bC))

    wmma_tile_kernel.__name__ = f"wmma_tile_{dtype_name}_swap{int(do_swap)}"

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        wmma_tile_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)

    launch.__name__ = f"launch_wmma_tile_{dtype_name}_swap{int(do_swap)}"
    return launch


@lru_cache(maxsize=4)
def build_fused_gemm_tn_module(dtype_name: str = "bfloat16") -> Callable[..., None]:
    """Zero-LDS fused ``C1 = (A0 @ B0.T) @ B1.T`` for 16×16×16 panels.

    GEMM0: A/B swapped → D0 N-major in VGPR. Narrow to act dtype. GEMM1:
    unswapped, D0 as A. Standard M-major store. No LDS transpose of D0.
    """
    if dtype_name not in ("bfloat16", "float16"):
        raise ValueError(f"supports bf16/fp16, got {dtype_name}")
    Elem = fx.BFloat16 if dtype_name == "bfloat16" else fx.Float16
    M = N = K = _TILE

    @flyc.kernel
    def fused_gemm_tn_kernel(A0: fx.Tensor, B0: fx.Tensor, B1: fx.Tensor, C1: fx.Tensor) -> None:
        tid = fx.thread_idx.x
        bA0 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(A0)),
            fx.make_layout((M, K), (K, 1)),
        )
        bB0 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(B0)),
            fx.make_layout((N, K), (K, 1)),
        )
        bB1 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(B1)),
            fx.make_layout((N, K), (K, 1)),
        )
        bC1 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(C1)),
            fx.make_layout((M, N), (N, 1)),
        )
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, Elem, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        frag_A0 = thr_mma.make_fragment_A(bA0)
        frag_B0 = thr_mma.make_fragment_B(bB0)
        frag_B1 = thr_mma.make_fragment_B(bB1)
        frag_C1 = thr_mma.make_fragment_C(bC1)
        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
        thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
        fx.copy(copy_a, thr_copy_A.partition_S(bA0), thr_copy_A.retile(frag_A0))
        fx.copy(copy_b, thr_copy_B.partition_S(bB0), thr_copy_B.retile(frag_B0))
        fx.copy(copy_b, thr_copy_B.partition_S(bB1), thr_copy_B.retile(frag_B1))

        a0 = fx.make_rmem_tensor(8, Elem)
        b0 = fx.make_rmem_tensor(8, Elem)
        c0 = fx.make_rmem_tensor(8, fx.Float32)
        a0.store(Vec(frag_A0.load()))
        b0.store(Vec(frag_B0.load()))
        c0.store(Vec.from_elements(_zeros8(), fx.Float32))
        # GEMM0 swapped → N-major D0 (no LDS transpose).
        fx.gemm(mma_atom, c0, [b0], [a0], c0)

        a1 = fx.make_rmem_tensor(8, Elem)
        c0v = Vec(c0.load())
        narrow = []
        for i in range_constexpr(8):
            narrow.append(fx.Float32(c0v[i]).to(Elem))
        a1.store(Vec.from_elements(narrow, Elem))

        b1 = fx.make_rmem_tensor(8, Elem)
        b1.store(Vec(frag_B1.load()))
        c1 = fx.make_rmem_tensor(8, fx.Float32)
        c1.store(Vec.from_elements(_zeros8(), fx.Float32))
        # GEMM1 unswapped: N-major D0 is already legal as A for Sequence ABI.
        fx.gemm(mma_atom, c1, [a1], [b1], c1)

        frag_C1.store(Vec(c1.load()))
        fx.copy(copy_c, thr_copy_C.retile(frag_C1), thr_copy_C.partition_S(bC1))

    fused_gemm_tn_kernel.__name__ = f"fused_gemm_tn_{dtype_name}"

    @flyc.jit
    def launch(
        A0: fx.Tensor,
        B0: fx.Tensor,
        B1: fx.Tensor,
        C1: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        fused_gemm_tn_kernel(A0, B0, B1, C1).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)

    launch.__name__ = f"launch_fused_gemm_tn_{dtype_name}"
    return launch


@lru_cache(maxsize=4)
def build_fused_swiglu_mlp_module(dtype_name: str = "bfloat16") -> Callable[..., None]:
    """Zero-LDS fused SwiGLU MLP for 16×16 panels (in-register SiLU×mul).

    ``Y = (silu(A0 @ Bg.T) * (A0 @ Bu.T)) @ Bd.T`` with no LDS/GMEM spill of
    the mid activation. GEMM0_gate and GEMM0_up swap A/B (N-major D in VGPR);
    Sequence ABI keeps GEMM1 **unswapped** (mid already legal as A).
    """
    if dtype_name not in ("bfloat16", "float16"):
        raise ValueError(f"supports bf16/fp16, got {dtype_name}")
    Elem = fx.BFloat16 if dtype_name == "bfloat16" else fx.Float16
    M = N = K = _TILE

    @flyc.kernel
    def fused_swiglu_mlp_kernel(
        A0: fx.Tensor,
        Bg: fx.Tensor,
        Bu: fx.Tensor,
        Bd: fx.Tensor,
        C1: fx.Tensor,
    ) -> None:
        tid = fx.thread_idx.x
        bA0 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(A0)),
            fx.make_layout((M, K), (K, 1)),
        )
        bBg = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(Bg)),
            fx.make_layout((N, K), (K, 1)),
        )
        bBu = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(Bu)),
            fx.make_layout((N, K), (K, 1)),
        )
        bBd = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(Bd)),
            fx.make_layout((N, K), (K, 1)),
        )
        bC1 = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(C1)),
            fx.make_layout((M, N), (N, 1)),
        )
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(M, N, K, Elem, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        frag_A0 = thr_mma.make_fragment_A(bA0)
        frag_Bg = thr_mma.make_fragment_B(bBg)
        frag_Bu = thr_mma.make_fragment_B(bBu)
        frag_Bd = thr_mma.make_fragment_B(bBd)
        frag_C1 = thr_mma.make_fragment_C(bC1)
        copy_a = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
        copy_b = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
        thr_copy_A = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)
        fx.copy(copy_a, thr_copy_A.partition_S(bA0), thr_copy_A.retile(frag_A0))
        fx.copy(copy_b, thr_copy_B.partition_S(bBg), thr_copy_B.retile(frag_Bg))
        fx.copy(copy_b, thr_copy_B.partition_S(bBu), thr_copy_B.retile(frag_Bu))
        fx.copy(copy_b, thr_copy_B.partition_S(bBd), thr_copy_B.retile(frag_Bd))

        a0 = fx.make_rmem_tensor(8, Elem)
        bg = fx.make_rmem_tensor(8, Elem)
        bu = fx.make_rmem_tensor(8, Elem)
        cg = fx.make_rmem_tensor(8, fx.Float32)
        cu = fx.make_rmem_tensor(8, fx.Float32)
        a0.store(Vec(frag_A0.load()))
        bg.store(Vec(frag_Bg.load()))
        bu.store(Vec(frag_Bu.load()))
        cg.store(Vec.from_elements(_zeros8(), fx.Float32))
        cu.store(Vec.from_elements(_zeros8(), fx.Float32))
        # GEMM0 gate + up: swapped → N-major D in VGPR (no LDS of mid).
        fx.gemm(mma_atom, cg, [bg], [a0], cg)
        fx.gemm(mma_atom, cu, [bu], [a0], cu)

        # In-register SiLU(gate)*up (same formula as rdna4_swiglu).
        one = fx.Float32(1.0)
        neg_log2e = fx.Float32(-1.4426950408889634)
        cgv = Vec(cg.load())
        cuv = Vec(cu.load())
        mid = []
        for i in range_constexpr(8):
            g = fx.Float32(cgv[i])
            u = fx.Float32(cuv[i])
            sigv = one / (one + fmath.exp2(g * neg_log2e))
            y = g * sigv * u
            mid.append(y.to(Elem))
        a1 = fx.make_rmem_tensor(8, Elem)
        a1.store(Vec.from_elements(mid, Elem))

        bd = fx.make_rmem_tensor(8, Elem)
        bd.store(Vec(frag_Bd.load()))
        c1 = fx.make_rmem_tensor(8, fx.Float32)
        c1.store(Vec.from_elements(_zeros8(), fx.Float32))
        # GEMM1 unswapped: N-major mid is already legal as A for Sequence ABI.
        fx.gemm(mma_atom, c1, [a1], [bd], c1)

        frag_C1.store(Vec(c1.load()))
        fx.copy(copy_c, thr_copy_C.retile(frag_C1), thr_copy_C.partition_S(bC1))

    fused_swiglu_mlp_kernel.__name__ = f"fused_swiglu_mlp_{dtype_name}"

    @flyc.jit
    def launch(
        A0: fx.Tensor,
        Bg: fx.Tensor,
        Bu: fx.Tensor,
        Bd: fx.Tensor,
        C1: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        fused_swiglu_mlp_kernel(A0, Bg, Bu, Bd, C1).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)

    launch.__name__ = f"launch_fused_swiglu_mlp_{dtype_name}"
    return launch


def _dtype_name(dt: torch.dtype) -> str:
    if dt == torch.bfloat16:
        return "bfloat16"
    if dt == torch.float16:
        return "float16"
    raise ValueError(f"expected bf16/fp16, got {dt}")


# Multi-wave LDS tile ladder: (reg_m, reg_n, reg_k, waves_m, waves_n) → BM×BN×BK.
# Prefers fat blocks on fat shapes; 16×16×32 covers panels after K pad to 2×BK.
_LDS_TILE_CANDIDATES = (
    (4, 4, 2, 2, 2),  # 128×128×32
    (4, 2, 2, 2, 2),  # 128×64×32
    (2, 4, 2, 2, 2),  # 64×128×32
    (2, 2, 2, 2, 2),  # 64×64×32
    (2, 2, 4, 2, 2),  # 64×64×64
    (2, 2, 2, 2, 1),  # 64×32×32
    (2, 2, 2, 1, 2),  # 32×64×32
    (2, 2, 2, 1, 1),  # 32×32×32
    (2, 1, 2, 1, 1),  # 32×16×32
    (1, 2, 2, 1, 1),  # 16×32×32
    (1, 1, 2, 1, 1),  # 16×16×32
)


def _lds_block_shape(tile: tuple[int, int, int, int, int]) -> tuple[int, int, int]:
    reg_m, reg_n, reg_k, waves_m, waves_n = tile
    return (
        WM * reg_m * waves_m,
        WN * reg_n * waves_n,
        WK * reg_k,
    )


def _lds_tile_geometry_ok(tile: tuple[int, int, int, int, int]) -> bool:
    """Mirror ``rdna_f16_gemm.create_wmma_gemm_module`` G2S thread constraints."""
    reg_m, reg_n, reg_k, waves_m, waves_n = tile
    if reg_k < 2 or reg_k % 2 != 0:
        return False
    bm, bn, bk = _lds_block_shape(tile)
    threads = waves_m * waves_n * WAVE
    load_vec = 8  # 128-bit / 16-bit elem
    if bk % load_vec != 0:
        return False
    thrs_k = bk // load_vec
    if thrs_k == 0 or threads % thrs_k != 0:
        return False
    thrs_m = threads // thrs_k
    return bm % thrs_m == 0 and bn % thrs_m == 0


def pick_nmajor_lds_tile(M: int, N: int, K: int) -> tuple[tuple[int, int, int, int, int], int, int, int, int, int, int]:
    """Pick a multi-wave LDS block tile and padded ``(Mp, Np, Kp)``.

    ``rdna_f16_gemm`` needs ``Kp >= 2 * BK`` for the prefetch pipeline and
    compile-time multiples of ``BM/BN/BK``. Returns
    ``(tile, Mp, Np, Kp, BM, BN, BK)``.
    """
    best = None
    best_key = None
    for tile in _LDS_TILE_CANDIDATES:
        if not _lds_tile_geometry_ok(tile):
            continue
        bm, bn, bk = _lds_block_shape(tile)
        mp = (M + bm - 1) // bm * bm
        np_ = (N + bn - 1) // bn * bn
        kp = (K + bk - 1) // bk * bk
        if kp < 2 * bk:
            kp = 2 * bk
        pad_vol = mp * np_ * kp - M * N * max(K, 1)
        # Prefer tiles the logical shape can fill; then larger BM×BN; then less pad.
        fills = int(M >= bm and N >= bn)
        key = (-fills, -(bm * bn), pad_vol, bm * bn * bk)
        if best is None or key < best_key:
            best = (tile, mp, np_, kp, bm, bn, bk)
            best_key = key
    if best is None:
        raise RuntimeError("no feasible LDS tile (internal)")
    return best


@lru_cache(maxsize=64)
def build_gemm_nmajor_lds_module(
    M: int,
    N: int,
    K: int,
    dtype_name: str = "bfloat16",
    out_name: str = "bfloat16",
    tile: tuple = (1, 1, 2, 1, 1),
) -> Callable[..., None]:
    """Compile multi-wave LDS-pipelined ``C = A @ B.T`` for fixed padded MNK.

    Wraps ``kernels.gemm.rdna_f16_gemm.create_wmma_gemm_module`` (double-buffered
    LDS, multi-wave WMMA). ``tile`` is ``(reg_m, reg_n, reg_k, waves_m, waves_n)``.
    """
    if dtype_name not in ("bfloat16", "float16"):
        raise ValueError(f"supports bf16/fp16, got {dtype_name}")
    if out_name not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"out supports bf16/fp16/f32, got {out_name}")
    from kernels.gemm.rdna_f16_gemm import create_wmma_gemm_module

    in_dtype = "bf16" if dtype_name == "bfloat16" else "f16"
    out_dtype = {"bfloat16": "bf16", "float16": "f16", "float32": "f32"}[out_name]
    reg_m, reg_n, reg_k, waves_m, waves_n = tile
    launch, bm, bn, bk = create_wmma_gemm_module(
        M,
        N,
        K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        reg_m=reg_m,
        reg_n=reg_n,
        reg_k=reg_k,
        waves_m=waves_m,
        waves_n=waves_n,
    )
    return launch, bm, bn, bk


def gemm_bf16_nmajor_lds(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K].T via multi-wave LDS-pipelined WMMA (gfx120x).

    Production building block for larger K. Pads M/N/K to the picked block tile
    (and to ``K_pad >= 2 * BLOCK_K``). For tiny panels prefer ``gemm_bf16_nmajor``
    (zero-LDS host-tiled 16×16) or ``fused_gemm_tn`` / ``fused_swiglu_mlp_inreg``.
    """
    require_gfx120x(a.device, what="gemm_bf16_nmajor_lds (gfx120x)")
    if a.ndim != 2 or b_nk.ndim != 2:
        raise ValueError("expects A[M,K], B[N,K]")
    if a.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"A must be bf16/fp16, got {a.dtype}")
    if b_nk.dtype != a.dtype:
        raise ValueError(f"B dtype {b_nk.dtype} must match A {a.dtype}")
    m, k = a.shape
    n, k2 = b_nk.shape
    if k != k2:
        raise ValueError(f"K mismatch {k} vs {k2}")
    if out_dtype is None:
        out_dtype = a.dtype

    tile, mp, np_, kp, _bm, _bn, _bk = pick_nmajor_lds_tile(m, n, k)
    if mp != m or np_ != n or kp != k:
        a_pad = torch.zeros((mp, kp), device=a.device, dtype=a.dtype)
        b_pad = torch.zeros((np_, kp), device=a.device, dtype=a.dtype)
        a_pad[:m, :k] = a
        b_pad[:n, :k] = b_nk
    else:
        a_pad, b_pad = a.contiguous(), b_nk.contiguous()

    out_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(out_dtype)
    if out_name is None:
        raise ValueError(f"out_dtype must be bf16/fp16/f32, got {out_dtype}")

    launch, _, _, _ = build_gemm_nmajor_lds_module(mp, np_, kp, _dtype_name(a.dtype), out_name, tile)
    c_full = torch.zeros((mp, np_), device=a.device, dtype=out_dtype)
    # Call JitFunction directly (not ``_run_compiled``): launch_gemm has a
    # default ``sr_seed`` and ``_run_compiled``'s cached ``_cf`` mishandles
    # the optional arg on the second invoke.
    launch(
        c_full,
        a_pad,
        b_pad,
        torch.cuda.current_stream(device=a.device),
    )
    result = c_full[:m, :n]
    if out is not None:
        out.copy_(result)
        return out
    return result.contiguous()


def _run_tile(
    a_tile: torch.Tensor,
    b_tile: torch.Tensor,
    *,
    swap_ab: bool = False,
) -> torch.Tensor:
    out = torch.zeros((_TILE, _TILE), device=a_tile.device, dtype=torch.float32)
    launch = build_wmma_tile_module(_dtype_name(a_tile.dtype), swap_ab=swap_ab)
    _run_compiled(
        launch,
        a_tile.contiguous(),
        b_tile.contiguous(),
        out,
        torch.cuda.current_stream(device=a_tile.device),
    )
    return out


def fused_gemm_tn(
    a0: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """``C1 = (A0 @ B0.T) @ B1.T`` via zero-LDS fused WMMA (16×16 panels only)."""
    require_gfx120x(a0.device, what="fused_gemm_tn (gfx120x)")
    for t, name in ((a0, "a0"), (b0, "b0"), (b1, "b1")):
        if t.shape != (_TILE, _TILE):
            raise ValueError(f"{name} must be [16,16], got {tuple(t.shape)}")
    if out_dtype is None:
        out_dtype = a0.dtype
    out = torch.zeros((_TILE, _TILE), device=a0.device, dtype=torch.float32)
    launch = build_fused_gemm_tn_module(_dtype_name(a0.dtype))
    _run_compiled(
        launch,
        a0.contiguous(),
        b0.contiguous(),
        b1.contiguous(),
        out,
        torch.cuda.current_stream(device=a0.device),
    )
    return out.to(out_dtype)


def gemm_bf16_nmajor(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K].T via host-tiled 16×16 WMMA (unswapped tiles).

    Named ``nmajor`` for the zero-LDS family this module belongs to; the
    single-GEMM path uses the layout-correct unswapped tile. For larger K
    prefer ``gemm_bf16_nmajor_lds`` (multi-wave LDS). See ``fused_gemm_tn``
    for the A/B-swap N-major fuse. Pads M/N/K up to 16.
    """
    require_gfx120x(a.device, what="gemm_bf16_nmajor (gfx120x)")
    if a.ndim != 2 or b_nk.ndim != 2:
        raise ValueError("expects A[M,K], B[N,K]")
    if a.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"A must be bf16/fp16, got {a.dtype}")
    if b_nk.dtype != a.dtype:
        raise ValueError(f"B dtype {b_nk.dtype} must match A {a.dtype}")
    m, k = a.shape
    n, k2 = b_nk.shape
    if k != k2:
        raise ValueError(f"K mismatch {k} vs {k2}")
    if out_dtype is None:
        out_dtype = a.dtype

    mp = (m + _TILE - 1) // _TILE * _TILE
    np_ = (n + _TILE - 1) // _TILE * _TILE
    kp = (k + _TILE - 1) // _TILE * _TILE
    if mp != m or np_ != n or kp != k:
        a_pad = torch.zeros((mp, kp), device=a.device, dtype=a.dtype)
        b_pad = torch.zeros((np_, kp), device=a.device, dtype=a.dtype)
        a_pad[:m, :k] = a
        b_pad[:n, :k] = b_nk
    else:
        a_pad, b_pad = a.contiguous(), b_nk.contiguous()

    acc = torch.zeros((mp, np_), device=a.device, dtype=torch.float32)
    for i in range(0, mp, _TILE):
        for j in range(0, np_, _TILE):
            tile_acc = torch.zeros((_TILE, _TILE), device=a.device, dtype=torch.float32)
            for k0 in range(0, kp, _TILE):
                tile_acc += _run_tile(
                    a_pad[i : i + _TILE, k0 : k0 + _TILE],
                    b_pad[j : j + _TILE, k0 : k0 + _TILE],
                    swap_ab=False,
                )
            acc[i : i + _TILE, j : j + _TILE] = tile_acc

    result = acc[:m, :n].to(out_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    try:
        from kernels.common.gfx120x_swiglu import build_silu_mul_module

        out = torch.empty_like(gate)
        launch = build_silu_mul_module(dtype=_dtype_name(gate.dtype))
        _run_compiled(
            launch,
            _ptr(gate),
            _ptr(up),
            _ptr(out),
            int(gate.numel()),
            torch.cuda.current_stream(device=gate.device),
        )
        return out
    except Exception:
        return (F.silu(gate.float()) * up.float()).to(gate.dtype)


def fused_swiglu_mlp_inreg(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """In-register SwiGLU MLP for ``K == FFN == 16`` panels (host-tiles M/N).

    Computes ``silu(x @ Wgate.T) * (x @ Wup.T)`` then ``@ Wdown.T`` with
    weights ``[N, K]`` (linear layout). Each 16×16 (M,N) tile keeps mid
    in VGPRs (no LDS/GMEM spill). Requires ``K == FFN == 16`` and ``M``,
    ``N_out`` multiples of 16 (e.g. 32×16). Larger ``K``/``FFN`` need an
    in-kernel K/FFN loop — use ``fused_swiglu_mlp_nmajor`` instead.
    """
    require_gfx120x(x.device, what="fused_swiglu_mlp_inreg (gfx120x)")
    if out_dtype is None:
        out_dtype = x.dtype
    orig = x.shape
    x2d = x.reshape(-1, orig[-1]).contiguous()
    m, k = x2d.shape
    ffn, k_g = w_gate.shape
    ffn_u, k_u = w_up.shape
    n_out, k_d = w_down.shape
    if w_gate.dtype != x.dtype or w_up.dtype != x.dtype or w_down.dtype != x.dtype:
        raise ValueError("weight dtypes must match x")
    if k != _TILE or k_g != _TILE or k_u != _TILE or k_d != _TILE:
        raise ValueError(
            "fused_swiglu_mlp_inreg requires K == 16 "
            f"(got xK={k} gateK={k_g} upK={k_u} downK={k_d}); "
            "use fused_swiglu_mlp_nmajor for other sizes"
        )
    if ffn != _TILE or ffn_u != _TILE:
        raise ValueError(
            "fused_swiglu_mlp_inreg requires FFN == 16 "
            f"(got gate={ffn} up={ffn_u}); "
            "use fused_swiglu_mlp_nmajor for other sizes"
        )
    if m % _TILE != 0 or n_out % _TILE != 0:
        raise ValueError(
            "fused_swiglu_mlp_inreg requires M and N_out multiples of 16 "
            f"(got M={m} N_out={n_out}); "
            "use fused_swiglu_mlp_nmajor for other sizes"
        )

    launch = build_fused_swiglu_mlp_module(_dtype_name(x.dtype))
    stream = torch.cuda.current_stream(device=x.device)
    w_gate_c = w_gate.contiguous()
    w_up_c = w_up.contiguous()
    w_down_c = w_down.contiguous()
    out_f32 = torch.zeros((m, n_out), device=x.device, dtype=torch.float32)
    # Host-tile along M and N: each launch is the true in-reg 16×16 fuse.
    # N>16 recomputes GEMM0+SiLU per N tile (mid not shared across launches).
    for i0 in range(0, m, _TILE):
        x_tile = x2d[i0 : i0 + _TILE, :].contiguous()
        for j0 in range(0, n_out, _TILE):
            wd_tile = w_down_c[j0 : j0 + _TILE, :].contiguous()
            tile_out = torch.zeros((_TILE, _TILE), device=x.device, dtype=torch.float32)
            _run_compiled(
                launch,
                x_tile,
                w_gate_c,
                w_up_c,
                wd_tile,
                tile_out,
                stream,
            )
            out_f32[i0 : i0 + _TILE, j0 : j0 + _TILE] = tile_out
    return out_f32.to(out_dtype).reshape(*orig[:-1], n_out)


@lru_cache(maxsize=32)
def build_fused_swiglu_mlp_lds_module(dtype_name: str, k: int, ffn: int) -> Callable[..., None]:
    """In-kernel SwiGLU: K and FFN loops, 16-wide mid kept in LDS.

    One wave owns one ``[16, K]`` row panel and one ``[16, FFN]`` down-proj
    panel. Gate and up accumulate in registers across K. The SiLU×mul result
    is the N-major fragment from the swapped GEMM0. It is stored to LDS in
    that fragment order (lane ``tid`` owns elements ``8*tid : 8*tid+8``),
    reloaded, and used as A of the unswapped GEMM1. ``K`` and ``FFN`` are
    compile-time multiples of 16. The launch tensors are ``X[16, K]``,
    ``Wg[FFN, K]``, ``Wu[FFN, K]``, ``Wd[16, FFN]``, ``Y[16, 16]`` f32.
    """
    if dtype_name not in ("bfloat16", "float16"):
        raise ValueError(f"supports bf16/fp16, got {dtype_name}")
    if k % _TILE != 0 or ffn % _TILE != 0 or k < _TILE or ffn < _TILE:
        raise ValueError(f"K and FFN must be positive multiples of 16, got K={k} FFN={ffn}")
    Elem = fx.BFloat16 if dtype_name == "bfloat16" else fx.Float16
    k_tiles = k // _TILE
    ffn_tiles = ffn // _TILE
    mid_elems = WAVE * 8

    @fx.struct
    class SharedStorage:
        mid: fx.Array[Elem, mid_elems, 16]

    @flyc.kernel
    def fused_swiglu_mlp_lds_kernel(
        X: fx.Tensor,
        Wg: fx.Tensor,
        Wu: fx.Tensor,
        Wd: fx.Tensor,
        Y: fx.Tensor,
    ) -> None:
        tid = fx.thread_idx.x
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(_TILE, _TILE, _TILE, Elem, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        copy_ab = fx.make_copy_atom(fx.rocdl.BufferCopy(Elem.width), Elem)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
        thr_copy_A = fx.make_tiled_copy_A(copy_ab, tiled_mma).get_slice(tid)
        thr_copy_B = fx.make_tiled_copy_B(copy_ab, tiled_mma).get_slice(tid)
        thr_copy_C = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(tid)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        mid_lds = lds.mid.view(fx.make_layout(mid_elems, 1))

        bY = fx.make_view(
            fx.get_iter(fx.rocdl.make_buffer_tensor(Y)),
            fx.make_layout((_TILE, _TILE), (_TILE, 1)),
        )
        frag_Y = thr_mma.make_fragment_C(bY)

        a0 = fx.make_rmem_tensor(8, Elem)
        bg = fx.make_rmem_tensor(8, Elem)
        bu = fx.make_rmem_tensor(8, Elem)
        cg = fx.make_rmem_tensor(8, fx.Float32)
        cu = fx.make_rmem_tensor(8, fx.Float32)
        a1 = fx.make_rmem_tensor(8, Elem)
        bd = fx.make_rmem_tensor(8, Elem)
        c1 = fx.make_rmem_tensor(8, fx.Float32)
        c1.store(Vec.from_elements(_zeros8(), fx.Float32))

        def tile_view(tensor: fx.Tensor, row_stride: int, row0: fx.Int32 | int, col0: fx.Int32 | int) -> fx.Tensor:
            base = fx.get_iter(fx.rocdl.make_buffer_tensor(tensor))
            off = fx.Int64(row0) * fx.Int64(row_stride) + fx.Int64(col0)
            return fx.make_view(
                fx.add_offset(base, off),
                fx.make_layout((_TILE, _TILE), (row_stride, 1)),
            )

        def load_ab(view_a: fx.Tensor, view_b: fx.Tensor, dest_a: fx.Tensor, dest_b: fx.Tensor) -> None:
            frag_a = thr_mma.make_fragment_A(view_a)
            frag_b = thr_mma.make_fragment_B(view_b)
            fx.copy(copy_ab, thr_copy_A.partition_S(view_a), thr_copy_A.retile(frag_a))
            fx.copy(copy_ab, thr_copy_B.partition_S(view_b), thr_copy_B.retile(frag_b))
            dest_a.store(Vec(frag_a.load()))
            dest_b.store(Vec(frag_b.load()))

        for ft, fstate in range(0, fx.Int32(ffn_tiles), 1, init=[c1.load()]):
            c1.store(fstate[0])
            cg.store(Vec.from_elements(_zeros8(), fx.Float32))
            cu.store(Vec.from_elements(_zeros8(), fx.Float32))
            f_row = ft * fx.Int32(_TILE)
            for kt, kstate in range(0, fx.Int32(k_tiles), 1, init=[cg.load(), cu.load()]):
                cg.store(kstate[0])
                cu.store(kstate[1])
                k_col = kt * fx.Int32(_TILE)
                vx = tile_view(X, k, fx.Int32(0), k_col)
                vg = tile_view(Wg, k, f_row, k_col)
                vu = tile_view(Wu, k, f_row, k_col)
                load_ab(vx, vg, a0, bg)
                fx.gemm(mma_atom, cg, [bg], [a0], cg)
                load_ab(vx, vu, a0, bu)
                fx.gemm(mma_atom, cu, [bu], [a0], cu)
                cg_k, cu_k = yield [cg.load(), cu.load()]
            cg.store(cg_k)
            cu.store(cu_k)

            one = fx.Float32(1.0)
            neg_log2e = fx.Float32(-1.4426950408889634)
            cgv = Vec(cg.load())
            cuv = Vec(cu.load())
            mid = []
            for i in range_constexpr(8):
                g = fx.Float32(cgv[i])
                u = fx.Float32(cuv[i])
                sigv = one / (one + fmath.exp2(g * neg_log2e))
                mid.append((g * sigv * u).to(Elem))
            a1.store(Vec.from_elements(mid, Elem))

            lane_base = tid * fx.Int32(8)
            stored = Vec(a1.load())
            for i in range_constexpr(8):
                mid_lds[lane_base + fx.Int32(i)] = stored[i]
            fx.gpu.barrier()
            reloaded = []
            for i in range_constexpr(8):
                reloaded.append(mid_lds[lane_base + fx.Int32(i)])
            a1.store(Vec.from_elements(reloaded, Elem))
            fx.gpu.barrier()

            vd = tile_view(Wd, ffn, fx.Int32(0), f_row)
            frag_d = thr_mma.make_fragment_B(vd)
            fx.copy(copy_ab, thr_copy_B.partition_S(vd), thr_copy_B.retile(frag_d))
            bd.store(Vec(frag_d.load()))
            fx.gemm(mma_atom, c1, [a1], [bd], c1)
            f_out = yield [c1.load()]
        c1.store(f_out)

        frag_Y.store(Vec(c1.load()))
        fx.copy(copy_c, thr_copy_C.retile(frag_Y), thr_copy_C.partition_S(bY))

    fused_swiglu_mlp_lds_kernel.__name__ = f"fused_swiglu_mlp_lds_{dtype_name}_k{k}_f{ffn}"

    @flyc.jit
    def launch(
        X: fx.Tensor,
        Wg: fx.Tensor,
        Wu: fx.Tensor,
        Wd: fx.Tensor,
        Y: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        fused_swiglu_mlp_lds_kernel(X, Wg, Wu, Wd, Y).launch(grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream)

    launch.__name__ = f"launch_fused_swiglu_mlp_lds_{dtype_name}_k{k}_f{ffn}"
    return launch


def fused_swiglu_mlp_lds(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """SwiGLU MLP with the mid in LDS inside one kernel per 16×16 output tile.

    ``K``, ``FFN``, ``M``, and ``N_out`` must be multiples of 16. Each launch
    is one wave. The K loop accumulates gate and up. SiLU×mul is written to
    LDS in fragment order and read back as GEMM1's A. Weights are ``[N, K]``.
    """
    require_gfx120x(x.device, what="fused_swiglu_mlp_lds (gfx120x)")
    if out_dtype is None:
        out_dtype = x.dtype
    orig = x.shape
    x2d = x.reshape(-1, orig[-1]).contiguous()
    m, k = x2d.shape
    ffn, k_g = w_gate.shape
    ffn_u, k_u = w_up.shape
    n_out, k_d = w_down.shape
    if w_gate.dtype != x.dtype or w_up.dtype != x.dtype or w_down.dtype != x.dtype:
        raise ValueError("weight dtypes must match x")
    if k != k_g or k != k_u or ffn != ffn_u or k_d != ffn:
        raise ValueError(
            f"shape mismatch x[*,{k}] gate{tuple(w_gate.shape)} up{tuple(w_up.shape)} down{tuple(w_down.shape)}"
        )
    if any(v % _TILE != 0 or v < _TILE for v in (m, k, ffn, n_out)):
        raise ValueError(
            "fused_swiglu_mlp_lds requires M, K, FFN, and N_out to be positive multiples of 16 "
            f"(got M={m} K={k} FFN={ffn} N_out={n_out})"
        )
    launch = build_fused_swiglu_mlp_lds_module(_dtype_name(x.dtype), k, ffn)
    stream = torch.cuda.current_stream(device=x.device)
    w_gate_c = w_gate.contiguous()
    w_up_c = w_up.contiguous()
    w_down_c = w_down.contiguous()
    out_f32 = torch.zeros((m, n_out), device=x.device, dtype=torch.float32)
    for i0 in range(0, m, _TILE):
        x_tile = x2d[i0 : i0 + _TILE, :].contiguous()
        for j0 in range(0, n_out, _TILE):
            wd_tile = w_down_c[j0 : j0 + _TILE, :].contiguous()
            tile_out = torch.zeros((_TILE, _TILE), device=x.device, dtype=torch.float32)
            _run_compiled(launch, x_tile, w_gate_c, w_up_c, wd_tile, tile_out, stream)
            out_f32[i0 : i0 + _TILE, j0 : j0 + _TILE] = tile_out
    return out_f32.to(out_dtype).reshape(*orig[:-1], n_out)


def fused_swiglu_mlp_nmajor(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """SwiGLU MLP host.

    ``K == FFN == 16`` with M and N_out multiples of 16 uses the in-register
    kernel. Other shapes whose M, K, FFN, and N_out are multiples of 16 use
    ``fused_swiglu_mlp_lds`` (K/FFN loop, mid in LDS). Anything else is two
    GEMMs with the mid in GMEM, which can pad. Weights are ``[N, K]``.
    """
    require_gfx120x(x.device, what="fused_swiglu_mlp_nmajor (gfx120x)")
    if out_dtype is None:
        out_dtype = x.dtype
    orig = x.shape
    x2d = x.reshape(-1, orig[-1]).contiguous()
    m, k = x2d.shape
    ffn = w_gate.shape[0]
    n_out = w_down.shape[0]
    aligned = all(v % _TILE == 0 and v >= _TILE for v in (m, k, ffn, n_out))
    if aligned and k == _TILE and ffn == _TILE:
        return fused_swiglu_mlp_inreg(x, w_gate, w_up, w_down, out_dtype=out_dtype)
    if aligned:
        return fused_swiglu_mlp_lds(x, w_gate, w_up, w_down, out_dtype=out_dtype)
    gemm = gemm_bf16_nmajor_lds if (k > _TILE or ffn > _TILE) else gemm_bf16_nmajor
    gate = gemm(x2d, w_gate, out_dtype=x.dtype)
    up = gemm(x2d, w_up, out_dtype=x.dtype)
    mid = _silu_mul(gate, up)
    y = gemm(mid, w_down, out_dtype=out_dtype)
    return y.reshape(*orig[:-1], w_down.shape[0])


def reference_swiglu_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Eager reference: two GEMMs + SiLU×mul + down (f32 math)."""
    orig = x.shape
    x2d = x.reshape(-1, orig[-1]).float()
    gate = x2d @ w_gate.float().T
    up = x2d @ w_up.float().T
    mid = F.silu(gate) * up
    y = mid @ w_down.float().T
    return y.reshape(*orig[:-1], w_down.shape[0])
