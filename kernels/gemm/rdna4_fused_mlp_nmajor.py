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
* ``gemm_bf16_nmajor`` -- ``A @ B.T`` entry. Every shape launches the multi-wave
  LDS device kernel (``gemm_bf16_nmajor_lds``); K is not accumulated with a
  host ``torch`` sum of 16×16 tiles.
* ``gemm_bf16_nmajor_lds`` / ``build_gemm_nmajor_lds_module`` -- multi-wave
  LDS-pipelined production ``A @ B.T`` (wraps ``rdna_f16_gemm`` double-buffered
  LDS WMMA). Pads M/N/K to the selected block tile; requires ``K_pad >= 2*BK``
  for the prefetch pipeline. Tile choice is autotuned over every feasible
  ``_LDS_TILE_CANDIDATES`` entry (untuned default = ``pick_nmajor_lds_tile``).
  bf16/fp16 in; bf16/fp16/f32 out.
* ``fused_swiglu_mlp_inreg`` / ``build_fused_swiglu_mlp_module`` -- in-register
  SwiGLU fuse for one-wave 16x16 panels. Host tiles along M/N when
  ``K == FFN == 16`` and ``M, N_out`` are multiples of 16. Per tile: GEMM0_gate
  + GEMM0_up (A/B-swapped) -> SiLU(gate)*up in registers -> narrow -> GEMM1
  unswapped -> M-major store. Layout: ``silu(x @ Wgate.T) * (x @ Wup.T)`` then
  ``@ Wdown.T`` with weights ``[N, K]``.
* ``fused_swiglu_mlp_nmajor`` -- SwiGLU MLP host. ``K == FFN == 16`` with
  M/N_out multiples of 16 uses ``fused_swiglu_mlp_inreg``. Every other positive
  shape uses ``fused_swiglu_mlp_lds`` (in-kernel K/FFN loop, mid staged in LDS).
  A K, FFN, M, or N_out that is not a multiple of 16 is zero-filled in that
  kernel. The host does not clone those operands.

Known limits
------------
* Multi-wave LDS GEMM is a single-GEMM production block (M-major D store), not
  an in-kernel A/B-swap N-major fuse across two GEMMs. The zero-LDS
  ``fused_gemm_tn`` / ``fused_swiglu_mlp_inreg`` paths keep the VGPR N-major fuse.
* In-kernel fused SwiGLU with a K/FFN loop keeps the 16-wide mid in LDS
  (``fused_swiglu_mlp_lds``). The fragment is spilled per lane (32×8),
  reloaded, then consumed as A of GEMM1. It is not a multi-wave LDS GEMM.
  Multiples of 16 keep the wide tiled copy. Any other positive shape
  scalar-fills that same fragment and issues the same WMMA.
* Host-tiled in-reg covers ``K == FFN == 16`` with ``M, N_out`` multiples of 16
  (one fused 16x16 launch per (M,N) tile; N>16 recomputes GEMM0+SiLU per N tile).
  A direct ``fused_swiglu_mlp_inreg`` call with ``K`` or ``FFN`` below 16 still
  soft-pads that cube. ``fused_swiglu_mlp_nmajor`` does not send those shapes there.
* ``rdna_f16_gemm`` LDS path needs ``K_pad >= 2 * BLOCK_K`` (prefetch); the
  nmajor GEMM host still zero-pads K when the logical K is shallower. SwiGLU
  does not use that pad.
* Production ``scaled_mm_fp8`` / ``int8_linear`` defaults are untouched.
* vs HIP baseline: honest miss (no fused-MLP HIP baseline). Microbench vs
  unfused FlyDSL gemm+SiLU is informational only.
"""

from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from flydsl.compiler.jit_argument import PointerJitArg
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from kernels.common.gfx120x_arch import require_gfx120x

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

        # In-register SiLU(gate)*up (same formula as gfx120x_swiglu).
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


def _padded_mnk(m: int, n: int, k: int, tile: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int, int]:
    """Pad ``(m, n, k)`` up to ``tile`` and to ``Kp >= 2 * BK`` (prefetch)."""
    bm, bn, bk = _lds_block_shape(tile)
    mp = (m + bm - 1) // bm * bm
    np_ = (n + bn - 1) // bn * bn
    kp = (k + bk - 1) // bk * bk
    if kp < 2 * bk:
        kp = 2 * bk
    return mp, np_, kp, bm, bn, bk


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
        mp, np_, kp, bm, bn, bk = _padded_mnk(M, N, K, tile)
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


def _nmajor_lds_shape_class(m: int, n: int, k: int) -> str:
    """Padded working-set class, not one measured ``(M, N, K)``.

    Shapes the heuristic pads to the same ``(Mp, Np, Kp)`` share a tuned tile.
    """
    _tile, mp, np_, kp, _bm, _bn, _bk = pick_nmajor_lds_tile(m, n, k)
    return f"{mp}x{np_}x{kp}"


_NMAJOR_LDS_TUNING_SCHEMA = 1


def _nmajor_lds_configs() -> list[Config]:
    configs = []
    for reg_m, reg_n, reg_k, waves_m, waves_n in _LDS_TILE_CANDIDATES:
        tile = (reg_m, reg_n, reg_k, waves_m, waves_n)
        if not _lds_tile_geometry_ok(tile):
            continue
        configs.append(
            Config(
                reg_m=reg_m,
                reg_n=reg_n,
                reg_k=reg_k,
                waves_m=waves_m,
                waves_n=waves_n,
            )
        )
    return configs


_NMAJOR_LDS_CONFIGS = _nmajor_lds_configs()


def _default_nmajor_lds_config(*args, **kwargs) -> Config:
    """Untuned tile: today's ``pick_nmajor_lds_tile`` heuristic."""
    a = kwargs["a"] if "a" in kwargs else args[0]
    b_nk = kwargs["b_nk"] if "b_nk" in kwargs else args[1]
    tile, *_rest = pick_nmajor_lds_tile(int(a.shape[0]), int(b_nk.shape[0]), int(a.shape[1]))
    reg_m, reg_n, reg_k, waves_m, waves_n = tile
    return Config(reg_m=reg_m, reg_n=reg_n, reg_k=reg_k, waves_m=waves_m, waves_n=waves_n)


@contextmanager
def _validate_nmajor_lds(sig_args):
    """NaN-poison C and raise if a candidate leaves non-finite values."""
    c = sig_args["c"]
    c.fill_(float("nan"))
    yield
    if c.numel() and not bool(torch.isfinite(c).all()):
        raise ValueError("nmajor LDS candidate left non-finite C")


def _prepare_nmajor_operands(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    m: int,
    n: int,
    k: int,
    tile: tuple[int, int, int, int, int],
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    from kernels.common.gfx120x_pad import ensure_contiguous

    mp, np_, kp, _bm, _bn, _bk = _padded_mnk(m, n, k, tile)
    from kernels.common.gfx120x_pad import device_pad

    def _pad2d(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        r, c = int(t.shape[0]), int(t.shape[1])
        pr, pc = rows - r, cols - c
        if pr == 0 and pc == 0:
            return ensure_contiguous(t, stream=stream)
        # F.pad layout: (left, right, top, bottom) → last dim then rows.
        return device_pad(ensure_contiguous(t, stream=stream), (0, pc, 0, pr), stream=stream)

    if m > 0 and k > 0:
        a_pad = _pad2d(a, mp, kp)
    else:
        a_pad = torch.empty((mp, kp), device=a.device, dtype=a.dtype)
    if n > 0 and k > 0:
        b_pad = _pad2d(b_nk, np_, kp)
    else:
        b_pad = torch.empty((np_, kp), device=a.device, dtype=a.dtype)
    return a_pad, b_pad, mp, np_, kp


@autotune(
    configs=_NMAJOR_LDS_CONFIGS,
    key=["dtype_name", "out_name", "shape_class", "tuning_schema"],
    default=_default_nmajor_lds_config,
    validate_hook=_validate_nmajor_lds,
)
def _launch_nmajor_lds(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    c: torch.Tensor,
    dtype_name: str,
    out_name: str,
    shape_class: str,
    tuning_schema: int,
    reg_m: int,
    reg_n: int,
    reg_k: int,
    waves_m: int,
    waves_n: int,
    stream=None,
) -> None:
    """Device ``C = A @ B.T`` for one autotune tile.

    ``shape_class`` and ``tuning_schema`` are cache axes only. They are not a
    measured ``(M, N, K)`` and are not read by the kernel.
    """
    del shape_class, tuning_schema
    m = int(a.shape[0])
    k = int(a.shape[1])
    n = int(b_nk.shape[0])
    tile = (int(reg_m), int(reg_n), int(reg_k), int(waves_m), int(waves_n))
    a_pad, b_pad, mp, np_, kp = _prepare_nmajor_operands(a, b_nk, m, n, k, tile, stream=stream)
    direct = c.is_contiguous() and tuple(c.shape) == (m, n) and mp == m and np_ == n
    c_full = c if direct else torch.full((mp, np_), float("nan"), device=c.device, dtype=c.dtype)
    launch, _, _, _ = build_gemm_nmajor_lds_module(mp, np_, kp, dtype_name, out_name, tile)
    if stream is None:
        launch(c_full, a_pad, b_pad)
    else:
        launch(c_full, a_pad, b_pad, stream)
    if c_full is not c:
        if stream is None:
            c.copy_(c_full[:m, :n])
        else:
            with torch.cuda.stream(stream):
                c.copy_(c_full[:m, :n])


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
    raw_launch, bm, bn, bk = create_wmma_gemm_module(
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

    # ``launch_gemm`` requires a stream. Give each specialization its own jit so
    # the host can omit the argument (default ``fx.Stream(None)``).
    def launch_nmajor_lds(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
        sr_seed: fx.Int32 = 0,
    ) -> None:
        raw_launch(arg_c, arg_a, arg_bt, stream, sr_seed)

    launch_nmajor_lds.__name__ = (
        f"launch_nmajor_lds_m{M}_n{N}_k{K}_{in_dtype}_{out_dtype}_r{reg_m}x{reg_n}x{reg_k}_w{waves_m}x{waves_n}"
    )
    return flyc.jit(launch_nmajor_lds), bm, bn, bk


def gemm_bf16_nmajor_lds(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K].T via multi-wave LDS-pipelined WMMA (gfx120x).

    Pads any positive bf16/fp16 M/N/K up to the selected block tile (and to
    ``K_pad >= 2 * BLOCK_K``). Untuned launches use ``pick_nmajor_lds_tile``.
    ``FLYDSL_AUTOTUNE=1`` searches every geometry-legal ``_LDS_TILE_CANDIDATES``
    entry. The cache key is the dtype names and that padded-shape class, not
    one measured ``(M, N, K)``.
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
    out_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }.get(out_dtype)
    if out_name is None:
        raise ValueError(f"out_dtype must be bf16/fp16/f32, got {out_dtype}")

    # Empty-K (or empty M/N) is a legal GEMM shape: C is all zeros. Soft-handle
    # without launching WMMA on uninitialized pads from _prepare_nmajor_operands.
    if m == 0 or n == 0 or k == 0:
        c = torch.zeros((m, n), device=a.device, dtype=out_dtype)
        if out is not None:
            if stream is None:
                out.copy_(c)
            else:
                with torch.cuda.stream(stream):
                    out.copy_(c)
            return out
        return c

    c = torch.empty((m, n), device=a.device, dtype=out_dtype)
    kw = dict(
        dtype_name=_dtype_name(a.dtype),
        out_name=out_name,
        shape_class=_nmajor_lds_shape_class(int(m), int(n), int(k)),
        tuning_schema=_NMAJOR_LDS_TUNING_SCHEMA,
    )
    if stream is not None:
        kw["stream"] = stream
    _launch_nmajor_lds(
        a,
        b_nk,
        c,
        **kw,
    )
    if out is not None:
        if stream is None:
            out.copy_(c)
        else:
            with torch.cuda.stream(stream):
                out.copy_(c)
        return out
    return c


def _run_tile(
    a_tile: torch.Tensor,
    b_tile: torch.Tensor,
    *,
    swap_ab: bool = False,
) -> torch.Tensor:
    out = torch.zeros((_TILE, _TILE), device=a_tile.device, dtype=torch.float32)
    launch = build_wmma_tile_module(_dtype_name(a_tile.dtype), swap_ab=swap_ab)
    launch(a_tile.contiguous(), b_tile.contiguous(), out)
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
    if b0.dtype != a0.dtype or b1.dtype != a0.dtype:
        raise ValueError(f"fused_gemm_tn dtypes must match, got a0={a0.dtype} b0={b0.dtype} b1={b1.dtype}")
    if out_dtype is None:
        out_dtype = a0.dtype
    out = torch.zeros((_TILE, _TILE), device=a0.device, dtype=torch.float32)
    launch = build_fused_gemm_tn_module(_dtype_name(a0.dtype))
    launch(a0.contiguous(), b0.contiguous(), b1.contiguous(), out)
    return out.to(out_dtype)


def gemm_bf16_nmajor(
    a: torch.Tensor,
    b_nk: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K].T on the LDS device kernel.

    Kept as the historical name. Every shape, including panels smaller than
    16, goes through ``gemm_bf16_nmajor_lds`` so K accumulation stays on device.
    """
    return gemm_bf16_nmajor_lds(a, b_nk, out=out, out_dtype=out_dtype)


@lru_cache(maxsize=8)
def _silu_launch_default_stream(dtype_name: str):
    """Wrap silu launch so callers may omit stream (defaults to current)."""
    from kernels.common.gfx120x_swiglu import build_silu_mul_module

    raw = build_silu_mul_module(dtype=dtype_name)

    def launch_silu_default_stream(
        gate: fx.Pointer,
        up: fx.Pointer,
        out: fx.Pointer,
        n_elems: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        raw(gate, up, out, n_elems, stream)

    launch_silu_default_stream.__name__ = f"launch_silu_mul_default_stream_{dtype_name}"
    return flyc.jit(launch_silu_default_stream)


def _silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    out = torch.empty_like(gate)
    launch = _silu_launch_default_stream(_dtype_name(gate.dtype))
    if stream is None:
        launch(_ptr(gate), _ptr(up), _ptr(out), int(gate.numel()))
    else:
        launch(_ptr(gate), _ptr(up), _ptr(out), int(gate.numel()), stream)
    return out


def fused_swiglu_mlp_inreg(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """In-register SwiGLU MLP for the 16-cube (host-tiles M/N).

    Computes ``silu(x @ Wgate.T) * (x @ Wup.T)`` then ``@ Wdown.T`` with
    weights ``[N, K]`` (linear layout). Each 16×16 (M,N) tile keeps mid
    in VGPRs (no LDS/GMEM spill). Soft-pads ``K``/``FFN``/``M``/``N_out``
    up to 16 when they are positive and ``<= 16``. Larger ``K``/``FFN`` need
    an in-kernel loop — use ``fused_swiglu_mlp_nmajor`` / ``_lds`` instead.
    """
    from kernels.common.gfx120x_pad import device_pad, ensure_contiguous

    require_gfx120x(x.device, what="fused_swiglu_mlp_inreg (gfx120x)")
    if out_dtype is None:
        out_dtype = x.dtype
    orig = x.shape
    x2d = ensure_contiguous(x.reshape(-1, orig[-1]), stream=stream)
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
    if k <= 0 or ffn <= 0 or k > _TILE or ffn > _TILE:
        raise ValueError(
            "fused_swiglu_mlp_inreg requires 0 < K,FFN <= 16 "
            f"(got K={k} FFN={ffn}); use fused_swiglu_mlp_nmajor for other sizes"
        )
    if m == 0 or n_out == 0:
        return torch.empty(*orig[:-1], n_out, device=x.device, dtype=out_dtype)
    # Soft pad into the 16-cube (prefer high-TOPS in-reg over raise / LDS).

    def _pad2d(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        r, c = int(t.shape[0]), int(t.shape[1])
        pr, pc = rows - r, cols - c
        if pr == 0 and pc == 0:
            return ensure_contiguous(t, stream=stream)
        return device_pad(ensure_contiguous(t, stream=stream), (0, pc, 0, pr), stream=stream)

    m_p = max(_TILE, ((int(m) + _TILE - 1) // _TILE) * _TILE)
    n_p = max(_TILE, ((int(n_out) + _TILE - 1) // _TILE) * _TILE)
    x2d = _pad2d(x2d, m_p, _TILE)
    w_gate_c = _pad2d(w_gate, _TILE, _TILE)
    w_up_c = _pad2d(w_up, _TILE, _TILE)
    w_down_c = _pad2d(w_down, n_p, _TILE)

    launch = build_fused_swiglu_mlp_module(_dtype_name(x.dtype))
    out_f32 = torch.zeros((m_p, n_p), device=x.device, dtype=torch.float32)
    # Host-tile along M and N: each launch is the true in-reg 16×16 fuse.
    # N>16 recomputes GEMM0+SiLU per N tile (mid not shared across launches).
    for i0 in range(0, m_p, _TILE):
        x_tile = ensure_contiguous(x2d[i0 : i0 + _TILE, :], stream=stream)
        for j0 in range(0, n_p, _TILE):
            wd_tile = ensure_contiguous(w_down_c[j0 : j0 + _TILE, :], stream=stream)
            tile_out = torch.zeros((_TILE, _TILE), device=x.device, dtype=torch.float32)
            if stream is None:
                launch(x_tile, w_gate_c, w_up_c, wd_tile, tile_out)
            else:
                launch(x_tile, w_gate_c, w_up_c, wd_tile, tile_out, stream)
            out_f32[i0 : i0 + _TILE, j0 : j0 + _TILE] = tile_out
    return out_f32[:m, :n_out].to(out_dtype).reshape(*orig[:-1], n_out)


@lru_cache(maxsize=32)
def build_fused_swiglu_mlp_lds_module(
    dtype_name: str,
    k: int,
    ffn: int,
    logical_k: int | None = None,
    logical_ffn: int | None = None,
    mask_rows: bool = False,
) -> Callable[..., None]:
    """In-kernel SwiGLU: K and FFN loops, 16-wide mid kept in LDS.

    One wave owns one ``[16, K]`` row panel and one ``[16, FFN]`` down-proj
    panel. Gate and up accumulate in registers across K. The SiLU×mul result
    is the N-major fragment from the swapped GEMM0. It is stored to LDS in
    that fragment order (lane ``tid`` owns elements ``8*tid : 8*tid+8``),
    reloaded, and used as A of the unswapped GEMM1. ``K`` and ``FFN`` are
    compile-time multiples of 16 (the WMMA width). ``logical_k`` / ``logical_ffn``
    default to those widths. When they are shorter, or ``mask_rows`` is set,
    the kernel scalar-fills the WMMA fragment and takes runtime panel row
    counts. The aligned launch tensors stay ``X[16, K]``, ``Wg[FFN, K]``,
    ``Wu[FFN, K]``, ``Wd[16, FFN]``, ``Y[16, 16]`` f32, with no extra arguments.
    """
    if logical_k is None:
        logical_k = k
    if logical_ffn is None:
        logical_ffn = ffn
    if dtype_name not in ("bfloat16", "float16"):
        raise ValueError(f"supports bf16/fp16, got {dtype_name}")
    if k % _TILE != 0 or ffn % _TILE != 0 or k < _TILE or ffn < _TILE:
        raise ValueError(f"K and FFN must be positive multiples of 16, got K={k} FFN={ffn}")
    if logical_k <= 0 or logical_k > k or logical_ffn <= 0 or logical_ffn > ffn:
        raise ValueError(f"logical_k={logical_k} logical_ffn={logical_ffn} must lie in 1..K={k} and 1..FFN={ffn}")
    if logical_k != k or logical_ffn != ffn or mask_rows:
        return _build_fused_swiglu_mlp_lds_tail(dtype_name, k, ffn, logical_k, logical_ffn)
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
            # scf.for yields an index. arith.muli rejects index * i32.
            f_row = fx.Int32(ft) * fx.Int32(_TILE)
            for kt, kstate in range(0, fx.Int32(k_tiles), 1, init=[cg.load(), cu.load()]):
                cg.store(kstate[0])
                cu.store(kstate[1])
                k_col = fx.Int32(kt) * fx.Int32(_TILE)
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


@lru_cache(maxsize=32)
def _build_fused_swiglu_mlp_lds_tail(
    dtype_name: str,
    k: int,
    ffn: int,
    logical_k: int,
    logical_ffn: int,
) -> Callable[..., None]:
    """Same SwiGLU loop as the aligned builder, with scalar WMMA fragments.

    ``k`` and ``ffn`` are the padded multiples of 16 the WMMA issues. ``logical_*``
    are the caller's sizes. RDNA4 A/B layout is ``M = lane % 16`` and
    ``K = (lane // 16) * 8 + val``. A wide copy of a short row reads the next
    row or past the allocation, so each val is one global element, or zero.
    """
    Elem = fx.BFloat16 if dtype_name == "bfloat16" else fx.Float16
    k_tiles = k // _TILE
    ffn_tiles = ffn // _TILE
    mid_elems = WAVE * 8

    @fx.struct
    class SharedStorage:
        mid: fx.Array[Elem, mid_elems, 16]

    @flyc.kernel
    def fused_swiglu_mlp_lds_tail_kernel(
        X: fx.Tensor,
        Wg: fx.Tensor,
        Wu: fx.Tensor,
        Wd: fx.Tensor,
        Y: fx.Tensor,
        x_rows: fx.Int32,
        d_rows: fx.Int32,
    ) -> None:
        tid = fx.thread_idx.x
        mma_atom = fx.make_mma_atom(fx.rocdl.WMMA(_TILE, _TILE, _TILE, Elem, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)
        copy_c = fx.make_copy_atom(fx.rocdl.BufferCopy(fx.Float32.width), fx.Float32)
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

        def as_elem(tensor: fx.Tensor) -> fx.Pointer:
            base = fx.get_iter(tensor)
            return fx.recast_iter(fx.PointerType.get(Elem.ir_type, base.address_space), base)

        x_ptr = as_elem(X)
        g_ptr = as_elem(Wg)
        u_ptr = as_elem(Wu)
        d_ptr = as_elem(Wd)
        lane = fx.Int32(tid)
        row = lane % fx.Int32(16)
        kbase = (lane // fx.Int32(16)) * fx.Int32(8)

        def fill_frag(
            ptr: fx.Pointer,
            row_stride: int,
            row0: fx.Int32 | int,
            col0: fx.Int32 | int,
            n_rows: fx.Int32 | int,
            n_cols: int,
            dest: fx.Tensor,
        ) -> None:
            elems = []
            for i in range_constexpr(8):
                one = Elem(0.0)
                g_row = fx.Int32(row0) + row
                g_col = fx.Int32(col0) + kbase + fx.Int32(i)
                if (g_row < fx.Int32(n_rows)) & (g_col < fx.Int32(n_cols)):
                    idx = fx.Int64(g_row) * fx.Int64(row_stride) + fx.Int64(g_col)
                    one = Vec(fx.make_view(fx.add_offset(ptr, idx), fx.make_layout(1, 1)).load())[0]
                elems.append(one)
            dest.store(Vec.from_elements(elems, Elem))

        for ft, fstate in range(0, fx.Int32(ffn_tiles), 1, init=[c1.load()]):
            c1.store(fstate[0])
            cg.store(Vec.from_elements(_zeros8(), fx.Float32))
            cu.store(Vec.from_elements(_zeros8(), fx.Float32))
            f_row = fx.Int32(ft) * fx.Int32(_TILE)
            for kt, kstate in range(0, fx.Int32(k_tiles), 1, init=[cg.load(), cu.load()]):
                cg.store(kstate[0])
                cu.store(kstate[1])
                k_col = fx.Int32(kt) * fx.Int32(_TILE)
                fill_frag(x_ptr, logical_k, fx.Int32(0), k_col, x_rows, logical_k, a0)
                fill_frag(g_ptr, logical_k, f_row, k_col, logical_ffn, logical_k, bg)
                fx.gemm(mma_atom, cg, [bg], [a0], cg)
                fill_frag(x_ptr, logical_k, fx.Int32(0), k_col, x_rows, logical_k, a0)
                fill_frag(u_ptr, logical_k, f_row, k_col, logical_ffn, logical_k, bu)
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

            fill_frag(d_ptr, logical_ffn, fx.Int32(0), f_row, d_rows, logical_ffn, bd)
            fx.gemm(mma_atom, c1, [a1], [bd], c1)
            f_out = yield [c1.load()]
        c1.store(f_out)

        frag_Y.store(Vec(c1.load()))
        fx.copy(copy_c, thr_copy_C.retile(frag_Y), thr_copy_C.partition_S(bY))

    fused_swiglu_mlp_lds_tail_kernel.__name__ = (
        f"fused_swiglu_mlp_lds_{dtype_name}_k{k}_f{ffn}_lk{logical_k}_lf{logical_ffn}"
    )

    @flyc.jit
    def launch(
        X: fx.Tensor,
        Wg: fx.Tensor,
        Wu: fx.Tensor,
        Wd: fx.Tensor,
        Y: fx.Tensor,
        x_rows: fx.Int32,
        d_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        fused_swiglu_mlp_lds_tail_kernel(X, Wg, Wu, Wd, Y, x_rows, d_rows).launch(
            grid=(1, 1, 1), block=(WAVE, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_fused_swiglu_mlp_lds_{dtype_name}_k{k}_f{ffn}_lk{logical_k}_lf{logical_ffn}"
    return launch


def fused_swiglu_mlp_lds(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """SwiGLU MLP with the mid in LDS inside one kernel per 16×16 output tile.

    Each launch is one wave. The K loop accumulates gate and up. SiLU×mul is
    written to LDS in fragment order and read back as GEMM1's A. Weights are
    ``[N, K]``. A dimension that is not a multiple of 16 stays at the caller's
    size; the kernel zero-fills lanes past that size. The per-tile ``Y`` is a
    16×16 scratch buffer, and the returned tensor has the caller's shape.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(x.device, what="fused_swiglu_mlp_lds (gfx120x)")
    if out_dtype is None:
        out_dtype = x.dtype
    orig = x.shape
    x2d = ensure_contiguous(x.reshape(-1, orig[-1]), stream=stream)
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
    if k <= 0 or ffn <= 0:
        raise ValueError(f"fused_swiglu_mlp_lds requires positive K and FFN, got K={k} FFN={ffn}")
    if m == 0 or n_out == 0:
        return torch.empty(*orig[:-1], n_out, device=x.device, dtype=out_dtype)

    def _ceil_tile(v: int) -> int:
        return max(_TILE, ((int(v) + _TILE - 1) // _TILE) * _TILE)

    k_p, ffn_p = _ceil_tile(k), _ceil_tile(ffn)
    mask_rows = (m % _TILE) != 0 or (n_out % _TILE) != 0
    aligned = k == k_p and ffn == ffn_p and not mask_rows
    w_gate_c = ensure_contiguous(w_gate, stream=stream)
    w_up_c = ensure_contiguous(w_up, stream=stream)
    w_down_c = ensure_contiguous(w_down, stream=stream)
    out_f32 = torch.empty((m, n_out), device=x.device, dtype=torch.float32)
    if aligned:
        launch = build_fused_swiglu_mlp_lds_module(_dtype_name(x.dtype), k, ffn)
    else:
        launch = build_fused_swiglu_mlp_lds_module(
            _dtype_name(x.dtype),
            k_p,
            ffn_p,
            logical_k=k,
            logical_ffn=ffn,
            mask_rows=mask_rows,
        )
    for i0 in range(0, m, _TILE):
        x_rows = min(_TILE, m - i0)
        x_tile = ensure_contiguous(x2d[i0 : i0 + x_rows, :], stream=stream)
        for j0 in range(0, n_out, _TILE):
            d_rows = min(_TILE, n_out - j0)
            wd_tile = ensure_contiguous(w_down_c[j0 : j0 + d_rows, :], stream=stream)
            tile_out = torch.zeros((_TILE, _TILE), device=x.device, dtype=torch.float32)
            if aligned:
                if stream is None:
                    launch(x_tile, w_gate_c, w_up_c, wd_tile, tile_out)
                else:
                    launch(x_tile, w_gate_c, w_up_c, wd_tile, tile_out, stream)
                out_f32[i0 : i0 + _TILE, j0 : j0 + _TILE] = tile_out
            else:
                if stream is None:
                    launch(x_tile, w_gate_c, w_up_c, wd_tile, tile_out, x_rows, d_rows)
                else:
                    launch(x_tile, w_gate_c, w_up_c, wd_tile, tile_out, x_rows, d_rows, stream)
                out_f32[i0 : i0 + x_rows, j0 : j0 + d_rows] = tile_out[:x_rows, :d_rows]
    return out_f32.to(out_dtype).reshape(*orig[:-1], n_out)


def _swiglu_leading_rows(x: torch.Tensor) -> int:
    rows = 1
    for size in x.shape[:-1]:
        rows *= int(size)
    return rows


def fused_swiglu_mlp_nmajor(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """SwiGLU MLP host.

    ``K == FFN == 16`` with ``M`` and ``N_out`` multiples of 16 uses the
    in-register cube. Every other positive shape uses the LDS kernel, which
    reads the caller's K and FFN and zero-fills the tail. Weights are ``[N, K]``.
    """
    require_gfx120x(x.device, what="fused_swiglu_mlp_nmajor (gfx120x)")
    if out_dtype is None:
        out_dtype = x.dtype
    k = int(x.shape[-1])
    ffn = int(w_gate.shape[0])
    m = _swiglu_leading_rows(x)
    n_out = int(w_down.shape[0])
    if k == _TILE and ffn == _TILE and m > 0 and n_out > 0 and m % _TILE == 0 and n_out % _TILE == 0:
        return fused_swiglu_mlp_inreg(x, w_gate, w_up, w_down, out_dtype=out_dtype, stream=stream)
    return fused_swiglu_mlp_lds(x, w_gate, w_up, w_down, out_dtype=out_dtype, stream=stream)
