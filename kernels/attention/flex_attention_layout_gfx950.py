# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Independent flash/flex-attention forward on the FlyDSL layout API (gfx950).

This is a from-scratch attention kernel written on the new CuTe-style layout API
(``fx.make_tiled_mma`` / ``make_fragment_{A,B,C}`` / ``fx.gemm`` / ``fx.copy`` /
swizzled LDS views), modelled on ``flydsl-examples/kernels/hgemm_layout_gfx950.py``
for the MMA/pipeline structure and ``kernels/norm/softmax_kernel.py`` for the
softmax numerics. It does NOT reuse the legacy raw-MFMA ``flash_attn_generic``
path.

Phase 0 (this file): a correct DENSE forward, single-stage, no flex hooks. One
workgroup computes one ``[BLOCK_M, D]`` query tile: load Q resident, loop over KV
``[BLOCK_N, D]`` tiles doing GEMM1 (S = Q@K^T), online softmax, the C->A bridge,
then GEMM2 (O += P@V); epilogue normalizes O by the row sum and stores it.

Target arch: gfx950 (CDNA4). Uses the cdna4 LDS transpose-read atom and the
gfx950 LDS swizzles; it is NOT expected to run on gfx942.
"""

from typing import Optional

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp
from flydsl.runtime.device import get_rocm_arch

GFX950_WAVE_SIZE = 64
GFX950_DMA_BYTES = 16
FLEX_DTYPE_BF16 = 2
FLEX_DTYPE_FP16 = 3

_LOG2E = 1.4426950408889634


@fx.struct
class FlexAttnParam:
    dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    head_dim: fx.Constexpr[int]
    num_heads_q: fx.Constexpr[int]
    num_heads_kv: fx.Constexpr[int]
    causal: fx.Constexpr[bool]
    # wave tiling
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    # mma shape (bf16/f16: 16x16x32 to match hgemm)
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]
    # derived
    block_threads: fx.Constexpr[int]
    gqa_group: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    n_kv_tiles: fx.Constexpr[int]  # seqlen_kv // block_n (KV loop is compile-time unrolled)


def make_flex_attn_param(
    seqlen_kv: int,
    dtype_id: int = FLEX_DTYPE_BF16,
    block_m: int = 32,
    block_n: int = 32,
    head_dim: int = 128,
    num_heads_q: int = 8,
    num_heads_kv: int = 8,
    causal: bool = False,
    m_waves: int = 2,
    n_waves: int = 1,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
) -> FlexAttnParam:
    if dtype_id not in (FLEX_DTYPE_BF16, FLEX_DTYPE_FP16):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if block_m <= 0 or block_n <= 0 or head_dim <= 0:
        raise ValueError("block_m, block_n, head_dim must be positive")
    # Phase 0 proven config (see plan progress log): fx.gemm with MFMA 16x16x16 has
    # a lowering bug on this build, so both GEMMs use mma_k=32 -> block_n multiple of
    # 32. The C-fragment slot->row map is only locked for block_m=32 (2 M-waves x
    # mma_m=16), one N-wave; larger block_m needs a per-slot row map (TODO perf).
    if (mma_m, mma_n, mma_k) != (16, 16, 32):
        raise ValueError("Phase 0 requires mma=16x16x32 (16x16x16 fx.gemm lowering bug)")
    if block_m != 32:
        raise ValueError("Phase 0 requires block_m == 32 (larger M needs per-slot row map)")
    if block_n % 32 != 0:
        raise ValueError("block_n must be a multiple of 32 (mma_k)")
    if (m_waves, n_waves) != (2, 1):
        raise ValueError("Phase 0 requires m_waves=2, n_waves=1")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError("num_heads_q must be divisible by num_heads_kv (GQA)")
    if head_dim % mma_k != 0:
        raise ValueError(f"head_dim ({head_dim}) must be divisible by mma_k ({mma_k})")
    if seqlen_kv % block_n != 0:
        raise ValueError(f"seqlen_kv ({seqlen_kv}) must be a multiple of block_n ({block_n})")

    in_dbytes = 2  # bf16/f16
    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE

    return FlexAttnParam(
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        head_dim=head_dim,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        causal=causal,
        m_waves=m_waves,
        n_waves=n_waves,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        block_threads=block_threads,
        gqa_group=num_heads_q // num_heads_kv,
        in_data_bytes=in_dbytes,
        n_kv_tiles=seqlen_kv // block_n,
    )


def make_flex_attn_kernel_name(param: FlexAttnParam) -> str:
    dtype_str = "fp16" if param.dtype_id == FLEX_DTYPE_FP16 else "bf16"
    name = f"flex_attn_{dtype_str}_m{param.block_m}n{param.block_n}d{param.head_dim}"
    name += f"_w{param.m_waves}x{param.n_waves}"
    name += "_causal" if param.causal else "_dense"
    return name


_FM = fx.arith.FastMathFlags.fast


def _elem_dtype(dtype_id):
    return fx.Float16 if dtype_id == FLEX_DTYPE_FP16 else fx.BFloat16


def _row_reduce(x, mode):
    # Per-query-row reduction over the key dimension: the 16 keys of an N-wave are
    # spread across the low 4 lane bits, so a shuffle_xor butterfly over offsets
    # 8/4/2/1 combines them. block_n stays within one N-wave (n_waves_n == 1).
    w = x
    for off in (8, 4, 2, 1):
        peer = w.shuffle_xor(off, GFX950_WAVE_SIZE)
        w = w.maximumf(peer) if mode == "max" else w.addf(peer, fastmath=_FM)
    return w


@flyc.kernel
def flex_attn_fwd_gfx950_kernel(
    o: fx.Tensor,       # [B, Sq, Hq, D]
    q: fx.Tensor,       # [B, Sq, Hq, D]
    k: fx.Tensor,       # [B, Skv, Hkv, D]
    v: fx.Tensor,       # [B, Skv, Hkv, D]
    seqlen_q: fx.Int32,
    seqlen_kv: fx.Int32,
    scale: fx.Float32,
    tiled_mma_qk: fx.TiledMma,
    tiled_mma_pv: fx.TiledMma,
    param: FlexAttnParam,
):
    block_m = param.block_m
    block_n = param.block_n
    head_dim = param.head_dim
    elem_dtype = _elem_dtype(param.dtype_id)

    tid = fx.thread_idx.x
    # grid.x = query-tile index within a (batch,head); grid.y = head; grid.z = batch.
    q_tile = fx.block_idx.x
    h_idx = fx.block_idx.y
    b_idx = fx.block_idx.z
    kv_head = h_idx // param.gqa_group

    q_start = q_tile * block_m
    n_kv_tiles = (seqlen_kv + block_n - 1) // block_n

    n_kv_tiles = param.n_kv_tiles  # compile-time: seqlen_kv // block_n (validated on host)
    mma_k = param.mma_k

    # ── LDS: only the P bridge buffer is needed (Q/K/V read global->reg directly;
    # V comes pre-transposed on the host as [.., D, Skv]). ────────────────────
    @fx.struct
    class SharedStorage:
        p: fx.Array[elem_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    sP = fx.make_view(storage.p.peek().ptr, fx.make_layout((block_m, block_n), (block_n, 1)))

    # ── per-(batch,head) [S, D] views of the BSHD tensors ─────────────────────
    # Element (b,s,h,d) at b*Sq*Hq*D + s*Hq*D + h*D + d.  q/o slice: base offset
    # b*Sq*Hq*D + h*D + q_start*Hq*D, row-stride Hq*D. k slice uses Hkv/kv_head.
    hq = param.num_heads_q
    hkv = param.num_heads_kv
    q_off = b_idx * seqlen_q * hq * head_dim + h_idx * head_dim + q_start * hq * head_dim
    o_off = q_off
    k_off = b_idx * seqlen_kv * hkv * head_dim + kv_head * head_dim
    # Vt is [B, Hkv, D, Skv] (host-transposed per head): element (b,h,d,s) at
    # b*Hkv*D*Skv + h*D*Skv + d*Skv + s.  This head's [D, Skv] slice:
    vt_off = b_idx * hkv * head_dim * seqlen_kv + kv_head * head_dim * seqlen_kv

    # Per-(batch,head) sub-views wrapped as BUFFER pointers so BufferCopy128b
    # legalizes (a plain make_view over a global ptr is not buffer-backed).
    q_it = fx.rocdl.make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(q)) + fx.Int32(q_off))
    o_it = fx.rocdl.make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(o)) + fx.Int32(o_off))
    k_it = fx.rocdl.make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(k)) + fx.Int32(k_off))
    vt_it = fx.rocdl.make_buffer_ptr(fx.recast_iter(elem_dtype, fx.get_iter(v)) + fx.Int32(vt_off))

    gQ = fx.make_view(q_it, fx.make_layout((block_m, head_dim), (hq * head_dim, 1)))
    gO = fx.make_view(o_it, fx.make_layout((block_m, head_dim), (hq * head_dim, 1)))
    gK = fx.make_view(k_it, fx.make_layout((seqlen_kv, head_dim), (hkv * head_dim, 1)))
    gVt = fx.make_view(vt_it, fx.make_layout((head_dim, seqlen_kv), (seqlen_kv, 1)))

    thr_qk = tiled_mma_qk.thr_slice(tid)
    thr_pv = tiled_mma_pv.thr_slice(tid)

    ca = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    uca = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)

    # Q resident: load once into the GEMM1 A-fragment (reused every KV tile).
    tcA = fx.make_tiled_copy_A(ca, tiled_mma_qk).get_slice(tid)
    frag_Q = thr_qk.make_fragment_A(gQ)
    fx.copy(ca, tcA.partition_S(gQ), tcA.retile(frag_Q))

    # Persistent O accumulator [block_m, head_dim] in registers across the KV loop.
    frag_O = thr_pv.make_fragment_C(gO)
    frag_O.fill(0.0)

    # C-fragment slot layout (MFMA 16x16x32, 2 M-waves x 1 N-wave, block_m=32,
    # block_n=32): 8 slots/lane; slot i -> row (i % npair), col-group (i // npair);
    # slots {r, r+npair} are the two key col-groups of row r. (see plan progress log)
    n_c = fx.size(thr_qk.partition_C(sP).shape).unpack()
    npair = n_c // 2
    m_i = [fx.Float32(float("-inf")) for _ in range_constexpr(npair)]
    l_i = [fx.Float32(0.0) for _ in range_constexpr(npair)]
    n_o = fx.size(frag_O.shape).unpack()
    o_slot_row = [i % npair for i in range_constexpr(n_o)]

    for kv in range_constexpr(n_kv_tiles):
        gK_tile = fx.slice(fx.zipped_divide(gK, (block_n, head_dim)), (None, kv))
        gVt_tile = fx.slice(fx.zipped_divide(gVt, (head_dim, block_n)), (None, kv))

        # GEMM1: S = Q @ K^T
        tcB = fx.make_tiled_copy_B(ca, tiled_mma_qk).get_slice(tid)
        frag_K = thr_qk.make_fragment_B(gK_tile)
        frag_S = thr_qk.make_fragment_C(sP)
        fx.copy(ca, tcB.partition_S(gK_tile), tcB.retile(frag_K))
        frag_S.fill(0.0)
        fx.gemm(tiled_mma_qk, frag_S, frag_Q, frag_K, frag_S)

        for i in range_constexpr(n_c):
            frag_S[i] = frag_S[i] * scale

        # online softmax (flash-attention): running max/sum + O rescale
        corr = [fx.Float32(1.0) for _ in range_constexpr(npair)]
        for r in range_constexpr(npair):
            tile_m = _row_reduce(frag_S[r].maximumf(frag_S[r + npair]), "max")
            m_new = m_i[r].maximumf(tile_m)
            corr[r] = fmath.exp2((m_i[r] - m_new) * fx.Float32(_LOG2E), fastmath=_FM)
            e0 = fmath.exp2((frag_S[r] - m_new) * fx.Float32(_LOG2E), fastmath=_FM)
            e1 = fmath.exp2((frag_S[r + npair] - m_new) * fx.Float32(_LOG2E), fastmath=_FM)
            frag_S[r] = e0
            frag_S[r + npair] = e1
            tile_l = _row_reduce(e0.addf(e1, fastmath=_FM), "sum")
            l_i[r] = l_i[r] * corr[r] + tile_l
            m_i[r] = m_new

        for i in range_constexpr(n_o):
            frag_O[i] = frag_O[i] * corr[o_slot_row[i]]

        # C->A bridge: write P (unnormalized exp) to LDS, read as GEMM2 A
        tr = thr_qk.partition_C(fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0))))
        tc = thr_qk.partition_C(fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1))))
        fx.gpu.barrier()
        for i in range_constexpr(n_c):
            sP[fx.get_scalar(tr[i]), fx.get_scalar(tc[i])] = frag_S[i].to(elem_dtype)
        fx.gpu.barrier()

        # GEMM2: O += P @ V
        tcA2 = fx.make_tiled_copy_A(uca, tiled_mma_pv).get_slice(tid)
        tcB2 = fx.make_tiled_copy_B(ca, tiled_mma_pv).get_slice(tid)
        frag_P = thr_pv.make_fragment_A(sP)
        frag_Vt = thr_pv.make_fragment_B(gVt_tile)
        fx.copy(uca, tcA2.partition_S(sP), tcA2.retile(frag_P))
        fx.copy(ca, tcB2.partition_S(gVt_tile), tcB2.retile(frag_Vt))
        fx.gemm(tiled_mma_pv, frag_O, frag_P, frag_Vt, frag_O)

    # epilogue: O /= l_i (per row), store this q-tile
    for i in range_constexpr(n_o):
        frag_O[i] = frag_O[i] * (fx.Float32(1.0) / l_i[o_slot_row[i]])
    thr_o = thr_pv.partition_C(gO)
    for i in range_constexpr(n_o):
        thr_o[i] = frag_O[i].to(elem_dtype)


@flyc.jit
def launch_flex_attn_gfx950(
    o: fx.Tensor,
    q: fx.Tensor,
    k: fx.Tensor,
    v: fx.Tensor,
    scale: fx.Float32,
    param: FlexAttnParam,
    stream: fx.Stream = fx.Stream(None),
):
    b = fx.Int32(fx.get_scalar(q.shape[0]))
    seqlen_q = fx.Int32(fx.get_scalar(q.shape[1]))
    hq = fx.Int32(fx.get_scalar(q.shape[2]))
    seqlen_kv = fx.Int32(fx.get_scalar(k.shape[1]))

    elem_dtype = _elem_dtype(param.dtype_id)

    # Both GEMMs use the same MFMA atom + wave layout as the validated probe:
    # make_tiled_mma(atom, (m_waves, n_waves, 1)) with no K-permutation tile.
    wave_layout = fx.make_layout(
        (param.m_waves, param.n_waves, 1), (param.n_waves, 1, 0)
    )
    mma_atom_qk = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
    )
    tiled_mma_qk = fx.make_tiled_mma(mma_atom_qk, wave_layout)
    mma_atom_pv = fx.make_mma_atom(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, elem_dtype)
    )
    tiled_mma_pv = fx.make_tiled_mma(mma_atom_pv, wave_layout)

    num_q_tiles = (seqlen_q + param.block_m - 1) // param.block_m

    flex_attn_fwd_gfx950_kernel._known_block_size = [param.block_threads, 1, 1]
    flex_attn_fwd_gfx950_kernel._func.__name__ = make_flex_attn_kernel_name(param)
    flex_attn_fwd_gfx950_kernel(
        o, q, k, v, seqlen_q, seqlen_kv, scale, tiled_mma_qk, tiled_mma_pv, param,
    ).launch(
        grid=(num_q_tiles, hq, b),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def flydsl_flex_attention_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    block_m: int = 32,
    block_n: int = 32,
    stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Dense flash-attention forward on the layout API (gfx950). Phase 0: no mods.

    q/k/v: ``[B, S, H, D]`` (BSHD), bf16/f16. Returns ``[B, Sq, Hq, D]``.
    """
    arch = get_rocm_arch()
    if not arch.startswith("gfx950"):
        raise RuntimeError(f"flex_attention_layout targets gfx950; got {arch!r}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q/k/v must be CUDA tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q/k/v must share dtype")
    if q.dim() != 4:
        raise ValueError(f"q must be 4D [B,S,H,D], got {q.dim()}D")

    dtype_id = FLEX_DTYPE_FP16 if q.dtype is torch.float16 else FLEX_DTYPE_BF16
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {q.dtype}")

    B, Sq, Hq, D = q.shape
    Skv, Hkv = k.shape[1], k.shape[2]
    if num_kv_heads is not None and num_kv_heads != Hkv:
        raise ValueError(f"num_kv_heads {num_kv_heads} != k head count {Hkv}")
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    if stream is None:
        stream = torch.cuda.current_stream()
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    param = make_flex_attn_param(
        seqlen_kv=Skv,
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        head_dim=D,
        num_heads_q=Hq,
        num_heads_kv=Hkv,
        causal=causal,
    )
    # V is pre-transposed on the host to [B, Hkv, D, Skv] (contiguous) so GEMM2's
    # B operand load is a plain K(block_n)-contiguous read. The in-kernel
    # transpose-read (cdna4 LDSReadTrans) is a later perf optimization.
    vt = v.permute(0, 2, 3, 1).contiguous()  # [B, Skv, Hkv, D] -> [B, Hkv, D, Skv]
    launch_flex_attn_gfx950(
        out.contiguous(), q.contiguous(), k.contiguous(), vt,
        fx.Float32(scale), param, stream,
    )
    return out
