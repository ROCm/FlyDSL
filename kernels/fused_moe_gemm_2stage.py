# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# ============================================================================
# FUSED-OPERATOR stage1 GEMM for the MoE megakernel ONLY (`compile_fused_moe_gemm1`,
# module-name prefix `mfma_fmoe1_...`).  This is the dispatch⊕GEMM builder consumed by
# `FusedMoEMegaStage1` (single launch).  The legacy baseline GEMM lives separately in
# `mixed_moe_gemm_2stage.py` (used by the ATOM perf baseline in the bench).
# ============================================================================

"""MoE stage-1 fused dispatch⊕GEMM kernel builder (FlyDSL MFMA, CDNA4 / MI355X).

`compile_fused_moe_gemm1` builds the persistent sparse-tile group-GEMM (gate/up + silu)
with the optional in-kernel dispatch prologue (`fuse_dispatch`):
  * ""          : plain GEMM (no in-kernel dispatch).
  * "fixedslot" : decode strict-phase (fixed-slot dispatch prologue, then persistent GEMM).
  * "handshake" : prefill producer/consumer overlap (block0-self-sufficient handshake +
                  per-expert payload_done gate).
See docs/moe_stage1_mega.md.

Mixed-precision support (a_dtype x b_dtype):
- fp8 x fp8, fp8 x fp4 (A8W4 on gfx950), fp4 x fp4,
  fp16 x fp16, int8 x int4, ...

A8W4 path is selected by `a_dtype='fp8', b_dtype='fp4'` plus
`gate_mode=GateMode.INTERLEAVE` + `a_scale_one=True` in stage1.
"""

import functools
import os
import types
from enum import Enum

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref, scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.kernels_common import validate_moe_dtypes
from kernels.layout_utils import crd2idx, idx2crd
from kernels.layout_utils import get as layout_get
from kernels.mfma_epilogues import c_shuffle_epilog
from kernels.mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    buffer_copy_gmem16_dwordx4,
    lds_store_4b_xor16,
    lds_store_8b_xor16,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    swizzle_xor16,
    tile_chunk_coord_i32,
)
# ---- Low-level cross-PE / atomic / fence device primitives for the FUSED megakernel dispatch
# prologue (fuse_dispatch != "").  Folded in here so the megakernel is self-contained; exposed via
# the `_epk` namespace so the prologue call-sites read `_epk.atomic_add_agent(...)` etc.  `_epkT` is
# the CALLABLE type factory (flydsl.expr.T, distinct from this file's flydsl.expr.typing.T).
from flydsl.expr import T as _epkT


def _epk_to_i64(v):
    return arith.extui(_epkT.i64(), arith.unwrap(v))


def _epk_to_ptr_global(v):
    return llvm.IntToPtrOp(llvm.PointerType.get(address_space=1), arith.unwrap(v)).result


def _epk_store_i32_system(addr_i64, offset, val):
    base = arith.unwrap(addr_i64)
    off = arith.unwrap(offset)
    val_ = arith.unwrap(val)
    _i64 = ir.IntegerType.get_signless(64)
    _i32 = ir.IntegerType.get_signless(32)
    _nuw = ir.Attribute.parse("#llvm.overflow<none>")
    off64 = llvm.ZExtOp(_i64, off).res if off.type == _i32 else off
    byte_off = llvm.MulOp(off64, llvm.ConstantOp(_i64, ir.IntegerAttr.get(_i64, 4)).result, _nuw).result
    addr = llvm.AddOp(base, byte_off, _nuw).result
    gptr = llvm.IntToPtrOp(llvm.PointerType.get(address_space=1), addr).result
    llvm.StoreOp(val_, gptr, alignment=4, ordering=llvm.AtomicOrdering.release, syncscope="one-as")


def _epk_fence_system_acquire():
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="one-as")


def _epk_fence_system_release():
    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="one-as")


# Agent (device) scope: only an L2 flush to device coherence (cheaper than the system/xGMI flush);
# the correct scope for same-GPU cross-block visibility.
def _epk_fence_agent_acquire():
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="agent-one-as")


def _epk_fence_agent_release():
    llvm.FenceOp(llvm.AtomicOrdering.release, syncscope="agent-one-as")


def _epk_atomic_add_agent(addr_i64, val):
    ptr = _epk_to_ptr_global(addr_i64)
    return llvm.AtomicRMWOp(llvm.AtomicBinOp.add, ptr, arith.unwrap(val),
                            llvm.AtomicOrdering.monotonic, syncscope="agent").res


def _epk_atomic_add_system(addr_i64, val):
    ptr = _epk_to_ptr_global(addr_i64)
    return llvm.AtomicRMWOp(llvm.AtomicBinOp.add, ptr, arith.unwrap(val),
                            llvm.AtomicOrdering.monotonic, syncscope="one-as").res


def _epk_readlane0(val_i32):
    return rocdl.readlane(_epkT.i32(), arith.unwrap(val_i32), arith.unwrap(arith.constant(0)))


_epk = types.SimpleNamespace(
    _to_i64=_epk_to_i64, store_i32_system=_epk_store_i32_system,
    fence_system_acquire=_epk_fence_system_acquire, fence_system_release=_epk_fence_system_release,
    fence_agent_acquire=_epk_fence_agent_acquire, fence_agent_release=_epk_fence_agent_release,
    atomic_add_agent=_epk_atomic_add_agent, atomic_add_system=_epk_atomic_add_system,
    _readlane0=_epk_readlane0,
)


class GateMode(str, Enum):
    """Gate/Up computation strategy for stage1 GEMM.

    SEPARATED:      Two separate B-tile streams (gate + up), default mode.
    MOCK_GATE_ONLY: Single B-tile stream over full [0, 2*inter_dim), simulates
                    gate-only by doubling grid X on top of SEPARATED layout.
                    Requires split-K (k_batch>1).  NOT true gate-only.
    GATE_ONLY:      Reserved for future true gate-only implementation.
    INTERLEAVE:     Weight rows interleave gate/up (gate[0], up[0], gate[1], ...).
                    pack_N=2 routes even/odd N subtiles.  NOT tied to split-K.
    """

    SEPARATED = "separated"
    MOCK_GATE_ONLY = "mock_gate_only"
    GATE_ONLY = "gate_only"
    INTERLEAVE = "interleave"


def _fence_system_acquire():
    """Acquire fence (one-as) -- pairs with the dispatch's release-store of
    payload_done so the gated GEMM observes freshly-landed payload (not stale L2)."""
    llvm.FenceOp(llvm.AtomicOrdering.acquire, syncscope="one-as")


def _barrier(vmcnt=63, lgkmcnt=63):
    """Emit s_waitcnt + s_barrier via inline asm.

    Bypasses LLVM SIInsertWaitcnts which would insert a conservative
    s_waitcnt vmcnt(0) lgkmcnt(0) before every S_BARRIER MI.
    """
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            wc.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(wc))
    parts.append("s_barrier")
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


@functools.lru_cache(maxsize=None)
def compile_fused_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    act: str = "silu",
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 1,
    sparse_tiles: bool = False,
    use_async_copy: bool = False,
    waves_per_eu: int = 4,
    k_batch: int = 1,
    b_nt: int = 0,
    gate_mode: GateMode = GateMode.SEPARATED,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    contiguous_io: bool = False,
    dedup_gather: bool = False,
    diag_scale_valu: bool = False,
    raw_a_scale: bool = False,
    overlap_gate: bool = False,
    rank: int = 0,
    experts_per_rank: int = 0,
    # ---- FUSED megakernel (dispatch ⊕ GEMM in one launch) ----------------------------------
    # fuse_dispatch: "" (default, shipping path -- byte-identical, prologue not traced) |
    #   "fixedslot" (decode strict-phase: handshake-free fixed-slot dispatch prologue, then a
    #   grid barrier, then the existing persistent sparse-tile GEMM, all in ONE kernel).
    # The prologue writes the SAME tile metadata (tile_row_base/sorted_expert_ids/num_valid) the
    # GEMM already consumes; A-input (arg_x) is the dispatch's local rx_em buffer.  All fuse_* are
    # host constants threaded only when fuse_dispatch != "".
    fuse_dispatch: str = "",
    fuse_npes: int = 0,
    fuse_topk: int = 0,
    fuse_cap: int = 0,
    fuse_mtpr: int = 0,
    fuse_scale_dim: int = 0,
    fuse_scale_type_size: int = 1,
):
    """Compile stage1 kernel (gate+up with silu/swiglu).

    ``contiguous_io`` (default False keeps the exact baseline behavior): when True the
    activation is assumed **physically expert-major contiguous** (DeepGEMM m-grouped), so the
    per-row ``sorted_token_ids`` gather is dropped (A row r = physical row ``bx_m+row_local``)
    and the output is written **expert-major contiguous** ``out[row]`` instead of scattered to
    ``out[token*topk+s]``.  Frees the per-row index registers + the ``lds_tid`` cache.  The
    caller passes ``out`` shaped ``[num_valid, 1, inter]`` and may pass a dummy ``sorted_token_ids``.

    GEMM: act(X @ W_gate.T, X @ W_up.T) -> [tokens*topk, inter_dim]
    Direct store (no atomic).  When k_batch>1 (split-K), each CTA
    computes a K-slice and atomically adds gate/up partials.
    Note: persist_m=1 (no persistence) is optimal for stage1 because K=model_dim
    is large, so each CTA is already compute-heavy. persist_m>1 serializes M blocks
    that the GPU can process in parallel.

    gate_mode controls the gate/up computation strategy — see GateMode enum.
    """
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    _state = {}

    validate_moe_dtypes(a_dtype, b_dtype)

    is_f16_a = a_dtype == "fp16"
    is_f16_b = b_dtype == "fp16"
    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"

    sort_block_m = max(32, tile_m)
    num_waves = min(4, tile_n // 32)
    total_threads = num_waves * 64
    pack_M = 1 if tile_m < 32 else 2
    n_per_wave = tile_n // num_waves
    pack_N = min(2, n_per_wave // 16)
    pack_K = 2
    scale_mn_pack = 2
    elem_bytes = 1
    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)
    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    if (tile_k_bytes % 64) != 0:
        raise ValueError(f"tile_k_bytes must be divisible by 64, got {tile_k_bytes}")

    out_s = str(out_dtype).strip().lower()
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    is_int4 = b_dtype == "int4"
    is_int8 = False

    def _x_elem_type():
        if is_f4_b:
            return T.f8 if is_f8_a else T.i8
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    def _w_elem_type():
        if is_f4_b:
            return T.i8
        return T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)

    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    mock_gate_only = gate_mode is GateMode.MOCK_GATE_ONLY
    gate_up_interleave = gate_mode is GateMode.INTERLEAVE

    # Padding semantics: model_dim and inter_dim INCLUDE padding.
    #   model_dim = model_dim_true + model_dim_pad   (K direction)
    #   inter_dim = inter_dim_true + inter_dim_pad   (N direction)
    # Tensor sizes use the padded dimensions (inter_dim, model_dim).
    # Padding only affects kernel internal logic and grid computation.
    _inter_dim_valid = inter_dim - inter_dim_pad

    # Split-K validation
    _is_splitk = k_batch > 1
    if mock_gate_only and not _is_splitk:
        raise ValueError("mock_gate_only requires k_batch > 1 (split-K)")
    if _is_splitk:
        _k_per_batch = model_dim // k_batch
        assert model_dim % k_batch == 0, f"model_dim={model_dim} not divisible by k_batch={k_batch}"
        assert _k_per_batch % tile_k == 0, f"K_per_batch={_k_per_batch} not divisible by tile_k={tile_k}"

        out_dtype = "bf16"
    else:
        _k_per_batch = model_dim
    _k_dim = _k_per_batch

    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(f"tile_m*tile_k*elem_bytes must be divisible by {total_threads}")
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _use_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _use_lds128 else 8
    lds_stride = tile_k + pad_k

    if use_cshuffle_epilog is None:
        _use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE1_CSHUFFLE", "1") in (
            "1",
            "true",
            "True",
            "YES",
            "yes",
        )
    else:
        _use_cshuffle_epilog = bool(use_cshuffle_epilog)

    _need_fp4 = out_dtype == "fp4"
    _need_fp8 = out_dtype == "fp8"
    _need_quant = _need_fp4 or _need_fp8
    _need_sort = _need_quant

    if _need_quant:
        _use_cshuffle_epilog = True

    _fp4q_tag = "_fp4q" if _need_fp4 else ""
    _fp8q_tag = "_fp8q" if _need_fp8 else ""
    _sort_tag = "_sort" if _need_sort else ""
    _async_tag = "_async" if use_async_copy else ""
    _sk_tag = f"_sk{k_batch}" if _is_splitk else ""
    _go_tag = "_go" if mock_gate_only else ""
    _gui_tag = "_gui" if gate_up_interleave else ""
    _as1_tag = "_as1" if a_scale_one else ""
    _xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
    # dedup_gather: expert-major OUTPUT (like contiguous_io) but A-input is gathered from the
    # deduped per-source-token buffer via srcmap row = sorted_token_ids[r] & 0xFFFFFF.  Kills the
    # fp8 embedding write-amp (Lever B) without an expand kernel.  Implies contiguous_io output.
    if dedup_gather:
        contiguous_io = True
    _ci_tag = "_ci" if contiguous_io else ""
    _dg_tag = "_dg" if dedup_gather else ""
    _dsv_tag = "_dsv" if diag_scale_valu else ""
    _ras_tag = "_ras" if raw_a_scale else ""
    _og_tag = f"_og{rank}x{experts_per_rank}" if overlap_gate else ""
    # CK-style hot-loop interleave (DEFAULT mode 3): prescribe sched_group_barrier so this
    # phase's B(VMEM)/A(DS) loads interleave into the MFMA stream at R MFMAs/load, hiding
    # load latency under compute (lifts MFMA util; ~2-3% GEMM, ~1-2% e2e at large bs, bit-exact).
    # Deployed CK-style hot-loop schedule: mode 3 (mixed A/B interleave), rate 3 MFMAs per
    # interleaved load (lifts MFMA util; ~2-3% GEMM, ~1-2% e2e at large bs, bit-exact).
    _isched = 3
    _ck_rate = 3
    _is_tag = f"_is{_isched}_r{_ck_rate}" if _isched else ""
    # The megakernel GEMM is ALWAYS persistent round-robin: grid_y CTAs each take a strided chunk
    # of M-tiles (bounded by device num_valid); sparse_tiles carries the per-tile row base.
    _persistent = True
    from aiter.jit.utils.chip_info import get_cu_num
    _cu_num = get_cu_num()
    _pm_tag = f"persist_cu{_cu_num}"
    _spt_tag = "_spt" if sparse_tiles else ""
    # ---- FUSED megakernel dispatch-prologue (host consts; only used when fuse_dispatch != "") ----
    #   "fixedslot" : decode strict-phase (fixed-slot dispatch prologue, then persistent GEMM).
    #   "handshake" : prefill PRODUCER/CONSUMER OVERLAP -- block0-self-sufficient counts-first
    #       handshake, then producer blocks (extra grid-x columns, by>=gx) write payload + signal
    #       payload_done, while consumer blocks (by<gx) run the persistent GEMM gated per-expert.
    _fuse = fuse_dispatch in ("fixedslot", "handshake")
    _fuse_fs = fuse_dispatch == "fixedslot"
    _fuse_hs = fuse_dispatch == "handshake"
    # DIAG only (coverage verification): count GEMM-body execs (per valid tile) into
    # addr_payload_done[0] -- fixedslot ONLY (its addr_payload_done is spare scratch); handshake's
    # addr_payload_done IS the live per-expert gate, so handshake coverage is verified by the
    # srcmap key-set bit-exact oracle instead (a dropped tile leaves rows uncomputed => mismatch).
    _fuse_tilecount = _fuse_fs and os.environ.get("FUSED_MEGA_TILECOUNT", "0") == "1"
    _fz_gy = 0  # default so the xcd-swizzle closure ref below is bound even for non-fused builds
    if _fuse:
        assert sparse_tiles, "fuse_dispatch requires sparse_tiles"
        assert _persistent, "fuse_dispatch requires a persistent grid"
        _fz_npes = int(fuse_npes)
        _fz_epr = int(experts_per_rank)
        _fz_k = int(fuse_topk)
        _fz_cap = int(fuse_cap)
        _fz_mtpr = int(fuse_mtpr)
        _fz_rank = int(rank)
        _fz_tile_m = int(sort_block_m)              # dispatch tile granularity == GEMM M-tile
        assert _fz_cap % _fz_tile_m == 0, f"fuse_cap({_fz_cap}) %% tile_m({_fz_tile_m}) != 0"
        _fz_total_experts = _fz_npes * _fz_epr
        _fz_sentinel = _fz_total_experts
        if is_f4_a:
            _fz_n_i32 = model_dim // 8
            _fz_nbytes = model_dim // 2
        else:                                       # fp8 activation
            _fz_n_i32 = model_dim // 4
            _fz_nbytes = model_dim
        _fz_scale_bytes = int(fuse_scale_dim) * int(fuse_scale_type_size)
        _fz_scale_n_i32 = (_fz_scale_bytes + 3) // 4 if _fz_scale_bytes > 0 else 0
        _fz_enable_scales = _fz_scale_bytes > 0
        _fz_safe_end_i32 = (_fz_n_i32 // 512) * 512
        # Co-resident grid: gx (N-tiles) * gy <= cu_num so the in-kernel atomic-epoch grid
        # barrier (dispatch prologue) is dead-lock-free (every block resident, >=1 block/CU).
        # gx MUST mirror the host launcher's gx formula EXACTLY (the facade passes
        # i32_inter_in = 2*inter_dim, so use that here too).
        _fz_inter_in = 2 * inter_dim
        _fz_pad_total = 2 * inter_dim_pad
        if mock_gate_only or gate_up_interleave:
            _fz_gx_static = (_fz_inter_in - _fz_pad_total + tile_n - 1) // tile_n
        else:
            _fz_gx_static = (_fz_inter_in - _fz_pad_total + 2 * tile_n - 1) // tile_n // 2
        # gy is the persistent (M-tile round-robin) dimension.  fixedslot (decode) is co-resident
        # (total = gx*gy <= cu_num) so its in-prologue arrival grid-barrier is deadlock-free;
        # handshake (prefill) has zero grid barriers in the producer/consumer phases and may
        # oversubscribe.  Env override (FUSED_MEGA_GY) forces gy for sweeps.
        if _fuse_hs:
            # PREFILL persistent producer/consumer overlap (docs/moe_stage1_mega.md §9).
            # The redesigned prologue is BLOCK0-SELF-SUFFICIENT (one block does the whole
            # counts-first handshake; producers/consumers gate ONLY on a monotonic meta_flag +
            # per-expert payload_done, never on each other) => the producer/consumer phases carry
            # ZERO grid barriers, so the grid may OVERSUBSCRIBE (total blocks > cu_num).  HW
            # wave-schedules producers (xGMI payload writes) ‖ consumers (MFMA); a producer's
            # store latency is hidden under a co-resident consumer's compute, and a retired
            # producer's CU is back-filled by a consumer wave.  (block0-self-sufficient: it never
            # waits on another block => no co-residency requirement, no deadlock.)
            #   np_cols  = MINIMUM producer columns that saturate the xGMI write (NOT the old
            #              gx/2 which halved consumer parallelism, §9.2 R1).  Sweep {8,16,32,64}.
            #   gy       = round(alpha * cu_num / (gx + np_cols))  => total = alpha*cu_num blocks.
            # DEFAULT alpha=1 (total==cu_num, CO-RESIDENT) validates the barrier-removal restructure
            # deadlock-free; set FUSED_MEGA_ALPHA=2 for the §9.5 oversubscribe perf mode (HW waves
            # overlap producer xGMI ‖ consumer MFMA).  FUSED_MEGA_GY overrides gy directly.
            _fz_np_cols = int(os.environ.get("FUSED_MEGA_NP_COLS", "0")) or 8
            _fz_alpha = max(1, int(os.environ.get("FUSED_MEGA_ALPHA", "1")))
            _fz_gy = max(1, (_fz_alpha * _cu_num) // max(1, _fz_gx_static + _fz_np_cols))
            _fz_gy_env = int(os.environ.get("FUSED_MEGA_GY", "0"))
            if _fz_gy_env > 0:
                _fz_gy = _fz_gy_env
        else:
            # rocprofv3: the mega is SYNC-bound (5-6% occupancy); the earlier 128-block cap left
            # HALF the GPU idle.  Default to gy = cu_num//gx so total ~= cu_num blocks (use all CUs,
            # max GEMM M-parallelism for the compute that exists).  Env override for sweeps.
            _fz_np_cols = 0
            _fz_gy_cap = max(1, _cu_num // max(1, _fz_gx_static))
            _fz_gy_env = int(os.environ.get("FUSED_MEGA_GY", "0"))
            _fz_gy = _fz_gy_env if _fz_gy_env > 0 else _fz_gy_cap
    _fuse_tag = f"_fz{fuse_dispatch}{fuse_npes}x{fuse_cap}x{fuse_topk}" if _fuse else ""
    module_name = (
        f"mfma_fmoe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}"
        f"_t{tile_m}x{tile_n}x{tile_k}_{_pm_tag}{_fp4q_tag}{_fp8q_tag}{_sort_tag}{_async_tag}{_sk_tag}{_go_tag}{_gui_tag}{_as1_tag}{_xcd_tag}{_ci_tag}{_dg_tag}{_dsv_tag}{_ras_tag}{_spt_tag}{_og_tag}{_is_tag}{_fuse_tag}_v33"
    ).replace("-", "_")

    # -- LDS sizing --
    _cshuffle_elem_bytes = 4 if _need_quant else (4 if out_is_f32 else 2)
    # fp4 A is packed in LDS (a_elem_vec_pack=2) when async-copy is on, so the
    # ping/pong A buffers only need the packed stride.  Stage1 previously sized
    # them with the *unpacked* lds_stride (2x A-LDS for a4w4), which capped
    # occupancy at 2 wg/CU.  Mirror stage2 and allocate at the effective stride.
    _eff_lds_stride_alloc = (
        lds_stride // a_elem_vec_pack
        if (use_async_copy and a_elem_vec_pack > 1)
        else lds_stride
    )
    _single_x_bytes = int(tile_m) * int(_eff_lds_stride_alloc) * int(a_elem_bytes)
    lds_out_bytes = _cshuffle_elem_bytes * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    lds_tid_bytes = int(tile_m) * 4
    _input_elems = _single_x_bytes if a_elem_bytes == 1 else (_single_x_bytes // 2)

    # Determine whether we need wave-group split for lds_out.
    # Standard layout: pong = max(input, lds_out) + tid, ping = input.
    # When this overflows, split lds_out into two halves across pong & ping.
    _GLOBAL_ALIGN = 1024
    _std_pong = max(_single_x_bytes, lds_out_bytes) + lds_tid_bytes
    _std_ping = _single_x_bytes
    _std_pong_aligned = allocator_pong._align(_std_pong, 128)
    _std_total = allocator_pong._align(_std_pong_aligned, _GLOBAL_ALIGN) + allocator_pong._align(_std_ping, 128)
    _lds_limit = {"gfx950": 163840, "gfx942": 65536}.get(gpu_arch, 0)

    _split_lds_out = _lds_limit > 0 and lds_out_bytes > 0 and _std_total > _lds_limit and num_waves >= 2

    if _split_lds_out:
        _half_out_bytes = _cshuffle_elem_bytes * int(tile_m) * (int(tile_n) // 2)
        _pong_buffer_bytes = max(_single_x_bytes, _half_out_bytes)
        _ping_buffer_bytes = max(_single_x_bytes, _half_out_bytes)
    else:
        _pong_buffer_bytes = max(_single_x_bytes, lds_out_bytes)
        _ping_buffer_bytes = _single_x_bytes

    def x_lds_elem():
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _pong_buffer_bytes
    _lds_tid_offset_pong = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = _lds_tid_offset_pong + lds_tid_bytes

    # raw_a_scale: stage this output tile's FULL A activation-scale into a small, single-buffered
    # LDS region ONCE in the per-tile prologue (outside the K-loop), so the K-loop reads it
    # coalesced from LDS instead of re-loading it uncoalesced from the row-major global scale_em
    # every K-iteration.  Allocated from the pong arena BEFORE the waves_per_eu LDS padding below,
    # so it is absorbed into the already-reserved occupancy slack -> total LDS (hence wave
    # occupancy) is unchanged.  Only active on the raw_a_scale path (pre-swizzled path untouched).
    _raw_a_scale_lds = bool(raw_a_scale and not a_scale_one)
    if _raw_a_scale_lds:
        _raw_sni_lds = model_dim // 128                      # i32 scale cols per scale_em row
        _scale_lds_rows = ((tile_m // 16) // pack_M) * 32    # m_repeat_packed * 32 rows
        _scale_lds_n_i32 = _scale_lds_rows * _raw_sni_lds
        _scale_lds_offset_pong = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = _scale_lds_offset_pong + _scale_lds_n_i32 * 4
        # cooperative-copy tiling: largest vec width in {4,2,1} that evenly tiles the region
        # across all threads (coalesced i32 dword[x4/x2] copies).
        _sc_cp_vec = 4
        while _sc_cp_vec > 1 and (_scale_lds_n_i32 % (total_threads * _sc_cp_vec)) != 0:
            _sc_cp_vec //= 2
        assert _scale_lds_n_i32 % (total_threads * _sc_cp_vec) == 0, (
            f"raw_a_scale LDS staging: {_scale_lds_n_i32} i32 not tileable by "
            f"{total_threads} threads x {_sc_cp_vec}")
        _sc_cp_iters = _scale_lds_n_i32 // (total_threads * _sc_cp_vec)
    else:
        _raw_sni_lds = 0
        _scale_lds_n_i32 = 0
        _scale_lds_offset_pong = 0
        _sc_cp_vec = 1
        _sc_cp_iters = 0

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes

    # Real per-block DATA LDS (pong+ping, 128B-aligned) BEFORE the waves_per_eu occupancy floor is
    # applied below.  For raw_a_scale this INCLUDES the scale-LDS staging region (else it does not);
    # the facade compares the two variants' values against the per-CU LDS limit to decide whether the
    # scale-LDS costs a thread-block of occupancy -> whether to external-pre-swizzle for prefill.
    _lds_data_bytes = (allocator_pong._align(allocator_pong.ptr, 128)
                       + allocator_ping._align(allocator_ping.ptr, 128))
    _lds_scale_bytes = _scale_lds_n_i32 * 4

    if waves_per_eu is not None and waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        _ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        _cur_lds = _pong_sz + _ping_sz
        if _cur_lds < _min_lds:
            allocator_ping.ptr += _min_lds - _cur_lds

    # Real per-block LDS AFTER the waves_per_eu floor -- this is what actually bounds LDS occupancy.
    # At small tiles the scale-LDS is absorbed by the floor (this total is identical with/without it,
    # so pre-swizzle would NOT raise occupancy); at large tiles the floor is slack and the scale-LDS
    # genuinely drops blocks/CU (e.g. 3->2).  The facade compares this total (raw vs pre-swizzle) to
    # gate the pre-swizzle precisely where it lifts occupancy.
    _lds_total_bytes = (allocator_pong._align(allocator_pong.ptr, 128)
                        + allocator_ping._align(allocator_ping.ptr, 128))

    kpack_bytes = 8 if is_int4 else 16
    out_elem_bytes = 4 if out_is_f32 else 2

    _e_vec_s1 = min(tile_n // 32, 8)
    if _need_quant:
        _e_vec_s1 = max(2, _e_vec_s1)
    # EXPERIMENT: force the CShuffle output vector width (must be pow2 in {2,4,8}).  Lets non-pow2
    # tile_n/32 (e.g. tile_n=192 -> auto e_vec=6 = illegal) use a smaller legal e_vec so gx=inter/192
    # can hit an XCD-aligned value (gx|cu).  Both _e_vec_s1 (cshuffle nlane) and the store e_vec read it.
    _evec_ovr = int(os.environ.get("FUSED_MEGA_EVEC", "0"))
    if _evec_ovr in (2, 4, 8):
        _e_vec_s1 = _evec_ovr
    _num_threads_per_quant_blk_s1 = 32 // _e_vec_s1
    _shuffle_dists_s1 = []
    _sh_val = 1
    while _sh_val < _num_threads_per_quant_blk_s1:
        _shuffle_dists_s1.append(_sh_val)
        _sh_val *= 2
    _num_shuffle_steps_s1 = len(_shuffle_dists_s1)

    # ---- Unified pipeline schedule (outside @flyc.kernel) ----
    # Each scheduling phase is a dict:
    #   mfma:      [(k_idx, mi_idx, ikxdl, imxdl, asv_idx), ...]
    #   a_reads:   [(k, mi), ...]       # A ds_read subtiles
    #   b_loads:   [('gate'/'up', ku, ni), ...]  # B VMEM loads
    #   has_scale: bool                  # A/B scale VMEM loads
    _pipe_m_repeat = tile_m // 16
    _pipe_k_unroll = tile_k_bytes // 128
    _pipe_k_unroll_packed = _pipe_k_unroll // pack_K
    _pipe_m_repeat_packed = _pipe_m_repeat // pack_M
    _pipe_num_acc_n = n_per_wave // 16

    # A ds_read groups: group by mi (same mi, all k values together)
    _pipe_a_groups = []
    for _mi in range(_pipe_m_repeat):
        _grp = []
        for _k in range(_pipe_k_unroll):
            _grp.append((_k, _mi))
            if len(_grp) == 2:
                _pipe_a_groups.append(_grp)
                _grp = []
        if _grp:
            _pipe_a_groups.append(_grp)

    # B VMEM loads: individual gate/up loads
    _pipe_b_loads = []
    for ku in range(_pipe_k_unroll):
        for ni in range(_pipe_num_acc_n):
            _pipe_b_loads.append(("gate", ku, ni))
            if not mock_gate_only and not gate_up_interleave:
                _pipe_b_loads.append(("up", ku, ni))

    # MFMA order: B-major (fix B, cycle all A tiles before next B)
    # Each entry: one (k, ni) pair; the compute function loops over all mi.
    # This keeps B operands (from VMEM) fixed while cycling A (from LDS, no wait).
    _pipe_num_acc_n_packed = _pipe_num_acc_n // pack_N
    _pipe_all_mfma = []
    for _ku128 in range(_pipe_k_unroll_packed):
        for _ni_packed in range(_pipe_num_acc_n_packed):
            for _ikxdl in range(pack_K):
                for _inxdl in range(pack_N):
                    _k_idx = _ku128 * pack_K + _ikxdl
                    _ni_idx = _ni_packed * pack_N + _inxdl
                    _pipe_all_mfma.append((_k_idx, _ni_idx, _ikxdl, _inxdl, _ku128))

    # Group MFMAs per scheduling phase (wider M -> more MFMAs per phase)
    _pipe_mfma_per_phase = max(1, len(_pipe_all_mfma) // 4)
    _pipe_n_phases = len(_pipe_all_mfma) // _pipe_mfma_per_phase

    # Build unified phase descriptors
    _a_groups_per_phase = (len(_pipe_a_groups) + _pipe_n_phases - 1) // _pipe_n_phases
    _pipe_phases = []
    _mfma_i = 0
    _a_i = 0
    for _p in range(_pipe_n_phases):
        _a_reads = []
        for _ in range(_a_groups_per_phase):
            if _a_i < len(_pipe_a_groups):
                _a_reads.extend(_pipe_a_groups[_a_i])
                _a_i += 1
        _phase = {
            "mfma": _pipe_all_mfma[_mfma_i : _mfma_i + _pipe_mfma_per_phase],
            "a_reads": _a_reads,
            "b_loads": [],
            "has_scale": (_p == 0),
        }
        _mfma_i += _pipe_mfma_per_phase
        _pipe_phases.append(_phase)

    # Distribute B loads evenly across phases 1..n-1 (phase 0 has scales)
    _bi = 0
    for _p in range(1, _pipe_n_phases):
        _rem_b = len(_pipe_b_loads) - _bi
        _rem_p = _pipe_n_phases - _p
        _n_b = (_rem_b + _rem_p - 1) // _rem_p if _rem_p > 0 else 0
        for _ in range(_n_b):
            if _bi < len(_pipe_b_loads):
                _pipe_phases[_p]["b_loads"].append(_pipe_b_loads[_bi])
                _bi += 1

    # Extract flat lists for kernel access (avoids dict access in AST rewriter)
    _pp_mfma = [p["mfma"] for p in _pipe_phases]
    _pp_a_reads = [p["a_reads"] for p in _pipe_phases]
    _pp_b_loads = [p["b_loads"] for p in _pipe_phases]
    _pp_has_scale = [p["has_scale"] for p in _pipe_phases]

    fp4_ratio = 2 if a_dtype == "fp4" else 1
    gui_ratio = 1 if gate_up_interleave else 2
    _vmcnt_before_barrier = tile_m // 32 // fp4_ratio + tile_n // 32 * gui_ratio

    if True:

        @flyc.kernel
        def moe_gemm1(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            arg_bias: fx.Tensor,
            arg_out_scale_sorted: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
            addr_payload_done: fx.Int64,   # [epr] i32 per-expert post-write counter (overlap gate)
            addr_expected_real: fx.Int64,  # [epr] i32 per-expert expected real count (overlap gate)
            addr_disp: fx.Int64,           # FUSED: ptr to FIXED-pointer table (op bufs + p2p; built once)
            i32_cur_tok: fx.Int32,         # FUSED: this-rank token count this launch
            addr_in_tok: fx.Int64,         # FUSED: per-step input pointers passed as SCALAR launch args
            addr_in_idx: fx.Int64,         # (NOT via the table) so CUDAGraph capture bakes the graph-
            addr_in_wts: fx.Int64,         # stable quant-output addresses -> no illegal in-capture H2D.
            addr_in_sc: fx.Int64,
        ):

            # ============================================================================
            # FUSED megakernel — dispatch PROLOGUE (decode, handshake-free fixed-slot).
            # Ported from make_lowlatency_dispatch_kernel (unified, no swizzle).  Runs on the
            # SAME co-resident persistent grid as the GEMM: flat block id over (gridDim.x N-tiles,
            # gridDim.y persistent), warps = num_waves.  Writes peers' expert-major buffers, one
            # cross-PE done-barrier, then a compact post-pass emitting tile_row_base /
            # sorted_expert_ids / num_valid (the SAME tensors the GEMM below consumes) + folds the
            # per-expert count reset.  A final grid barrier broadcasts the metadata to all blocks.
            # Guarded by const_expr(_fuse_fs): traced ONLY for the decode fixed-slot scheme
            # (byte-identical when not fused / handshake).
            # ============================================================================
            if const_expr(_fuse_fs):
                _crfa = buffer_ops.create_buffer_resource_from_addr
                _rdisp = _crfa(addr_disp)

                def _dp(_i):
                    return buffer_ops.buffer_load(_rdisp, arith.constant(_i), vec_width=1, dtype=T.i64)

                _a_tok = addr_in_tok; _a_idx = addr_in_idx; _a_wts = addr_in_wts; _a_sc = addr_in_sc
                _a_gb1 = _dp(4); _a_run = _dp(5); _a_done2 = _dp(6); _a_cnt = _dp(7)
                _p_rx = _dp(8); _p_sc = _dp(9); _p_idx = _dp(10); _p_wts = _dp(11)
                _p_sm = _dp(12); _p_run = _dp(13); _p_done2 = _dp(14)
                _a_se = _dp(15); _a_trb = _dp(16); _a_nv = _dp(17)
                _a_meta = _dp(18)   # metadata-ready flag (monotonic; block0 release / consumers acquire)

                _tid = fx.thread_idx.x
                _lane = _tid & 63
                _warp = _tid >> 6
                _gdx = fx.grid_dim.x
                _flat = fx.block_idx.y * _gdx + fx.block_idx.x
                _gwid = _flat * num_waves + _warp
                _nblk = _gdx * fx.grid_dim.y
                _gwn = _nblk * num_waves
                _wl = i32_cur_tok * arith.constant(_fz_k)
                _c_epr = arith.constant(_fz_epr)
                _c_cap = arith.constant(_fz_cap)
                _r_idx = _crfa(_a_idx)
                _r_wts = _crfa(_a_wts)

                # NOTE: no initial GB1 — payload writes are independent (peer atomics), so they need
                # no "all blocks started" sync.  The launch epoch (for the cross-PE done signal) is
                # derived at the FIRST grid barrier (the post-write barrier below).  This removes one
                # of three grid barriers (decode overhead is dominated by grid-barrier cost).
                _bn = _epk._to_i64(_nblk)

                # ---- write: each (token,expert) -> fixed slot le*cap + atomic(running[le]) ----
                for _wk in range(_gwid, _wl, _gwn):
                    _src_tok = _wk // arith.constant(_fz_k)
                    _k_slot = _wk % arith.constant(_fz_k)
                    _expert = buffer_ops.buffer_load(_r_idx, _wk, vec_width=1, dtype=T.i32)
                    _is_valid = arith.cmpi(CmpIPredicate.ult, _expert, arith.constant(_fz_total_experts))
                    _dest_pe = _expert // _c_epr
                    _le = _expert % _c_epr
                    _off_l0 = arith.constant(0)
                    if _lane == 0:
                        if _is_valid:
                            _run_remote = buffer_ops.buffer_load(_crfa(_p_run), _dest_pe, vec_width=1, dtype=T.i64)
                            _off_l0 = _epk.atomic_add_system(_run_remote + _epk._to_i64(_le) * 4, arith.constant(1))
                    _off = _epk._readlane0(_off_l0)
                    _in_range = arith.cmpi(CmpIPredicate.ult, _off, _c_cap)
                    _do_pub = arith.select(_is_valid, _in_range, _is_valid)
                    _slot = _le * _c_cap + _off

                    if _lane == 0:
                        if _do_pub:
                            _wt_val = buffer_ops.buffer_load(_r_wts, _wk, vec_width=1, dtype=T.f32)
                            _src_enc = (arith.constant(_fz_rank * _fz_mtpr) + _src_tok) | (_k_slot << arith.constant(24))
                            _idx_remote = buffer_ops.buffer_load(_crfa(_p_idx), _dest_pe, vec_width=1, dtype=T.i64)
                            buffer_ops.buffer_store(_expert, _crfa(_idx_remote), _slot)
                            _wts_remote = buffer_ops.buffer_load(_crfa(_p_wts), _dest_pe, vec_width=1, dtype=T.i64)
                            buffer_ops.buffer_store(arith.bitcast(T.i32, _wt_val), _crfa(_wts_remote), _slot)
                            _sm_remote = buffer_ops.buffer_load(_crfa(_p_sm), _dest_pe, vec_width=1, dtype=T.i64)
                            buffer_ops.buffer_store(_src_enc, _crfa(_sm_remote), _slot)

                    if const_expr(_fz_enable_scales):
                        if _lane < _fz_scale_n_i32:
                            if _do_pub:
                                _sc_val = buffer_ops.buffer_load(_crfa(_a_sc), _src_tok * arith.constant(_fz_scale_n_i32) + _lane, vec_width=1, dtype=T.i32)
                                _sc_remote = buffer_ops.buffer_load(_crfa(_p_sc), _dest_pe, vec_width=1, dtype=T.i64)
                                buffer_ops.buffer_store(_sc_val, _crfa(_sc_remote), _slot * arith.constant(_fz_scale_n_i32) + _lane)

                    # token embedding copy (all lanes, 16B v4i32 chunks)
                    _tok_remote_base = buffer_ops.buffer_load(_crfa(_p_rx), _dest_pe, vec_width=1, dtype=T.i64)
                    _rsrc_src = _crfa(_a_tok + _epk._to_i64(_src_tok) * _fz_nbytes)
                    _rsrc_dst = _crfa(_tok_remote_base + _epk._to_i64(_slot) * _fz_nbytes)
                    _lane_off = _lane * 4
                    if const_expr(_fz_n_i32 >= 512 and _fz_safe_end_i32 > 0):
                        _ce_main = arith.select(_do_pub, arith.constant(_fz_safe_end_i32), _lane_off)
                        for _co in range(_lane_off, _ce_main, 512):
                            _va = buffer_ops.buffer_load(_rsrc_src, _co, vec_width=4, dtype=T.i32)
                            _vb = buffer_ops.buffer_load(_rsrc_src, _co + 256, vec_width=4, dtype=T.i32)
                            buffer_ops.buffer_store(_va, _rsrc_dst, _co)
                            buffer_ops.buffer_store(_vb, _rsrc_dst, _co + 256)
                    if const_expr(_fz_safe_end_i32 < _fz_n_i32):
                        _ce_tail = arith.select(_do_pub, arith.constant(_fz_n_i32), _lane_off)
                        for _co in range(_lane_off + _fz_safe_end_i32, _ce_tail, 256):
                            _va = buffer_ops.buffer_load(_rsrc_src, _co, vec_width=4, dtype=T.i32)
                            buffer_ops.buffer_store(_va, _rsrc_dst, _co)
                    elif const_expr(_fz_n_i32 < 512):
                        _ce_small = arith.select(_do_pub, arith.constant(_fz_n_i32), _lane_off)
                        for _co in range(_lane_off, _ce_small, 256):
                            _va = buffer_ops.buffer_load(_rsrc_src, _co, vec_width=4, dtype=T.i32)
                            buffer_ops.buffer_store(_va, _rsrc_dst, _co)

                # ---- post-write ARRIVAL (N->1; only block0 waits) — NOT a symmetric grid barrier ----
                # Persistent-kernel philosophy: no whole-grid full sync.  Each block, after its warps
                # finish writing (CTA-internal fx.barrier), snapshots meta_flag (= L-1, stable because
                # block0 is still waiting for THIS block's arrival) and arrives (atomic_add gb1).  Only
                # block0 spins for all arrivals (release/acquire => sees every block's writes at agent
                # scope), then does cross-PE + post-pass, then RELEASE-stores meta_flag = epoch.  Other
                # blocks DON'T spin here; they wait on the meta_flag (1 writer / N readers, cache-shared
                # -> no 256-way atomic-RMW storm).  gb1 is now incremented exactly ONCE/block/launch, so
                # gb1//nblk == launch count == meta_flag, keeping the cross-PE epoch + flag in lockstep.
                fx.barrier()
                _e0_i32 = arith.constant(0)
                if _tid == 0:
                    _e0_i32 = buffer_ops.buffer_load(_crfa(_a_meta), 0, vec_width=1, dtype=T.i32)  # = L-1
                    _one2 = arith.constant(1, type=T.i64)
                    _epk.fence_agent_release()
                    _epk.atomic_add_agent(_a_gb1, _one2)         # arrive (no spin here)

                # ---- block0: wait all arrivals -> cross-PE done-barrier -> post-pass -> publish flag ----
                if _gwid == 0:
                    _one2b = arith.constant(1, type=T.i64)
                    _gb1now = buffer_ops.buffer_load(_crfa(_a_gb1), 0, vec_width=1, dtype=T.i64)
                    _tg2 = ((arith.ArithValue(_gb1now, signed=False) - _one2b) // _bn + _one2b) * _bn
                    mori_shmem.int64_wait_until_equals(_a_gb1, _tg2)
                    _epk.fence_agent_acquire()
                    _epoch_i32 = arith.trunci(T.i32, arith.unwrap(_tg2 // _bn))
                    # cross-PE done-barrier (my writes flushed to peers; wait peers')
                    _epk.fence_system_release()
                    for _dpe in range(_lane, _fz_npes, 64):
                        _d2 = buffer_ops.buffer_load(_crfa(_p_done2), _dpe, vec_width=1, dtype=T.i64)
                        _epk.store_i32_system(_d2, arith.constant(_fz_rank), _epoch_i32)
                    for _spe in range(_lane, _fz_npes, 64):
                        mori_shmem.int32_wait_until_equals(_a_done2 + _epk._to_i64(_spe) * 4, _epoch_i32)
                    _epk.fence_system_acquire()
                    # post-pass: emit ONLY occupied tiles compactly + fold count reset
                    _r_run = _crfa(_a_run)
                    _r_se = _crfa(_a_se)
                    _r_trb = _crfa(_a_trb)
                    _r_nv = _crfa(_a_nv)
                    _r_cnt = _crfa(_a_cnt)
                    _ctm = arith.constant(_fz_tile_m)
                    _ctm1 = arith.constant(_fz_tile_m - 1)
                    _acc = arith.constant(0)
                    for _le2 in range(_fz_epr):                       # constexpr unroll (epr small)
                        _cnt = buffer_ops.buffer_load(_r_run, _le2, vec_width=1, dtype=T.i32)
                        _ntiles = (_cnt + _ctm1) // _ctm
                        _se_val = _le2 + arith.constant(_fz_rank * _fz_epr)
                        _trb_base = _le2 * arith.constant(_fz_cap)
                        for _t in range(_lane, _ntiles, 64):
                            _ci = _acc + _t
                            buffer_ops.buffer_store(_se_val, _r_se, _ci)
                            buffer_ops.buffer_store(_trb_base + _t * _ctm, _r_trb, _ci)
                        _acc = _acc + _ntiles
                    _nvv = _acc * _ctm
                    buffer_ops.buffer_store(_nvv, _r_nv, arith.constant(0))
                    buffer_ops.buffer_store(_nvv, _r_nv, arith.constant(1))
                    for _lei in range(_lane, _fz_epr, 64):
                        _cnt2 = buffer_ops.buffer_load(_r_run, _lei, vec_width=1, dtype=T.i32)
                        buffer_ops.buffer_store(_cnt2, _r_cnt, _lei)
                        buffer_ops.buffer_store(arith.constant(0), _r_run, _lei)
                    _epk.fence_system_release()
                    # publish metadata-ready: release-store meta_flag = epoch (1 writer, no atomic storm)
                    _epk.fence_agent_release()
                    buffer_ops.buffer_store(_epoch_i32, _crfa(_a_meta), arith.constant(0))

                # ---- all blocks: wait metadata ready (acquire; N readers spin a cacheline) ----
                if _tid == 0:
                    mori_shmem.int32_wait_until_greater_than(_a_meta, _e0_i32)
                    _epk.fence_agent_acquire()
                fx.barrier()

            # ============================================================================
            # FUSED megakernel — HANDSHAKE OVERLAP prologue (prefill producer/consumer).
            # Ported from make_handshake_dispatch_kernel (P0 hist -> count all-gather -> CMP
            # my_base/expected_real/metadata).  Then PRODUCER blocks (block_idx.x >= gx) write the
            # DENSE payload and signal payload_done[le] per token; CONSUMER blocks (block_idx.x < gx)
            # fall through to the persistent GEMM, which gates each tile on payload_done[le] >=
            # expected_real[le] (overlap_gate) -> GEMM compute hides the dispatch payload movement.
            # ============================================================================
            if const_expr(_fuse_hs):
                _crfa = buffer_ops.create_buffer_resource_from_addr
                _rd = _crfa(addr_disp)

                def _dph(_i):
                    return buffer_ops.buffer_load(_rd, arith.constant(_i), vec_width=1, dtype=T.i64)

                _a_tok = addr_in_tok; _a_idx = addr_in_idx; _a_wts = addr_in_wts; _a_sc = addr_in_sc
                _a_gb1 = _dph(4); _a_lh = _dph(5); _a_ob = _dph(6); _a_bc = _dph(7)
                _a_mb = _dph(8); _a_cd = _dph(9); _a_cnt = _dph(10)
                _a_se = _dph(11); _a_trb = _dph(12); _a_nv = _dph(13)
                _p_rx = _dph(14); _p_sc = _dph(15); _p_idx = _dph(16); _p_wts = _dph(17)
                _p_sm = _dph(18); _p_bc = _dph(19); _p_cd = _dph(20); _p_pd = _dph(21)
                _a_inv = _dph(22); _a_lpx = _dph(23)   # counting-sort: inv[dense]=wk, local_prefix[ge]

                _tid = fx.thread_idx.x
                _lane = _tid & 63
                _warp = _tid >> 6
                _gdx = fx.grid_dim.x
                _bxb = fx.block_idx.x
                _byb = fx.block_idx.y
                _flat = _byb * _gdx + _bxb
                _wl = i32_cur_tok * arith.constant(_fz_k)
                _c_epr = arith.constant(_fz_epr)
                _c_te = arith.constant(_fz_total_experts)
                _r_idx = _crfa(_a_idx)
                _r_wts = _crfa(_a_wts)
                _r_lh = _crfa(_a_lh)
                _r_ob = _crfa(_a_ob)
                _r_bc = _crfa(_a_bc)
                _r_mb = _crfa(_a_mb)
                _a_meta = _dph(24)   # metadata-ready flag (per-launch reset; block0 sets, all wait)

                # ====================================================================
                # PREFILL handshake (docs/moe_stage1_mega.md §9.3/§9.6): BLOCK0-SELF-SUFFICIENT.
                # The flat-id-0 block ALONE runs the whole counts-first handshake (P0 hist ->
                # cross-PE count all-gather -> CMP metadata -> SCT inv).  It waits on NO other block,
                # so the grid may OVERSUBSCRIBE (total>cu_num) without deadlock (§9.7).  block0 then
                # release-stores meta_flag (per-launch reset to 0 => ABSOLUTE "wait >= 1":
                # deadlock-free under any wave order, 1 writer / N readers, §9.2 R2).  Producers/
                # consumers acquire meta_flag, then run with ZERO grid barriers (only per-expert
                # payload_done couples them, §9.1).
                # Structure: P0/SCT are TOP-LEVEL loops gated by a block0-only loop bound (non-block0
                # -> empty range); PUB/CMP/epoch/publish are warp0-serial under `if _gwid == 0`.
                # (A for-loop nested inside a runtime `if` does not trace cleanly in this DSL, so the
                # block0 gate is pushed into the loop BOUND, matching the dispatch-kernel idiom.)
                # Barriers are TOP-LEVEL (every workgroup self-syncs).
                # ====================================================================
                _gwid = _flat * num_waves + _warp
                _is_b0 = (_flat == 0)
                _c_nw = arith.constant(num_waves)
                _wl_b0 = arith.select(_is_b0, _wl, arith.constant(0))   # block0 -> _wl, else empty

                # ---- launch epoch: block0/warp0/lane0 bumps gb1 by 1 (monotonic, NEVER reset ->
                #      cross-PE epoch lockstep + CUDAGraph-safe). ----
                if _gwid == 0:
                    if _lane == 0:
                        _epk.fence_agent_release()
                        _ = _epk.atomic_add_agent(_a_gb1, arith.constant(1, type=T.i64))
                fx.barrier()

                # ---- P0: block0's warps build local_hist[ge] + off_buf[wk] (top-level loop, block0-
                #      gated bound; lane0 of each warp does the atomic). ----
                for _wk in range(_warp, _wl_b0, _c_nw):
                    _expert = buffer_ops.buffer_load(_r_idx, _wk, vec_width=1, dtype=T.i32)
                    _is_valid = arith.cmpi(CmpIPredicate.ult, _expert, _c_te)
                    _off_l0 = arith.constant(0)
                    if _lane == 0:
                        if _is_valid:
                            _off_l0 = _epk.atomic_add_agent(_a_lh + _epk._to_i64(_expert) * 4, arith.constant(1))
                        buffer_ops.buffer_store(_epk._readlane0(_off_l0), _r_ob, _wk)
                rocdl.s_waitcnt(0)
                fx.barrier()
                if _gwid == 0:
                    _epk.fence_agent_release()
                    _epk.fence_agent_acquire()
                fx.barrier()

                # ---- PUB + CMP: block0/warp0 (cross-PE count all-gather -> my_base + tile metadata +
                #      expected_real + local_prefix).  Sequential in warp0 (PUB's system-acquire makes
                #      peers' bigcnt visible to CMP without an extra barrier). ----
                if _gwid == 0:
                    _epoch_i32 = arith.trunci(
                        T.i32, arith.unwrap(buffer_ops.buffer_load(_crfa(_a_gb1), 0, vec_width=1, dtype=T.i64)))
                    _epk.fence_system_release()
                    _base_row = arith.constant(_fz_rank * _fz_total_experts)
                    _te4 = (_fz_total_experts // 4) * 4
                    for _p in range_constexpr(_fz_npes):
                        _bc_remote = buffer_ops.buffer_load(_crfa(_p_bc), _p, vec_width=1, dtype=T.i64)
                        _rdst = _crfa(_bc_remote)
                        if const_expr(_te4 > 0):
                            for _off in range(_lane * 4, _te4, 256):
                                _v4 = buffer_ops.buffer_load(_r_lh, _off, vec_width=4, dtype=T.i32)
                                buffer_ops.buffer_store(_v4, _rdst, _base_row + _off)
                        if const_expr(_te4 < _fz_total_experts):
                            for _off in range(_lane + _te4, _fz_total_experts, 64):
                                _v1 = buffer_ops.buffer_load(_r_lh, _off, vec_width=1, dtype=T.i32)
                                buffer_ops.buffer_store(_v1, _rdst, _base_row + _off)
                    _epk.fence_system_release()
                    for _p in range(_lane, _fz_npes, 64):
                        _cd_remote = buffer_ops.buffer_load(_crfa(_p_cd), _p, vec_width=1, dtype=T.i64)
                        _epk.store_i32_system(_cd_remote, arith.constant(_fz_rank), _epoch_i32)
                    for _s in range(_lane, _fz_npes, 64):
                        mori_shmem.int32_wait_until_equals(_a_cd + _epk._to_i64(_s) * 4, _epoch_i32)
                    _epk.fence_system_acquire()
                    _ctm = arith.constant(_fz_tile_m)
                    _ctm1 = arith.constant(_fz_tile_m - 1)
                    for _d in range(_lane, _fz_npes, 64):
                        _eb = arith.constant(0)
                        for _le in range_constexpr(_fz_epr):
                            _ge = _d * _c_epr + arith.constant(_le)
                            _cs = arith.constant(0)
                            _sp = arith.constant(0)
                            for _s in range_constexpr(_fz_npes):
                                _v = buffer_ops.buffer_load(_r_bc, arith.constant(_s * _fz_total_experts) + _ge,
                                                            vec_width=1, dtype=T.i32)
                                _cs = _cs + _v
                                if const_expr(_s < _fz_rank):
                                    _sp = _sp + _v
                            buffer_ops.buffer_store(_eb + _sp, _r_mb, _ge)
                            _eb = _eb + ((_cs + _ctm1) // _ctm) * _ctm
                    _r_se = _crfa(_a_se)
                    _r_trb = _crfa(_a_trb)
                    _r_nv = _crfa(_a_nv)
                    _r_cnt = _crfa(_a_cnt)
                    _acc = arith.constant(0)
                    _ebase = arith.constant(0)
                    for _le2 in range(_fz_epr):
                        _cnt = arith.constant(0)
                        for _s in range_constexpr(_fz_npes):
                            _cnt = _cnt + buffer_ops.buffer_load(
                                _r_bc, arith.constant(_s * _fz_total_experts + _fz_rank * _fz_epr) + _le2,
                                vec_width=1, dtype=T.i32)
                        _ntiles = (_cnt + _ctm1) // _ctm
                        _se_val = _le2 + arith.constant(_fz_rank * _fz_epr)
                        for _t in range(_lane, _ntiles, 64):
                            _ci = _acc + _t
                            buffer_ops.buffer_store(_se_val, _r_se, _ci)
                            buffer_ops.buffer_store(_ebase + _t * _ctm, _r_trb, _ci)
                        if _lane == 0:
                            buffer_ops.buffer_store(_cnt, _r_cnt, _le2)   # expected_real[le]
                        _acc = _acc + _ntiles
                        _ebase = _ebase + _ntiles * _ctm
                    _nvv = _acc * _ctm
                    buffer_ops.buffer_store(_nvv, _r_nv, arith.constant(0))
                    buffer_ops.buffer_store(_nvv, _r_nv, arith.constant(1))
                    # local_prefix[ge] = exclusive prefix-sum of local_hist over global experts.
                    _r_lpx = _crfa(_a_lpx)
                    _r_lh2 = _crfa(_a_lh)
                    _pfx = arith.constant(0)
                    for _gg in range(_fz_total_experts):     # python range -> unrolled; _pfx carried
                        buffer_ops.buffer_store(_pfx, _r_lpx, _gg)
                        _pfx = _pfx + buffer_ops.buffer_load(_r_lh2, _gg, vec_width=1, dtype=T.i32)
                    rocdl.s_waitcnt(0)
                fx.barrier()
                if _gwid == 0:
                    _epk.fence_agent_release()
                    _epk.fence_agent_acquire()
                fx.barrier()

                # ---- SCT (counting sort): block0's warps build inv[local_prefix[e]+off_buf[wk]]=wk
                #      (top-level loop, block0-gated bound). ----
                _r_inv = _crfa(_a_inv)
                _r_lpx2 = _crfa(_a_lpx)
                _r_ob2 = _crfa(_a_ob)
                for _swk in range(_warp, _wl_b0, _c_nw):
                    _sxe = buffer_ops.buffer_load(_r_idx, _swk, vec_width=1, dtype=T.i32)
                    _sxv = arith.cmpi(CmpIPredicate.ult, _sxe, _c_te)
                    if _lane == 0:
                        if _sxv:
                            _sxpx = buffer_ops.buffer_load(_r_lpx2, _sxe, vec_width=1, dtype=T.i32)
                            _sxof = buffer_ops.buffer_load(_r_ob2, _swk, vec_width=1, dtype=T.i32)
                            buffer_ops.buffer_store(_swk, _r_inv, _sxpx + _sxof)
                rocdl.s_waitcnt(0)
                fx.barrier()

                # ---- publish meta_flag = 1 (release): metadata + inv ready for producers/consumers ----
                if _gwid == 0:
                    if _lane == 0:
                        _epk.fence_system_release()
                        _epk.fence_agent_release()
                        buffer_ops.buffer_store(arith.constant(1, type=T.i32), _crfa(_a_meta), arith.constant(0))

                # ---- ALL blocks: wait metadata ready (per-launch reset to 0 => wait >= 1;
                #      ABSOLUTE threshold, deadlock-free under oversubscribe, §9.7). ----
                if _tid == 0:
                    mori_shmem.int32_wait_until_greater_than(_a_meta, arith.constant(0, type=T.i32))
                    _epk.fence_agent_acquire()
                fx.barrier()

                # ---- PRODUCER blocks (block_idx.x >= gx): EXPERT-GROUPED write + per-expert signal ----
                # Each producer block OWNS global experts {ge : ge % nprod == pbid} and, per owned
                # expert, scans all work_idx (strided across the block's warps) writing its tokens
                # DENSELY, then signals payload_done[le] += local_hist[ge] ONCE (block-uniform expert
                # loop -> block barrier safe).  Contention on the receiver's payload_done[le] drops to
                # npes (one source-block per (rank,ge)) vs the per-token version's thousands -> the
                # producer dispatch stays fast AND the consumer gets EARLY per-expert readiness.
                # Loop var names are _q-prefixed to avoid AST closure-capture clashes with P0/CMP.
                _gxc = arith.constant(_fz_gx_static)
                _is_prod = arith.cmpi(CmpIPredicate.uge, _bxb, _gxc)
                _qnprod = (_gdx - _gxc) * fx.grid_dim.y          # producer block count = np_cols*gy
                _qpbid = (_bxb - _gxc) * fx.grid_dim.y + _byb    # producer flat id (valid if producer)
                _qstart = arith.select(_is_prod, _qpbid, _c_te)  # consumers: empty owned-expert loop
                _r_inv2 = _crfa(_a_inv)
                _r_lpx3 = _crfa(_a_lpx)
                for _qge in range(_qstart, _c_te, _qnprod):
                    _qle = _qge % _c_epr
                    _qdest = _qge // _c_epr
                    _qcnt = buffer_ops.buffer_load(_r_lh, _qge, vec_width=1, dtype=T.i32)    # #tokens this rank->ge
                    _qpx = buffer_ops.buffer_load(_r_lpx3, _qge, vec_width=1, dtype=T.i32)   # dense base in inv
                    _qmb = buffer_ops.buffer_load(_r_mb, _qge, vec_width=1, dtype=T.i32)     # receiver slot base
                    # warps split this expert's REAL tokens (no scan, no match gate)
                    for _qoff in range(_warp, _qcnt, arith.constant(num_waves)):
                        _qwk = buffer_ops.buffer_load(_r_inv2, _qpx + _qoff, vec_width=1, dtype=T.i32)
                        _qslot = _qmb + _qoff
                        _qsrc = _qwk // arith.constant(_fz_k)
                        _qks = _qwk % arith.constant(_fz_k)
                        if _lane == 0:
                            _qidxr = buffer_ops.buffer_load(_crfa(_p_idx), _qdest, vec_width=1, dtype=T.i64)
                            buffer_ops.buffer_store(_qge, _crfa(_qidxr), _qslot)
                        if _lane == 1:
                            _qwt = buffer_ops.buffer_load(_r_wts, _qwk, vec_width=1, dtype=T.f32)
                            _qwtsr = buffer_ops.buffer_load(_crfa(_p_wts), _qdest, vec_width=1, dtype=T.i64)
                            buffer_ops.buffer_store(arith.bitcast(T.i32, _qwt), _crfa(_qwtsr), _qslot)
                        if _lane == 2:
                            _qenc = (arith.constant(_fz_rank * _fz_mtpr) + _qsrc) | (_qks << arith.constant(24))
                            _qsmr = buffer_ops.buffer_load(_crfa(_p_sm), _qdest, vec_width=1, dtype=T.i64)
                            buffer_ops.buffer_store(_qenc, _crfa(_qsmr), _qslot)
                        if const_expr(_fz_enable_scales):
                            if _lane < _fz_scale_n_i32:
                                _qsc = buffer_ops.buffer_load(_crfa(_a_sc), _qsrc * arith.constant(_fz_scale_n_i32) + _lane, vec_width=1, dtype=T.i32)
                                _qscr = buffer_ops.buffer_load(_crfa(_p_sc), _qdest, vec_width=1, dtype=T.i64)
                                buffer_ops.buffer_store(_qsc, _crfa(_qscr), _qslot * arith.constant(_fz_scale_n_i32) + _lane)
                        _qtrb = buffer_ops.buffer_load(_crfa(_p_rx), _qdest, vec_width=1, dtype=T.i64)
                        _qsrcr = _crfa(_a_tok + _epk._to_i64(_qsrc) * _fz_nbytes)
                        _qdstr = _crfa(_qtrb + _epk._to_i64(_qslot) * _fz_nbytes)
                        _qloff = _lane * 4
                        if const_expr(_fz_n_i32 >= 512 and _fz_safe_end_i32 > 0):
                            for _qco in range(_qloff, arith.constant(_fz_safe_end_i32), 512):
                                _qva = buffer_ops.buffer_load(_qsrcr, _qco, vec_width=4, dtype=T.i32)
                                _qvb = buffer_ops.buffer_load(_qsrcr, _qco + 256, vec_width=4, dtype=T.i32)
                                buffer_ops.buffer_store(_qva, _qdstr, _qco)
                                buffer_ops.buffer_store(_qvb, _qdstr, _qco + 256)
                        if const_expr(_fz_safe_end_i32 < _fz_n_i32):
                            for _qco in range(_qloff + _fz_safe_end_i32, arith.constant(_fz_n_i32), 256):
                                _qva = buffer_ops.buffer_load(_qsrcr, _qco, vec_width=4, dtype=T.i32)
                                buffer_ops.buffer_store(_qva, _qdstr, _qco)
                        elif const_expr(_fz_n_i32 < 512):
                            for _qco in range(_qloff, arith.constant(_fz_n_i32), 256):
                                _qva = buffer_ops.buffer_load(_qsrcr, _qco, vec_width=4, dtype=T.i32)
                                buffer_ops.buffer_store(_qva, _qdstr, _qco)
                    # all warps finished writing this expert's tokens -> publish + signal ONCE.
                    # s_waitcnt(0) + system-release fence BEFORE the payload_done atomic so the
                    # consumer's acquire-gate observes freshly-landed payload (not stale L2).
                    fx.barrier()
                    rocdl.s_waitcnt(0)
                    if _tid == 0:
                        _epk.fence_system_release()
                        _qpdr = buffer_ops.buffer_load(_crfa(_p_pd), _qdest, vec_width=1, dtype=T.i64)
                        _epk.atomic_add_system(_qpdr + _epk._to_i64(_qle) * 4, _qcnt)

            tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
            n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
            k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())

            x_elem = T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec16_elems = 16 if a_elem_bytes == 1 else 8
            vec16_x = T.vec(vec16_elems, x_elem)
            vec2_i64 = T.vec(2, i64)

            acc_init = arith.constant_vector(0.0, vec4_f32)

            # --- Stage1 dimension mapping ---
            # X: [tokens, model_dim] -- M = sorted tokens, K = model_dim
            # W: [E*2*inter_dim, model_dim] gate portion -- N = inter_dim
            # Out: [tokens*topk, inter_dim]

            # B preshuffle layout: [E*2*inter_dim, model_dim]
            # Gate rows for expert e: [e*2*inter_dim, e*2*inter_dim + inter_dim)
            c_n_total = arith.constant(experts * (2 * inter_dim), index=True)
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in // pack_K,
                kpack_bytes=kpack_bytes,
                elem_bytes=b_elem_bytes,
                # k_major=True,
            )
            layout_b = b_layout.layout_b

            # A-scale: [sorted_size, K/32] -- pre-scattered by caller into sorted layout
            # Same as stage2: indexed by sorted_row position, not by token_id.
            sorted_m = size_expert_ids_in * arith.constant(sort_block_m, index=True)
            layout_a_scale = make_preshuffle_scale_layout(
                arith, c_mn=sorted_m, c_k=arith.constant(model_dim, index=True)
            )
            # B-scale: [E*2*inter_dim, K/32]
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=arith.constant(model_dim, index=True)
            )

            _eff_lds_stride = lds_stride
            _eff_tile_k_bytes = tile_k_bytes
            if const_expr(use_async_copy and a_elem_vec_pack > 1):
                _eff_lds_stride = lds_stride // a_elem_vec_pack
                _eff_tile_k_bytes = tile_k_bytes // a_elem_vec_pack

            shape_lds = fx.make_shape(tile_m, _eff_lds_stride)
            stride_lds = fx.make_stride(_eff_lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            by = gpu.block_id("x")  # tile along inter_dim (N)
            bx_persist = gpu.block_id("y")  # persistent WG index

            if const_expr(xcd_swizzle > 0):
                _NUM_XCDS_S1 = 8
                _c1_sw = arith.constant(1, index=True)
                _c_tn_sw = arith.constant(tile_n, index=True)
                _c_idp_sw = arith.constant(2 * inter_dim_pad, index=True)
                if const_expr(mock_gate_only or gate_up_interleave):
                    _gx = (n_in - _c_idp_sw + _c_tn_sw - _c1_sw) / _c_tn_sw
                else:
                    _c2_sw = arith.constant(2, index=True)
                    _gx = (n_in - _c_idp_sw + _c2_sw * _c_tn_sw - _c1_sw) / _c_tn_sw / _c2_sw
                _gy = arith.constant(_cu_num, index=True)

                _linear_id = bx_persist * _gx + by
                _num_wgs = _gx * _gy

                _c_xcds = arith.constant(_NUM_XCDS_S1, index=True)
                _wgs_per_xcd = _num_wgs / _c_xcds
                _wgid = (_linear_id % _c_xcds) * _wgs_per_xcd + (_linear_id / _c_xcds)

                _WGM_S1 = xcd_swizzle
                _c_wgm = arith.constant(_WGM_S1, index=True)
                _num_wgid_in_group = _c_wgm * _gx
                _group_id = _wgid / _num_wgid_in_group
                _first_pid_m = _group_id * _c_wgm
                _remaining_m = _gy - _first_pid_m
                _cmp_m = arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
                _group_size_m = arith.select(_cmp_m, _remaining_m, _c_wgm)

                _wgid_in_group = _wgid % _num_wgid_in_group
                bx_persist = _first_pid_m + (_wgid_in_group % _group_size_m)
                by = _wgid_in_group / _group_size_m

            by_n = by * arith.constant(tile_n, index=True)

            k_base_idx = arith.index(0)
            if const_expr(_is_splitk):
                bz = gpu.block_id("z")  # K-batch id
                k_base_idx = bz * arith.constant(_k_dim, index=True)

            k_blocks16 = arith.constant(_eff_tile_k_bytes // 16, index=True)
            layout_tx_wave_lane = fx.make_layout((num_waves, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            base_ptr_pong = allocator_pong.get_base()
            base_ptr_ping = allocator_ping.get_base()
            lds_x_pong = SmemPtr(base_ptr_pong, lds_pong_offset, x_lds_elem(), shape=(_input_elems,)).get()
            lds_x_ping = SmemPtr(base_ptr_ping, lds_ping_offset, x_lds_elem(), shape=(_input_elems,)).get()
            _lds_out_elem_type = T.f32 if _need_quant else (T.bf16 if out_is_bf16 else T.f16)
            if const_expr(_split_lds_out and _use_cshuffle_epilog):
                _half_out_elems = int(tile_m) * (int(tile_n) // 2)
                lds_out = SmemPtr(
                    base_ptr_pong,
                    lds_pong_offset,
                    _lds_out_elem_type,
                    shape=(_half_out_elems,),
                ).get()
                lds_out_B = SmemPtr(
                    base_ptr_ping,
                    lds_ping_offset,
                    _lds_out_elem_type,
                    shape=(_half_out_elems,),
                ).get()
            else:
                lds_out = (
                    SmemPtr(
                        base_ptr_pong,
                        lds_pong_offset,
                        _lds_out_elem_type,
                        shape=(tile_m * tile_n,),
                    ).get()
                    if _use_cshuffle_epilog
                    else None
                )
                lds_out_B = None
            lds_tid = SmemPtr(base_ptr_pong, _lds_tid_offset_pong, T.i32, shape=(tile_m,)).get()
            # raw_a_scale: row-major (NO swizzle) single-buffered LDS view of this tile's A-scale.
            lds_a_scale = (
                SmemPtr(base_ptr_pong, _scale_lds_offset_pong, T.i32, shape=(_scale_lds_n_i32,)).get()
                if _raw_a_scale_lds
                else None
            )

            # Buffer resources
            c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
            c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)

            # X: [tokens, model_dim]
            x_nbytes_idx = (tokens_in * k_in * c_elem_bytes) / c_a_pack
            x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
            x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False, num_records_bytes=x_nbytes_i32)

            # W: [experts, 2*inter_dim, model_dim]; fp4 packs 2 elements per byte.
            w_nbytes_s1 = (
                (experts * (2 * inter_dim) * model_dim) // 2
                if is_f4_b
                else (experts * (2 * inter_dim) * model_dim * b_elem_bytes)
            )
            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False, num_records_bytes=w_nbytes_s1)

            # Out: [tokens*topk, inter_dim]
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=arith.constant(4, type=T.i32),
            )
            num_valid_i32 = buffer_ops.buffer_load(numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32)

            sx_rsrc = 1
            sw_rsrc = 1
            if const_expr(not (is_f16_a or a_scale_one)):
                # A scale: [sorted_size, model_dim/32] pre-scattered by caller
                c32 = arith.constant(32, index=True)
                kblk = k_in / c32
                sx_nbytes_idx = sorted_m * kblk
                sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                )

            if const_expr(not is_f16_b):
                c32 = arith.constant(32, index=True)
                kblk_w = k_in / c32
                mn_w = arith.constant(experts * (2 * inter_dim), index=True)
                sw_nbytes_idx = mn_w * kblk_w
                sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
                sw_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_w, max_size=False, num_records_bytes=sw_nbytes_i32
                )

            sorted_nbytes_idx = size_expert_ids_in * arith.constant(sort_block_m * 4, index=True)
            sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_i32,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_i32
            )

            eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
            eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
            )
            # bias: [experts, 2*inter_dim] f32 -> bytes = experts * 2*inter_dim * 4
            bias_nbytes_s1 = experts * (2 * inter_dim) * 4
            bias_rsrc = (
                buffer_ops.create_buffer_resource(arg_bias, max_size=False, num_records_bytes=bias_nbytes_s1)
                if enable_bias
                else None
            )

            # Sorted-scale buffer resource for fused mxfp4 quantization
            _sorted_scale_cols = inter_dim // 32
            _sorted_scale_cols_i32 = arith.constant(_sorted_scale_cols, type=T.i32)
            sorted_scale_rsrc = None
            if const_expr(_need_sort):
                _sort_rows_idx = size_expert_ids_in * arith.constant(sort_block_m, index=True)
                _sort_padded_rows = (
                    (_sort_rows_idx + arith.constant(255, index=True))
                    / arith.constant(256, index=True)
                    * arith.constant(256, index=True)
                )
                _sort_padded_cols = arith.constant(((_sorted_scale_cols + 7) // 8) * 8, index=True)
                _sort_scale_nbytes = arith.index_cast(T.i32, _sort_padded_rows * _sort_padded_cols)
                sorted_scale_rsrc = buffer_ops.create_buffer_resource(
                    arg_out_scale_sorted, max_size=False, num_records_bytes=_sort_scale_nbytes
                )

            # ---- persistent round-robin tile loop: grid_y CTAs, each takes a strided chunk of
            # ceil(num_valid/sort_block_m)/grid_y M-tiles (bounded by device num_valid). ----
            _c0_p = arith.constant(0, index=True)
            _c1_p = arith.constant(1, index=True)
            # persistent round-robin: grid_y CTAs each take a strided chunk of M-tiles, bounded by
            # device num_valid (stride = actual grid_y, host-capped so small bs leaves no idle CTAs).
            _c_cu_p = arith.index_cast(ir.IndexType.get(), gpu.grid_dim.y)
            _c_sbm_p = arith.constant(sort_block_m, index=True)
            _num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)
            _total_m_tiles = (_num_valid_idx + _c_sbm_p - _c1_p) / _c_sbm_p
            _tiles_per_block = (_total_m_tiles + _c_cu_p - _c1_p) / _c_cu_p
            _for_persist = scf.ForOp(_c0_p, _tiles_per_block, _c1_p)
            _for_ip = ir.InsertionPoint(_for_persist.body)
            _for_ip.__enter__()
            _mi_p = _for_persist.induction_variable
            # Strided round-robin: CTA k does tiles {k, k+grid_y, ...} (adjacent CTAs hit adjacent
            # tiles -> per-wave B L2 reuse).
            bx = bx_persist + _mi_p * _c_cu_p
            if const_expr(sparse_tiles):
                # sparse fixed-slot layout: per-tile row base = tile_row_base[bx]
                # (carried in arg_sorted_token_ids); rows within a tile stay contiguous.
                _trb = buffer_ops.buffer_load(sorted_rsrc, bx, vec_width=1, dtype=T.i32)
                bx_m = arith.index_cast(ir.IndexType.get(), _trb)
            else:
                bx_m = bx * arith.constant(sort_block_m, index=True)

            # Block validity: sparse uses the tile index vs tile count (bx_m is a sparse base,
            # so bx_m<num_valid would be wrong); dense uses the row base vs num_valid rows.
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            if const_expr(sparse_tiles):
                blk_valid = arith.cmpi(CmpIPredicate.ult,
                                       arith.index_cast(T.i32, bx),
                                       arith.index_cast(T.i32, _total_m_tiles))
            else:
                blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)
            if const_expr(_fuse_hs):
                # handshake OVERLAP: only CONSUMER columns (by < gx) run the GEMM; PRODUCER columns
                # (by >= gx) already wrote payload in the prologue -> skip GEMM compute cheaply.
                _is_consumer = arith.cmpi(CmpIPredicate.ult, arith.index_cast(T.i32, by),
                                          arith.constant(_fz_gx_static, type=T.i32))
                blk_valid = arith.andi(blk_valid, _is_consumer)
            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=T.i32)
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            exp_valid = arith.cmpi(CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32))

            def _moe_gemm1_body():
                # Gate expert offset: first inter_dim rows of each expert's 2*inter_dim block
                expert_off_idx = expert_idx * arith.constant(2 * inter_dim, index=True)

                # X loading -- KEY DIFFERENCE from stage2: X row = token_id only
                x_load_bytes = 16
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4

                c_k_div4 = ((k_in / c_a_pack) * arith.constant(int(a_elem_bytes), index=True)) / arith.index(4)
                tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (4 * int(a_elem_vec_pack))
                layout_x_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), stride=(tile_k_dwords, 1))
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = arith.constant(topk)
                mask24 = arith.constant(0xFFFFFF)
                tokens_i32 = arith.index_cast(T.i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                def load_x(idx_i32):
                    idx_elem = idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
                    return buffer_copy_gmem16_dwordx4(
                        buffer_ops,
                        vector,
                        elem_type=x_elem,
                        idx_i32=idx_elem,
                        rsrc=x_rsrc,
                        vec_elems=vec16_elems,
                    )

                # Decode sorted token ids -- stage1: X row = token_id (not t*topk+s)
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                # Also store token_id and slot_id for output indexing

                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    if const_expr(dedup_gather):
                        # Lever B: A row r = dedup_rx[srcmap_em[r] & 0xFFFFFF].  arg_sorted_token_ids
                        # carries srcmap_em; low 24b = src_global = the per-source-token dedup row.
                        # Out-of-range / stale padding rows fall back to row 0 (OOB also reads 0 via
                        # the buffer resource); those output rows are >= num_valid and discarded.
                        fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32)
                        g_i32 = arith.andi(fused_i, mask24)
                        g_idx = arith.index_cast(ir.IndexType.get(), g_i32)
                        x_row_base_div4.append(g_idx * c_k_div4)
                    elif const_expr(contiguous_io):
                        # expert-major contiguous: A row r IS physical row (bx_m+row_local);
                        # no sorted_token_ids gather.
                        x_row_base_div4.append(sorted_row_i * c_k_div4)
                    else:
                        fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32)
                        t_i32 = arith.andi(fused_i, mask24)
                        s_i32 = arith.shrui(fused_i, arith.constant(24))
                        t_valid = arith.cmpi(CmpIPredicate.ult, t_i32, tokens_i32)
                        s_valid = arith.cmpi(CmpIPredicate.ult, s_i32, topk_i32)
                        ts_valid = arith.andi(t_valid, s_valid)
                        t_safe = arith.select(ts_valid, t_i32, arith.constant(0))
                        # KEY: X row base uses token_id only (not t*topk+s)
                        t_idx = arith.index_cast(ir.IndexType.get(), t_safe)
                        x_row_base_div4.append(t_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = ((base_k / c_a_pack) * arith.constant(int(a_elem_bytes), index=True)) / arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        parts.append(vector.bitcast(T.vec(4, i32), x_vec))
                    return parts

                # Wave/lane decomposition (identical to stage2)
                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)
                row_a_lds = lane_mod_16
                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_n_id = wave_id % arith.constant(num_waves, index=True)
                n_tile_base = wave_n_id * c_n_per_wave

                # N-tile precompute for gate AND up weights
                gate_n_intra_list = []
                gate_n_blk_list = []
                up_n_intra_list = []
                up_n_blk_list = []
                col_g_list = []
                c_n0_static = experts * (2 * inter_dim) // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                inter_idx = arith.constant(inter_dim, index=True)

                for i in range_constexpr(num_acc_n):
                    offset = i * 16
                    c_offset = arith.constant(offset, index=True)
                    if const_expr(not gate_up_interleave):
                        col_g = by_n + n_tile_base + c_offset + lane_mod_16
                        col_g_list.append(col_g)

                    global_n = by_n + n_tile_base + c_offset + lane_mod_16
                    # Gate/interleave: rows [expert_off, expert_off + 2*inter_dim)
                    gate_row_w = expert_off_idx + global_n
                    gate_coord = idx2crd(gate_row_w, layout_n_blk_intra)
                    gate_n_blk_list.append(layout_get(gate_coord, 0))
                    gate_n_intra_list.append(layout_get(gate_coord, 1))
                    if const_expr(not mock_gate_only and not gate_up_interleave):
                        up_row_w = gate_row_w + inter_idx
                        up_coord = idx2crd(up_row_w, layout_n_blk_intra)
                        up_n_blk_list.append(layout_get(up_coord, 0))
                        up_n_intra_list.append(layout_get(up_coord, 1))

                if const_expr(gate_up_interleave):
                    _gui_num_acc_n_out = num_acc_n // pack_N
                    for _gui_i in range_constexpr(_gui_num_acc_n_out):
                        _gui_offset = _gui_i * 16
                        _gui_c_offset = arith.constant(_gui_offset, index=True)
                        _gui_col_g = (by_n + n_tile_base) // arith.constant(2, index=True) + _gui_c_offset + lane_mod_16
                        col_g_list.append(_gui_col_g)

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                _K_per_ku = tile_k // k_unroll
                _pad_k_elems = (model_dim_pad % tile_k) if (not _is_splitk and model_dim_pad > 0) else 0
                _pad_ku_skip = _pad_k_elems // _K_per_ku
                _tail_ku = k_unroll - _pad_ku_skip
                _tail_ku_packed = (_tail_ku + pack_K - 1) // pack_K if _pad_ku_skip > 0 else None

                # B load for gate and up separately
                def load_b_packs_k64(base_k, ku: int, n_blk, n_intra):
                    c64 = arith.constant(64, index=True)
                    base_k_bytes = base_k * arith.constant(int(b_elem_bytes), index=True)
                    k0 = base_k_bytes // c64 + arith.constant(ku, index=True)
                    k1 = lane_div_16
                    coord_pack = (n_blk, k0, k1, n_intra, arith.constant(0, index=True))
                    idx_pack = crd2idx(coord_pack, layout_b)
                    vec_elems = kpack_bytes // int(b_elem_bytes)
                    b16 = _buffer_load_vec(
                        buffer_ops,
                        vector,
                        w_rsrc,
                        idx_pack,
                        elem_type=_w_elem_type(),
                        vec_elems=vec_elems,
                        elem_bytes=b_elem_bytes,
                        offset_in_bytes=(b_elem_bytes == 1),
                        cache_modifier=b_nt,
                    )
                    b_i64x2 = vector.bitcast(vec2_i64, b16)
                    b0 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                    b1 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
                    return b0, b1

                def load_b_tile(base_k, ku_limit=k_unroll):
                    """Load B tiles. Returns (gate_b_tile, up_b_tile).
                    When mock_gate_only or gate_up_interleave, up_b_tile is None."""
                    gate_b_tile = []
                    up_b_tile = [] if (not mock_gate_only and not gate_up_interleave) else None
                    for ku in range_constexpr(ku_limit):
                        g_packs0, g_packs1 = [], []
                        u_packs0, u_packs1 = [], []
                        for ni in range_constexpr(num_acc_n):
                            gb0, gb1 = load_b_packs_k64(base_k, ku, gate_n_blk_list[ni], gate_n_intra_list[ni])
                            g_packs0.append(gb0)
                            g_packs1.append(gb1)
                            if const_expr(not mock_gate_only and not gate_up_interleave):
                                ub0, ub1 = load_b_packs_k64(base_k, ku, up_n_blk_list[ni], up_n_intra_list[ni])
                                u_packs0.append(ub0)
                                u_packs1.append(ub1)
                        gate_b_tile.append((g_packs0, g_packs1))
                        if const_expr(not mock_gate_only and not gate_up_interleave):
                            up_b_tile.append((u_packs0, u_packs1))
                    return gate_b_tile, up_b_tile

                # Pre-compute scale base element indices (K-loop invariant).
                # idx = mni * stride_n0 + ku * stride_k0 + k_lane * stride_klane + n_lane
                # Split into: base_elem = mni * stride_n0 + lane_elem (invariant)
                #              k_elem    = ku * stride_k0             (per-iteration)
                _scale_lane_elem = lane_div_16 * layout_b_scale.stride_klane + lane_mod_16

                _gate_scale_bases = []
                _up_scale_bases = []
                for _ni in range_constexpr(num_acc_n_packed):
                    _col_base = by_n + n_tile_base + arith.constant(_ni * 16 * pack_N, index=True)
                    _gate_mni = (expert_off_idx + _col_base) // arith.constant(32, index=True)
                    _gate_scale_bases.append(_gate_mni * layout_b_scale.stride_n0 + _scale_lane_elem)
                    if const_expr(not mock_gate_only and not gate_up_interleave):
                        _up_mni = (expert_off_idx + inter_idx + _col_base) // arith.constant(32, index=True)
                        _up_scale_bases.append(_up_mni * layout_b_scale.stride_n0 + _scale_lane_elem)

                if const_expr(not a_scale_one):
                    _a_scale_bases = []
                    # Pre-swizzled (non-raw) A-scale in the SPARSE fixed-slot path is COMPACT-by-tile:
                    # the sparse-swizzle writes occupied tile bx at compact m-tile bx*(sort_block_m/32),
                    # decoupled from the SPARSE row base bx_m (= tile_row_base[bx]).  So index the
                    # scale by the compact persistent-loop counter bx, NOT bx_m.  (For dense/identity
                    # tile_row_base, bx_m == bx*sort_block_m so this is identical to the old formula;
                    # raw_a_scale reads from LDS and ignores _a_scale_bases entirely.)
                    _sbm_scale_mtiles = sort_block_m // (scale_mn_pack * 16)   # 32-row m-tiles per tile
                    for _mi in range_constexpr(m_repeat_packed):
                        if const_expr(sparse_tiles and not raw_a_scale):
                            _a_mni = _mi + bx * arith.constant(_sbm_scale_mtiles, index=True)
                        else:
                            _a_mni = _mi + bx_m // scale_mn_pack // 16
                        _a_scale_bases.append(_a_mni * layout_a_scale.stride_n0 + _scale_lane_elem)
                    # raw_a_scale (B): A-scale comes from row-major scale_em (no swizzle kernel).
                    # The diagnostic proved the repack VALU is free; the cost is the (uncoalesced,
                    # re-fetched-every-K) global loads.  So we now stage the tile's full scale into
                    # LDS once in the prologue and read it from LDS here (see _load_a_scale_i32 +
                    # the prologue cooperative copy).  CK i32 = [s(r0,c0), s(r0+16,c0), s(r0,c0+4),
                    # s(r0+16,c0+4)].
                    if const_expr(raw_a_scale):
                        _raw_sni = model_dim // 128       # scale_em i32 cols/row (= sblocks/4)
                        _raw_klane_sh = arith.index_cast(T.i32, lane_div_16) * arith.constant(8, type=T.i32)

                def _load_a_scale_i32(_mi, _koff):
                    if const_expr(raw_a_scale):
                        _k1 = _koff // layout_b_scale.stride_k0
                        _ca = _k1 * arith.constant(2, index=True)
                        # Staged in LDS by the per-tile prologue (row-major, no swizzle): read the
                        # two vec2-i32 from LDS instead of re-loading uncoalesced from global.
                        # Row-local = global row - bx_m = _mi*32 + lane_mod_16 (and +16); col = _ca.
                        # The repack (shift/mask/ori) below is UNCHANGED -- only the load source moved.
                        _rl0 = arith.constant(_mi * 32, index=True) + lane_mod_16
                        _rl16 = arith.constant(_mi * 32 + 16, index=True) + lane_mod_16
                        _idx0 = _rl0 * arith.constant(_raw_sni, index=True) + _ca
                        _idx16 = _rl16 * arith.constant(_raw_sni, index=True) + _ca
                        _v0 = vector.load_op(T.vec(2, T.i32), lds_a_scale, [_idx0])
                        _v1 = vector.load_op(T.vec(2, T.i32), lds_a_scale, [_idx16])
                        _ia = vector.extract(_v0, static_position=[0], dynamic_position=[])
                        _ib = vector.extract(_v0, static_position=[1], dynamic_position=[])
                        _ic = vector.extract(_v1, static_position=[0], dynamic_position=[])
                        _id = vector.extract(_v1, static_position=[1], dynamic_position=[])
                        _m8 = arith.constant(0xFF, type=T.i32)
                        _ba = arith.andi(arith.shrui(_ia, _raw_klane_sh), _m8)
                        _bc = arith.andi(arith.shrui(_ic, _raw_klane_sh), _m8)
                        _bb = arith.andi(arith.shrui(_ib, _raw_klane_sh), _m8)
                        _bd = arith.andi(arith.shrui(_id, _raw_klane_sh), _m8)
                        return arith.ori(arith.ori(_ba, arith.shli(_bc, arith.constant(8, type=T.i32))),
                                         arith.ori(arith.shli(_bb, arith.constant(16, type=T.i32)),
                                                   arith.shli(_bd, arith.constant(24, type=T.i32))))
                    return buffer_ops.buffer_load(sx_rsrc, _a_scale_bases[_mi] + _koff,
                                                  vec_width=1, dtype=T.i32, cache_modifier=0)

                def _diag_valu(s):
                    # DIAGNOSTIC: same byte-extract+repack VALU as B's per-token repack, but on
                    # the already-loaded i32 (no extra load).  Permuted (not identity) so it isn't
                    # optimized away.  Isolates "is the repack VALU the bottleneck?" (correctness
                    # irrelevant under this flag).
                    if const_expr(diag_scale_valu):
                        _m8 = arith.constant(0xFF, type=T.i32)
                        _b0 = arith.andi(s, _m8)
                        _b1 = arith.andi(arith.shrui(s, arith.constant(8, type=T.i32)), _m8)
                        _b2 = arith.andi(arith.shrui(s, arith.constant(16, type=T.i32)), _m8)
                        _b3 = arith.andi(arith.shrui(s, arith.constant(24, type=T.i32)), _m8)
                        return arith.ori(arith.ori(_b0, arith.shli(_b3, arith.constant(8, type=T.i32))),
                                         arith.ori(arith.shli(_b1, arith.constant(16, type=T.i32)),
                                                   arith.shli(_b2, arith.constant(24, type=T.i32))))
                    return s

                _c16_idx = arith.constant(16, index=True)
                _c2_idx = arith.constant(2, index=True)
                _scale_mask_lo = arith.constant(0xFF, type=T.i32)

                _m_half_idx = arith.constant(0, type=T.i32)
                _m_half_i32 = arith.constant(0, type=T.i32)
                _scale_shift = arith.constant(0, type=T.i32)
                _scale_shift_hi = arith.constant(0, type=T.i32)
                _n_half_idx = arith.constant(0, type=T.i32)
                _n_half_i32 = arith.constant(0, type=T.i32)
                _bscale_shift = arith.constant(0, type=T.i32)
                _bscale_shift_hi = arith.constant(0, type=T.i32)
                if const_expr(pack_M < scale_mn_pack):
                    _m_half_idx = (bx_m // _c16_idx) % _c2_idx
                    _m_half_i32 = arith.index_cast(T.i32, _m_half_idx)
                    _scale_shift = _m_half_i32 * arith.constant(8, type=T.i32)
                    _scale_shift_hi = _scale_shift + arith.constant(16, type=T.i32)

                if const_expr(pack_N < scale_mn_pack):
                    _n_half_idx = (n_tile_base // _c16_idx) % _c2_idx
                    _n_half_i32 = arith.index_cast(T.i32, _n_half_idx)
                    _bscale_shift = _n_half_i32 * arith.constant(8, type=T.i32)
                    _bscale_shift_hi = _bscale_shift + arith.constant(16, type=T.i32)

                def _rearrange_a_scale(raw_i32):
                    """Rearrange scale bytes for pack_M=1: extract m_half's k0,k1 bytes."""
                    if const_expr(pack_M >= scale_mn_pack):
                        return raw_i32
                    b_k0 = arith.andi(arith.shrui(raw_i32, _scale_shift), _scale_mask_lo)
                    b_k1 = arith.andi(arith.shrui(raw_i32, _scale_shift_hi), _scale_mask_lo)
                    return arith.ori(b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32)))

                def _rearrange_b_scale(raw_i32):
                    """Rearrange scale bytes for pack_N=1: extract n_half's k0,k1 bytes."""
                    if const_expr(pack_N >= scale_mn_pack):
                        return raw_i32
                    b_k0 = arith.andi(arith.shrui(raw_i32, _bscale_shift), _scale_mask_lo)
                    b_k1 = arith.andi(arith.shrui(raw_i32, _bscale_shift_hi), _scale_mask_lo)
                    return arith.ori(b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32)))

                if const_expr(a_scale_one):
                    _as1_const = arith.constant(0x7F7F7F7F, type=T.i32)
                    _as1_vec = vector.from_elements(T.vec(1, T.i32), [_as1_const])

                def prefetch_ab_scale_tile(base_k, ku_packed_limit=k_unroll_packed):
                    a_scale_tile = []
                    gate_b_scale = []
                    up_b_scale = [] if (not mock_gate_only and not gate_up_interleave) else None
                    for ku in range_constexpr(ku_packed_limit):
                        k_off = (ku + base_k) * layout_b_scale.stride_k0
                        for mi in range_constexpr(m_repeat_packed):
                            if const_expr(a_scale_one):
                                a_scale_tile.append(_as1_vec)
                            else:
                                s = _diag_valu(_rearrange_a_scale(_load_a_scale_i32(mi, k_off)))
                                a_scale_tile.append(vector.from_elements(T.vec(1, T.i32), [s]))
                        for ni in range_constexpr(num_acc_n_packed):
                            gs = buffer_ops.buffer_load(
                                sw_rsrc,
                                _gate_scale_bases[ni] + k_off,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=0,
                            )
                            gs = _rearrange_b_scale(gs)
                            gate_b_scale.append(vector.from_elements(T.vec(1, T.i32), [gs]))
                            if const_expr(not mock_gate_only and not gate_up_interleave):
                                us = buffer_ops.buffer_load(
                                    sw_rsrc,
                                    _up_scale_bases[ni] + k_off,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=0,
                                )
                                us = _rearrange_b_scale(us)
                                up_b_scale.append(vector.from_elements(T.vec(1, T.i32), [us]))
                    return [a_scale_tile, gate_b_scale, up_b_scale]

                _lds_base_zero = arith.index(0)

                def store_x_tile_to_lds(vec_x_in_parts, lds_buffer):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if const_expr(x_load_bytes == 16):
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_buffer,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=_lds_base_zero,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                if const_expr(use_async_copy):
                    _dma_bytes = 16
                    _wave_size = 64
                    _eff_bytes_per_buffer = int(tile_m) * int(_eff_lds_stride) * int(a_elem_bytes)
                    _num_dma_loads = max(1, _eff_bytes_per_buffer // (total_threads * _dma_bytes))

                    def dma_x_tile_to_lds(base_k, lds_buffer):
                        c4_idx = arith.index(4)
                        base_k_div4 = ((base_k / c_a_pack) * arith.constant(int(elem_bytes), index=True)) / arith.index(
                            4
                        )

                        lds_ptr_i64 = None
                        for i in range_constexpr(_num_dma_loads):
                            row_local_i = x_row_local[i]
                            col_local_i32_i = x_col_local_i32[i]
                            col_local_sw = swizzle_xor16(row_local_i, col_local_i32_i * c4_idx, k_blocks16)
                            row_k_dw = x_row_base_div4[i] + base_k_div4
                            global_byte_idx = row_k_dw * c4_idx + col_local_sw
                            global_offset = arith.index_cast(T.i32, global_byte_idx)

                            if const_expr(i == 0):
                                lds_addr = memref.extract_aligned_pointer_as_index(
                                    lds_buffer
                                ) + wave_id * arith.constant(_wave_size * _dma_bytes, index=True)
                                lds_ptr_i64 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                            else:
                                lds_ptr_i64 = lds_ptr_i64 + arith.constant(total_threads * _dma_bytes, type=T.i64)

                            lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                            lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                            rocdl.raw_ptr_buffer_load_lds(
                                x_rsrc,
                                lds_ptr,
                                arith.constant(_dma_bytes, type=T.i32),
                                global_offset,
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                                arith.constant(0, type=T.i32),
                            )

                    def prefetch_x_to_lds(base_k, lds_buffer):
                        dma_x_tile_to_lds(base_k, lds_buffer)

                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                    col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                    col_base_swz = col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes / arith.index(2))
                    idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
                    loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    return a0, a1

                def prefetch_full_a_from_lds(lds_buffer, ku_limit=k_unroll):
                    """Load entire A tile from LDS into registers before compute."""
                    a_regs = []
                    for k_idx in range_constexpr(ku_limit):
                        col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack
                        for mi_idx in range_constexpr(m_repeat):
                            mi_val = arith.constant(mi_idx * 16, index=True)
                            curr_row = row_a_lds + mi_val
                            a0, a1 = lds_load_packs_k64(curr_row, col_base, lds_buffer)
                            if const_expr(is_f8_a):
                                a2, a3 = lds_load_packs_k64(curr_row, col_base + 64, lds_buffer)
                                a_regs.append((a0, a1, a2, a3))
                            else:
                                a_regs.append((a0, a1))
                    return a_regs

                # Compute tile: gate + up MFMA interleaved, same A data, different B data.
                # Two accumulator sets; after all K tiles, acc = acc_gate + acc_up (f32 add).
                def compute_tile(
                    acc_gate_in,
                    acc_up_in,
                    gate_b_tile_in,
                    up_b_tile_in,
                    a_tile_regs,
                    a_scale=None,
                    gate_b_scale=None,
                    up_b_scale=None,
                    *,
                    prefetch_epilogue=False,
                    ku_count=k_unroll,
                ):
                    gate_list = list(acc_gate_in)
                    _single_b = mock_gate_only or gate_up_interleave
                    up_list = None if _single_b else list(acc_up_in)
                    mfma_res_ty = vec4_f32
                    epilogue_pf = None
                    bias_pf = None
                    if const_expr(prefetch_epilogue):
                        if const_expr(enable_bias):
                            bias_pf = []
                            for ni in range_constexpr(num_acc_n):
                                if const_expr(gate_up_interleave):
                                    _logical_col = (
                                        (by_n + n_tile_base) // arith.constant(2, index=True)
                                        + arith.constant((ni // 2) * 16, index=True)
                                        + lane_mod_16
                                    )
                                    _up_off = inter_idx if (ni % 2 == 1) else arith.constant(0, index=True)
                                    bias_offset = expert_off_idx + _up_off + _logical_col
                                else:
                                    global_n = by_n + n_tile_base + arith.constant(ni * 16, index=True) + lane_mod_16
                                    bias_offset = expert_off_idx + global_n
                                bias_pf.append(buffer_ops.buffer_load(bias_rsrc, bias_offset, vec_width=1, dtype=f32))
                        tw_pf = None
                        if const_expr(doweight_stage1):
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                            ii_idx_list_pf = [arith.constant(ii, index=True) for ii in range(4)]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.constant(mi * 16, index=True)
                                for ii in range_constexpr(4):
                                    row_off_pf = lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    sorted_row_pf = bx_m + mi_base_pf + row_off_pf
                                    tw_pf.append(
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc,
                                            sorted_row_pf,
                                            vec_width=1,
                                            dtype=f32,
                                        )
                                    )
                        epilogue_pf = (None, tw_pf, bias_pf)

                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    _eff_packed = (ku_count + pack_K - 1) // pack_K
                    # B-major: fix B (ni), cycle A (mi) -- B from VMEM stays
                    # in registers while A from LDS is repacked per mi.
                    for ku128 in range_constexpr(_eff_packed):
                        for ni in range_constexpr(num_acc_n_packed):
                            gate_bs_i32 = gate_b_scale[ku128 * num_acc_n_packed + ni]
                            gate_bs_val = vector.extract(
                                gate_bs_i32,
                                static_position=[0],
                                dynamic_position=[],
                            )
                            if const_expr(not _single_b):
                                up_bs_i32 = up_b_scale[ku128 * num_acc_n_packed + ni]
                                up_bs_val = vector.extract(up_bs_i32, static_position=[0], dynamic_position=[])
                            for ikxdl in range_constexpr(pack_K):
                                k_idx = ku128 * pack_K + ikxdl
                                if const_expr(k_idx < ku_count):
                                    gate_bp0, gate_bp1 = gate_b_tile_in[k_idx]
                                    if const_expr(not _single_b):
                                        up_bp0, up_bp1 = up_b_tile_in[k_idx]
                                    for inxdl in range_constexpr(pack_N):
                                        ni_idx = ni * pack_N + inxdl
                                        gb0 = gate_bp0[ni_idx]
                                        gb1 = gate_bp1[ni_idx]
                                        gb128 = pack_i64x4_to_i32x8(gb0, gb1, c0_i64, c0_i64)
                                        if const_expr(not _single_b):
                                            ub0 = up_bp0[ni_idx]
                                            ub1 = up_bp1[ni_idx]
                                            ub128 = pack_i64x4_to_i32x8(ub0, ub1, c0_i64, c0_i64)
                                        for mi in range_constexpr(m_repeat_packed):
                                            a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                                            a_scale_val = vector.extract(
                                                a_scale_i32,
                                                static_position=[0],
                                                dynamic_position=[],
                                            )
                                            for imxdl in range_constexpr(pack_M):
                                                mi_idx = mi * pack_M + imxdl
                                                _a_reg_idx = k_idx * m_repeat + mi_idx
                                                if const_expr(is_f8_a):
                                                    a0, a1, a2, a3 = a_tile_regs[_a_reg_idx]
                                                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                                                else:
                                                    a0, a1 = a_tile_regs[_a_reg_idx]
                                                    a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)
                                                acc_idx = mi_idx * num_acc_n + ni_idx
                                                gate_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        gb128,
                                                        gate_list[acc_idx],
                                                        cbsz,
                                                        blgp,
                                                        ikxdl * pack_M + imxdl,
                                                        a_scale_val,
                                                        ikxdl * pack_N + inxdl,
                                                        gate_bs_val,
                                                    ],
                                                )
                                                if const_expr(not _single_b):
                                                    up_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                        mfma_res_ty,
                                                        [
                                                            a128,
                                                            ub128,
                                                            up_list[acc_idx],
                                                            cbsz,
                                                            blgp,
                                                            ikxdl * pack_M + imxdl,
                                                            a_scale_val,
                                                            ikxdl * pack_N + inxdl,
                                                            up_bs_val,
                                                        ],
                                                    )
                    return gate_list, up_list, epilogue_pf

                def load_a_subtile(k_idx, mi_idx, lds_buffer):
                    """Load a single A sub-tile from LDS (one ds_read)."""
                    col_base = col_offset_base + (k_idx * 128) // a_elem_vec_pack
                    mi_val = arith.constant(mi_idx * 16, index=True)
                    curr_row = row_a_lds + mi_val
                    a0, a1 = lds_load_packs_k64(curr_row, col_base, lds_buffer)
                    if const_expr(is_f8_a):
                        a2, a3 = lds_load_packs_k64(curr_row, col_base + 64, lds_buffer)
                        return (a0, a1, a2, a3)
                    else:
                        return (a0, a1)

                _single_b_pipe = mock_gate_only or gate_up_interleave

                def compute_bmajor_mfma_phase(
                    all_a_tiles,
                    gate_b_single,
                    up_b_single,
                    a_scale_vals,
                    gate_bs_val,
                    up_bs_val,
                    gate_list,
                    up_list,
                    k_idx,
                    ni_idx,
                    ikxdl,
                    inxdl,
                ):
                    """B-major MFMA: fix one B (ni), cycle all A tiles (mi).

                    Packs B once and reuses across all mi iterations.
                    A tiles come from LDS (already available, no VMEM wait).

                    all_a_tiles: flat list indexed by [k*m_repeat + mi].
                    gate_b_single/up_b_single: (b0, b1) for one specific ni.
                      When _single_b_pipe (mock_gate_only or interleave), up_b_single is None.
                    a_scale_vals: list of A scale scalars indexed by mi_packed.
                    """
                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def _pack(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    mfma_res_ty = vec4_f32
                    gb128 = _pack(gate_b_single[0], gate_b_single[1], c0_i64, c0_i64)
                    if const_expr(not _single_b_pipe):
                        ub128 = _pack(up_b_single[0], up_b_single[1], c0_i64, c0_i64)

                    for mi_p in range_constexpr(m_repeat_packed):
                        a_scale_val = a_scale_vals[mi_p]
                        for imxdl in range_constexpr(pack_M):
                            mi_idx = mi_p * pack_M + imxdl
                            a_reg = all_a_tiles[k_idx * m_repeat + mi_idx]

                            if const_expr(is_f8_a):
                                a128 = _pack(a_reg[0], a_reg[1], a_reg[2], a_reg[3])
                            else:
                                a128 = _pack(a_reg[0], a_reg[1], c0_i64, c0_i64)

                            acc_idx = mi_idx * num_acc_n + ni_idx
                            gate_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                mfma_res_ty,
                                [
                                    a128,
                                    gb128,
                                    gate_list[acc_idx],
                                    cbsz,
                                    blgp,
                                    ikxdl * pack_M + imxdl,
                                    a_scale_val,
                                    ikxdl * pack_N + inxdl,
                                    gate_bs_val,
                                ],
                            )
                            if const_expr(not _single_b_pipe):
                                up_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    mfma_res_ty,
                                    [
                                        a128,
                                        ub128,
                                        up_list[acc_idx],
                                        cbsz,
                                        blgp,
                                        ikxdl * pack_M + imxdl,
                                        a_scale_val,
                                        ikxdl * pack_N + inxdl,
                                        up_bs_val,
                                    ],
                                )

                def _interleaved_half(
                    lds_read,
                    lds_write,
                    next_k_dma_py,
                    next_k_load,
                    prev_a_tile,
                    prev_gate_w,
                    prev_up_w,
                    prev_a_scale,
                    prev_gate_bs,
                    prev_up_bs,
                    acc_gate,
                    acc_up,
                ):
                    """One flatmm-style interleaved half-iteration (deep pipeline).

                    Generalized for arbitrary m_repeat (block_m=32, 64, ...).
                    DMA targets lds_write (OTHER buffer) while ds_read uses
                    lds_read (already DMA'd in previous half).

                    Interleaving schedule (per half):
                      Phase 0: scale VMEM + 2 ds_read(A) -> 4 MFMA(prev)
                      Phase 1..N: B VMEM(distributed) + 2 ds_read(A, if avail) -> 4 MFMA(prev)
                      Phase N+1..: remaining B VMEM -> 4 MFMA(prev)
                    """
                    _abs_k = k_base_idx + arith.constant(next_k_load, index=True)
                    _bk = _abs_k // arith.constant(2, index=True)
                    _sk = _abs_k // arith.constant(pack_K * 128, index=True)
                    _k_off = _sk * layout_b_scale.stride_k0

                    rocdl.sched_barrier(0)
                    rocdl.s_waitcnt(_vmcnt_before_barrier)
                    _barrier()
                    rocdl.sched_barrier(0)

                    # DMA A to OTHER buffer (for next half), non-blocking
                    _abs_k_dma = k_base_idx + arith.constant(next_k_dma_py, index=True)
                    if const_expr(use_async_copy and next_k_dma_py < int(_k_dim)):
                        prefetch_x_to_lds(_abs_k_dma, lds_write)
                    if const_expr(not use_async_copy):
                        _x_regs = load_x_tile(_abs_k_dma)

                    # ---- Extract previous scale values ----
                    _prev_asvs = []
                    for _mi_p in range_constexpr(m_repeat_packed):
                        _prev_asvs.append(
                            vector.extract(
                                prev_a_scale[_mi_p],
                                static_position=[0],
                                dynamic_position=[],
                            )
                        )
                    _prev_gsv_list = []
                    for _gs_ni in range_constexpr(num_acc_n_packed):
                        _prev_gsv_list.append(
                            vector.extract(
                                prev_gate_bs[_gs_ni],
                                static_position=[0],
                                dynamic_position=[],
                            )
                        )
                    if const_expr(not _single_b_pipe):
                        _prev_usv_list = []
                        for _us_ni in range_constexpr(num_acc_n_packed):
                            _prev_usv_list.append(
                                vector.extract(
                                    prev_up_bs[_us_ni],
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                            )

                    # ---- Execute phases from unified schedule ----
                    _a_all = {}
                    _b_gate_all = {}
                    _b_up_all = {}

                    for _p in range_constexpr(_pipe_n_phases):
                        # Scale VMEM loads (phase 0 only)
                        if const_expr(_pp_has_scale[_p]):
                            _new_as_list = []
                            for _mi_p in range_constexpr(m_repeat_packed):
                                if const_expr(a_scale_one):
                                    _new_as_list.append(_as1_const)
                                else:
                                    _new_as_list.append(_diag_valu(_rearrange_a_scale(_load_a_scale_i32(_mi_p, _k_off))))
                            _new_gs_list = []
                            for _gs_ni in range_constexpr(num_acc_n_packed):
                                _gs_raw = buffer_ops.buffer_load(
                                    sw_rsrc,
                                    _gate_scale_bases[_gs_ni] + _k_off,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=0,
                                )
                                _new_gs_list.append(_rearrange_b_scale(_gs_raw))
                            if const_expr(not _single_b_pipe):
                                _new_us_list = []
                                for _us_ni in range_constexpr(num_acc_n_packed):
                                    _us_raw = buffer_ops.buffer_load(
                                        sw_rsrc,
                                        _up_scale_bases[_us_ni] + _k_off,
                                        vec_width=1,
                                        dtype=T.i32,
                                        cache_modifier=0,
                                    )
                                    _new_us_list.append(_rearrange_b_scale(_us_raw))

                        # B VMEM loads
                        for _b_j in range_constexpr(len(_pp_b_loads[_p])):
                            _b_type, _b_ku, _b_ni = _pp_b_loads[_p][_b_j]
                            if const_expr(_b_type == "gate"):
                                _b_gate_all[(_b_ku, _b_ni)] = load_b_packs_k64(
                                    _bk,
                                    _b_ku,
                                    gate_n_blk_list[_b_ni],
                                    gate_n_intra_list[_b_ni],
                                )
                            else:
                                _b_up_all[(_b_ku, _b_ni)] = load_b_packs_k64(
                                    _bk,
                                    _b_ku,
                                    up_n_blk_list[_b_ni],
                                    up_n_intra_list[_b_ni],
                                )

                        # A ds_reads
                        if const_expr(_isched == 0):
                            rocdl.sched_barrier(0)
                        for _a_j in range_constexpr(len(_pp_a_reads[_p])):
                            _ak, _ami = _pp_a_reads[_p][_a_j]
                            _a_all[(_ak, _ami)] = load_a_subtile(
                                _ak,
                                _ami,
                                lds_read,
                            )
                        if const_expr(_isched == 0):
                            rocdl.sched_barrier(0)

                        # MFMAs on prev data
                        rocdl.s_setprio(1)
                        for _m_j in range_constexpr(len(_pp_mfma[_p])):
                            _k_idx, _ni_idx, _ikxdl, _inxdl, _ku128 = _pp_mfma[_p][_m_j]
                            _ni_packed_idx = _ni_idx // pack_N
                            _up_b_single = (
                                (
                                    prev_up_w[_k_idx][0][_ni_idx],
                                    prev_up_w[_k_idx][1][_ni_idx],
                                )
                                if not _single_b_pipe
                                else None
                            )
                            compute_bmajor_mfma_phase(
                                prev_a_tile,
                                (
                                    prev_gate_w[_k_idx][0][_ni_idx],
                                    prev_gate_w[_k_idx][1][_ni_idx],
                                ),
                                _up_b_single,
                                _prev_asvs,
                                _prev_gsv_list[_ni_packed_idx],
                                (_prev_usv_list[_ni_packed_idx] if not _single_b_pipe else None),
                                acc_gate,
                                acc_up,
                                _k_idx,
                                _ni_idx,
                                _ikxdl,
                                _inxdl,
                            )
                        rocdl.s_setprio(0)
                        if const_expr(_isched >= 2):
                            # CK-style: prescribe interleave of this phase's loads into its MFMA
                            # stream (R MFMAs per load), spreading VMEM(B)/DS(A) latency under
                            # compute; final big group drains remaining MFMAs.  Counts are
                            # best-effort (over-claim -> no-op, under-claim -> implicit tail group).
                            _na_s = len(_pp_a_reads[_p])
                            _nb_s = len(_pp_b_loads[_p])
                            if const_expr(_isched == 3):
                                # mode 3: alternate A(ds)/B(vmem) evenly across the MFMAs
                                for _i_s in range_constexpr(max(_na_s, _nb_s)):
                                    if const_expr(_i_s < _na_s):
                                        rocdl.sched_mfma(_ck_rate)
                                        rocdl.sched_dsrd(1)
                                    if const_expr(_i_s < _nb_s):
                                        rocdl.sched_mfma(_ck_rate)
                                        rocdl.sched_vmem(1)
                            else:
                                for _ in range_constexpr(_na_s):
                                    rocdl.sched_mfma(_ck_rate)
                                    rocdl.sched_dsrd(1)
                                for _ in range_constexpr(_nb_s):
                                    rocdl.sched_mfma(_ck_rate)
                                    rocdl.sched_vmem(1)
                            rocdl.sched_mfma(256)
                        rocdl.sched_barrier(0)

                    # ---- Assemble loaded data for next half-iteration ----
                    cur_a_tile = []
                    for _k in range_constexpr(k_unroll):
                        for _mi in range_constexpr(m_repeat):
                            cur_a_tile.append(_a_all[(_k, _mi)])

                    cur_gate_w = []
                    cur_up_w = None if _single_b_pipe else []
                    for ku in range_constexpr(k_unroll):
                        g_packs0, g_packs1 = [], []
                        u_packs0, u_packs1 = [], []
                        for ni in range_constexpr(num_acc_n):
                            g = _b_gate_all[(ku, ni)]
                            g_packs0.append(g[0])
                            g_packs1.append(g[1])
                            if const_expr(not _single_b_pipe):
                                u = _b_up_all[(ku, ni)]
                                u_packs0.append(u[0])
                                u_packs1.append(u[1])
                        cur_gate_w.append((g_packs0, g_packs1))
                        if const_expr(not _single_b_pipe):
                            cur_up_w.append((u_packs0, u_packs1))

                    cur_a_scale = []
                    for _mi_p in range_constexpr(m_repeat_packed):
                        cur_a_scale.append(
                            vector.from_elements(
                                T.vec(1, T.i32),
                                [_new_as_list[_mi_p]],
                            )
                        )
                    cur_gate_bs = []
                    for _gs_ni in range_constexpr(num_acc_n_packed):
                        cur_gate_bs.append(vector.from_elements(T.vec(1, T.i32), [_new_gs_list[_gs_ni]]))
                    if const_expr(not _single_b_pipe):
                        cur_up_bs = []
                        for _us_ni in range_constexpr(num_acc_n_packed):
                            cur_up_bs.append(vector.from_elements(T.vec(1, T.i32), [_new_us_list[_us_ni]]))
                    else:
                        cur_up_bs = None

                    if const_expr(not use_async_copy):
                        store_x_tile_to_lds(_x_regs, lds_write)

                    return (
                        cur_a_tile,
                        cur_gate_w,
                        cur_up_w,
                        cur_a_scale,
                        cur_gate_bs,
                        cur_up_bs,
                        acc_gate,
                        acc_up,
                    )

                # Pipeline (split ping/pong allocators)
                rocdl.sched_barrier(0)

                # raw_a_scale: stage this tile's FULL A-scale into LDS ONCE, BEFORE the K-loop /
                # async-X pipeline begins.  Placing it here keeps these cooperative VMEM loads out
                # of the in-flight X-DMA vmcnt chain, so the A ping/pong pipeline stays undisturbed.
                # Coalesced: consecutive threads copy consecutive i32 columns of row-major global
                # scale_em (via sx_rsrc, row base = bx_m*_raw_sni) into the row-major scale LDS.
                # ONE workgroup barrier publishes it to all waves.  The whole body runs under
                # blk_valid && exp_valid (block-uniform), so padded tiles never reach here, and
                # sx_rsrc is a bounded resource (any stray global read returns 0).
                if const_expr(_raw_a_scale_lds):
                    _sc_row_base = bx_m * arith.constant(_raw_sni_lds, index=True)
                    _sc_tx = tx * arith.constant(_sc_cp_vec, index=True)
                    for _j in range_constexpr(_sc_cp_iters):
                        _sc_f = _sc_tx + arith.constant(_j * total_threads * _sc_cp_vec, index=True)
                        _sc_v = buffer_ops.buffer_load(
                            sx_rsrc, _sc_row_base + _sc_f,
                            vec_width=_sc_cp_vec, dtype=T.i32, cache_modifier=0,
                        )
                        if const_expr(_sc_cp_vec == 1):
                            _sc_v = vector.from_elements(T.vec(1, T.i32), [_sc_v])
                        vector.store(_sc_v, lds_a_scale, [_sc_f], alignment=_sc_cp_vec * 4)
                    gpu.barrier()

                k0 = k_base_idx
                if const_expr(use_async_copy):
                    prefetch_x_to_lds(k0, lds_x_pong)
                else:
                    x_regs0 = load_x_tile(k0)
                    store_x_tile_to_lds(x_regs0, lds_x_pong)
                rocdl.sched_barrier(0)
                _k0_scale = k_base_idx // arith.constant(pack_K * 128, index=True)
                a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(_k0_scale)
                if const_expr(not contiguous_io):
                    _c_tile_m_idx = arith.constant(tile_m, index=True)
                    _tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, _c_tile_m_idx)
                    _if_tid = scf.IfOp(_tid_in_range)
                    with ir.InsertionPoint(_if_tid.then_block):
                        _tid_row = bx_m + tx
                        _tid_val = buffer_ops.buffer_load(sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32)
                        _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                        vector.store(_tid_vec1, lds_tid, [tx])
                        scf.YieldOp([])

                acc_gate = [acc_init] * num_acc_n * m_repeat
                acc_up = [acc_init] * num_acc_n * m_repeat if not _single_b_pipe else None

                _k1 = k_base_idx + arith.constant(tile_k, index=True)
                rocdl.sched_barrier(0)
                if const_expr(use_async_copy):
                    prefetch_x_to_lds(_k1, lds_x_ping)
                else:
                    _x_regs_prime = load_x_tile(_k1)
                    store_x_tile_to_lds(_x_regs_prime, lds_x_ping)

                _k0_b = k_base_idx // arith.constant(2, index=True)
                gate_w0, up_w0 = load_b_tile(_k0_b)
                # Prime the deep pipeline: DMA K=tile_k -> ping (1 tile ahead)
                if const_expr(use_async_copy):
                    rocdl.s_waitcnt(0)
                gpu.barrier()
                rocdl.sched_barrier(0)
                a_tile_pong = prefetch_full_a_from_lds(lds_x_pong)

                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(6)

                num_k_tiles_py = int(_k_dim) // int(tile_k)
                odd_k_tiles = (num_k_tiles_py % 2) == 1
                tail_tiles = 1 if odd_k_tiles else 2
                k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                if const_expr(k_main2_py < 0):
                    k_main2_py = 0

                gate_w_pong = gate_w0
                up_w_pong = up_w0

                rocdl.sched_barrier(0)

                if const_expr(k_main2_py > 0):
                    for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                        next_k_load_1 = k_iv_py + tile_k
                        next_k_load_2 = k_iv_py + tile_k * 2
                        next_k_dma_1 = k_iv_py + tile_k * 2
                        next_k_dma_2 = k_iv_py + tile_k * 3

                        # Half 1: read ping (DMA'd prev half), DMA->pong, MFMA(pong)
                        (
                            a_tile_ping,
                            gate_w_ping,
                            up_w_ping,
                            a_scale_ping,
                            gate_bs_ping,
                            up_bs_ping,
                            acc_gate,
                            acc_up,
                        ) = _interleaved_half(
                            lds_x_ping,
                            lds_x_pong,
                            next_k_dma_1,
                            next_k_load_1,
                            a_tile_pong,
                            gate_w_pong,
                            up_w_pong,
                            a_scale_pong,
                            gate_bs_pong,
                            up_bs_pong,
                            acc_gate,
                            acc_up,
                        )

                        # Half 2: read pong (DMA'd Half 1), DMA->ping, MFMA(ping)
                        (
                            a_tile_pong,
                            gate_w_pong,
                            up_w_pong,
                            a_scale_pong,
                            gate_bs_pong,
                            up_bs_pong,
                            acc_gate,
                            acc_up,
                        ) = _interleaved_half(
                            lds_x_pong,
                            lds_x_ping,
                            next_k_dma_2,
                            next_k_load_2,
                            a_tile_ping,
                            gate_w_ping,
                            up_w_ping,
                            a_scale_ping,
                            gate_bs_ping,
                            up_bs_ping,
                            acc_gate,
                            acc_up,
                        )

                # _wave_mod2_b = wave_id % arith.constant(2, index=True)
                # _wave_odd = arith.cmpi(
                #     CmpIPredicate.eq, _wave_mod2_b, arith.constant(1, index=True)
                # )
                # _if_wave_odd = scf.IfOp(_wave_odd)
                # with ir.InsertionPoint(_if_wave_odd.then_block):
                #     # gpu.barrier()
                #     _barrier()
                #     scf.YieldOp([])

                if const_expr(odd_k_tiles):
                    acc_gate, acc_up, epilogue_pf = compute_tile(
                        acc_gate,
                        acc_up,
                        gate_w_pong,
                        up_w_pong,
                        a_tile_pong,
                        a_scale_pong,
                        gate_bs_pong,
                        up_bs_pong,
                        prefetch_epilogue=True,
                        ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll,
                    )
                else:
                    _k_tail_rel = arith.constant(_k_dim - tile_k, index=True)
                    k_tail1 = k_base_idx + _k_tail_rel
                    x_regs_ping = []
                    if const_expr(use_async_copy):
                        prefetch_x_to_lds(k_tail1, lds_x_ping)
                    else:
                        x_regs_ping = load_x_tile(k_tail1)
                    if const_expr(_pad_ku_skip > 0):
                        gate_w_ping, up_w_ping = load_b_tile(
                            k_tail1 // arith.constant(2, index=True),
                            ku_limit=_tail_ku,
                        )
                        a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                            k_tail1 // arith.constant(pack_K * 128, index=True),
                            ku_packed_limit=_tail_ku_packed,
                        )
                    else:
                        gate_w_ping, up_w_ping = load_b_tile(k_tail1 // arith.constant(2, index=True))
                        a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                            k_tail1 // arith.constant(pack_K * 128, index=True)
                        )
                    acc_gate, acc_up, _ = compute_tile(
                        acc_gate,
                        acc_up,
                        gate_w_pong,
                        up_w_pong,
                        a_tile_pong,
                        a_scale_pong,
                        gate_bs_pong,
                        up_bs_pong,
                    )
                    if const_expr(not use_async_copy):
                        store_x_tile_to_lds(x_regs_ping, lds_x_ping)
                    rocdl.s_waitcnt(0)
                    _barrier()
                    if const_expr(_pad_ku_skip > 0):
                        a_tile_ping = prefetch_full_a_from_lds(lds_x_ping, ku_limit=_tail_ku)
                    else:
                        a_tile_ping = prefetch_full_a_from_lds(lds_x_ping)
                    acc_gate, acc_up, epilogue_pf = compute_tile(
                        acc_gate,
                        acc_up,
                        gate_w_ping,
                        up_w_ping,
                        a_tile_ping,
                        a_scale_ping,
                        gate_bs_ping,
                        up_bs_ping,
                        prefetch_epilogue=True,
                        ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll,
                    )

                bias_pf = None
                if const_expr(epilogue_pf is not None):
                    _, _, bias_pf = epilogue_pf

                # Activation helpers (f32 element-wise on vec4_f32)
                def _silu_elem(g):
                    """silu(x) = x * sigmoid(x); HW fast path: exp2, rcp"""
                    neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                    t = g * neg_log2e
                    emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                    one = arith.constant(1.0, type=f32)
                    den = one + emu
                    sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                    return g * sig

                def _silu_mul_vec4(gate_v4, up_v4):
                    """Element-wise silu(gate) * up on vec4_f32."""
                    result_elems = []
                    for ei in range_constexpr(4):
                        g = vector.extract(gate_v4, static_position=[ei], dynamic_position=[])
                        u = vector.extract(up_v4, static_position=[ei], dynamic_position=[])
                        result_elems.append(_silu_elem(g) * u)
                    return vector.from_elements(vec4_f32, result_elems)

                def _swiglu_mul_vec4(gate_v4, up_v4):
                    """Element-wise swiglu(gate, up) on vec4_f32.
                    swiglu(g, u) = g * sigmoid(alpha * g) * (u + 1)
                    with clamping: gate <= limit, -limit <= up <= limit.
                    """
                    result_elems = []
                    _alpha = arith.constant(1.702, type=f32)
                    _limit = arith.constant(7.0, type=f32)
                    _neg_limit = arith.constant(-7.0, type=f32)
                    _one = arith.constant(1.0, type=f32)
                    _neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                    for ei in range_constexpr(4):
                        g = vector.extract(gate_v4, static_position=[ei], dynamic_position=[])
                        u = vector.extract(up_v4, static_position=[ei], dynamic_position=[])
                        g = arith.minimumf(g, _limit)
                        u = arith.minimumf(u, _limit)
                        u = arith.maximumf(u, _neg_limit)
                        t = g * _alpha * _neg_log2e
                        emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                        den = _one + emu
                        sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                        result_elems.append(g * sig * (u + _one))
                    return vector.from_elements(vec4_f32, result_elems)

                def _act_vec4(gate_v4, up_v4):
                    """Dispatch activation based on `act` parameter."""
                    if const_expr(act == "swiglu"):
                        return _swiglu_mul_vec4(gate_v4, up_v4)
                    else:
                        return _silu_mul_vec4(gate_v4, up_v4)

                # Add bias to raw GEMM accumulators before activation.
                # bias layout: [E, 2*inter_dim] flat f32 (non-interleaved: gate then up).
                # For gate_up_interleave, map physical column to logical bias offset.
                if const_expr(enable_bias and not _is_splitk):
                    if const_expr(bias_pf is not None):
                        _bias_gate_vals = bias_pf
                    else:
                        _bias_gate_vals = []
                        for _ni in range_constexpr(num_acc_n):
                            if const_expr(gate_up_interleave):
                                _logical_col = (
                                    (by_n + n_tile_base) // arith.constant(2, index=True)
                                    + arith.constant((_ni // 2) * 16, index=True)
                                    + lane_mod_16
                                )
                                _up_off = inter_idx if (_ni % 2 == 1) else arith.constant(0, index=True)
                                _bias_off = expert_off_idx + _up_off + _logical_col
                            else:
                                _bn = by_n + n_tile_base + arith.constant(_ni * 16, index=True) + lane_mod_16
                                _bias_off = expert_off_idx + _bn
                            _bias_gate_vals.append(buffer_ops.buffer_load(bias_rsrc, _bias_off, vec_width=1, dtype=f32))
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            _bsplat = vector.from_elements(vec4_f32, [_bias_gate_vals[_ni]] * 4)
                            acc_gate[_aidx] = arith.addf(acc_gate[_aidx], _bsplat)

                    if const_expr(not (mock_gate_only or gate_up_interleave)):
                        _bias_up_vals = []
                        for _ni in range_constexpr(num_acc_n):
                            _bn = by_n + n_tile_base + arith.constant(_ni * 16, index=True) + lane_mod_16
                            _bias_up_vals.append(
                                buffer_ops.buffer_load(
                                    bias_rsrc,
                                    expert_off_idx + inter_idx + _bn,
                                    vec_width=1,
                                    dtype=f32,
                                )
                            )
                        for _mi in range_constexpr(m_repeat):
                            for _ni in range_constexpr(num_acc_n):
                                _aidx = _mi * num_acc_n + _ni
                                _bsplat = vector.from_elements(vec4_f32, [_bias_up_vals[_ni]] * 4)
                                acc_up[_aidx] = arith.addf(acc_up[_aidx], _bsplat)

                if const_expr(gate_up_interleave and not _is_splitk):
                    _gui_out_n = num_acc_n // pack_N
                    acc = [None] * (_gui_out_n * m_repeat)
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(_gui_out_n):
                            _g_idx = _mi * num_acc_n + _ni * pack_N
                            _u_idx = _g_idx + 1
                            _out_idx = _mi * _gui_out_n + _ni
                            acc[_out_idx] = _act_vec4(acc_gate[_g_idx], acc_gate[_u_idx])
                elif const_expr(not _is_splitk):
                    acc = [None] * (int(num_acc_n) * int(m_repeat))
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            acc[_aidx] = _silu_mul_vec4(acc_gate[_aidx], acc_up[_aidx])

                # ---- Epilogue: CShuffle + direct store (accumulate=False) ----
                # Output: out[(t*topk+s) * inter_dim + col] = silu(gate) * up
                # For split-K: skip silu, output gate/up separately with atomic add
                tw_pf = None
                bias_pf = None
                if const_expr(epilogue_pf is not None):
                    _, tw_pf, bias_pf = epilogue_pf

                mask24_i32 = arith.constant(0xFFFFFF)
                topk_i32_v = topk_i32
                tokens_i32_v = tokens_i32

                from flydsl._mlir.dialects import fly as _fly

                _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
                out_base_ptr = _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty, arg_out)
                out_base_i64 = llvm.ptrtoint(T.i64, out_base_ptr)
                out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

                if const_expr(lds_out is None):
                    raise RuntimeError("CShuffle epilogue requires lds_out")

                _apply_weight = doweight_stage1 and not _is_splitk

                def write_row_to_lds(
                    *,
                    mi: int,
                    ii: int,
                    row_in_tile,
                    row,
                    row_base_lds,
                    col_base_local,
                    num_acc_n: int,
                    lds_out,
                ):
                    if const_expr(_apply_weight):
                        tw_idx = (mi * 4) + ii
                        if const_expr(tw_pf is not None):
                            tw = tw_pf[tw_idx]
                        else:
                            tw = buffer_ops.buffer_load(sorted_w_rsrc, row, vec_width=1, dtype=f32)
                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                        if const_expr(_apply_weight):
                            v = v * tw
                        if const_expr(_need_quant):
                            lds_idx = row_base_lds + col_local
                            vec1_f32 = T.vec(1, f32)
                            v1 = vector.from_elements(vec1_f32, [v])
                            vector.store(v1, lds_out, [lds_idx], alignment=4)
                        else:
                            v_out = arith.trunc_f(out_elem(), v)
                            lds_idx = row_base_lds + col_local
                            vec1_out = T.vec(1, out_elem())
                            v1 = vector.from_elements(vec1_out, [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                _out_row_stride = (
                    inter_dim * 2 * out_elem_bytes
                    if _is_splitk
                    else (inter_dim // 2 if _need_fp4 else (inter_dim if _need_fp8 else inter_dim * out_elem_bytes))
                )

                def precompute_row(*, row_local, row):
                    row_i32 = arith.index_cast(T.i32, row)
                    row_valid = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                    if const_expr(contiguous_io):
                        # expert-major contiguous output: out[row] (no token*topk+s scatter).
                        row_byte_base = out_base_idx + row * arith.constant(_out_row_stride, index=True)
                        return ((row_i32, row_byte_base), row_valid)
                    fused2 = memref.load(lds_tid, [row_local])
                    t = fused2 & mask24_i32
                    s = fused2 >> 24
                    t_ok = arith.cmpi(CmpIPredicate.ult, t, tokens_i32_v)
                    s_ok = arith.cmpi(CmpIPredicate.ult, s, topk_i32_v)
                    row_valid = arith.andi(row_valid, arith.andi(t_ok, s_ok))
                    t_idx = arith.index_cast(ir.IndexType.get(), t)
                    s_idx = arith.index_cast(ir.IndexType.get(), s)
                    ts_idx = t_idx * arith.constant(topk, index=True) + s_idx
                    row_byte_base = out_base_idx + ts_idx * arith.constant(_out_row_stride, index=True)
                    return ((fused2, row_byte_base), row_valid)

                def _idx_to_llvm_ptr(idx_val, addr_space=1):
                    idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
                    i64_v = arith.index_cast(T.i64, idx_v)
                    i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
                    ptr_ty = ir.Type.parse(f"!llvm.ptr<{addr_space}>")
                    return llvm.inttoptr(ptr_ty, i64_raw)

                _e_vec = _e_vec_s1
                _e_vec_sk = 2
                _cshuffle_nlane = min(32, tile_n // _e_vec)
                _cshuffle_nlane_sk = min(32, tile_n // _e_vec_sk)
                _num_threads_per_quant_blk = _num_threads_per_quant_blk_s1

                _c0_i32 = arith.constant(0, type=T.i32)
                _c1_i32 = arith.constant(1, type=T.i32)
                _c2_i32 = arith.constant(2, type=T.i32)
                _c3_i32 = arith.constant(3, type=T.i32)
                _c4_i32 = arith.constant(4, type=T.i32)
                _c5_i32 = arith.constant(5, type=T.i32)
                _c7_i32 = arith.constant(7, type=T.i32)
                _c15_i32 = arith.constant(15, type=T.i32)
                _c21_i32 = arith.constant(21, type=T.i32)
                _c23_i32 = arith.constant(23, type=T.i32)
                _c28_i32 = arith.constant(28, type=T.i32)
                _c31_i32 = arith.constant(31, type=T.i32)
                _c32_i32 = arith.constant(32, type=T.i32)
                _c64_i32 = arith.constant(64, type=T.i32)
                _c126_i32 = arith.constant(126, type=T.i32)
                _c127_i32 = arith.constant(127, type=T.i32)
                _c254_i32 = arith.constant(254, type=T.i32)
                _c256_i32 = arith.constant(256, type=T.i32)
                _c0xFF_i32 = arith.constant(0xFF, type=T.i32)
                _c0x200000_i32 = arith.constant(0x200000, type=T.i32)
                _c0xFF800000_i32 = arith.constant(0xFF800000, type=T.i32)
                _c0x400000_i32 = arith.constant(0x400000, type=T.i32)
                _c0x7FFFFF_i32 = arith.constant(0x7FFFFF, type=T.i32)
                _c0x80000000_i32 = arith.constant(0x80000000, type=T.i32)
                _c0_f32 = arith.constant(0.0, type=T.f32)

                _c8_i32 = arith.constant(8, type=T.i32)
                _fp_headroom = 2 if _need_fp4 else (8 if _need_fp8 else 0)
                _c_headroom_i32 = arith.constant(_fp_headroom, type=T.i32)

                def _f32_to_e2m1(qx_f32):
                    """Convert a scaled f32 value to fp4 (e2m1) 4-bit integer."""
                    qx = qx_f32.bitcast(T.i32)
                    s = qx & _c0x80000000_i32
                    e = (qx >> _c23_i32) & _c0xFF_i32
                    m = qx & _c0x7FFFFF_i32
                    adj_exp = arith.maxsi(_c126_i32 - e, _c0_i32)
                    m_denorm = (_c0x400000_i32 | (m >> _c1_i32)) >> adj_exp
                    is_denorm = arith.cmpi(CmpIPredicate.ult, e, _c127_i32)
                    m = arith.select(is_denorm, m_denorm, m)
                    e = arith.maxsi(e - _c126_i32, _c0_i32)
                    combined = (e << _c2_i32) | (m >> _c21_i32)
                    rounded = (combined + _c1_i32) >> _c1_i32
                    e2m1 = arith.minui(rounded, _c7_i32)
                    return (s >> _c28_i32) | e2m1

                if const_expr(_need_sort):
                    _n32_sort = _sorted_scale_cols_i32 * _c32_i32

                # Mutable slot for split-K N-offset (gate=0, up=inter_dim)
                _sk_n_offset = [0]

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    fused, row_byte_base = row_ctx
                    if const_expr(_need_quant and not _is_splitk):
                        frag_vals = []
                        for i in range_constexpr(_e_vec):
                            frag_vals.append(vector.extract(frag, static_position=[i], dynamic_position=[]))

                        local_max = _c0_f32
                        for i in range_constexpr(_e_vec):
                            abs_v = llvm.call_intrinsic(f32, "llvm.fabs.f32", [frag_vals[i]], [], [])
                            local_max = arith.maximumf(local_max, abs_v)

                        for _si in range_constexpr(_num_shuffle_steps_s1):
                            off = arith.constant(_shuffle_dists_s1[_si], type=T.i32)
                            peer = local_max.shuffle_xor(off, _c64_i32)
                            local_max = arith.maximumf(local_max, peer)

                        max_i32 = local_max.bitcast(T.i32)
                        max_rounded = (max_i32 + _c0x200000_i32) & _c0xFF800000_i32
                        exp_field = max_rounded >> _c23_i32
                        e8m0_biased = arith.maxsi(exp_field - _c_headroom_i32, _c0_i32)

                        quant_exp = _c254_i32 - e8m0_biased
                        quant_scale = (quant_exp << _c23_i32).bitcast(T.f32)

                        if const_expr(_need_fp4):
                            fp4_vals = []
                            for i in range_constexpr(_e_vec):
                                scaled_v = frag_vals[i] * quant_scale
                                fp4_vals.append(_f32_to_e2m1(scaled_v))

                            packed_i32 = fp4_vals[0] | (fp4_vals[1] << _c4_i32)
                            for k in range_constexpr(1, _e_vec // 2):
                                byte_k = fp4_vals[2 * k] | (fp4_vals[2 * k + 1] << _c4_i32)
                                packed_i32 = packed_i32 | (byte_k << arith.constant(k * 8, type=T.i32))

                            ptr_addr_idx = row_byte_base + col_g0 / arith.constant(2, index=True)
                            out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                            _pack_bytes = _e_vec // 2
                            if const_expr(_pack_bytes == 1):
                                store_val = arith.TruncIOp(T.i8, packed_i32)
                                store_raw = store_val._value if hasattr(store_val, "_value") else store_val
                                llvm.StoreOp(store_raw, out_ptr_v, alignment=1, nontemporal=True)
                            elif const_expr(_pack_bytes == 2):
                                store_val = arith.TruncIOp(T.i16, packed_i32)
                                store_raw = store_val._value if hasattr(store_val, "_value") else store_val
                                llvm.StoreOp(store_raw, out_ptr_v, alignment=2, nontemporal=True)
                            else:
                                packed_raw = packed_i32._value if hasattr(packed_i32, "_value") else packed_i32
                                llvm.StoreOp(packed_raw, out_ptr_v, alignment=4, nontemporal=True)

                        elif const_expr(_need_fp8):
                            scaled_vals = []
                            for i in range_constexpr(_e_vec):
                                scaled_vals.append(frag_vals[i] * quant_scale)

                            ptr_addr_idx = row_byte_base + col_g0
                            if const_expr(_e_vec <= 4):
                                packed_i32 = _c0_i32
                                for _w in range_constexpr(_e_vec // 2):
                                    packed_i32 = rocdl.cvt_pk_fp8_f32(
                                        T.i32,
                                        scaled_vals[2 * _w],
                                        scaled_vals[2 * _w + 1],
                                        packed_i32,
                                        _w,
                                    )
                                out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                                if const_expr(_e_vec == 2):
                                    store_val = arith.TruncIOp(T.i16, packed_i32)
                                    store_raw = store_val._value if hasattr(store_val, "_value") else store_val
                                    llvm.StoreOp(
                                        store_raw,
                                        out_ptr_v,
                                        alignment=2,
                                        nontemporal=True,
                                    )
                                else:
                                    packed_raw = packed_i32._value if hasattr(packed_i32, "_value") else packed_i32
                                    llvm.StoreOp(
                                        packed_raw,
                                        out_ptr_v,
                                        alignment=4,
                                        nontemporal=True,
                                    )
                            else:
                                for _wg in range_constexpr(_e_vec // 4):
                                    _b = _wg * 4
                                    packed_w = _c0_i32
                                    packed_w = rocdl.cvt_pk_fp8_f32(
                                        T.i32,
                                        scaled_vals[_b],
                                        scaled_vals[_b + 1],
                                        packed_w,
                                        0,
                                    )
                                    packed_w = rocdl.cvt_pk_fp8_f32(
                                        T.i32,
                                        scaled_vals[_b + 2],
                                        scaled_vals[_b + 3],
                                        packed_w,
                                        1,
                                    )
                                    word_ptr = ptr_addr_idx + arith.constant(_wg * 4, index=True)
                                    out_ptr_v = _idx_to_llvm_ptr(word_ptr)
                                    packed_raw = packed_w._value if hasattr(packed_w, "_value") else packed_w
                                    llvm.StoreOp(
                                        packed_raw,
                                        out_ptr_v,
                                        alignment=4,
                                        nontemporal=True,
                                    )

                        if const_expr(_need_sort):
                            col_g0_i32 = arith.index_cast(T.i32, col_g0)
                            is_scale_writer = arith.cmpi(CmpIPredicate.eq, col_g0_i32 & _c31_i32, _c0_i32)
                            _if_scale = scf.IfOp(is_scale_writer)
                            with ir.InsertionPoint(_if_scale.then_block):
                                row_i32_s = arith.index_cast(T.i32, row)
                                col_s_i32 = col_g0_i32 >> _c5_i32
                                d0 = row_i32_s >> _c5_i32
                                d1 = (row_i32_s >> _c4_i32) & _c1_i32
                                d2 = row_i32_s & _c15_i32
                                d3 = col_s_i32 >> _c3_i32
                                d4 = (col_s_i32 >> _c2_i32) & _c1_i32
                                d5 = col_s_i32 & _c3_i32
                                byte_off = (
                                    d0 * _n32_sort + d3 * _c256_i32 + d5 * _c64_i32 + d2 * _c4_i32 + d4 * _c2_i32 + d1
                                )
                                e8m0_i8 = arith.TruncIOp(T.i8, e8m0_biased)
                                buffer_ops.buffer_store(
                                    e8m0_i8,
                                    sorted_scale_rsrc,
                                    byte_off,
                                    offset_is_bytes=True,
                                )
                                scf.YieldOp([])
                    elif const_expr(_is_splitk):
                        col_idx = col_g0 + arith.constant(_sk_n_offset[0], index=True)
                        byte_off_col = col_idx * arith.constant(out_elem_bytes, index=True)
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            out_ptr_v,
                            frag_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=_e_vec_sk * out_elem_bytes,
                        )
                    else:
                        col_idx = col_g0
                        byte_off_col = col_idx * arith.constant(out_elem_bytes, index=True)
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.StoreOp(
                            frag_v,
                            out_ptr_v,
                            alignment=_e_vec * out_elem_bytes,
                            nontemporal=True,
                        )

                _frag_elem = (
                    ir.F32Type.get() if _need_quant else (ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get())
                )

                if const_expr(gate_up_interleave and not _is_splitk):
                    # gui without splitk: acc has activation applied, halved N
                    _gui_eff_n = _gui_out_n
                    _gui_tile_n = tile_n // 2
                    _gui_cshuffle_nlane = min(32, _gui_tile_n // _e_vec)
                    _gui_by_n = by_n / arith.constant(2, index=True)
                    _gui_n_tile_base = n_tile_base / arith.constant(2, index=True)
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=_gui_tile_n,
                        e_vec=_e_vec,
                        cshuffle_nlane=_gui_cshuffle_nlane,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=_gui_eff_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=_gui_by_n,
                        n_tile_base=_gui_n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
                elif const_expr(mock_gate_only or (gate_up_interleave and _is_splitk)):
                    # mock_gate_only: single pass, by_n covers full [0, 2*inter_dim)
                    _eff_e_vec = _e_vec_sk
                    acc = acc_gate
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_eff_e_vec,
                        cshuffle_nlane=_cshuffle_nlane_sk,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        lds_out_split=lds_out_B,
                    )
                elif const_expr(_is_splitk):
                    # Two-pass epilogue: gate then up, each with atomic add
                    _eff_e_vec = _e_vec_sk

                    # Pass 1: gate
                    acc = acc_gate
                    _sk_n_offset[0] = 0
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_eff_e_vec,
                        cshuffle_nlane=_cshuffle_nlane_sk,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        lds_out_split=lds_out_B,
                    )

                    gpu.barrier()

                    # Pass 2: up
                    acc = acc_up
                    _sk_n_offset[0] = inter_dim
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_eff_e_vec,
                        cshuffle_nlane=_cshuffle_nlane_sk,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        lds_out_split=lds_out_B,
                    )
                else:
                    c_shuffle_epilog(
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=_e_vec,
                        cshuffle_nlane=_cshuffle_nlane,
                        block_size=total_threads,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=_frag_elem,
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                        lds_out_split=lds_out_B,
                    )

            _if_blk = scf.IfOp(blk_valid)
            with ir.InsertionPoint(_if_blk.then_block):
                _ifexpert_of = scf.IfOp(exp_valid)
                with ir.InsertionPoint(_ifexpert_of.then_block):
                    if const_expr(overlap_gate or _fuse_hs):
                        # comm/compute overlap gate: spin until ALL senders' tokens for
                        # this (local) expert have landed, then acquire-fence so this CTA
                        # reads fresh payload.  blk_valid+exp_valid are block-uniform, so
                        # every thread here gates identically (no deadlock).
                        _le_idx = arith.index_cast(
                            ir.IndexType.get(),
                            arith.subi(expert_i32, arith.constant(rank * experts_per_rank, type=T.i32)))
                        _exp_rsrc = buffer_ops.create_buffer_resource_from_addr(addr_expected_real)
                        _exp_cnt = buffer_ops.buffer_load(
                            _exp_rsrc, _le_idx, vec_width=1, dtype=T.i32)
                        _thr = arith.subi(_exp_cnt, arith.constant(1, type=T.i32))
                        _le_i64 = arith.index_cast(T.i64, _le_idx)
                        _pd_addr = addr_payload_done + _le_i64 * arith.constant(4, type=T.i64)
                        mori_shmem.int32_wait_until_greater_than(_pd_addr, _thr)
                        _fence_system_acquire()
                    if const_expr(_fuse_tilecount):
                        # DIAG: count this (valid tile, valid expert) GEMM-body exec into
                        # addr_payload_done[0] (agent atomic, tid0 only) -> #tiles actually computed.
                        if tx == arith.constant(0, index=True):
                            _epk.atomic_add_agent(addr_payload_done, arith.constant(1))
                    _moe_gemm1_body()
                    scf.YieldOp([])
                scf.YieldOp([])

            gpu.barrier()
            scf.YieldOp([])
            _for_ip.__exit__(None, None, None)

    # -- Host launcher --
    _cache_tag = (
        module_name,
        a_dtype,
        b_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage1,
        act,
        enable_bias,
        model_dim_pad,
        inter_dim_pad,
        use_cshuffle_epilog,
        persist_m,
        use_async_copy,
        waves_per_eu,
        k_batch,
        gate_mode,
        a_scale_one,
        xcd_swizzle,
    )

    def _launch_body(arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w, arg_sorted_token_ids,
                     arg_expert_ids, arg_sorted_weights, arg_max_token_ids, arg_bias,
                     arg_out_scale_sorted, i32_tokens_in, i32_inter_in, i32_k_in,
                     i32_size_expert_ids_in, addr_payload_done, addr_expected_real, stream,
                     addr_disp=None, i32_cur_tok=None, addr_in_tok=None, addr_in_idx=None,
                     addr_in_wts=None, addr_in_sc=None):
        _ = _cache_tag
        if addr_disp is None:
            addr_disp = fx.Int64(0)
        if i32_cur_tok is None:
            i32_cur_tok = fx.Int32(0)
        if addr_in_tok is None:
            addr_in_tok = fx.Int64(0); addr_in_idx = fx.Int64(0)
            addr_in_wts = fx.Int64(0); addr_in_sc = fx.Int64(0)
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        inter_in = arith.index_cast(ir.IndexType.get(), i32_inter_in.ir_value())
        tile_n_index = arith.constant(tile_n, index=True)
        inter_dim_pad_total = arith.constant(2 * inter_dim_pad, index=True)
        if const_expr(mock_gate_only or gate_up_interleave):
            gx = (inter_in - inter_dim_pad_total + tile_n_index - 1) / tile_n_index
        else:
            gx = (inter_in - inter_dim_pad_total + 2 * tile_n_index - 1) / tile_n_index / arith.constant(2, index=True)
        if const_expr(_fuse_hs):
            # handshake OVERLAP: append np_cols PRODUCER columns to grid_x (by>=gx).  Consumers
            # (by<gx) run the GEMM; producers write payload + signal payload_done.
            gx = gx + arith.constant(_fz_np_cols, index=True)
        if const_expr(_fuse):
            # FUSED megakernel grid_y (host-fixed constant).  fixedslot (decode): co-resident
            # (total=gx*gy<=cu_num) so the in-prologue arrival grid-sync is deadlock-free.
            # handshake (prefill): may OVERSUBSCRIBE (total>cu_num) -- the producer/consumer phases
            # have ZERO grid barriers (block0-self-sufficient + meta_flag), so HW wave-scheduling
            # overlaps producer xGMI writes with consumer MFMA (§9.3).  _fz_gy is sized per scheme.
            gy = arith.constant(_fz_gy, index=True)
        else:
            # non-fused persistent fallback (general callers): grid_y = min(cu_num, max_blocks).
            _c_cu_l = arith.constant(_cu_num, index=True)
            _seid_l = arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
            gy = arith.select(arith.cmpi(CmpIPredicate.ult, _c_cu_l, _seid_l), _c_cu_l, _seid_l)

        moe_gemm1(
            arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w, arg_sorted_token_ids,
            arg_expert_ids, arg_sorted_weights, arg_max_token_ids, arg_bias,
            arg_out_scale_sorted, i32_tokens_in, i32_inter_in, i32_k_in,
            i32_size_expert_ids_in, addr_payload_done, addr_expected_real, addr_disp, i32_cur_tok,
            addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
        ).launch(grid=(gx, gy, k_batch), block=(total_threads, 1, 1), stream=stream)

    if _fuse:
        # Fused launcher.  addr_payload_done/addr_expected_real are 0 for fixedslot (strict-phase,
        # no gate) and the real per-expert buffers for handshake (overlap_gate).
        @flyc.jit
        def launch_mixed_moe_gemm1(
            arg_out: fx.Tensor, arg_x: fx.Tensor, arg_w: fx.Tensor, arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor, arg_sorted_token_ids: fx.Tensor, arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor, arg_max_token_ids: fx.Tensor, arg_bias: fx.Tensor,
            arg_out_scale_sorted: fx.Tensor, i32_tokens_in: fx.Int32, i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32, i32_size_expert_ids_in: fx.Int32,
            addr_payload_done: fx.Int64, addr_expected_real: fx.Int64,
            addr_disp: fx.Int64, i32_cur_tok: fx.Int32,
            addr_in_tok: fx.Int64, addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64,
            stream: fx.Stream,
        ):
            _launch_body(arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w, arg_sorted_token_ids,
                         arg_expert_ids, arg_sorted_weights, arg_max_token_ids, arg_bias,
                         arg_out_scale_sorted, i32_tokens_in, i32_inter_in, i32_k_in,
                         i32_size_expert_ids_in, addr_payload_done, addr_expected_real, stream,
                         addr_disp=addr_disp, i32_cur_tok=i32_cur_tok, addr_in_tok=addr_in_tok,
                         addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc)
    elif overlap_gate:
        @flyc.jit
        def launch_mixed_moe_gemm1(
            arg_out: fx.Tensor, arg_x: fx.Tensor, arg_w: fx.Tensor, arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor, arg_sorted_token_ids: fx.Tensor, arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor, arg_max_token_ids: fx.Tensor, arg_bias: fx.Tensor,
            arg_out_scale_sorted: fx.Tensor, i32_tokens_in: fx.Int32, i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32, i32_size_expert_ids_in: fx.Int32,
            addr_payload_done: fx.Int64, addr_expected_real: fx.Int64, stream: fx.Stream,
        ):
            _launch_body(arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w, arg_sorted_token_ids,
                         arg_expert_ids, arg_sorted_weights, arg_max_token_ids, arg_bias,
                         arg_out_scale_sorted, i32_tokens_in, i32_inter_in, i32_k_in,
                         i32_size_expert_ids_in, addr_payload_done, addr_expected_real, stream)
    else:
        @flyc.jit
        def launch_mixed_moe_gemm1(
            arg_out: fx.Tensor, arg_x: fx.Tensor, arg_w: fx.Tensor, arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor, arg_sorted_token_ids: fx.Tensor, arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor, arg_max_token_ids: fx.Tensor, arg_bias: fx.Tensor,
            arg_out_scale_sorted: fx.Tensor, i32_tokens_in: fx.Int32, i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32, i32_size_expert_ids_in: fx.Int32, stream: fx.Stream,
        ):
            _launch_body(arg_out, arg_x, arg_w, arg_scale_x, arg_scale_w, arg_sorted_token_ids,
                         arg_expert_ids, arg_sorted_weights, arg_max_token_ids, arg_bias,
                         arg_out_scale_sorted, i32_tokens_in, i32_inter_in, i32_k_in,
                         i32_size_expert_ids_in, fx.Int64(0), fx.Int64(0), stream)

    # Expose the resource sizing the facade's occupancy gate needs (host-only; no effect on the
    # compiled kernel).  lds_total_bytes = real per-block LDS AFTER the waves_per_eu floor (what
    # bounds occupancy -> the gate input); lds_data_bytes = pre-floor DATA LDS; lds_scale_bytes =
    # the raw_a_scale scale-LDS staging region (0 when pre-swizzled).
    launch_mixed_moe_gemm1.lds_total_bytes = int(_lds_total_bytes)
    launch_mixed_moe_gemm1.lds_data_bytes = int(_lds_data_bytes)
    launch_mixed_moe_gemm1.lds_scale_bytes = int(_lds_scale_bytes)
    launch_mixed_moe_gemm1.raw_a_scale = bool(raw_a_scale)
    return launch_mixed_moe_gemm1
