# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused intranode MoE stage1 (A8W4): **dispatch_pull + MFMA GEMM + activation** in one ``@flyc.kernel``.

Vendors ``compile_mixed_moe_gemm1`` / ``moe_gemm1`` from ``mixed_moe_gemm_2stage.py``.  GEMM reads
``pull_row`` built by ``kernels.moe_metadata_dispatch_recv_meta`` (separate launch), then P2P-loads
peer ``x_staging`` rows into LDS and runs MFMA.

"""

import os
import functools
from enum import Enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl._mlir.ir import IntegerType as _IntTy_mlir
from flydsl.expr.typing import T

from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl, const_expr
from flydsl._mlir.dialects import llvm, scf, memref
from flydsl._mlir.dialects.arith import CmpIPredicate

from kernels.mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    lds_store_8b_xor16,
    lds_store_4b_xor16,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    tile_chunk_coord_i32,
    swizzle_xor16,
)
from kernels.mfma_epilogues import c_shuffle_epilog
from kernels.layout_utils import crd2idx, idx2crd, get as layout_get
from kernels.kernels_common import _if_then, validate_moe_dtypes
from kernels.moe_mxfp4_sort_flydsl import mxfp4_sort_const_geom
from kernels.moe_sorting_flydsl import moe_sorting_const_sizes
from kernels.dispatch_combine_intranode_kernel import (
    _lv_unwrap,
    _to_idx,
    atomic_add_global_at,
    buffer_load as _dc_buffer_load,
    buffer_store as _dc_buffer_store,
    create_buffer_resource_from_addr,
    store_i32_system,
)


def _global_device_ptr_from_i64(addr_i64):
    """i64 → !llvm.ptr<1> (global address space)."""
    return llvm.IntToPtrOp(
        llvm.PointerType.get(address_space=1), _lv_unwrap(addr_i64)
    ).result


def atomic_add_global_acq_rel(addr_i64, val):
    """Global atomic fetch-and-add (acq_rel). Pair with ``load_i32_global_acquire`` for device-wide counters."""
    ptr = _global_device_ptr_from_i64(addr_i64)
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add, ptr, _lv_unwrap(val), llvm.AtomicOrdering.acq_rel
    ).res


def load_i32_global_acquire(addr_i64):
    """Global i32 load with acquire semantics (for spinning on a device-wide counter)."""
    ptr = _global_device_ptr_from_i64(addr_i64)
    _i32 = _IntTy_mlir.get_signless(32)
    return llvm.LoadOp(
        _i32, ptr, alignment=4, ordering=llvm.AtomicOrdering.acquire
    ).result


def _i64_cell_addr(addr_base, word_off_i32):
    """addr_base (i64 mlir) + word_off_i32 * 4 → byte offset cell address (i64)."""
    _i64 = _IntTy_mlir.get_signless(64)
    off64 = arith.zext_i64(word_off_i32)
    c4 = arith.constant(4, type=_i64)
    mul = arith.muli(off64, c4)
    return arith.addi(_lv_unwrap(addr_base), mul)


def load_i32_global_acquire_cell(addr_base, word_index_i32):
    cell_i64 = _i64_cell_addr(addr_base, word_index_i32)
    ptr = llvm.IntToPtrOp(
        llvm.PointerType.get(address_space=1), cell_i64
    ).result
    _i32 = _IntTy_mlir.get_signless(32)
    return llvm.LoadOp(
        _i32, ptr, alignment=4, ordering=llvm.AtomicOrdering.acquire
    ).result


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



# ---- Vendored from mixed_moe_gemm_2stage.py: compile_mixed_moe_gemm1 (lines 102-2694) ----
# SPDX-License-Identifier: Apache-2.0 (same as source file)
@functools.lru_cache(maxsize=None)
def compile_fused_dispatch_gather_gemm1_a8w4(
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
    waves_per_eu: int = 4,
    k_batch: int = 1,
    b_nt: int = 0,
    gate_mode: GateMode = GateMode.SEPARATED,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    compile_rank: int = 0,
    use_local_pool: bool = False,
    pull_gemm_overlap: bool = False,
    pull_gemm_producer_planes: int = 1,
    max_tok_per_rank: int = 0,
):
    """Compile intranode stage1 GEMM (P2P pull or local pool + gate/up MFMA).

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

    if gpu_arch != "gfx950":
        raise ValueError(
            "compile_fused_dispatch_gather_gemm1_a8w4 targets gfx950 only; "
            f"HIP arch={gpu_arch!r}"
        )

    if int(model_dim) % 4 != 0:
        raise ValueError(
            "intranode dispatch_pull requires model_dim % 4 == 0 (4-byte row chunks), "
            f"got model_dim={model_dim}"
        )

    is_f16_a = a_dtype == "fp16"
    is_f16_b = b_dtype == "fp16"
    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"

    sort_block_m = max(32, tile_m)
    _compile_rank = int(compile_rank)
    _use_local_pool = bool(use_local_pool)
    _pull_gemm_overlap = bool(pull_gemm_overlap)
    _overlap_producer_planes = max(1, int(pull_gemm_producer_planes))
    if _pull_gemm_overlap:
        if not _use_local_pool:
            raise ValueError("pull_gemm_overlap requires use_local_pool=True")
        if int(max_tok_per_rank) <= 0:
            raise ValueError("pull_gemm_overlap requires max_tok_per_rank")
        if int(k_batch) != 1:
            raise ValueError("pull_gemm_overlap P0 requires k_batch=1")
        if int(tile_m) % 32 != 0:
            raise ValueError("pull_gemm_overlap requires tile_m divisible by 32")
        if _overlap_producer_planes < 1:
            raise ValueError("pull_gemm_producer_planes must be >= 1")
        xcd_swizzle = 0
    _overlap_max_padded = 0
    _overlap_max_m_blocks = 0
    _overlap_ws_row_src_base = 0
    _overlap_ws_arrival_base = 0
    _overlap_mx_gn = 0
    _overlap_mx_stride_o0 = 0
    _overlap_mx_stride_o1 = 0
    _overlap_mx_stride_o2 = 0
    if _pull_gemm_overlap:
        _overlap_max_padded, _overlap_max_m_blocks, _ = moe_sorting_const_sizes(
            max_tok_per_rank=int(max_tok_per_rank),
            experts_per_token=int(topk),
            num_experts=int(experts),
            block_size=int(tile_m),
        )
        _overlap_ws_pool_slot_base = 3 * int(experts)
        _overlap_ws_row_src_base = _overlap_ws_pool_slot_base + int(_overlap_max_padded)
        _overlap_ws_block_row_start_base = _overlap_ws_row_src_base + int(_overlap_max_padded)
        _overlap_ws_block_valid_base = _overlap_ws_block_row_start_base + int(_overlap_max_m_blocks)
        _overlap_ws_arrival_base = _overlap_ws_block_valid_base + int(_overlap_max_m_blocks)
        (
            _,
            _overlap_mx_gn,
            _,
            _,
            _,
            _,
            _,
            _overlap_mx_strides,
        ) = mxfp4_sort_const_geom(
            max_sorted_rows=int(_overlap_max_padded),
            scale_mx_blocks=int(model_dim) // 32,
        )
        _overlap_mx_stride_o0, _overlap_mx_stride_o1, _overlap_mx_stride_o2, _ = _overlap_mx_strides
    num_waves = min(4, tile_n // 32)
    total_threads = num_waves * 64
    _chunks_pg = int(model_dim) // 4
    _iters_pg = (_chunks_pg + int(total_threads) - 1) // int(total_threads)
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
        assert (
            model_dim % k_batch == 0
        ), f"model_dim={model_dim} not divisible by k_batch={k_batch}"
        assert (
            _k_per_batch % tile_k == 0
        ), f"K_per_batch={_k_per_batch} not divisible by tile_k={tile_k}"

        out_dtype = "bf16"
    else:
        _k_per_batch = model_dim
    _k_dim = _k_per_batch

    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            f"tile_m*tile_k*elem_bytes must be divisible by {total_threads}"
        )
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
    _sk_tag = f"_sk{k_batch}" if _is_splitk else ""
    _go_tag = "_go" if mock_gate_only else ""
    _gui_tag = "_gui" if gate_up_interleave else ""
    _as1_tag = "_as1" if a_scale_one else ""
    _xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
    _pool_tag = "_pool" if _use_local_pool else ""
    module_name = (
        f"mfma_moe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}"
        f"_t{tile_m}x{tile_n}x{tile_k}_pm{persist_m}{_fp4q_tag}{_fp8q_tag}{_sort_tag}{_sk_tag}{_go_tag}{_gui_tag}{_as1_tag}{_xcd_tag}{_pool_tag}_v32"
    ).replace("-", "_")

    # -- LDS sizing --
    _cshuffle_elem_bytes = 4 if _need_quant else (4 if out_is_f32 else 2)
    _single_x_bytes = int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = (
        _cshuffle_elem_bytes * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )
    lds_tid_bytes = int(tile_m) * 4
    _input_elems = _single_x_bytes if a_elem_bytes == 1 else (_single_x_bytes // 2)

    # Determine whether we need wave-group split for lds_out.
    # Standard layout: pong = max(input, lds_out) + tid, ping = input.
    # When this overflows, split lds_out into two halves across pong & ping.
    _GLOBAL_ALIGN = 1024
    _std_pong = max(_single_x_bytes, lds_out_bytes) + lds_tid_bytes
    _std_ping = _single_x_bytes
    _std_pong_aligned = allocator_pong._align(_std_pong, 128)
    _std_total = allocator_pong._align(
        _std_pong_aligned, _GLOBAL_ALIGN
    ) + allocator_pong._align(_std_ping, 128)
    _lds_limit = {"gfx950": 163840, "gfx942": 65536}.get(gpu_arch, 0)

    _split_lds_out = (
        _lds_limit > 0
        and lds_out_bytes > 0
        and _std_total > _lds_limit
        and num_waves >= 2
    )

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
    _pull_tbl_bytes = 0

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes

    if waves_per_eu is not None and waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        _ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        _cur_lds = _pong_sz + _ping_sz
        if _cur_lds < _min_lds:
            allocator_ping.ptr += _min_lds - _cur_lds

    kpack_bytes = 8 if is_int4 else 16
    out_elem_bytes = 4 if out_is_f32 else 2

    _e_vec_s1 = min(tile_n // 32, 8)
    if _need_quant:
        _e_vec_s1 = max(2, _e_vec_s1)
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
        def moe_fused_stage1_a8w4(
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
            addr_p2p_base_x: fx.Int64,
            addr_peer_x_staging: fx.Int64,
            i32_peer_npes: fx.Int32,
            addr_pull_row_i64: fx.Int64,
            i32_pool_base: fx.Int32,
            addr_p2p_scale: fx.Int64,
            addr_sort_workspace: fx.Int64,
        ):

            tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
            n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
            k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
            )

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

            shape_lds = fx.make_shape(tile_m, _eff_lds_stride)
            stride_lds = fx.make_stride(_eff_lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            by = gpu.block_id("x")  # tile along inter_dim (N)
            bx_persist_grid = gpu.block_id("y")
            bx_persist = bx_persist_grid  # persistent WG index
            producer_plane = arith.cmpi(
                CmpIPredicate.eq,
                arith.constant(0, index=True),
                arith.constant(1, index=True),
            )
            if const_expr(_pull_gemm_overlap):
                c0_idx_overlap = arith.constant(0, index=True)
                producer_planes_idx = arith.index(_overlap_producer_planes)
                producer_plane = arith.cmpi(
                    CmpIPredicate.ult, bx_persist_grid, producer_planes_idx
                )
                bx_persist = arith.select(
                    producer_plane,
                    c0_idx_overlap,
                    bx_persist_grid - producer_planes_idx,
                )

            if const_expr(xcd_swizzle > 0 and not _pull_gemm_overlap):
                _NUM_XCDS_S1 = 8
                _c1_sw = arith.constant(1, index=True)
                _c_tn_sw = arith.constant(tile_n, index=True)
                _c_idp_sw = arith.constant(2 * inter_dim_pad, index=True)
                if const_expr(mock_gate_only or gate_up_interleave):
                    _gx = (n_in - _c_idp_sw + _c_tn_sw - _c1_sw) / _c_tn_sw
                else:
                    _c2_sw = arith.constant(2, index=True)
                    _gx = (
                        (n_in - _c_idp_sw + _c2_sw * _c_tn_sw - _c1_sw)
                        / _c_tn_sw
                        / _c2_sw
                    )
                _c_pm_sw = arith.constant(persist_m, index=True)
                _gy = (size_expert_ids_in + _c_pm_sw - _c1_sw) / _c_pm_sw

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
            lds_x_pong = SmemPtr(
                base_ptr_pong, lds_pong_offset, x_lds_elem(), shape=(_input_elems,)
            ).get()
            lds_x_ping = SmemPtr(
                base_ptr_ping, lds_ping_offset, x_lds_elem(), shape=(_input_elems,)
            ).get()
            _lds_out_elem_type = (
                T.f32 if _need_quant else (T.bf16 if out_is_bf16 else T.f16)
            )
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
            lds_tid = SmemPtr(
                base_ptr_pong, _lds_tid_offset_pong, T.i32, shape=(tile_m,)
            ).get()
            # Buffer resources
            c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
            c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)

            # X: [tokens, model_dim]
            x_nbytes_idx = (tokens_in * k_in * c_elem_bytes) / c_a_pack
            x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_i32
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # Out: [tokens*topk, inter_dim]
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=arith.constant(4, type=T.i32),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
            )

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

            sorted_nbytes_idx = size_expert_ids_in * arith.constant(
                sort_block_m * 4, index=True
            )
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
            overlap_ws_rsrc = 1
            overlap_p2p_x_rsrc = 1
            overlap_p2p_scale_rsrc = 1
            overlap_staging_rsrc = 1
            if const_expr(_pull_gemm_overlap):
                overlap_ws_rsrc = create_buffer_resource_from_addr(
                    _lv_unwrap(addr_sort_workspace)
                )
                overlap_p2p_x_rsrc = create_buffer_resource_from_addr(
                    _lv_unwrap(addr_p2p_base_x)
                )
                overlap_p2p_scale_rsrc = create_buffer_resource_from_addr(
                    _lv_unwrap(addr_p2p_scale)
                )
                overlap_staging_rsrc = create_buffer_resource_from_addr(
                    _lv_unwrap(addr_peer_x_staging)
                )
            bias_rsrc = (
                buffer_ops.create_buffer_resource(arg_bias, max_size=False)
                if enable_bias
                else None
            )

            # Sorted-scale buffer resource for fused mxfp4 quantization
            _sorted_scale_cols = inter_dim // 32
            _sorted_scale_cols_i32 = arith.constant(_sorted_scale_cols, type=T.i32)
            sorted_scale_rsrc = None
            if const_expr(_need_sort):
                sorted_scale_rsrc = buffer_ops.create_buffer_resource(
                    arg_out_scale_sorted, max_size=False
                )

            def _run_overlap_pull_producer():
                tx_i32_p = arith.index_cast(T.i32, tx)
                grid_x_idx_p = arith.index_cast(ir.IndexType.get(), gpu.grid_dim.x.ir_value())
                producer_start_idx = bx_persist_grid * grid_x_idx_p + by
                producer_stride_idx = grid_x_idx_p * arith.index(_overlap_producer_planes)
                total_threads_i32 = arith.constant(total_threads, type=T.i32)
                c0_i32_p = arith.constant(0, type=T.i32)
                c1_i32_p = arith.constant(1, type=T.i32)
                cm1_i32_p = arith.constant(-1, type=T.i32)
                c8_i32_p = arith.constant(8, type=T.i32)
                c64_i32_p = arith.constant(64, type=T.i32)
                c_ff_i32_p = arith.constant(0xFF, type=T.i32)
                mask20_i32_p = arith.constant(0xFFFFF, type=T.i32)
                token_bytes_i32_p = arith.constant(model_dim * int(a_elem_bytes), type=T.i32)
                pool_base_i32_p = arith.index_cast(T.i32, i32_pool_base.ir_value())
                row_src_base_i32_p = arith.constant(_overlap_ws_row_src_base, type=T.i32)
                arrival_base_i32_p = arith.constant(_overlap_ws_arrival_base, type=T.i32)
                scale_cols_i32_p = arith.constant(model_dim // 32, type=T.i32)
                stride_o0_i32_p = arith.constant(_overlap_mx_stride_o0, type=T.i32)
                stride_o1_i32_p = arith.constant(_overlap_mx_stride_o1, type=T.i32)
                stride_o2_i32_p = arith.constant(_overlap_mx_stride_o2, type=T.i32)
                words_per_scale_m_i32_p = arith.constant(_overlap_mx_gn * 64, type=T.i32)
                scale_m_groups_per_block_i32_p = arith.constant(tile_m // 32, type=T.i32)

                for block_idx in range(producer_start_idx, size_expert_ids_in, producer_stride_idx):
                    block_i32 = arith.index_cast(T.i32, block_idx)
                    row_start_i32 = arith.muli(
                        block_i32, arith.constant(sort_block_m, type=T.i32)
                    )
                    remaining_i32 = arith.subi(num_valid_i32, row_start_i32)
                    has_rows = arith.cmpi(CmpIPredicate.sgt, remaining_i32, c0_i32_p)
                    if has_rows:
                        full_block = arith.cmpi(
                            CmpIPredicate.uge,
                            remaining_i32,
                            arith.constant(sort_block_m, type=T.i32),
                        )
                        valid_m_i32 = arith.select(
                            full_block,
                            arith.constant(sort_block_m, type=T.i32),
                            remaining_i32,
                        )
                        for work_base in range_constexpr(0, sort_block_m * 64, total_threads):
                            work_i32 = arith.addi(
                                tx_i32_p, arith.constant(work_base, type=T.i32)
                            )
                            local_row_i32 = arith.divui(work_i32, c64_i32_p)
                            lane_i32 = arith.remui(work_i32, c64_i32_p)
                            pool_row_i32 = arith.addi(row_start_i32, local_row_i32)
                            dst_row_z = arith.addi(pool_base_i32_p, pool_row_i32)
                            dst_row_off_z = arith.muli(dst_row_z, token_bytes_i32_p)
                            r_dst_z = create_buffer_resource_from_addr(
                                arith.addi(
                                    _lv_unwrap(addr_peer_x_staging),
                                    arith.zext_i64(dst_row_off_z),
                                )
                            )
                            for chunk_base_z in range_constexpr(0, (model_dim * int(a_elem_bytes)) // 16, 64):
                                chunk_i32_z = arith.addi(
                                    lane_i32,
                                    arith.constant(chunk_base_z, type=T.i32),
                                )
                                in_chunk_z = arith.cmpi(
                                    CmpIPredicate.ult,
                                    chunk_i32_z,
                                    arith.constant(
                                        (model_dim * int(a_elem_bytes)) // 16,
                                        type=T.i32,
                                    ),
                                )
                                if in_chunk_z:
                                    elem_off_z = arith.muli(
                                        chunk_i32_z, arith.constant(4, type=T.i32)
                                    )
                                    z = arith.constant_vector(0, T.vec(4, T.i32))
                                    buffer_ops.buffer_store(z, r_dst_z, elem_off_z)
                            row_in_valid = arith.cmpi(
                                CmpIPredicate.ult, local_row_i32, valid_m_i32
                            )
                            if row_in_valid:
                                row_src = buffer_ops.buffer_load(
                                    overlap_ws_rsrc,
                                    arith.addi(row_src_base_i32_p, pool_row_i32),
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                                row_src_valid = arith.cmpi(
                                    CmpIPredicate.ne, row_src, cm1_i32_p
                                )
                                if row_src_valid:
                                    src_pe = arith.andi(row_src, c_ff_i32_p)
                                    src_tok = arith.andi(
                                        arith.shrui(row_src, c8_i32_p), mask20_i32_p
                                    )
                                    src_base = buffer_ops.buffer_load(
                                        overlap_p2p_x_rsrc,
                                        src_pe,
                                        vec_width=1,
                                        dtype=T.i64,
                                    )
                                    src_row_off = arith.muli(src_tok, token_bytes_i32_p)
                                    dst_row = arith.addi(pool_base_i32_p, pool_row_i32)
                                    dst_row_off = arith.muli(dst_row, token_bytes_i32_p)
                                    r_src = create_buffer_resource_from_addr(
                                        arith.addi(src_base, arith.zext_i64(src_row_off))
                                    )
                                    r_dst = create_buffer_resource_from_addr(
                                        arith.addi(
                                            _lv_unwrap(addr_peer_x_staging),
                                            arith.zext_i64(dst_row_off),
                                        )
                                    )
                                    for chunk_base in range_constexpr(0, (model_dim * int(a_elem_bytes)) // 16, 64):
                                        chunk_i32 = arith.addi(
                                            lane_i32,
                                            arith.constant(chunk_base, type=T.i32),
                                        )
                                        in_chunk = arith.cmpi(
                                            CmpIPredicate.ult,
                                            chunk_i32,
                                            arith.constant(
                                                (model_dim * int(a_elem_bytes)) // 16,
                                                type=T.i32,
                                            ),
                                        )
                                        if in_chunk:
                                            elem_off = arith.muli(
                                                chunk_i32, arith.constant(4, type=T.i32)
                                            )
                                            v = buffer_ops.buffer_load(
                                                r_src, elem_off, vec_width=4, dtype=T.i32
                                            )
                                            buffer_ops.buffer_store(v, r_dst, elem_off)

                        scale_start_m = arith.divui(
                            row_start_i32, arith.constant(32, type=T.i32)
                        )
                        scale_start_word = arith.muli(scale_start_m, words_per_scale_m_i32_p)
                        for scale_base in range_constexpr(
                            0, (tile_m // 32) * _overlap_mx_gn * 64, total_threads
                        ):
                            scale_local_i32 = arith.addi(
                                tx_i32_p, arith.constant(scale_base, type=T.i32)
                            )
                            work_i32 = arith.addi(scale_start_word, scale_local_i32)
                            prog_m = arith.divui(work_i32, stride_o0_i32_p)
                            rem0 = arith.remui(work_i32, stride_o0_i32_p)
                            prog_n = arith.divui(rem0, stride_o1_i32_p)
                            rem1 = arith.remui(rem0, stride_o1_i32_p)
                            ln = arith.divui(rem1, stride_o2_i32_p)
                            lm = arith.remui(rem1, stride_o2_i32_p)
                            pack = c0_i32_p
                            for i in range_constexpr(4):
                                row_i = arith.addi(
                                    arith.muli(prog_m, arith.constant(32, type=T.i32)),
                                    arith.addi(
                                        arith.constant((i % 2) * 16, type=T.i32), lm
                                    ),
                                )
                                col_i = arith.addi(
                                    arith.muli(prog_n, arith.constant(8, type=T.i32)),
                                    arith.addi(
                                        arith.constant((i // 2) * 4, type=T.i32), ln
                                    ),
                                )
                                row_ok = arith.cmpi(CmpIPredicate.ult, row_i, num_valid_i32)
                                col_ok = arith.cmpi(
                                    CmpIPredicate.ult, col_i, scale_cols_i32_p
                                )
                                if arith.andi(row_ok, col_ok):
                                    row_src = buffer_ops.buffer_load(
                                        overlap_ws_rsrc,
                                        arith.addi(row_src_base_i32_p, row_i),
                                        vec_width=1,
                                        dtype=T.i32,
                                    )
                                    if arith.cmpi(CmpIPredicate.ne, row_src, cm1_i32_p):
                                        src_pe = arith.andi(row_src, c_ff_i32_p)
                                        src_tok = arith.andi(
                                            arith.shrui(row_src, c8_i32_p), mask20_i32_p
                                        )
                                        src_scale_base = buffer_ops.buffer_load(
                                            overlap_p2p_scale_rsrc,
                                            src_pe,
                                            vec_width=1,
                                            dtype=T.i64,
                                        )
                                        r_src_scale = create_buffer_resource_from_addr(
                                            src_scale_base
                                        )
                                        src_off = arith.addi(
                                            arith.muli(src_tok, scale_cols_i32_p), col_i
                                        )
                                        b8 = buffer_ops.buffer_load(
                                            r_src_scale, src_off, vec_width=1, dtype=T.i8
                                        )
                                        pack = arith.ori(
                                            _lv_unwrap(pack),
                                            arith.shli(
                                                arith.extui(T.i32, b8),
                                                arith.constant(i * 8, type=T.i32),
                                            ),
                                        )
                            buffer_ops.buffer_store(_lv_unwrap(pack), sx_rsrc, work_i32)

                        gpu.barrier()
                        rocdl.s_waitcnt(0)
                        if arith.cmpi(CmpIPredicate.eq, tx_i32_p, c0_i32_p):
                            atomic_add_global_acq_rel(
                                _i64_cell_addr(
                                    addr_sort_workspace,
                                    arith.addi(arrival_base_i32_p, block_i32),
                                ),
                                c1_i32_p,
                            )
                        gpu.barrier()

            if const_expr(_pull_gemm_overlap):
                if producer_plane:
                    _run_overlap_pull_producer()

            # ---- persist_m loop (same pattern as stage2) ----
            _PERSIST_M = persist_m
            _c0_p = arith.constant(0, index=True)
            _c1_p = arith.constant(1, index=True)
            _c_pm = arith.constant(_PERSIST_M, index=True)
            _for_persist = scf.ForOp(_c0_p, _c_pm, _c1_p)
            _for_ip = ir.InsertionPoint(_for_persist.body)
            _for_ip.__enter__()
            _mi_p = _for_persist.induction_variable
            bx = bx_persist * _c_pm + _mi_p
            bx_m = bx * arith.constant(sort_block_m, index=True)

            # Block validity
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)
            if const_expr(_pull_gemm_overlap):
                blk_valid = arith.andi(
                    blk_valid,
                    arith.cmpi(
                        CmpIPredicate.uge,
                        bx_persist_grid,
                        arith.index(_overlap_producer_planes),
                    ),
                )
            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, bx, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            exp_valid = arith.cmpi(
                CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32)
            )

            def _moe_gemm1_body():
                # Gate expert offset: first inter_dim rows of each expert's 2*inter_dim block
                expert_off_idx = expert_idx * arith.constant(2 * inter_dim, index=True)

                # X loading -- KEY DIFFERENCE from stage2: X row = token_id only
                x_load_bytes = 16
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4

                c_k_div4 = (
                    (k_in / c_a_pack) * arith.constant(int(a_elem_bytes), index=True)
                ) / arith.index(4)
                tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                    4 * int(a_elem_vec_pack)
                )
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
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

                x_col_local_i32 = []
                x_row_local = []

                _staging_i64 = _lv_unwrap(addr_peer_x_staging)
                _staging_x_rsrc = create_buffer_resource_from_addr(
                    _staging_i64
                )

                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                x_sorted_row = []
                for i in range_constexpr(num_x_loads):
                    sorted_row_i = bx_m + x_row_local[i]
                    x_sorted_row.append(sorted_row_i)

                if const_expr(_use_local_pool):
                    pool_base_i32 = arith.index_cast(
                        T.i32, i32_pool_base.ir_value()
                    )

                    def load_x_tile(base_k):
                        base_k_div4 = (
                            (base_k / c_a_pack)
                            * arith.constant(int(a_elem_bytes), index=True)
                        ) / arith.index(4)
                        parts = []
                        for i in range_constexpr(num_x_loads):
                            pool_row_i32 = arith.index_cast(T.i32, x_sorted_row[i])
                            gmem_row_i32 = arith.addi(pool_base_i32, pool_row_i32)
                            gmem_row = arith.index_cast(ir.IndexType.get(), gmem_row_i32)
                            idx_i32 = arith.index_cast(
                                T.i32,
                                arith.addi(
                                    arith.addi(
                                        arith.muli(gmem_row, c_k_div4),
                                        base_k_div4,
                                    ),
                                    arith.index_cast(
                                        ir.IndexType.get(), x_col_local_i32[i]
                                    ),
                                ),
                            )
                            x_vec = buffer_copy_gmem16_dwordx4(
                                buffer_ops,
                                vector,
                                elem_type=x_elem,
                                idx_i32=idx_i32,
                                rsrc=_staging_x_rsrc,
                                vec_elems=vec16_elems,
                            )
                            parts.append(vector.bitcast(T.vec(4, i32), x_vec))
                        return parts

                else:
                    _cr_b = arith.constant(_compile_rank, type=T.i32)
                    _ff_b = arith.constant(0xFF, type=T.i32)
                    _npes_b = arith.index_cast(
                        T.i32, i32_peer_npes.ir_value()
                    )
                    r_pull_b = create_buffer_resource_from_addr(
                        _lv_unwrap(addr_pull_row_i64)
                    )
                    r_p2p_b = create_buffer_resource_from_addr(
                        _lv_unwrap(addr_p2p_base_x)
                    )
                    _c24_i32 = arith.constant(24, type=T.i32)

                    def _pull_packed_for_sorted_row(sorted_row_i):
                        fused_i = buffer_ops.buffer_load(
                            sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                        )
                        t_i32 = arith.andi(fused_i, mask24)
                        kp_i32 = arith.shrui(fused_i, _c24_i32)
                        t_valid = arith.cmpi(
                            CmpIPredicate.ult, t_i32, tokens_i32
                        )
                        kp_valid = arith.cmpi(
                            CmpIPredicate.ult, kp_i32, topk_i32
                        )
                        t_safe = arith.select(
                            t_valid, t_i32, arith.constant(0, type=T.i32)
                        )
                        ts_valid = arith.andi(t_valid, kp_valid)
                        lut_idx = arith.addi(
                            arith.muli(t_safe, topk_i32), kp_i32
                        )
                        packed = arith.select(
                            ts_valid,
                            buffer_ops.buffer_load(
                                r_pull_b, lut_idx, vec_width=1, dtype=T.i32
                            ),
                            arith.constant(0, type=T.i32),
                        )
                        has_pull = arith.cmpi(
                            CmpIPredicate.ne, packed, arith.constant(0, type=T.i32)
                        )
                        local_pk = arith.addi(
                            _cr_b,
                            arith.shli(t_safe, arith.constant(8, type=T.i32)),
                        )
                        return arith.select(has_pull, packed, local_pk)

                    _max_pes_pf = 8
                    _p2p_base_pe = []
                    for _pe in range_constexpr(_max_pes_pf):
                        _p2p_base_pe.append(
                            buffer_ops.buffer_load(
                                r_p2p_b,
                                arith.constant(_pe, index=True),
                                vec_width=1,
                                dtype=T.i64,
                            )
                        )
                    _p2p_x_rsrc_pe = []
                    for _pe in range_constexpr(_max_pes_pf):
                        _p2p_x_rsrc_pe.append(
                            create_buffer_resource_from_addr(_p2p_base_pe[_pe])
                        )

                    def _p2p_x_rsrc_for_spe(spe_i32):
                        rsrc = _p2p_x_rsrc_pe[0]
                        for _pe in range_constexpr(1, _max_pes_pf):
                            use_pe = arith.cmpi(
                                CmpIPredicate.eq,
                                spe_i32,
                                arith.constant(_pe, type=T.i32),
                            )
                            rsrc = arith.select(use_pe, _p2p_x_rsrc_pe[_pe], rsrc)
                        return rsrc

                    x_pk_raw = []
                    for i in range_constexpr(num_x_loads):
                        x_pk_raw.append(_pull_packed_for_sorted_row(x_sorted_row[i]))

                    def _pk_for_load_slot(slot_i: int):
                        pk = x_pk_raw[slot_i]
                        sr_i = x_sorted_row[slot_i]
                        for j in range_constexpr(slot_i):
                            same = arith.cmpi(
                                CmpIPredicate.eq, sr_i, x_sorted_row[j]
                            )
                            pk = arith.select(same, x_pk_raw[j], pk)
                        return pk

                    def load_x_tile(base_k):
                        base_k_div4 = (
                            (base_k / c_a_pack)
                            * arith.constant(int(a_elem_bytes), index=True)
                        ) / arith.index(4)
                        parts = []
                        for i in range_constexpr(num_x_loads):
                            pk_i = _pk_for_load_slot(i)
                            spe_i = arith.andi(pk_i, _ff_b)
                            stok_i = arith.shrui(
                                pk_i, arith.constant(8, type=T.i32)
                            )
                            stok_idx = arith.index_cast(ir.IndexType.get(), stok_i)
                            idx_i32 = arith.index_cast(
                                T.i32,
                                arith.addi(
                                    arith.addi(
                                        arith.muli(stok_idx, c_k_div4), base_k_div4
                                    ),
                                    arith.index_cast(
                                        ir.IndexType.get(), x_col_local_i32[i]
                                    ),
                                ),
                            )
                            pe_ok = arith.cmpi(
                                CmpIPredicate.ult, spe_i, _npes_b
                            )
                            peer_rsrc = _p2p_x_rsrc_for_spe(spe_i)
                            load_rsrc = arith.select(
                                pe_ok, peer_rsrc, _staging_x_rsrc
                            )
                            x_vec = buffer_copy_gmem16_dwordx4(
                                buffer_ops,
                                vector,
                                elem_type=x_elem,
                                idx_i32=idx_i32,
                                rsrc=load_rsrc,
                                vec_elems=vec16_elems,
                            )
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
                        _gui_col_g = (
                            (by_n + n_tile_base) // arith.constant(2, index=True)
                            + _gui_c_offset
                            + lane_mod_16
                        )
                        col_g_list.append(_gui_col_g)

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                _K_per_ku = tile_k // k_unroll
                _pad_k_elems = (
                    (model_dim_pad % tile_k)
                    if (not _is_splitk and model_dim_pad > 0)
                    else 0
                )
                _pad_ku_skip = _pad_k_elems // _K_per_ku
                _tail_ku = k_unroll - _pad_ku_skip
                _tail_ku_packed = (
                    (_tail_ku + pack_K - 1) // pack_K if _pad_ku_skip > 0 else None
                )

                # B load for gate and up separately
                def load_b_packs_k64(base_k, ku: int, n_blk, n_intra):
                    c64 = arith.constant(64, index=True)
                    base_k_bytes = base_k * arith.constant(
                        int(b_elem_bytes), index=True
                    )
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
                    b0 = vector.extract(
                        b_i64x2, static_position=[0], dynamic_position=[]
                    )
                    b1 = vector.extract(
                        b_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return b0, b1

                def load_b_tile(base_k, ku_limit=k_unroll):
                    """Load B tiles. Returns (gate_b_tile, up_b_tile).
                    When mock_gate_only or gate_up_interleave, up_b_tile is None."""
                    gate_b_tile = []
                    up_b_tile = (
                        [] if (not mock_gate_only and not gate_up_interleave) else None
                    )
                    for ku in range_constexpr(ku_limit):
                        g_packs0, g_packs1 = [], []
                        u_packs0, u_packs1 = [], []
                        for ni in range_constexpr(num_acc_n):
                            gb0, gb1 = load_b_packs_k64(
                                base_k, ku, gate_n_blk_list[ni], gate_n_intra_list[ni]
                            )
                            g_packs0.append(gb0)
                            g_packs1.append(gb1)
                            if const_expr(
                                not mock_gate_only and not gate_up_interleave
                            ):
                                ub0, ub1 = load_b_packs_k64(
                                    base_k, ku, up_n_blk_list[ni], up_n_intra_list[ni]
                                )
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
                _scale_lane_elem = (
                    lane_div_16 * layout_b_scale.stride_klane + lane_mod_16
                )

                _gate_scale_bases = []
                _up_scale_bases = []
                for _ni in range_constexpr(num_acc_n_packed):
                    _col_base = (
                        by_n
                        + n_tile_base
                        + arith.constant(_ni * 16 * pack_N, index=True)
                    )
                    _gate_mni = (expert_off_idx + _col_base) // arith.constant(
                        32, index=True
                    )
                    _gate_scale_bases.append(
                        _gate_mni * layout_b_scale.stride_n0 + _scale_lane_elem
                    )
                    if const_expr(not mock_gate_only and not gate_up_interleave):
                        _up_mni = (
                            expert_off_idx + inter_idx + _col_base
                        ) // arith.constant(32, index=True)
                        _up_scale_bases.append(
                            _up_mni * layout_b_scale.stride_n0 + _scale_lane_elem
                        )

                if const_expr(not a_scale_one):
                    _a_scale_bases = []
                    for _mi in range_constexpr(m_repeat_packed):
                        _a_mni = _mi + bx_m // scale_mn_pack // 16
                        _a_scale_bases.append(
                            _a_mni * layout_a_scale.stride_n0 + _scale_lane_elem
                        )

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
                    b_k0 = arith.andi(
                        arith.shrui(raw_i32, _scale_shift), _scale_mask_lo
                    )
                    b_k1 = arith.andi(
                        arith.shrui(raw_i32, _scale_shift_hi), _scale_mask_lo
                    )
                    return arith.ori(
                        b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32))
                    )

                def _rearrange_b_scale(raw_i32):
                    """Rearrange scale bytes for pack_N=1: extract n_half's k0,k1 bytes."""
                    if const_expr(pack_N >= scale_mn_pack):
                        return raw_i32
                    b_k0 = arith.andi(
                        arith.shrui(raw_i32, _bscale_shift), _scale_mask_lo
                    )
                    b_k1 = arith.andi(
                        arith.shrui(raw_i32, _bscale_shift_hi), _scale_mask_lo
                    )
                    return arith.ori(
                        b_k0, arith.shli(b_k1, arith.constant(8, type=T.i32))
                    )

                if const_expr(a_scale_one):
                    _as1_const = arith.constant(0x7F7F7F7F, type=T.i32)
                    _as1_vec = vector.from_elements(T.vec(1, T.i32), [_as1_const])

                def prefetch_ab_scale_tile(base_k, ku_packed_limit=k_unroll_packed):
                    a_scale_tile = []
                    gate_b_scale = []
                    up_b_scale = (
                        [] if (not mock_gate_only and not gate_up_interleave) else None
                    )
                    for ku in range_constexpr(ku_packed_limit):
                        k_off = (ku + base_k) * layout_b_scale.stride_k0
                        for mi in range_constexpr(m_repeat_packed):
                            if const_expr(a_scale_one):
                                a_scale_tile.append(_as1_vec)
                            else:
                                if const_expr(False):
                                    _scale_cols = arith.constant(model_dim // 32, index=True)
                                    _row0 = (
                                        bx_m
                                        + arith.constant(mi * scale_mn_pack * 16, index=True)
                                        + lane_mod_16
                                    )
                                    _row1 = _row0 + arith.constant(16, index=True)
                                    _col0 = (
                                        (ku + base_k) * arith.constant(8, index=True)
                                        + lane_div_16
                                    )
                                    _col1 = _col0 + arith.constant(4, index=True)
                                    _b00 = arith.extui(
                                        T.i32,
                                        buffer_ops.buffer_load(
                                            sx_rsrc,
                                            _row0 * _scale_cols + _col0,
                                            vec_width=1,
                                            dtype=T.i8,
                                            cache_modifier=0,
                                        ),
                                    )
                                    _b10 = arith.extui(
                                        T.i32,
                                        buffer_ops.buffer_load(
                                            sx_rsrc,
                                            _row1 * _scale_cols + _col0,
                                            vec_width=1,
                                            dtype=T.i8,
                                            cache_modifier=0,
                                        ),
                                    )
                                    _b01 = arith.extui(
                                        T.i32,
                                        buffer_ops.buffer_load(
                                            sx_rsrc,
                                            _row0 * _scale_cols + _col1,
                                            vec_width=1,
                                            dtype=T.i8,
                                            cache_modifier=0,
                                        ),
                                    )
                                    _b11 = arith.extui(
                                        T.i32,
                                        buffer_ops.buffer_load(
                                            sx_rsrc,
                                            _row1 * _scale_cols + _col1,
                                            vec_width=1,
                                            dtype=T.i8,
                                            cache_modifier=0,
                                        ),
                                    )
                                    s = arith.ori(
                                        arith.ori(
                                            _b00,
                                            arith.shli(_b10, arith.constant(8, type=T.i32)),
                                        ),
                                        arith.ori(
                                            arith.shli(_b01, arith.constant(16, type=T.i32)),
                                            arith.shli(_b11, arith.constant(24, type=T.i32)),
                                        ),
                                    )
                                else:
                                    s = buffer_ops.buffer_load(
                                        sx_rsrc,
                                        _a_scale_bases[mi] + k_off,
                                        vec_width=1,
                                        dtype=T.i32,
                                        cache_modifier=0,
                                    )
                                s = _rearrange_a_scale(s)
                                a_scale_tile.append(
                                    vector.from_elements(T.vec(1, T.i32), [s])
                                )
                        for ni in range_constexpr(num_acc_n_packed):
                            gs = buffer_ops.buffer_load(
                                sw_rsrc,
                                _gate_scale_bases[ni] + k_off,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=0,
                            )
                            gs = _rearrange_b_scale(gs)
                            gate_b_scale.append(
                                vector.from_elements(T.vec(1, T.i32), [gs])
                            )
                            if const_expr(
                                not mock_gate_only and not gate_up_interleave
                            ):
                                us = buffer_ops.buffer_load(
                                    sw_rsrc,
                                    _up_scale_bases[ni] + k_off,
                                    vec_width=1,
                                    dtype=T.i32,
                                    cache_modifier=0,
                                )
                                us = _rearrange_b_scale(us)
                                up_b_scale.append(
                                    vector.from_elements(T.vec(1, T.i32), [us])
                                )
                    return [a_scale_tile, gate_b_scale, up_b_scale]

                _lds_base_zero = arith.index(0)

                def store_x_tile_to_lds(vec_x_in_parts, lds_buffer):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]

                        def _do_lds_store():
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

                        _do_lds_store()

                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes / arith.index(2))
                    )
                    idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
                    loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
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
                                a2, a3 = lds_load_packs_k64(
                                    curr_row, col_base + 64, lds_buffer
                                )
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
                                        (by_n + n_tile_base)
                                        // arith.constant(2, index=True)
                                        + arith.constant((ni // 2) * 16, index=True)
                                        + lane_mod_16
                                    )
                                    _up_off = (
                                        inter_idx
                                        if (ni % 2 == 1)
                                        else arith.constant(0, index=True)
                                    )
                                    bias_offset = (
                                        expert_off_idx + _up_off + _logical_col
                                    )
                                else:
                                    global_n = (
                                        by_n
                                        + n_tile_base
                                        + arith.constant(ni * 16, index=True)
                                        + lane_mod_16
                                    )
                                    bias_offset = expert_off_idx + global_n
                                bias_pf.append(
                                    buffer_ops.buffer_load(
                                        bias_rsrc, bias_offset, vec_width=1, dtype=f32
                                    )
                                )
                        tw_pf = None
                        if const_expr(doweight_stage1):
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                            ii_idx_list_pf = [
                                arith.constant(ii, index=True) for ii in range(4)
                            ]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.constant(mi * 16, index=True)
                                for ii in range_constexpr(4):
                                    row_off_pf = (
                                        lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    )
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
                                up_bs_val = vector.extract(
                                    up_bs_i32, static_position=[0], dynamic_position=[]
                                )
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
                                        gb128 = pack_i64x4_to_i32x8(
                                            gb0, gb1, c0_i64, c0_i64
                                        )
                                        if const_expr(not _single_b):
                                            ub0 = up_bp0[ni_idx]
                                            ub1 = up_bp1[ni_idx]
                                            ub128 = pack_i64x4_to_i32x8(
                                                ub0, ub1, c0_i64, c0_i64
                                            )
                                        for mi in range_constexpr(m_repeat_packed):
                                            a_scale_i32 = a_scale[
                                                ku128 * m_repeat_packed + mi
                                            ]
                                            a_scale_val = vector.extract(
                                                a_scale_i32,
                                                static_position=[0],
                                                dynamic_position=[],
                                            )
                                            for imxdl in range_constexpr(pack_M):
                                                mi_idx = mi * pack_M + imxdl
                                                _a_reg_idx = k_idx * m_repeat + mi_idx
                                                if const_expr(is_f8_a):
                                                    a0, a1, a2, a3 = a_tile_regs[
                                                        _a_reg_idx
                                                    ]
                                                    a128 = pack_i64x4_to_i32x8(
                                                        a0, a1, a2, a3
                                                    )
                                                else:
                                                    a0, a1 = a_tile_regs[_a_reg_idx]
                                                    a128 = pack_i64x4_to_i32x8(
                                                        a0, a1, c0_i64, c0_i64
                                                    )
                                                acc_idx = mi_idx * num_acc_n + ni_idx
                                                gate_list[acc_idx] = (
                                                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
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
                                                )
                                                if const_expr(not _single_b):
                                                    up_list[acc_idx] = (
                                                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
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
                                up_list[acc_idx] = (
                                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
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

                    # Prefetch A to OTHER buffer (for next half)
                    _abs_k_dma = k_base_idx + arith.constant(next_k_dma_py, index=True)
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
                                    _raw_as = buffer_ops.buffer_load(
                                        sx_rsrc,
                                        _a_scale_bases[_mi_p] + _k_off,
                                        vec_width=1,
                                        dtype=T.i32,
                                        cache_modifier=0,
                                    )
                                    _new_as_list.append(_rearrange_a_scale(_raw_as))
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
                        rocdl.sched_barrier(0)
                        for _a_j in range_constexpr(len(_pp_a_reads[_p])):
                            _ak, _ami = _pp_a_reads[_p][_a_j]
                            _a_all[(_ak, _ami)] = load_a_subtile(
                                _ak,
                                _ami,
                                lds_read,
                            )
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
                                (
                                    _prev_usv_list[_ni_packed_idx]
                                    if not _single_b_pipe
                                    else None
                                ),
                                acc_gate,
                                acc_up,
                                _k_idx,
                                _ni_idx,
                                _ikxdl,
                                _inxdl,
                            )
                        rocdl.s_setprio(0)
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
                        cur_gate_bs.append(
                            vector.from_elements(
                                T.vec(1, T.i32), [_new_gs_list[_gs_ni]]
                            )
                        )
                    if const_expr(not _single_b_pipe):
                        cur_up_bs = []
                        for _us_ni in range_constexpr(num_acc_n_packed):
                            cur_up_bs.append(
                                vector.from_elements(
                                    T.vec(1, T.i32), [_new_us_list[_us_ni]]
                                )
                            )
                    else:
                        cur_up_bs = None

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

                k0 = k_base_idx
                x_regs0 = load_x_tile(k0)
                store_x_tile_to_lds(x_regs0, lds_x_pong)
                rocdl.sched_barrier(0)
                _k0_scale = k_base_idx // arith.constant(pack_K * 128, index=True)
                a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(
                    _k0_scale
                )
                _c_tile_m_idx = arith.constant(tile_m, index=True)
                _tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, _c_tile_m_idx)
                _if_tid = scf.IfOp(_tid_in_range)
                with ir.InsertionPoint(_if_tid.then_block):
                    _tid_row = bx_m + tx
                    _tid_val = buffer_ops.buffer_load(
                        sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32
                    )
                    _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                    vector.store(_tid_vec1, lds_tid, [tx])
                    scf.YieldOp([])

                acc_gate = [acc_init] * num_acc_n * m_repeat
                acc_up = (
                    [acc_init] * num_acc_n * m_repeat if not _single_b_pipe else None
                )

                _k1 = k_base_idx + arith.constant(tile_k, index=True)
                rocdl.sched_barrier(0)
                _x_regs_prime = load_x_tile(_k1)
                store_x_tile_to_lds(_x_regs_prime, lds_x_ping)

                _k0_b = k_base_idx // arith.constant(2, index=True)
                gate_w0, up_w0 = load_b_tile(_k0_b)
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
                        gate_w_ping, up_w_ping = load_b_tile(
                            k_tail1 // arith.constant(2, index=True)
                        )
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
                    store_x_tile_to_lds(x_regs_ping, lds_x_ping)
                    rocdl.s_waitcnt(0)
                    _barrier()
                    if const_expr(_pad_ku_skip > 0):
                        a_tile_ping = prefetch_full_a_from_lds(
                            lds_x_ping, ku_limit=_tail_ku
                        )
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
                        g = vector.extract(
                            gate_v4, static_position=[ei], dynamic_position=[]
                        )
                        u = vector.extract(
                            up_v4, static_position=[ei], dynamic_position=[]
                        )
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
                        g = vector.extract(
                            gate_v4, static_position=[ei], dynamic_position=[]
                        )
                        u = vector.extract(
                            up_v4, static_position=[ei], dynamic_position=[]
                        )
                        g = arith.minimumf(g, _limit)
                        u = arith.minimumf(u, _limit)
                        u = arith.maximumf(u, _neg_limit)
                        t = g * _alpha * _neg_log2e
                        emu = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.exp2.f32", [t], [], []
                        )
                        den = _one + emu
                        sig = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [den], [], []
                        )
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
                                    (by_n + n_tile_base)
                                    // arith.constant(2, index=True)
                                    + arith.constant((_ni // 2) * 16, index=True)
                                    + lane_mod_16
                                )
                                _up_off = (
                                    inter_idx
                                    if (_ni % 2 == 1)
                                    else arith.constant(0, index=True)
                                )
                                _bias_off = expert_off_idx + _up_off + _logical_col
                            else:
                                _bn = (
                                    by_n
                                    + n_tile_base
                                    + arith.constant(_ni * 16, index=True)
                                    + lane_mod_16
                                )
                                _bias_off = expert_off_idx + _bn
                            _bias_gate_vals.append(
                                buffer_ops.buffer_load(
                                    bias_rsrc, _bias_off, vec_width=1, dtype=f32
                                )
                            )
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            _bsplat = vector.from_elements(
                                vec4_f32, [_bias_gate_vals[_ni]] * 4
                            )
                            acc_gate[_aidx] = arith.addf(acc_gate[_aidx], _bsplat)

                    if const_expr(not (mock_gate_only or gate_up_interleave)):
                        _bias_up_vals = []
                        for _ni in range_constexpr(num_acc_n):
                            _bn = (
                                by_n
                                + n_tile_base
                                + arith.constant(_ni * 16, index=True)
                                + lane_mod_16
                            )
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
                                _bsplat = vector.from_elements(
                                    vec4_f32, [_bias_up_vals[_ni]] * 4
                                )
                                acc_up[_aidx] = arith.addf(acc_up[_aidx], _bsplat)

                if const_expr(gate_up_interleave and not _is_splitk):
                    _gui_out_n = num_acc_n // pack_N
                    acc = [None] * (_gui_out_n * m_repeat)
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(_gui_out_n):
                            _g_idx = _mi * num_acc_n + _ni * pack_N
                            _u_idx = _g_idx + 1
                            _out_idx = _mi * _gui_out_n + _ni
                            acc[_out_idx] = _act_vec4(
                                acc_gate[_g_idx], acc_gate[_u_idx]
                            )
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
                out_base_ptr = _fly.extract_aligned_pointer_as_index(
                    _llvm_ptr_ty, arg_out
                )
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
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=f32
                            )
                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(
                            acc[acc_idx], static_position=[ii], dynamic_position=[]
                        )
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
                    else (
                        inter_dim // 2
                        if _need_fp4
                        else (inter_dim if _need_fp8 else inter_dim * out_elem_bytes)
                    )
                )

                def precompute_row(*, row_local, row):
                    fused2 = memref.load(lds_tid, [row_local])
                    row_i32 = arith.index_cast(T.i32, row)
                    row_valid0 = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                    t = fused2 & mask24_i32
                    s = fused2 >> 24
                    t_ok = arith.cmpi(CmpIPredicate.ult, t, tokens_i32_v)
                    s_ok = arith.cmpi(CmpIPredicate.ult, s, topk_i32_v)
                    ts_ok = arith.andi(t_ok, s_ok)
                    if const_expr(_use_local_pool):
                        row_valid = arith.andi(row_valid0, ts_ok)
                        row_byte_base = out_base_idx + row * arith.constant(
                            _out_row_stride, index=True
                        )
                    else:
                        row_valid = arith.andi(row_valid0, ts_ok)
                        t_idx = arith.index_cast(ir.IndexType.get(), t)
                        s_idx = arith.index_cast(ir.IndexType.get(), s)
                        ts_idx = t_idx * arith.constant(topk, index=True) + s_idx
                        row_byte_base = out_base_idx + ts_idx * arith.constant(
                            _out_row_stride, index=True
                        )
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
                            frag_vals.append(
                                vector.extract(
                                    frag, static_position=[i], dynamic_position=[]
                                )
                            )

                        local_max = _c0_f32
                        for i in range_constexpr(_e_vec):
                            abs_v = llvm.call_intrinsic(
                                f32, "llvm.fabs.f32", [frag_vals[i]], [], []
                            )
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
                                byte_k = fp4_vals[2 * k] | (
                                    fp4_vals[2 * k + 1] << _c4_i32
                                )
                                packed_i32 = packed_i32 | (
                                    byte_k << arith.constant(k * 8, type=T.i32)
                                )

                            ptr_addr_idx = row_byte_base + col_g0 / arith.constant(
                                2, index=True
                            )
                            out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                            _pack_bytes = _e_vec // 2
                            if const_expr(_pack_bytes == 1):
                                store_val = arith.TruncIOp(T.i8, packed_i32)
                                store_raw = (
                                    store_val._value
                                    if hasattr(store_val, "_value")
                                    else store_val
                                )
                                llvm.StoreOp(
                                    store_raw, out_ptr_v, alignment=1, nontemporal=True
                                )
                            elif const_expr(_pack_bytes == 2):
                                store_val = arith.TruncIOp(T.i16, packed_i32)
                                store_raw = (
                                    store_val._value
                                    if hasattr(store_val, "_value")
                                    else store_val
                                )
                                llvm.StoreOp(
                                    store_raw, out_ptr_v, alignment=2, nontemporal=True
                                )
                            else:
                                packed_raw = (
                                    packed_i32._value
                                    if hasattr(packed_i32, "_value")
                                    else packed_i32
                                )
                                llvm.StoreOp(
                                    packed_raw, out_ptr_v, alignment=4, nontemporal=True
                                )

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
                                    store_raw = (
                                        store_val._value
                                        if hasattr(store_val, "_value")
                                        else store_val
                                    )
                                    llvm.StoreOp(
                                        store_raw,
                                        out_ptr_v,
                                        alignment=2,
                                        nontemporal=True,
                                    )
                                else:
                                    packed_raw = (
                                        packed_i32._value
                                        if hasattr(packed_i32, "_value")
                                        else packed_i32
                                    )
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
                                    word_ptr = ptr_addr_idx + arith.constant(
                                        _wg * 4, index=True
                                    )
                                    out_ptr_v = _idx_to_llvm_ptr(word_ptr)
                                    packed_raw = (
                                        packed_w._value
                                        if hasattr(packed_w, "_value")
                                        else packed_w
                                    )
                                    llvm.StoreOp(
                                        packed_raw,
                                        out_ptr_v,
                                        alignment=4,
                                        nontemporal=True,
                                    )

                        if const_expr(_need_sort):
                            col_g0_i32 = arith.index_cast(T.i32, col_g0)
                            is_scale_writer = arith.cmpi(
                                CmpIPredicate.eq, col_g0_i32 & _c31_i32, _c0_i32
                            )
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
                                    d0 * _n32_sort
                                    + d3 * _c256_i32
                                    + d5 * _c64_i32
                                    + d2 * _c4_i32
                                    + d4 * _c2_i32
                                    + d1
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
                        byte_off_col = col_idx * arith.constant(
                            out_elem_bytes, index=True
                        )
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
                        byte_off_col = col_idx * arith.constant(
                            out_elem_bytes, index=True
                        )
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
                    ir.F32Type.get()
                    if _need_quant
                    else (ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get())
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

            def _wait_overlap_block_ready():
                if const_expr(_pull_gemm_overlap):
                    c0_wait = arith.constant(0, type=T.i32)
                    c1_wait = arith.constant(1, type=T.i32)
                    tx_i32_wait = arith.index_cast(T.i32, tx)
                    _if_wait_t0 = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, tx_i32_wait, c0_wait)
                    )
                    with ir.InsertionPoint(_if_wait_t0.then_block):
                        arrival_idx = arith.addi(
                            arith.constant(_overlap_ws_arrival_base, type=T.i32),
                            arith.divui(
                                bx_m_i32, arith.constant(sort_block_m, type=T.i32)
                            ),
                        )
                        init_cur = load_i32_global_acquire_cell(
                            addr_sort_workspace, arrival_idx
                        )
                        w = scf.WhileOp([T.i32], [init_cur])
                        before = ir.Block.create_at_start(w.before, [T.i32])
                        after = ir.Block.create_at_start(w.after, [T.i32])
                        with ir.InsertionPoint(before):
                            cur = before.arguments[0]
                            need_wait = arith.cmpi(CmpIPredicate.ne, cur, c1_wait)
                            scf.ConditionOp(need_wait, [cur])
                        with ir.InsertionPoint(after):
                            nxt = load_i32_global_acquire_cell(
                                addr_sort_workspace, arrival_idx
                            )
                            scf.YieldOp([nxt])
                        scf.YieldOp([])
                    gpu.barrier()

            _if_blk = scf.IfOp(blk_valid)
            with ir.InsertionPoint(_if_blk.then_block):
                _ifexpert_of = scf.IfOp(exp_valid)
                with ir.InsertionPoint(_ifexpert_of.then_block):
                    _wait_overlap_block_ready()
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
        waves_per_eu,
        k_batch,
        gate_mode,
        a_scale_one,
        xcd_swizzle,
        _compile_rank,
        _use_local_pool,
        _pull_gemm_overlap,
        _overlap_producer_planes,
        int(max_tok_per_rank),
    )

    def _prepare_launch_grid(
        i32_inter_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        inter_in = arith.index_cast(ir.IndexType.get(), i32_inter_in.ir_value())
        tile_n_index = arith.constant(tile_n, index=True)
        inter_dim_pad_total = arith.constant(2 * inter_dim_pad, index=True)
        if const_expr(mock_gate_only or gate_up_interleave):
            gx = (inter_in - inter_dim_pad_total + tile_n_index - 1) / tile_n_index
        else:
            gx = (
                (inter_in - inter_dim_pad_total + 2 * tile_n_index - 1)
                / tile_n_index
                / arith.constant(2, index=True)
            )
        _c_pm_l = arith.constant(persist_m, index=True)
        gy = (
            arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
            + _c_pm_l
            - arith.constant(1, index=True)
        ) / _c_pm_l
        return gx, gy

    def _finalize_allocators():
        _ = _cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

    if _use_local_pool and _pull_gemm_overlap:

        @flyc.jit
        def launch_fused_stage1_a8w4(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_max_token_ids: fx.Tensor,
            arg_bias: fx.Tensor,
            arg_out_scale_sorted: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
            addr_x_staging: fx.Int64,
            i32_pool_base: fx.Int32,
            addr_p2p_base_x: fx.Int64,
            addr_p2p_scale: fx.Int64,
            addr_sort_workspace: fx.Int64,
            stream: fx.Stream,
        ):
            _finalize_allocators()
            gx, gy = _prepare_launch_grid(i32_inter_in, i32_size_expert_ids_in)
            moe_fused_stage1_a8w4(
                arg_out,
                arg_x,
                arg_w,
                arg_scale_x,
                arg_scale_w,
                arg_sorted_token_ids,
                arg_expert_ids,
                arg_sorted_weights,
                arg_max_token_ids,
                arg_bias,
                arg_out_scale_sorted,
                i32_tokens_in,
                i32_inter_in,
                i32_k_in,
                i32_size_expert_ids_in,
                addr_p2p_base_x,
                addr_x_staging,
                fx.Int32(0),
                fx.Int64(0),
                i32_pool_base,
                addr_p2p_scale,
                addr_sort_workspace,
            ).launch(
                grid=(
                    gx,
                    gy + arith.index(_overlap_producer_planes),
                    k_batch,
                ),
                block=(total_threads, 1, 1),
                stream=stream,
            )

    elif _use_local_pool:

        @flyc.jit
        def launch_fused_stage1_a8w4(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_max_token_ids: fx.Tensor,
            arg_bias: fx.Tensor,
            arg_out_scale_sorted: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
            addr_x_staging: fx.Int64,
            i32_pool_base: fx.Int32,
            stream: fx.Stream,
        ):
            _finalize_allocators()
            gx, gy = _prepare_launch_grid(i32_inter_in, i32_size_expert_ids_in)
            moe_fused_stage1_a8w4(
                arg_out,
                arg_x,
                arg_w,
                arg_scale_x,
                arg_scale_w,
                arg_sorted_token_ids,
                arg_expert_ids,
                arg_sorted_weights,
                arg_max_token_ids,
                arg_bias,
                arg_out_scale_sorted,
                i32_tokens_in,
                i32_inter_in,
                i32_k_in,
                i32_size_expert_ids_in,
                fx.Int64(0),
                addr_x_staging,
                fx.Int32(0),
                fx.Int64(0),
                i32_pool_base,
                fx.Int64(0),
                fx.Int64(0),
            ).launch(grid=(gx, gy, k_batch), block=(total_threads, 1, 1), stream=stream)

    else:

        @flyc.jit
        def launch_fused_stage1_a8w4(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_max_token_ids: fx.Tensor,
            arg_bias: fx.Tensor,
            arg_out_scale_sorted: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_inter_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
            addr_p2p_base_x: fx.Int64,
            addr_peer_x_staging: fx.Int64,
            i32_peer_npes: fx.Int32,
            addr_pull_row_i64: fx.Int64,
            stream: fx.Stream,
        ):
            _finalize_allocators()
            gx, gy = _prepare_launch_grid(i32_inter_in, i32_size_expert_ids_in)
            moe_fused_stage1_a8w4(
                arg_out,
                arg_x,
                arg_w,
                arg_scale_x,
                arg_scale_w,
                arg_sorted_token_ids,
                arg_expert_ids,
                arg_sorted_weights,
                arg_max_token_ids,
                arg_bias,
                arg_out_scale_sorted,
                i32_tokens_in,
                i32_inter_in,
                i32_k_in,
                i32_size_expert_ids_in,
                addr_p2p_base_x,
                addr_peer_x_staging,
                i32_peer_npes,
                addr_pull_row_i64,
                fx.Int32(0),
                fx.Int64(0),
                fx.Int64(0),
            ).launch(grid=(gx, gy, k_batch), block=(total_threads, 1, 1), stream=stream)

    return launch_fused_stage1_a8w4


def compile_dispatch_pull_gemm1_a8w4(**kwargs):
    """Compile intranode stage1 GEMM (P2P pull + MFMA). Alias of compile_fused_dispatch_gather_gemm1_a8w4."""
    return compile_fused_dispatch_gather_gemm1_a8w4(**kwargs)


def compile_dispatch_gemm1_local_a8w4(**kwargs):
    """Compile intranode stage1 GEMM that reads A from the local x_staging pool."""
    return compile_fused_dispatch_gather_gemm1_a8w4(**kwargs, use_local_pool=True)


def compile_dispatch_gemm1_pull_pool_overlap_a8w4(**kwargs):
    """Compile experimental stage1 GEMM with in-kernel pull producer + local-pool GEMM consumer."""
    return compile_fused_dispatch_gather_gemm1_a8w4(
        **kwargs, use_local_pool=True, pull_gemm_overlap=True
    )

