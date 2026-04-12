"""FlyDSL Paged Attention Decode with Persistent Scheduling — FP8.

Extends pa_decode_sw_fp8.py with persistent scheduling (PS) mode:
- Grid = (num_SM, 1, 4) so each CTA handles one 256-token sub-tile of a 1024-token KV page
- Outer work loop iterates over pre-computed worklist from get_pa_metadata_v1
- Inner KV loop iterates pages from kv_page_indices instead of block_tables
- Supports split-reduce for load balancing across CUs

Requires: aiter's get_pa_metadata_v1 (module_pa_metadata.so)
"""

from __future__ import annotations
import math
import torch
import functools
from contextlib import contextmanager
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, vector, gpu, rocdl, buffer_ops, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
try:
    from flydsl.autotune import autotune, Config
except ImportError:
    pass
from flydsl._mlir.dialects import arith as _mlir_arith
from flydsl._mlir.dialects import scf


@contextmanager
def _if_then(if_op):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    """Compat helper for SCF IfOp else-region across old/new Python APIs."""
    if getattr(if_op, "else_block", None) is None:
        raise RuntimeError("IfOp has no else block")
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield if_op.else_block
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


# ── Kernel geometry constants ────────────────────────────────────────
QUERY_GROUP_SIZE = 16
HEAD_SIZE = 128
KV_BLOCK_SIZE = 1024     # physical page size (matches SP3 kBlockSize)
KV_COMPUTE_BLOCK = 256   # tile size (matches SP3 kTileKV)
NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 256
MFMA_N = 16
MFMA_K = 32

TOKENS_PER_WARP = KV_COMPUTE_BLOCK // NUM_WARPS  # 64
TLOOP = TOKENS_PER_WARP // MFMA_N                # 4
ROWS_PER_WARP = WARP_SIZE // MFMA_N              # 4
FP8_ELEMS_16B = 16                                # 16 FP8 per 16-byte load
QKHE_PER_FETCH = FP8_ELEMS_16B * ROWS_PER_WARP   # 64
QKHELOOP = HEAD_SIZE // QKHE_PER_FETCH           # 2

VHELOOP = HEAD_SIZE // MFMA_N // NUM_WARPS        # 2
VTLOOP = NUM_WARPS                                # 4

# LDS sizes
PROB_ROW_STRIDE_BYTES = 40  # 32 data + 8 padding -> 0 bank conflict
LDS_LOGITS_BYTES = NUM_WARPS * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES  # 10240
LDS_SOFTMAX_BYTES = 2 * NUM_WARPS * MFMA_N * 4     # 512

FP8_MAX = 240.0
LOG2E = 1.4426950408889634

# Number of loop-carried K values (i64)
_N_K = TLOOP * QKHELOOP * 2  # 16
# Number of loop-carried V values (i64)
_N_V = VHELOOP * VTLOOP * 2  # 16

# Tiles per block (1024 tokens / 256 tokens per tile = 4, matches SP3 kNumBlockTiles)
TILES_PER_BLOCK = KV_BLOCK_SIZE // KV_COMPUTE_BLOCK  # 4

_PACKED_FP8_QUERY_DTYPES = tuple(
    dtype
    for dtype in (
        torch.uint8,
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e4m3fn", None),
    )
    if dtype is not None
)


def _pack_i32_pair_to_i64(a_i32, b_i32):
    v = vector.from_elements(T.vec(2, T.i32), [a_i32, b_i32])
    v1 = vector.bitcast(T.vec(1, T.i64), v)
    return vector.extract(v1, static_position=[0])


def _compute_valid_block_limit(
    context_len,
    kv_start_abs_tok,
    tile_token_offset,
    num_blocks_in_work,
    *,
    block_size: int = KV_BLOCK_SIZE,
):
    """Return how many 1024-token physical blocks this 256-token split can touch."""
    tile_start_tok = kv_start_abs_tok + tile_token_offset
    remaining_tokens = context_len - tile_start_tok
    if all(
        isinstance(v, int)
        for v in (context_len, kv_start_abs_tok, tile_token_offset, num_blocks_in_work)
    ):
        remaining_pos = max(remaining_tokens, 0)
        valid_blocks = (remaining_pos + block_size - 1) // block_size
        return min(valid_blocks, num_blocks_in_work)

    c_zero_i32 = arith.constant(0, type=T.i32)
    c_bs_i32 = arith.constant(block_size, type=T.i32)
    c_one_i32 = arith.constant(1, type=T.i32)
    remaining_pos = arith.select(remaining_tokens > c_zero_i32, remaining_tokens, c_zero_i32)
    valid_blocks = (remaining_pos + c_bs_i32 - c_one_i32) // c_bs_i32
    return arith.select(valid_blocks < num_blocks_in_work, valid_blocks, num_blocks_in_work)


def _expand_pa_metadata_for_block_splits(
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    query_length: int,
    *,
    block_split_factor: int = TILES_PER_BLOCK,
):
    """Expand PA metadata so each 1024-token work tile reduces 4 block-split partials.

    `get_pa_metadata_v1()` only materializes split partials and uses `partial_idx=-1`
    for direct tiles that write final output directly. With `grid_z=4`, every work item
    becomes four partials, so direct tiles must also participate in the reduce stage.
    """

    dev = work_info.device
    valid_work = int(work_indptr[-1].item())
    work_info_cpu = work_info[:valid_work].cpu()

    if valid_work == 0:
        empty_reduce_indptr = torch.zeros(1, dtype=torch.int32, device=dev)
        empty_reduce_final_map = torch.empty((0, 2), dtype=torch.int32, device=dev)
        empty_reduce_partial_map = torch.empty((0,), dtype=torch.int32, device=dev)
        return work_info[:0].contiguous(), empty_reduce_indptr, empty_reduce_final_map, empty_reduce_partial_map

    group_order = []
    group_slot_keys = {}
    group_slot_seen = {}
    row_slot_keys = []

    for wi in range(valid_work):
        row = work_info_cpu[wi]
        q_start = int(row[2].item())
        q_end = int(row[3].item())
        orig_partial_idx = int(row[1].item())
        group_key = (q_start, q_end)
        if group_key not in group_slot_keys:
            group_order.append(group_key)
            group_slot_keys[group_key] = []
            group_slot_seen[group_key] = set()

        if orig_partial_idx >= 0:
            slot_key = ("split", orig_partial_idx)
        else:
            slot_key = ("direct", q_start, q_end)

        if slot_key not in group_slot_seen[group_key]:
            group_slot_seen[group_key].add(slot_key)
            group_slot_keys[group_key].append(slot_key)
        row_slot_keys.append(slot_key)

    slot_id_by_key = {}
    next_slot_id = 0
    for group_key in group_order:
        for slot_key in group_slot_keys[group_key]:
            if slot_key not in slot_id_by_key:
                slot_id_by_key[slot_key] = next_slot_id
                next_slot_id += 1

    for wi, slot_key in enumerate(row_slot_keys):
        work_info_cpu[wi, 1] = slot_id_by_key[slot_key] * query_length

    reduce_indptr_cpu = torch.zeros(len(group_order) + 1, dtype=torch.int32)
    reduce_final_map_cpu = torch.empty((len(group_order), 2), dtype=torch.int32)
    reduce_partial_map_entries = []
    running = 0

    for group_idx, group_key in enumerate(group_order):
        q_start, q_end = group_key
        reduce_final_map_cpu[group_idx, 0] = q_start
        reduce_final_map_cpu[group_idx, 1] = q_end
        for slot_key in group_slot_keys[group_key]:
            slot_id = slot_id_by_key[slot_key]
            base_row = slot_id * query_length * block_split_factor
            for block_split_idx in range(block_split_factor):
                reduce_partial_map_entries.append(base_row + block_split_idx * query_length)
                running += 1
        reduce_indptr_cpu[group_idx + 1] = running

    work_info_out = work_info_cpu.to(device=dev).contiguous()
    reduce_indptr = reduce_indptr_cpu.to(device=dev)
    reduce_final_map = reduce_final_map_cpu.to(device=dev)
    reduce_partial_map = torch.tensor(
        reduce_partial_map_entries, dtype=torch.int32, device=dev
    )
    return work_info_out, reduce_indptr, reduce_final_map, reduce_partial_map


# =====================================================================
# compile_pa_decode_ps — Persistent Scheduling PA decode kernel
# =====================================================================
@functools.lru_cache(maxsize=256)
def compile_pa_decode_ps(
    softmax_scale=None,
    trans_v=False,
    needs_mask=True,
    query_group_size=QUERY_GROUP_SIZE,
    per_token_kv=False,
    query_length: int = 1,
    query_input_dtype: str = "packed_fp8",
):
    """Compile a PS-mode PA decode kernel.

    Unlike compile_pa_decode_sw, this does NOT bake in num_seqs/num_kv_heads/num_partitions
    because PS mode uses dynamic work distribution. Grid = (num_sm, 1, 4).
    """
    arch = get_hip_arch()
    query_packed_fp8 = query_input_dtype == "packed_fp8"
    query_load_is_bf16 = query_input_dtype == "bf16"
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)
    _softmax_scale = float(softmax_scale)
    _bs = KV_BLOCK_SIZE  # 1024 for PS mode (matches SP3 kBlockSize)

    # Note: waves_per_eu=4 causes agpr=0 regression on current build (0c1805f).
    # Leave empty to let LLVM decide — gets agpr=128, vgpr=96, ~203us.
    CompilationContext._compile_hints.data = {}

    # LDS allocation
    # Extra LDS for cross-warp v_scale_max reduction (per_token_kv only):
    # NUM_WARPS floats per lane16id slot, aligned to same layout as softmax data.
    LDS_VMAX_BYTES = NUM_WARPS * MFMA_N * 4 if per_token_kv else 0  # 256 or 0
    LDS_SOFTMAX_TOTAL = LDS_SOFTMAX_BYTES + LDS_VMAX_BYTES
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_ps_smem")
    logits_off = 0
    allocator.ptr = LDS_LOGITS_BYTES
    softmax_off = LDS_LOGITS_BYTES
    allocator.ptr += LDS_SOFTMAX_TOTAL

    # ── @flyc.kernel ─────────────────────────────────────────────────
    @flyc.kernel
    def pa_decode_ps_kernel(
        out_ptr: fx.Tensor,           # output [batch, num_q_heads, head_size]
        partial_out_ptr: fx.Tensor,   # partial output [num_partials, 1, nhead, head_dim] fp32
        partial_lse_ptr: fx.Tensor,   # partial LSE [num_partials, 1, nhead, 1] fp32
        query_ptr: fx.Tensor,         # queries [batch, num_q_heads, head_size]
        key_cache_ptr: fx.Tensor,     # key cache
        value_cache_ptr: fx.Tensor,   # value cache
        context_lengths_ptr: fx.Tensor,  # [batch] int32
        query_scale_ptr: fx.Tensor,
        key_scale_ptr: fx.Tensor,
        value_scale_ptr: fx.Tensor,
        work_indptr_ptr: fx.Tensor,   # [num_sm + 1] int32
        work_info_ptr: fx.Tensor,     # [num_work, 8] int32 (flattened to 1D)
        kv_page_indices_ptr: fx.Tensor,  # [total_pages] int32
        kv_indptr_ptr: fx.Tensor,     # [num_seqs + 1] int32 — prefix sum of pages per seq
        stride_q_seq: Int32,
        stride_q_head: Int32,
        stride_k_block: Int32,
        stride_k_head: Int32,
        stride_v_block: Int32,
        stride_v_head: Int32,
        stride_out_seq: Int32,
        stride_out_head: Int32,
        stride_po_partial: Int32,     # stride for partial_output partial dim (nhead * head_dim)
        stride_pl_partial: Int32,     # stride for partial_lse partial dim (nhead)
        stride_ks_block: Int32,       # key_scale stride for block dim (num_kv_heads * KV_BLOCK_SIZE); 0 for per-tensor
        stride_ks_head: Int32,        # key_scale stride for head dim (KV_BLOCK_SIZE); 0 for per-tensor
        stride_po_ql: Int32,          # stride for partial_output query-length dim (num_query_heads * head_size)
        stride_pl_ql: Int32,          # stride for partial_lse query-length dim (num_query_heads)
    ):
        tid = gpu.thread_idx.x
        cu_id = gpu.block_idx.x  # CU index (0..num_sm-1)

        # ── Thread decomposition ──
        lane16id = tid & arith.constant(15, type=T.i32)
        lane4id = tid & arith.constant(3, type=T.i32)
        rowid = (tid >> arith.constant(4, type=T.i32)) & arith.constant(3, type=T.i32)
        warp_id = tid >> arith.constant(6, type=T.i32)

        # ── Buffer resources ──
        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(key_cache_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(value_cache_ptr, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(partial_out_ptr, max_size=True)
        pl_rsrc = buffer_ops.create_buffer_resource(partial_lse_ptr, max_size=True)
        cl_rsrc = buffer_ops.create_buffer_resource(context_lengths_ptr, max_size=True)
        wi_rsrc = buffer_ops.create_buffer_resource(work_indptr_ptr, max_size=True)
        winfo_rsrc = buffer_ops.create_buffer_resource(work_info_ptr, max_size=True)
        kpi_rsrc = buffer_ops.create_buffer_resource(kv_page_indices_ptr, max_size=True)
        kvindptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr_ptr, max_size=True)

        qs_rsrc = buffer_ops.create_buffer_resource(query_scale_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        q_scale_val = buffer_ops.buffer_load(qs_rsrc, arith.constant(0, type=T.i32), vec_width=1)
        k_scale_val = buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1)
        v_scale_val = buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1)

        # ── LDS views ──
        smem_base = allocator.get_base()
        logits_lds_i32 = SmemPtr(smem_base, logits_off, T.i32, shape=(LDS_LOGITS_BYTES // 4,)).get()
        softmax_lds_f32 = SmemPtr(smem_base, softmax_off, T.f32, shape=(LDS_SOFTMAX_TOTAL // 4,)).get()
        logits_lds_i64 = SmemPtr(smem_base, logits_off, T.i64, shape=(LDS_LOGITS_BYTES // 8,)).get()

        # ── Constants ──
        c_kb = stride_k_block
        c_kh = stride_k_head
        c_vb = stride_v_block
        c_vh = stride_v_head

        _softmax_q_scale = arith.constant(_softmax_scale, type=T.f32) * q_scale_val
        _scale = _softmax_q_scale * k_scale_val  # per-tensor only; per-token uses per-token k_scale
        c_w = arith.constant(WARP_SIZE, type=T.i32)
        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = arith.constant(0.0, type=T.f32)
        c_cps = arith.constant(KV_COMPUTE_BLOCK, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_bs = arith.constant(_bs, type=T.i32)
        c_tpb = arith.constant(TILES_PER_BLOCK, type=T.i32)

        local_qhead_idx = warp_id * arith.constant(4, type=T.i32) + rowid

        # ── Work loop bounds ──
        work_start = buffer_ops.buffer_load(wi_rsrc, cu_id, vec_width=1, dtype=T.i32)
        work_end = buffer_ops.buffer_load(wi_rsrc, cu_id + c_one, vec_width=1, dtype=T.i32)

        # ════════════════════════════════════════════════════════════
        # Outer work loop — iterate over assigned work items
        # Each work item = one (batch, kv_head_range, kv_page_range)
        # ════════════════════════════════════════════════════════════
        _work_start_idx = arith.index_cast(T.index, arith.unwrap(work_start))
        _work_end_idx = arith.index_cast(T.index, arith.unwrap(work_end))
        _work_step = arith.index(1)

        for _wi in range(_work_start_idx, _work_end_idx, _work_step):
            work_idx = arith.index_cast(T.i32, _wi)

            # ── Load work_info[work_idx] — 8 × int32 ──
            info_base = work_idx * arith.constant(8, type=T.i32)
            batch_idx = buffer_ops.buffer_load(winfo_rsrc, info_base, vec_width=1, dtype=T.i32)
            partial_idx = buffer_ops.buffer_load(winfo_rsrc, info_base + c_one, vec_width=1, dtype=T.i32)
            kv_start = buffer_ops.buffer_load(winfo_rsrc, info_base + arith.constant(4, type=T.i32), vec_width=1, dtype=T.i32)
            kv_end = buffer_ops.buffer_load(winfo_rsrc, info_base + arith.constant(5, type=T.i32), vec_width=1, dtype=T.i32)
            q_head_range = buffer_ops.buffer_load(winfo_rsrc, info_base + arith.constant(7, type=T.i32), vec_width=1, dtype=T.i32)

            # Absolute token offset for the first page of this work item within its sequence.
            # kv_start is an absolute index into kv_page_indices; kv_indptr[batch_idx] is
            # the page index where this sequence starts.  Their difference * KV_BLOCK_SIZE
            # gives the token offset from sequence start to the first token we process.
            kv_indptr_batch = buffer_ops.buffer_load(kvindptr_rsrc, batch_idx, vec_width=1, dtype=T.i32)
            kv_start_abs_tok = (kv_start - kv_indptr_batch) * c_bs

            # Derive kv_head from q_head_range
            q_head_start = q_head_range & arith.constant(0xFFFF, type=T.i32)
            kv_h = q_head_start // arith.constant(query_group_size, type=T.i32)

            # Context length for this sequence
            context_len = buffer_ops.buffer_load(cl_rsrc, batch_idx, vec_width=1, dtype=T.i32)
            # Head offsets for K and V cache
            _k_head_off = kv_h * c_kh
            _v_head_off = kv_h * c_vh

            # ── Tiles per block constant for tile-based loop (SP3-matching) ──
            c_four = arith.constant(4, type=T.i32)

            # ── Pre-computed thread-invariant offset components ──
            # K: thread base for token index (constant across all K loads)
            _k_tok_thread_base = warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32) + lane16id
            # K: stride constants in dwords (compile-time)
            _c_tok_stride_dw = arith.constant(FP8_ELEMS_16B // 4, type=T.i32)  # 4
            _c_he_stride_dw = arith.constant(_bs * FP8_ELEMS_16B // 4, type=T.i32)  # 4096
            # K: per-qkhe head element offset (only depends on rowid, compile-time qkhe)
            _k_he_off_dw = [rowid * _c_he_stride_dw + arith.constant(qkhe * 4, type=T.i32) * _c_he_stride_dw
                            for qkhe in range(QKHELOOP)]

            # V: thread-invariant head element for each vhe (constant across all V loads)
            _vhead_elems = [arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                            + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id
                            for vhe in range(VHELOOP)]
            # V: thread-invariant token offsets for each vt (only rowid varies)
            _v_tok_thread_off = [arith.constant(vt * TOKENS_PER_WARP, type=T.i32)
                                 + rowid * arith.constant(MFMA_N, type=T.i32)
                                 for vt in range(VTLOOP)]
            # V: pre-computed per-vhe head element dword offsets (for trans_v)
            if trans_v:
                _vhead_elem_dw = [_vhead_elems[vhe] * arith.constant(FP8_ELEMS_16B // 4, type=T.i32)
                                  for vhe in range(VHELOOP)]
            else:
                _vhead_elem_dw = [_vhead_elems[vhe] * arith.constant(_bs // 4, type=T.i32)
                                  for vhe in range(VHELOOP)]

            # Softmax: thread-invariant kv_tok base (partition_start added at call site)
            _kv_tok_thread_base = warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32) + rowid * c_four

            # Prob pack: thread-invariant LDS write base
            _rowid_8x8 = rowid // arith.constant(2, type=T.i32)
            _offset_in_slot = rowid % arith.constant(2, type=T.i32)
            _prob_wr_thread_base = (warp_id * arith.constant(4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                    + lane16id * arith.constant(PROB_ROW_STRIDE_BYTES, type=T.i32)
                                    + _rowid_8x8 * arith.constant(8, type=T.i32)
                                    + _offset_in_slot * c_four)

            # PV prob read: thread-invariant LDS read base for j=0 and j=1
            # j=0: offset_raw=rowid*4, p_off1=0, p_off2=rowid → base = rowid*640 + lane16id*40
            # j=1: offset_raw=rowid*4+2, p_off1=1, p_off2=rowid → base = rowid*640 + lane16id*40 + 8
            _pv_prob_read_base = (rowid * arith.constant(MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                  + lane16id * arith.constant(PROB_ROW_STRIDE_BYTES, type=T.i32))

            # LDS softmax offsets (thread-invariant)
            _sm_max_off = arith.index_cast(T.index,
                warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
            _sm_sum_off = arith.index_cast(T.index,
                arith.constant(NUM_WARPS * MFMA_N, type=T.i32) + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
            # Cross-warp softmax read offsets (compile-time warp index)
            _sm_rd_max_offs = [arith.index_cast(T.index,
                arith.constant(w * MFMA_N, type=T.i32) + lane16id) for w in range(NUM_WARPS)]
            _sm_rd_sum_offs = [arith.index_cast(T.index,
                arith.constant(NUM_WARPS * MFMA_N + w * MFMA_N, type=T.i32) + lane16id) for w in range(NUM_WARPS)]
            # Per-token v_scale_max cross-warp offsets (written in _qk_and_intra_softmax,
            # read in _cross_warp_softmax_and_prob_pack after the barrier).
            # Layout: float32 index = 2*NUM_WARPS*MFMA_N + warp_id*MFMA_N + lane16id
            if per_token_kv:
                _sm_vmax_wr_off = arith.index_cast(T.index,
                    arith.constant(2 * NUM_WARPS * MFMA_N, type=T.i32)
                    + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
                _sm_vmax_rd_offs = [arith.index_cast(T.index,
                    arith.constant(2 * NUM_WARPS * MFMA_N + w * MFMA_N, type=T.i32) + lane16id)
                    for w in range(NUM_WARPS)]

            # ════════════════════════════════════════════════════════
            # Helper: load K data for one tile using pre-computed block base
            # k_block_base_dw = (phys_block * c_kb + _k_head_off) // 4
            # tile_token_offset_i32 = token offset within block (0, 256, 512, 768)
            # ════════════════════════════════════════════════════════
            def _load_k_flat_ps(k_block_base_dw, tile_token_offset_i32):
                """Load K data for one tile using pre-computed block base address."""
                k_flat = []
                tile_tok_base = tile_token_offset_i32 + _k_tok_thread_base

                for td in range_constexpr(TLOOP):
                    kbo = tile_tok_base + arith.constant(td * MFMA_N, type=T.i32)
                    kbo_dw = k_block_base_dw + kbo * _c_tok_stride_dw
                    for qkhe in range_constexpr(QKHELOOP):
                        ka_dw = kbo_dw + _k_he_off_dw[qkhe]
                        k4 = buffer_ops.buffer_load(k_rsrc, ka_dw,
                                                    vec_width=4, dtype=T.i32)
                        k_flat.append(_pack_i32_pair_to_i64(
                            vector.extract(k4, static_position=[0]),
                            vector.extract(k4, static_position=[1])))
                        k_flat.append(_pack_i32_pair_to_i64(
                            vector.extract(k4, static_position=[2]),
                            vector.extract(k4, static_position=[3])))
                return k_flat

            def _load_q_words(q_base_elem):
                if query_packed_fp8:
                    q_off = q_base_elem + lane16id * arith.constant(FP8_ELEMS_16B, type=T.i32)
                    q_vec = buffer_ops.buffer_load(
                        q_rsrc,
                        q_off // arith.constant(4, type=T.i32),
                        vec_width=4,
                        dtype=T.i32,
                    )
                    return [
                        vector.extract(q_vec, static_position=[0], dynamic_position=[]),
                        vector.extract(q_vec, static_position=[1], dynamic_position=[]),
                        vector.extract(q_vec, static_position=[2], dynamic_position=[]),
                        vector.extract(q_vec, static_position=[3], dynamic_position=[]),
                    ]

                q_elem = q_base_elem + lane16id * arith.constant(FP8_ELEMS_16B, type=T.i32)
                q_words = []
                for qwi in range_constexpr(4):
                    q_src = buffer_ops.buffer_load(
                        q_rsrc,
                        q_elem + arith.constant(qwi * 4, type=T.i32),
                        vec_width=4,
                        dtype=T.bf16 if query_load_is_bf16 else T.f16,
                    )
                    q_f32 = _mlir_arith.ExtFOp(T.f32x4, q_src).result
                    p0 = vector.extract(q_f32, static_position=[0], dynamic_position=[])
                    p1 = vector.extract(q_f32, static_position=[1], dynamic_position=[])
                    p2 = vector.extract(q_f32, static_position=[2], dynamic_position=[])
                    p3 = vector.extract(q_f32, static_position=[3], dynamic_position=[])
                    lo = rocdl.cvt_pk_fp8_f32(
                        T.i32, p0, p1, arith.constant(0, type=T.i32), False
                    )
                    pk = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)
                    q_words.append(pk)
                return q_words

            def _unflatten_k(k_flat):
                return [[k_flat[td * (QKHELOOP * 2) + j]
                          for j in range(QKHELOOP * 2)]
                         for td in range(TLOOP)]

            def _unpack_i64_to_i32_pair(i64_val):
                """Reverse of _pack_i32_pair_to_i64."""
                v1 = vector.from_elements(T.vec(1, T.i64), [i64_val])
                v2 = vector.bitcast(T.vec(2, T.i32), v1)
                return vector.extract(v2, static_position=[0]), vector.extract(v2, static_position=[1])

            # ════════════════════════════════════════════════════════
            # Helper: load V data for one tile — returns vec4<i32> directly (no i64)
            # ════════════════════════════════════════════════════════
            def _load_v_ops(v_block_base_dw, tile_token_offset_i32):
                """Load V data for one tile using pre-computed block base and vhead_elems."""
                result = []
                for vhe in range_constexpr(VHELOOP):
                    vhe_data = []
                    for vt in range_constexpr(VTLOOP):
                        v_token_in_block = tile_token_offset_i32 + _v_tok_thread_off[vt]
                        if trans_v:
                            vt_group = v_token_in_block // arith.constant(FP8_ELEMS_16B, type=T.i32)
                            va_dw = v_block_base_dw + vt_group * arith.constant(HEAD_SIZE * FP8_ELEMS_16B // 4, type=T.i32) + _vhead_elem_dw[vhe]
                        else:
                            va_dw = v_block_base_dw + _vhead_elem_dw[vhe] + v_token_in_block // c_four
                        v_4xi32 = buffer_ops.buffer_load(
                            v_rsrc, va_dw,
                            vec_width=4, dtype=T.i32)
                        vhe_data.append(v_4xi32)
                    result.append(vhe_data)
                return result

            def _load_v_flat_ps(v_block_base_dw, tile_token_offset_i32):
                """Load V data for one tile as flat i64 list using pre-computed block base."""
                v_flat = []
                for vhe in range_constexpr(VHELOOP):
                    for vt in range_constexpr(VTLOOP):
                        v_token_in_block = tile_token_offset_i32 + _v_tok_thread_off[vt]
                        if trans_v:
                            vt_group = v_token_in_block // arith.constant(FP8_ELEMS_16B, type=T.i32)
                            va_dw = v_block_base_dw + vt_group * arith.constant(HEAD_SIZE * FP8_ELEMS_16B // 4, type=T.i32) + _vhead_elem_dw[vhe]
                        else:
                            va_dw = v_block_base_dw + _vhead_elem_dw[vhe] + v_token_in_block // c_four
                        v_4xi32 = buffer_ops.buffer_load(
                            v_rsrc, va_dw,
                            vec_width=4, dtype=T.i32)
                        v_flat.append(_pack_i32_pair_to_i64(
                            vector.extract(v_4xi32, static_position=[0]),
                            vector.extract(v_4xi32, static_position=[1])))
                        v_flat.append(_pack_i32_pair_to_i64(
                            vector.extract(v_4xi32, static_position=[2]),
                            vector.extract(v_4xi32, static_position=[3])))
                return v_flat

            def _unflatten_v(v_flat):
                """Convert flat i64 list back to v_data[vhe][vt] = vec4<i32>."""
                result = []
                fi = 0
                for _vhe in [0, 1]:  # VHELOOP=2, use literal to avoid DSL range()
                    vhe_data = []
                    for _vt in [0, 1, 2, 3]:  # VTLOOP=4
                        lo0, lo1 = _unpack_i64_to_i32_pair(v_flat[fi])
                        hi0, hi1 = _unpack_i64_to_i32_pair(v_flat[fi + 1])
                        v_4xi32 = vector.from_elements(T.vec(4, T.i32), [lo0, lo1, hi0, hi1])
                        vhe_data.append(v_4xi32)
                        fi = fi + 2
                    result.append(vhe_data)
                return result

            # ════════════════════════════════════════════════════════
            # Helper: compute one tile body (QK → softmax → PV)
            # phys_block_i32 = physical block index (from kv_page_indices)
            # tile_token_offset_i32 = token offset within block (0, 256, 512, 768)
            # partition_start = global token offset for masking
            # ════════════════════════════════════════════════════════
            def _compute_partition_ps(phys_block_i32, tile_token_offset_i32,
                                      partition_start, k_ops_all, v_ops_all,
                                      running_max, running_sum, out0, out1,
                                      prefetch_k_block=None, prefetch_k_tile_off=None,
                                      prefetch_v_block=None, prefetch_v_tile_off=None):

                # V data passed in as v_ops_all[vhe][vt] = vec4<i32>
                v_data_prefetched = v_ops_all

                # ── QK MFMA ──
                d_out = []
                for td in range_constexpr(TLOOP):
                    acc = arith.constant_vector(0.0, T.f32x4)
                    for k_step in range_constexpr(QKHELOOP * 2):
                        acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [k_ops_all[td][k_step], q_frags[k_step], acc, 0, 0, 0])
                    d_out.append(acc * vector.broadcast(T.f32x4, _scale))

                # ── Intra-warp softmax ──
                qk_max = NEG_INF
                for td in range_constexpr(TLOOP):
                    for i in range_constexpr(4):
                        kv_tok = (partition_start
                                  + warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32)
                                  + rowid * arith.constant(4, type=T.i32)
                                  + arith.constant(td * MFMA_N + i, type=T.i32))
                        s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                        s = arith.select(kv_tok < context_len, s, NEG_INF)
                        qk_max = qk_max.maximumf(s)
                for sh in [32, 16]:
                    qk_max = qk_max.maximumf(qk_max.shuffle_xor(arith.constant(sh, type=T.i32), c_w))

                sm_max_off = arith.index_cast(T.index,
                    warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
                vector.store(vector.from_elements(T.vec(1, T.f32), [qk_max]), softmax_lds_f32, [sm_max_off])
                exp_sum = ZERO_F
                for td in range_constexpr(TLOOP):
                    for i in range_constexpr(4):
                        kv_tok = (partition_start
                                  + warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32)
                                  + rowid * arith.constant(4, type=T.i32)
                                  + arith.constant(td * MFMA_N + i, type=T.i32))
                        s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                        diff = s - qk_max
                        p = (diff * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                        p = arith.select(kv_tok < context_len, p, ZERO_F)
                        exp_sum = exp_sum + p
                        d_out[td] = vector.insert(p, d_out[td], static_position=[i], dynamic_position=[])
                for sh in [32, 16]:
                    exp_sum = exp_sum + exp_sum.shuffle_xor(arith.constant(sh, type=T.i32), c_w)

                # ── Cross-warp via LDS ──
                sm_sum_off = arith.index_cast(T.index,
                    arith.constant(NUM_WARPS * MFMA_N, type=T.i32) + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
                vector.store(vector.from_elements(T.vec(1, T.f32), [exp_sum]), softmax_lds_f32, [sm_sum_off])

                partition_max = NEG_INF
                partition_sum = ZERO_F
                warp_rescale_factors = []
                rd_max_offs = []
                for w in range_constexpr(NUM_WARPS):
                    rd_max_off = arith.index_cast(T.index,
                        arith.constant(w * MFMA_N, type=T.i32) + lane16id)
                    rd_max_offs.append(rd_max_off)
                gpu.barrier()
                for w in range_constexpr(NUM_WARPS):
                    w_max = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [rd_max_offs[w]]), static_position=[0])
                    partition_max = partition_max.maximumf(w_max)
                    warp_rescale_factors.append(w_max)
                for w in range_constexpr(NUM_WARPS):
                    diff_w = warp_rescale_factors[w] - partition_max
                    diff_w = arith.select(partition_max > NEG_INF, diff_w, ZERO_F)
                    wf = (diff_w * arith.constant(LOG2E, type=T.f32)).exp2(
                        fastmath=arith.FastMathFlags.fast)
                    rd_sum_off = arith.index_cast(T.index,
                        arith.constant(NUM_WARPS * MFMA_N + w * MFMA_N, type=T.i32) + lane16id)
                    w_sum = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [rd_sum_off]), static_position=[0])
                    partition_sum = partition_sum + w_sum * wf
                    warp_rescale_factors[w] = wf
                my_warp_rescale = warp_rescale_factors[0]
                for w in range_constexpr(1, NUM_WARPS):
                    my_warp_rescale = arith.select(
                        warp_id == arith.constant(w, type=T.i32),
                        warp_rescale_factors[w], my_warp_rescale)

                # ── Online softmax update ──
                new_running_max = running_max.maximumf(partition_max)
                accum_scale = arith.select(running_max > NEG_INF,
                    ((running_max - new_running_max) * arith.constant(LOG2E, type=T.f32)).exp2(
                        fastmath=arith.FastMathFlags.fast),
                    ZERO_F)
                part_to_new = arith.select(partition_max > NEG_INF,
                    ((partition_max - new_running_max) * arith.constant(LOG2E, type=T.f32)).exp2(
                        fastmath=arith.FastMathFlags.fast),
                    ZERO_F)
                running_sum = accum_scale * running_sum + partition_sum * part_to_new
                running_max = new_running_max
                out0 = out0 * vector.broadcast(T.f32x4, accum_scale)
                out1 = out1 * vector.broadcast(T.f32x4, accum_scale)

                # ── Prob FP8 pack + LDS store ──
                prob_scale = my_warp_rescale * part_to_new
                for td in range_constexpr(TLOOP):
                    d_out[td] = d_out[td] * vector.broadcast(T.f32x4, prob_scale)
                for td in range_constexpr(TLOOP):
                    p0 = vector.extract(d_out[td], static_position=[0], dynamic_position=[])
                    p1 = vector.extract(d_out[td], static_position=[1], dynamic_position=[])
                    p2 = vector.extract(d_out[td], static_position=[2], dynamic_position=[])
                    p3 = vector.extract(d_out[td], static_position=[3], dynamic_position=[])
                    lo = rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, arith.constant(0, type=T.i32), False)
                    pk = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)
                    rowid_8x8 = rowid // arith.constant(2, type=T.i32)
                    offset_in_slot = rowid % arith.constant(2, type=T.i32)
                    byte_base = (warp_id * arith.constant(4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                 + arith.constant(td * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                 + lane16id * arith.constant(PROB_ROW_STRIDE_BYTES, type=T.i32)
                                 + rowid_8x8 * arith.constant(8, type=T.i32)
                                 + offset_in_slot * arith.constant(4, type=T.i32))
                    i32_off = byte_base // arith.constant(4, type=T.i32)
                    pk_vec = vector.from_elements(T.vec(1, T.i32), [pk])
                    vector.store(pk_vec, logits_lds_i32, [arith.index_cast(T.index, i32_off)])

                # ── K prefetch for next tile (after prob pack, overlaps PV) ──
                if prefetch_k_block is not None:
                    k_prefetched = _load_k_flat_ps(prefetch_k_block, prefetch_k_tile_off)
                else:
                    k_prefetched = None

                # ── V prefetch for next tile ──
                if prefetch_v_block is not None:
                    v_prefetched = _load_v_ops(prefetch_v_block, prefetch_v_tile_off)
                else:
                    v_prefetched = None

                # ── PV MFMA ──
                pv_results = [arith.constant_vector(0.0, T.f32x4)
                              for _ in range_constexpr(VHELOOP)]
                v_i64s = []
                p_i64s = []
                for vhe in range_constexpr(VHELOOP):
                    for vt in range_constexpr(VTLOOP):
                        v_4xi32 = v_data_prefetched[vhe][vt]
                        for j in range_constexpr(2):
                            v_i64 = _pack_i32_pair_to_i64(
                                vector.extract(v_4xi32, static_position=[j * 2]),
                                vector.extract(v_4xi32, static_position=[j * 2 + 1]))
                            v_i64s.append(v_i64)
                            offset_raw = rowid * arith.constant(4, type=T.i32) + arith.constant(j * 2, type=T.i32)
                            p_off1 = (offset_raw % arith.constant(ROWS_PER_WARP, type=T.i32)) // arith.constant(2, type=T.i32)
                            p_off2 = offset_raw // arith.constant(ROWS_PER_WARP, type=T.i32)
                            p_byte = (arith.constant(vt * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                      + p_off2 * arith.constant(MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                      + lane16id * arith.constant(PROB_ROW_STRIDE_BYTES, type=T.i32)
                                      + p_off1 * arith.constant(8, type=T.i32))
                            p_i32_idx = p_byte // arith.constant(4, type=T.i32)
                            if vhe == 0 and vt == 0 and j == 0:
                                gpu.barrier()
                            pw0 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                                [arith.index_cast(T.index, p_i32_idx)]), static_position=[0])
                            pw1 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                                [arith.index_cast(T.index, p_i32_idx + arith.constant(1, type=T.i32))]), static_position=[0])
                            p_i64 = _pack_i32_pair_to_i64(pw0, pw1)
                            p_i64s.append(p_i64)
                for vhe in range_constexpr(VHELOOP):
                    tmp_out = arith.constant_vector(0.0, T.f32x4)
                    for vt in range_constexpr(VTLOOP):
                        for j in range_constexpr(2):
                            tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4, [v_i64s[vhe * VTLOOP * 2 + vt * 2 + j], p_i64s[vhe * VTLOOP * 2 + vt * 2 + j], tmp_out, 0, 0, 0])
                            pv_results[vhe] = tmp_out
                out0 = out0 + pv_results[0] * vector.broadcast(T.f32x4, v_scale_val)
                out1 = out1 + pv_results[1] * vector.broadcast(T.f32x4, v_scale_val)
                return running_max, running_sum, out0, out1, k_prefetched, v_prefetched

            # ── Phase helpers for barrier-merged pipeline ──

            def _qk_and_intra_softmax(k_ops, partition_start, v_block_base_dw, tile_token_offset_i32, phys_block=None):
                """Phase 1: QK MFMA + intra-warp softmax + LDS write max/sum.
                Returns (d_out, va_dws, v_scale_vecs) where va_dws is the V address
                array (V data loading is deferred to _pv_mfma), and v_scale_vecs is
                a list of f32x4 for each td (per-token mode) or None (per-tensor mode).
                """
                va_dws = []
                for vt in range_constexpr(VTLOOP):
                    vhe_data = []
                    for vhe in range_constexpr(VHELOOP):
                        v_token_in_block = tile_token_offset_i32 + _v_tok_thread_off[vt]
                        if trans_v:
                            vt_group = v_token_in_block // arith.constant(FP8_ELEMS_16B, type=T.i32)
                            va_dw = v_block_base_dw + vt_group * arith.constant(HEAD_SIZE * FP8_ELEMS_16B // 4, type=T.i32) + _vhead_elem_dw[vhe]
                        else:
                            va_dw = v_block_base_dw + _vhead_elem_dw[vhe] + v_token_in_block // c_four
                        vhe_data.append(va_dw)
                    va_dws.append(vhe_data)

                # Per-token: pre-load k_scale and v_scale for all td steps
                # (issued early to overlap with QK MFMA for memory latency hiding)
                if per_token_kv:
                    # scale tensor layout: [num_blocks, num_kv_heads, kv_block_size, 1] in f32
                    # Token dim stride=1 (contiguous f32).  MFMA 16x16x32 row `rowid`
                    # maps to tokens [rowid*4 .. rowid*4+3], which are contiguous, so each
                    # thread can directly vector-load its 4 scale values (buffer_load_dwordx4)
                    # instead of scalar-load + 4× ds_bpermute broadcast.
                    scale_block_base = phys_block * stride_ks_block + kv_h * stride_ks_head
                    _scale_tok_base_vec = (tile_token_offset_i32
                                          + warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32)
                                          + rowid * c_four)
                    k_scale_vecs = []
                    v_scale_vecs = []
                    for td in range_constexpr(TLOOP):
                        tok_off_vec = scale_block_base + _scale_tok_base_vec + arith.constant(td * MFMA_N, type=T.i32)
                        k_scale_vecs.append(buffer_ops.buffer_load(
                            ks_rsrc, tok_off_vec, vec_width=4, dtype=T.f32))
                        v_scale_vecs.append(buffer_ops.buffer_load(
                            vs_rsrc, tok_off_vec, vec_width=4, dtype=T.f32))
                else:
                    v_scale_vecs = None

                # V data loading is deferred to _pv_mfma to reduce peak VGPR pressure.
                # Only V addresses (va_dws) are kept live across the softmax phases.
                d_out = []
                for td in range_constexpr(TLOOP):
                    acc = arith.constant_vector(0.0, T.f32x4)
                    for k_step in range_constexpr(QKHELOOP * 2):
                        acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [k_ops[td][k_step], q_frags[k_step], acc, 0, 0, 0])
                    if per_token_kv:
                        # per-token: scale by (softmax_scale * q_scale) * k_scale_per_token
                        d_out.append(
                            acc
                            * (
                                k_scale_vecs[td]
                                * vector.broadcast(T.f32x4, _softmax_q_scale)
                            )
                        )
                    else:
                        d_out.append(acc * vector.broadcast(T.f32x4, _scale))

                # Pre-compute kv_tok base for this tile (add compile-time td*MFMA_N+i offsets)
                kv_tok_base = partition_start + _kv_tok_thread_base if needs_mask or query_length > 1 else None
                qk_max = NEG_INF
                for td in range_constexpr(TLOOP):
                    for i in range_constexpr(4):
                        s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                        if needs_mask or query_length > 1:
                            kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                            s = arith.select(kv_tok < causal_bound, s, NEG_INF)
                        qk_max = qk_max.maximumf(s)
                for sh in [32, 16]:
                    qk_max = qk_max.maximumf(qk_max.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
                vector.store(vector.from_elements(T.vec(1, T.f32), [qk_max]), softmax_lds_f32, [_sm_max_off])
                exp_sum = ZERO_F
                for td in range_constexpr(TLOOP):
                    for i in range_constexpr(4):
                        s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                        diff = s - qk_max
                        p = (diff * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                        if needs_mask or query_length > 1:
                            kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                            p = arith.select(kv_tok < causal_bound, p, ZERO_F)
                        exp_sum = exp_sum + p
                        d_out[td] = vector.insert(p, d_out[td], static_position=[i], dynamic_position=[])
                    rocdl.sched_barrier(0)
                for sh in [32, 16]:
                    exp_sum = exp_sum + exp_sum.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
                vector.store(vector.from_elements(T.vec(1, T.f32), [exp_sum]), softmax_lds_f32, [_sm_sum_off])

                # Per-token: write per-warp v_scale_max to LDS for cross-warp reduction.
                if per_token_kv:
                    v_max_warp = ZERO_F
                    for td in range_constexpr(TLOOP):
                        vs = v_scale_vecs[td]
                        for i in range_constexpr(4):
                            if needs_mask or query_length > 1:
                                kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                                vs_i = vector.extract(vs, static_position=[i], dynamic_position=[])
                                vs_i = arith.select(kv_tok < causal_bound, vs_i, ZERO_F)
                                vs = vector.insert(vs_i, vs, static_position=[i], dynamic_position=[])
                        v_max_warp = v_max_warp.maximumf(vector.reduction(T.f32, "maxnumf", vs))
                        rocdl.sched_barrier(0)
                    for sh in [32, 16]:
                        v_max_warp = v_max_warp.maximumf(
                            v_max_warp.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
                    vector.store(vector.from_elements(T.vec(1, T.f32), [v_max_warp]),
                                 softmax_lds_f32, [_sm_vmax_wr_off])

                return d_out, va_dws, v_scale_vecs

            def _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, o0, o1, v_scale_vecs=None):
                """Phase 2: Cross-warp reduction + online softmax + prob pack to LDS.
                ASSUMES barrier already called to make softmax LDS visible.
                When needs_mask=False, removes -inf guards (partition_max always finite).
                The rmax > NEG_INF guard is also safe to remove: exp2((-inf - x)*LOG2E) = 0.0 in IEEE754.

                Per-token V scale (per_token_kv=True): v_scale_vecs[td] = f32x4 of per-token v_scale.
                Uses a tile-wide (256-token) v_scale_max read from LDS (all 4 warps wrote their
                per-warp v_max in _qk_and_intra_softmax before the barrier).  This is required
                because _pv_mfma reads ALL warps' probabilities from LDS and applies a single
                v_correction — so all warps must normalize with the SAME v_max.

                Returns (rmax, rsum, o0, o1, v_correction) where v_correction is the factor to
                multiply the PV MFMA output by: v_scale_max/FP8_MAX for per-token, v_scale_val for per-tensor.
                """
                # ── Step 1: Read only max from LDS, compute partition_max + rescale factors ──
                partition_max = NEG_INF
                warp_rescale_factors = []
                for w in range_constexpr(NUM_WARPS):
                    w_max = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [_sm_rd_max_offs[w]]), static_position=[0])
                    partition_max = partition_max.maximumf(w_max)
                    warp_rescale_factors.append(w_max)
                for w in range_constexpr(NUM_WARPS):
                    diff_w = warp_rescale_factors[w] - partition_max
                    if needs_mask:
                        diff_w = arith.select(partition_max > NEG_INF, diff_w, ZERO_F)
                    warp_rescale_factors[w] = (diff_w * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                my_warp_rescale = warp_rescale_factors[0]
                for w in range_constexpr(1, NUM_WARPS):
                    my_warp_rescale = arith.select(
                        warp_id == arith.constant(w, type=T.i32),
                        warp_rescale_factors[w], my_warp_rescale)

                # ── Step 2: Online softmax rescale (no sum/vmax LDS reads needed yet) ──
                new_rmax = rmax.maximumf(partition_max)
                if needs_mask:
                    accum_scale = arith.select(rmax > NEG_INF,
                        ((rmax - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast),
                        ZERO_F)
                    part_to_new = arith.select(partition_max > NEG_INF,
                        ((partition_max - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast),
                        ZERO_F)
                else:
                    accum_scale = ((rmax - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                    part_to_new = ((partition_max - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                o0 = o0 * vector.broadcast(T.f32x4, accum_scale)
                o1 = o1 * vector.broadcast(T.f32x4, accum_scale)

                # ── Step 3: Barrier prevents LLVM from hoisting sum/vmax reads into Step 2,
                #    reducing peak VGPR by separating the rescale computation from LDS reads ──
                rocdl.sched_barrier(0)

                # ── Step 4: Read sum from LDS, compute partition_sum + rsum ──
                partition_sum = ZERO_F
                for w in range_constexpr(NUM_WARPS):
                    w_sum = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [_sm_rd_sum_offs[w]]), static_position=[0])
                    partition_sum = partition_sum + w_sum * warp_rescale_factors[w]
                rsum = accum_scale * rsum + partition_sum * part_to_new
                rmax = new_rmax

                # ── Step 5: Read vmax from LDS (per-token), prob scale + FP8 pack ──
                if per_token_kv and v_scale_vecs is not None:
                    v_max_global = ZERO_F
                    for w in range_constexpr(NUM_WARPS):
                        w_vmax = vector.extract(
                            vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [_sm_vmax_rd_offs[w]]),
                            static_position=[0])
                        v_max_global = v_max_global.maximumf(w_vmax)
                    v_max_safe = v_max_global + arith.constant(1e-8, type=T.f32)
                    c_fp8_max = arith.constant(FP8_MAX, type=T.f32)
                    # Use rocdl.rcp to avoid div_scale/div_fixup chain (~6 VGPR temporaries).
                    norm_factor = c_fp8_max * rocdl.rcp(T.f32, v_max_safe)
                    prob_scale = my_warp_rescale  # = exp(qk_max_warp - partition_max)
                    v_correction = v_max_global * rocdl.rcp(T.f32, c_fp8_max) * part_to_new
                    for td in range_constexpr(TLOOP):
                        # Apply per-tile softmax renormalization and normalize v_scale to FP8 range
                        d_out[td] = d_out[td] * (
                            v_scale_vecs[td]
                            * vector.broadcast(T.f32x4, prob_scale * norm_factor)
                        )
                else:
                    prob_scale = my_warp_rescale * part_to_new
                    v_correction = v_scale_val
                    for td in range_constexpr(TLOOP):
                        d_out[td] = d_out[td] * vector.broadcast(T.f32x4, prob_scale)

                for td in range_constexpr(TLOOP):
                    p0 = vector.extract(d_out[td], static_position=[0], dynamic_position=[])
                    p1 = vector.extract(d_out[td], static_position=[1], dynamic_position=[])
                    p2 = vector.extract(d_out[td], static_position=[2], dynamic_position=[])
                    p3 = vector.extract(d_out[td], static_position=[3], dynamic_position=[])
                    lo = rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, arith.constant(0, type=T.i32), False)
                    pk = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)
                    byte_base = _prob_wr_thread_base + arith.constant(td * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                    i32_off = byte_base // c_four
                    pk_vec = vector.from_elements(T.vec(1, T.i32), [pk])
                    vector.store(pk_vec, logits_lds_i32, [arith.index_cast(T.index, i32_off)])
                return rmax, rsum, o0, o1, v_correction

            def _pv_mfma(va_dws, o0, o1, v_correction):
                """Phase 3: V data load + LDS read probs + PV MFMA.
                ASSUMES barrier already called to make prob LDS visible.
                Loads V data from global memory (deferred from Phase 1a to reduce
                peak VGPR), reads prob data from LDS, and executes PV MFMAs.
                v_correction: scalar to multiply PV output by.
                  Per-tensor: v_scale_val (constant).
                  Per-token: v_scale_max/FP8_MAX (per-warp, computed in _cross_warp_softmax_and_prob_pack).
                """
                # Load V data here (deferred from _qk_and_intra_softmax to avoid
                # 32 VGPRs being live across the entire softmax + cross-warp phases).
                v_ops = []
                for vt in range_constexpr(VTLOOP):
                    vhe_data = []
                    for vhe in range_constexpr(VHELOOP):
                        v_4xi32 = buffer_ops.buffer_load(
                            v_rsrc, va_dws[vt][vhe],
                            vec_width=4, dtype=T.i32)
                        vhe_data.append(v_4xi32)
                    v_ops.append(vhe_data)
                pv_results = [arith.constant_vector(0.0, T.f32x4) for _ in range_constexpr(VHELOOP)]
                v_i64s = []
                p_i64s = []
                for vhe in range_constexpr(VHELOOP):
                    for vt in range_constexpr(VTLOOP):
                        v_4xi32 = v_ops[vt][vhe]
                        for j in range_constexpr(2):
                            v_i64 = _pack_i32_pair_to_i64(
                                vector.extract(v_4xi32, static_position=[j * 2]),
                                vector.extract(v_4xi32, static_position=[j * 2 + 1]))
                            v_i64s.append(v_i64)
                            # p_byte = vt*2560 + _pv_prob_read_base + j*8
                            p_byte = (arith.constant(vt * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                      + _pv_prob_read_base
                                      + arith.constant(j * 8, type=T.i32))
                            p_i32_idx = p_byte // c_four
                            pw0 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                                [arith.index_cast(T.index, p_i32_idx)]), static_position=[0])
                            pw1 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                                [arith.index_cast(T.index, p_i32_idx + c_one)]), static_position=[0])
                            p_i64 = _pack_i32_pair_to_i64(pw0, pw1)
                            p_i64s.append(p_i64)
                for vhe in range_constexpr(VHELOOP):
                    tmp_out = arith.constant_vector(0.0, T.f32x4)
                    for vt in range_constexpr(VTLOOP):
                        for j in range_constexpr(2):
                            tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4, [v_i64s[vhe * VTLOOP * 2 + vt * 2 + j], p_i64s[vhe * VTLOOP * 2 + vt * 2 + j], tmp_out, 0, 0, 0])
                            pv_results[vhe] = tmp_out
                o0 = o0 + pv_results[0] * vector.broadcast(T.f32x4, v_correction)
                o1 = o1 + pv_results[1] * vector.broadcast(T.f32x4, v_correction)
                return o0, o1

            # ════════════════════════════════════════════════════════
            # Inner KV loop — one CTA processes one 256-token sub-tile
            # across all 1024-token physical blocks in the work item.
            # ════════════════════════════════════════════════════════
            def _unwrap(v):
                return v.ir_value() if hasattr(v, 'ir_value') else v

            def _pack_state(rmax, rsum, o0, o1, phys_block, k_flat):
                return [_unwrap(v) for v in [rmax, rsum, o0, o1, phys_block] + k_flat]

            def _unpack_state(state):
                return state[0], state[1], state[2], state[3], state[4], list(state[5:5 + _N_K])

            def _process_block_split(phys_block, block_idx_in_work, rmax, rsum, o0, o1,
                                     tile_token_offset_i32, k_ops, next_phys_block=None):
                """Process one 256-token block split inside a 1024-token KV page.

                Pipeline (inlined from _qk_and_intra_softmax / _cross_warp_softmax_and_prob_pack / _pv_mfma):
                  Phase 1: K/V scale load + V load + QK MFMA + intra-warp softmax + LDS write max/sum
                  K next prefetch (overlapped with Phase 1 softmax)
                  ── barrier ──
                  Phase 2: Cross-warp softmax reduction + online rescale + prob FP8 pack → LDS
                  ── barrier ──
                  Phase 3: Prob LDS read + PV MFMA + accumulate
                """
                partition_start = kv_start_abs_tok + block_idx_in_work * c_bs + tile_token_offset_i32

                # ════════════════════════════════════════════════════════════
                # Phase 1b: Per-token K/V scale load (1 dwordx4 per scale + ds_bpermute)
                # Each lane loads 4 consecutive scales at lane16id*4 offset,
                # covering all 64 tokens (TLOOP*MFMA_N) in a single load per scale.
                # ds_bpermute redistributes to each thread's rowid-specific values per td.
                # ════════════════════════════════════════════════════════════
                if per_token_kv:
                    scale_block_base = phys_block * stride_ks_block + kv_h * stride_ks_head
                    _scale_tok_base_lane = (tile_token_offset_i32
                                          + warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32)
                                          + lane16id * c_four)
                    scale_off = scale_block_base + _scale_tok_base_lane
                    k_scale_raw = buffer_ops.buffer_load(ks_rsrc, scale_off, vec_width=4, dtype=T.f32)
                    v_scale_raw = buffer_ops.buffer_load(vs_rsrc, scale_off, vec_width=4, dtype=T.f32)
                    # Extract elements as i32 for ds_bpermute source
                    k_raw_i32 = [arith.bitcast(T.i32, vector.extract(k_scale_raw, static_position=[i], dynamic_position=[]))
                                 for i in range_constexpr(4)]
                    v_raw_i32 = [arith.bitcast(T.i32, vector.extract(v_scale_raw, static_position=[i], dynamic_position=[]))
                                 for i in range_constexpr(4)]
                    # Redistribute via ds_bpermute: thread (rowid=R) reads from lane (R + td*4)
                    k_scale_vecs = []
                    v_scale_vecs = []
                    for td in range_constexpr(TLOOP):
                        src_lane = rowid + arith.constant(td * 4, type=T.i32)
                        bperm_addr = src_lane * c_four  # byte address for ds_bpermute
                        k_elems = []
                        v_elems = []
                        for i in range_constexpr(4):
                            k_elems.append(arith.bitcast(T.f32, rocdl.ds_bpermute(T.i32, bperm_addr, k_raw_i32[i])))
                            v_elems.append(arith.bitcast(T.f32, rocdl.ds_bpermute(T.i32, bperm_addr, v_raw_i32[i])))
                        k_scale_vecs.append(vector.from_elements(T.f32x4, k_elems))
                        v_scale_vecs.append(vector.from_elements(T.f32x4, v_elems))
                else:
                    v_scale_vecs = None

                # ════════════════════════════════════════════════════════════
                # Phase 1b2: V data load (issued before QK MFMA for latency hiding)
                # ════════════════════════════════════════════════════════════
                v_block_base_dw = (phys_block * c_vb + _v_head_off) // c_four
                v_ops = []
                for vt in range_constexpr(VTLOOP):
                    vhe_data = []
                    for vhe in range_constexpr(VHELOOP):
                        v_token_in_block = tile_token_offset_i32 + _v_tok_thread_off[vt]
                        if trans_v:
                            vt_group = v_token_in_block // arith.constant(FP8_ELEMS_16B, type=T.i32)
                            va_dw = v_block_base_dw + vt_group * arith.constant(HEAD_SIZE * FP8_ELEMS_16B // 4, type=T.i32) + _vhead_elem_dw[vhe]
                        else:
                            va_dw = v_block_base_dw + _vhead_elem_dw[vhe] + v_token_in_block // c_four
                        v_4xi32 = buffer_ops.buffer_load(
                            v_rsrc, va_dw,
                            vec_width=4, dtype=T.i32)
                        vhe_data.append(v_4xi32)
                    v_ops.append(vhe_data)

                # ════════════════════════════════════════════════════════════
                # Phase 1c: QK MFMA
                # ════════════════════════════════════════════════════════════
                d_out = []
                for td in range_constexpr(TLOOP):
                    acc = arith.constant_vector(0.0, T.f32x4)
                    for k_step in range_constexpr(QKHELOOP * 2):
                        acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [k_ops[td][k_step], q_frags[k_step], acc, 0, 0, 0])
                    if per_token_kv:
                        d_out.append(
                            acc
                            * (
                                k_scale_vecs[td]
                                * vector.broadcast(T.f32x4, _softmax_q_scale)
                            )
                        )
                    else:
                        d_out.append(acc * vector.broadcast(T.f32x4, _scale))

                # ════════════════════════════════════════════════════════════
                # Phase 1d: Intra-warp softmax — max reduction
                # ════════════════════════════════════════════════════════════
                kv_tok_base = partition_start + _kv_tok_thread_base if needs_mask or query_length > 1 else None
                qk_max = NEG_INF
                for td in range_constexpr(TLOOP):
                    for i in range_constexpr(4):
                        s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                        if needs_mask or query_length > 1:
                            kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                            s = arith.select(kv_tok < causal_bound, s, NEG_INF)
                        qk_max = qk_max.maximumf(s)
                for sh in [32, 16]:
                    qk_max = qk_max.maximumf(qk_max.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
                vector.store(vector.from_elements(T.vec(1, T.f32), [qk_max]), softmax_lds_f32, [_sm_max_off])

                # ════════════════════════════════════════════════════════════
                # Phase 1e: Intra-warp softmax — exp + sum
                # ════════════════════════════════════════════════════════════
                exp_sum = ZERO_F
                for td in range_constexpr(TLOOP):
                    for i in range_constexpr(4):
                        s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                        diff = s - qk_max
                        p = (diff * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                        if needs_mask or query_length > 1:
                            kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                            p = arith.select(kv_tok < causal_bound, p, ZERO_F)
                        exp_sum = exp_sum + p
                        d_out[td] = vector.insert(p, d_out[td], static_position=[i], dynamic_position=[])
                    rocdl.sched_barrier(0)
                for sh in [32, 16]:
                    exp_sum = exp_sum + exp_sum.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
                vector.store(vector.from_elements(T.vec(1, T.f32), [exp_sum]), softmax_lds_f32, [_sm_sum_off])

                # ════════════════════════════════════════════════════════════
                # Phase 1f: Per-token v_scale_max write to LDS
                # ════════════════════════════════════════════════════════════
                if per_token_kv:
                    v_max_warp = ZERO_F
                    for td in range_constexpr(TLOOP):
                        vs = v_scale_vecs[td]
                        for i in range_constexpr(4):
                            if needs_mask or query_length > 1:
                                kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                                vs_i = vector.extract(vs, static_position=[i], dynamic_position=[])
                                vs_i = arith.select(kv_tok < causal_bound, vs_i, ZERO_F)
                                vs = vector.insert(vs_i, vs, static_position=[i], dynamic_position=[])
                        v_max_warp = v_max_warp.maximumf(vector.reduction(T.f32, "maxnumf", vs))
                        rocdl.sched_barrier(0)
                    for sh in [32, 16]:
                        v_max_warp = v_max_warp.maximumf(
                            v_max_warp.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
                    vector.store(vector.from_elements(T.vec(1, T.f32), [v_max_warp]),
                                 softmax_lds_f32, [_sm_vmax_wr_off])

                # ══════════════════ barrier ═════════════════════════════════
                gpu.barrier()

                # ════════════════════════════════════════════════════════════
                # Phase 2a: Cross-warp max reduction + rescale factors
                # ════════════════════════════════════════════════════════════
                partition_max = NEG_INF
                warp_rescale_factors = []
                for w in range_constexpr(NUM_WARPS):
                    w_max = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [_sm_rd_max_offs[w]]), static_position=[0])
                    partition_max = partition_max.maximumf(w_max)
                    warp_rescale_factors.append(w_max)
                for w in range_constexpr(NUM_WARPS):
                    diff_w = warp_rescale_factors[w] - partition_max
                    if needs_mask:
                        diff_w = arith.select(partition_max > NEG_INF, diff_w, ZERO_F)
                    warp_rescale_factors[w] = (diff_w * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                my_warp_rescale = warp_rescale_factors[0]
                for w in range_constexpr(1, NUM_WARPS):
                    my_warp_rescale = arith.select(
                        warp_id == arith.constant(w, type=T.i32),
                        warp_rescale_factors[w], my_warp_rescale)

                # ════════════════════════════════════════════════════════════
                # Phase 2b: Online softmax rescale (accumulator correction)
                # ════════════════════════════════════════════════════════════
                new_rmax = rmax.maximumf(partition_max)
                if needs_mask:
                    accum_scale = arith.select(rmax > NEG_INF,
                        ((rmax - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast),
                        ZERO_F)
                    part_to_new = arith.select(partition_max > NEG_INF,
                        ((partition_max - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast),
                        ZERO_F)
                else:
                    accum_scale = ((rmax - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                    part_to_new = ((partition_max - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                o0 = o0 * vector.broadcast(T.f32x4, accum_scale)
                o1 = o1 * vector.broadcast(T.f32x4, accum_scale)

                # sched_barrier: prevent LLVM from hoisting sum/vmax reads into rescale
                rocdl.sched_barrier(0)

                # ════════════════════════════════════════════════════════════
                # Phase 2c: Cross-warp sum reduction + running sum update
                # ════════════════════════════════════════════════════════════
                partition_sum = ZERO_F
                for w in range_constexpr(NUM_WARPS):
                    w_sum = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [_sm_rd_sum_offs[w]]), static_position=[0])
                    partition_sum = partition_sum + w_sum * warp_rescale_factors[w]
                rsum = accum_scale * rsum + partition_sum * part_to_new
                rmax = new_rmax

                # ════════════════════════════════════════════════════════════
                # Phase 2d: v_scale correction + prob scale + FP8 pack → LDS
                # ════════════════════════════════════════════════════════════
                if per_token_kv and v_scale_vecs is not None:
                    v_max_global = ZERO_F
                    for w in range_constexpr(NUM_WARPS):
                        w_vmax = vector.extract(
                            vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [_sm_vmax_rd_offs[w]]),
                            static_position=[0])
                        v_max_global = v_max_global.maximumf(w_vmax)
                    v_max_safe = v_max_global + arith.constant(1e-8, type=T.f32)
                    c_fp8_max = arith.constant(FP8_MAX, type=T.f32)
                    norm_factor = c_fp8_max * rocdl.rcp(T.f32, v_max_safe)
                    prob_scale = my_warp_rescale
                    v_correction = v_max_global * rocdl.rcp(T.f32, c_fp8_max) * part_to_new
                    for td in range_constexpr(TLOOP):
                        d_out[td] = d_out[td] * (
                            v_scale_vecs[td]
                            * vector.broadcast(T.f32x4, prob_scale * norm_factor)
                        )
                else:
                    prob_scale = my_warp_rescale * part_to_new
                    v_correction = v_scale_val
                    for td in range_constexpr(TLOOP):
                        d_out[td] = d_out[td] * vector.broadcast(T.f32x4, prob_scale)

                # ════════════════════════════════════════════════════════════
                # Phase 2e: Prob FP8 pack + LDS write
                # ════════════════════════════════════════════════════════════
                for td in range_constexpr(TLOOP):
                    p0 = vector.extract(d_out[td], static_position=[0], dynamic_position=[])
                    p1 = vector.extract(d_out[td], static_position=[1], dynamic_position=[])
                    p2 = vector.extract(d_out[td], static_position=[2], dynamic_position=[])
                    p3 = vector.extract(d_out[td], static_position=[3], dynamic_position=[])
                    lo = rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, arith.constant(0, type=T.i32), False)
                    pk = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)
                    byte_base = _prob_wr_thread_base + arith.constant(td * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                    i32_off = byte_base // c_four
                    pk_vec = vector.from_elements(T.vec(1, T.i32), [pk])
                    vector.store(pk_vec, logits_lds_i32, [arith.index_cast(T.index, i32_off)])

                # ══════════════════ barrier ═════════════════════════════════
                gpu.barrier()

                # ════════════════════════════════════════════════════════════
                # Phase 3a: Prob LDS read + V unpack → i64 operands
                # ════════════════════════════════════════════════════════════
                pv_results = [arith.constant_vector(0.0, T.f32x4) for _ in range_constexpr(VHELOOP)]
                v_i64s = []
                p_i64s = []
                for vhe in range_constexpr(VHELOOP):
                    for vt in range_constexpr(VTLOOP):
                        v_4xi32 = v_ops[vt][vhe]
                        for j in range_constexpr(2):
                            v_i64 = _pack_i32_pair_to_i64(
                                vector.extract(v_4xi32, static_position=[j * 2]),
                                vector.extract(v_4xi32, static_position=[j * 2 + 1]))
                            v_i64s.append(v_i64)
                            p_byte = (arith.constant(vt * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                      + _pv_prob_read_base
                                      + arith.constant(j * 8, type=T.i32))
                            p_i32_idx = p_byte // c_four
                            pw0 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                                [arith.index_cast(T.index, p_i32_idx)]), static_position=[0])
                            pw1 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                                [arith.index_cast(T.index, p_i32_idx + c_one)]), static_position=[0])
                            p_i64 = _pack_i32_pair_to_i64(pw0, pw1)
                            p_i64s.append(p_i64)

                # ════════════════════════════════════════════════════════════
                # K next prefetch (overlapped with prob LDS reads)
                # ════════════════════════════════════════════════════════════
                if next_phys_block is not None:
                    next_k_base = (next_phys_block * c_kb + _k_head_off) // c_four
                    k_next_flat = _load_k_flat_ps(next_k_base, tile_token_offset_i32)
                else:
                    k_next_flat = None

                # ════════════════════════════════════════════════════════════
                # Phase 3b: PV MFMA + accumulate output
                # ════════════════════════════════════════════════════════════
                for vhe in range_constexpr(VHELOOP):
                    tmp_out = arith.constant_vector(0.0, T.f32x4)
                    for vt in range_constexpr(VTLOOP):
                        for j in range_constexpr(2):
                            tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4, [v_i64s[vhe * VTLOOP * 2 + vt * 2 + j], p_i64s[vhe * VTLOOP * 2 + vt * 2 + j], tmp_out, 0, 0, 0])
                            pv_results[vhe] = tmp_out
                o0 = o0 + pv_results[0] * vector.broadcast(T.f32x4, v_correction)
                o1 = o1 + pv_results[1] * vector.broadcast(T.f32x4, v_correction)

                return rmax, rsum, o0, o1, k_next_flat


            # Metadata remaps every work tile into a partial slot shared across q-head ranges.
            # grid_z then expands each slot into 4 block-split partials.
            c_ql = arith.constant(query_length, type=T.i32)
            c_zero_i32 = arith.constant(0, type=T.i32)
            block_split_idx = gpu.block_idx.z
            tile_token_offset = block_split_idx * c_cps
            _partial_ge_zero = partial_idx >= c_zero_i32
            _po_row_base = arith.select(
                _partial_ge_zero,
                partial_idx * c_tpb + block_split_idx * c_ql + c_ql,
                c_zero_i32,
            )

            # Clip the block loop to physical blocks whose 256-token split still overlaps
            # the sequence. Fully masked splits fall through to the zero/-inf writeback path.
            num_blocks_in_work = kv_end - kv_start
            last_block_idx_val = num_blocks_in_work - c_one
            _valid_blocks_in_work = _compute_valid_block_limit(
                context_len=context_len,
                kv_start_abs_tok=kv_start_abs_tok,
                tile_token_offset=tile_token_offset,
                num_blocks_in_work=num_blocks_in_work,
            )
            _has_valid_blocks = arith.cmpi(arith.CmpIPredicate.sgt, _valid_blocks_in_work, c_zero_i32)
            _loop_start_g = arith.index(0)
            _loop_stop_g = arith.index_cast(T.index, arith.unwrap(_valid_blocks_in_work))
            _loop_step_g = arith.index(1)
            _mtp_groups = math.ceil(query_length * query_group_size / 16)
            _total_pairs = query_length * query_group_size

            _if_valid = scf.IfOp(_has_valid_blocks, has_else=True)
            with _if_then(_if_valid):
                first_phys_block = buffer_ops.buffer_load(
                    kpi_rsrc, kv_start, vec_width=1, dtype=T.i32
                )
                # ── MTP groups: Python compile-time loop — one MLIR KV-loop per group ──
                # Use range_constexpr so AST rewriter keeps this as a plain Python loop
                for _mtp_g in range_constexpr(_mtp_groups):
                    _g_off = _mtp_g * 16  # compile-time offset into (qi, qhi) pair space

                    # ── Lane pair for MFMA output (qi_val and qhi_pos) ──
                    _lane_pair_raw = lane16id + arith.constant(_g_off, type=T.i32)
                    _c_total_pairs = arith.constant(_total_pairs, type=T.i32)
                    _c_pair_max = arith.constant(_total_pairs - 1, type=T.i32)
                    _c_ql_m1 = arith.constant(query_length - 1, type=T.i32)
                    _lane_pair = arith.select(_lane_pair_raw < _c_total_pairs, _lane_pair_raw, _c_pair_max)
                    _qi_raw = _lane_pair // arith.constant(query_group_size, type=T.i32)
                    qi_val = arith.select(_qi_raw < _c_ql_m1, _qi_raw, _c_ql_m1)
                    qhi_pos = _lane_pair % arith.constant(query_group_size, type=T.i32)
                    # MTP causal bound for this lane's qi_val token
                    causal_bound = context_len + arith.constant(1 - query_length, type=T.i32) + qi_val

                    # ── local_qhead_idx pair for Q loading ──
                    _lqh_pair_raw = local_qhead_idx + arith.constant(_g_off, type=T.i32)
                    _lqh_pair = arith.select(_lqh_pair_raw < _c_total_pairs, _lqh_pair_raw, _c_pair_max)
                    _lqi_raw = _lqh_pair // arith.constant(query_group_size, type=T.i32)
                    qi_for_q = arith.select(_lqi_raw < _c_ql_m1, _lqi_raw, _c_ql_m1)
                    local_qhead_idx_for_q = _lqh_pair % arith.constant(query_group_size, type=T.i32)

                    # ── Q load into LDS for this mtp_g pass ──
                    # Between passes: barrier ensures prev pass's LDS prob-reads are done
                    if _mtp_g > 0:
                        gpu.barrier()
                        SmemPtr._view_cache = None
                    q_row = batch_idx * arith.constant(query_length, type=T.i32) + qi_for_q
                    q_base = q_row * stride_q_seq + (kv_h * arith.constant(query_group_size, type=T.i32) + local_qhead_idx_for_q) * stride_q_head
                    offset1 = lane16id // arith.constant(4, type=T.i32)
                    lds_q_base = (offset1 * arith.constant(2048, type=T.i32)
                                  + lane4id * arith.constant(512, type=T.i32)
                                  + local_qhead_idx * arith.constant(32, type=T.i32))
                    q_w0, q_w1, q_w2, q_w3 = _load_q_words(q_base)
                    v01 = vector.from_elements(T.vec(2, T.i32), [q_w0, q_w1])
                    v23 = vector.from_elements(T.vec(2, T.i32), [q_w2, q_w3])
                    lds_q_i32 = lds_q_base // arith.constant(4, type=T.i32)
                    vector.store(v01, logits_lds_i32, [arith.index_cast(T.index, lds_q_i32)])
                    vector.store(v23, logits_lds_i32,
                        [arith.index_cast(T.index, lds_q_i32 + arith.constant(2, type=T.i32))])

                    q_frags = []
                    gpu.barrier()
                    for qkhe in range_constexpr(QKHELOOP):
                        for qkr in range_constexpr(2):
                            lds_rd_byte = (arith.constant(qkhe * 2048, type=T.i32)
                                           + rowid * arith.constant(512, type=T.i32)
                                           + lane16id * arith.constant(32, type=T.i32)
                                           + arith.constant(qkr * 8, type=T.i32))
                            lds_rd_base = lds_rd_byte // arith.constant(8, type=T.i32)
                            q_v1 = vector.load_op(T.vec(1, T.i64), logits_lds_i64,
                                [arith.index_cast(T.index, lds_rd_base)])
                            q_frags.append(vector.extract(q_v1, static_position=[0]))
                    SmemPtr._view_cache = None

                    # ── K init: load this CTA's 256-token block split for the first block ──
                    first_k_base = (first_phys_block * c_kb + _k_head_off) // c_four
                    k_flat = _load_k_flat_ps(first_k_base, tile_token_offset)

                    init_state = _pack_state(
                        NEG_INF, ZERO_F,
                        arith.constant_vector(0.0, T.f32x4),
                        arith.constant_vector(0.0, T.f32x4),
                        first_phys_block,
                        k_flat,
                    )

                    for ib, state in range(_loop_start_g, _loop_stop_g, _loop_step_g,
                                           init=init_state):
                        running_max, running_sum, out0, out1, phys_block, k_flat = _unpack_state(state)
                        block_idx = arith.index_cast(T.i32, ib)

                        # Prefetch next iteration's phys_block (current one comes from loop-carry)
                        next_idx_raw = block_idx + c_one
                        next_idx_clamped = arith.select(
                            next_idx_raw < num_blocks_in_work, next_idx_raw, last_block_idx_val)
                        next_phys_block = buffer_ops.buffer_load(kpi_rsrc,
                            kv_start + next_idx_clamped, vec_width=1, dtype=T.i32)

                        k_ops = _unflatten_k(k_flat)

                        running_max, running_sum, out0, out1, k_next_flat = _process_block_split(
                            phys_block,
                            block_idx,
                            running_max,
                            running_sum,
                            out0,
                            out1,
                            tile_token_offset,
                            k_ops,
                            next_phys_block=next_phys_block,
                        )

                        results = yield _pack_state(running_max, running_sum, out0, out1, next_phys_block, k_next_flat)

                    running_max, running_sum, out0, out1, _, _ = _unpack_state(results)
                    SmemPtr._view_cache = None

                    # ── Normalize output ──
                    safe_sum = arith.select(running_sum > ZERO_F, running_sum,
                                            arith.constant(1.0, type=T.f32))
                    # Use rocdl.rcp instead of arith.divf to avoid the LLVM
                    # div_scale/div_fmas/div_fixup chain (~6 VGPRs of temporaries).
                    # v_rcp_f32 has <1 ULP error, sufficient for softmax normalization.
                    inv_sum = rocdl.rcp(T.f32, safe_sum)
                    out0_norm = out0 * vector.broadcast(T.f32x4, inv_sum)
                    out1_norm = out1 * vector.broadcast(T.f32x4, inv_sum)
                    outelems_norm = [out0_norm, out1_norm]

                    for vhe in range_constexpr(VHELOOP):
                        hs_base = (arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                                   + warp_id * arith.constant(MFMA_N, type=T.i32)
                                   + rowid * arith.constant(4, type=T.i32))
                        # qhi_pos: mtp_g-based head position within kv_head group
                        qhead = kv_h * arith.constant(query_group_size, type=T.i32) + qhi_pos
                        _po_row = _po_row_base + qi_val
                        po_off = (_po_row * stride_po_ql
                                  + qhead * arith.constant(HEAD_SIZE, type=T.i32)
                                  + hs_base)

                        # pa_reduce_v1 expects normalized partial output from every block split.
                        buffer_ops.buffer_store(outelems_norm[vhe], po_rsrc,
                            po_off * arith.constant(4, type=T.i32), offset_is_bytes=True)

                    # ── LSE ──
                    safe_sum_lse = arith.select(running_sum > ZERO_F, running_sum,
                                                arith.constant(1.0, type=T.f32))
                    from flydsl._mlir.dialects import math as _mlir_math
                    log_sum = _mlir_math.log(safe_sum_lse, fastmath=arith.FastMathFlags.fast)
                    lse_val = running_max + log_sum
                    qhead_lse = kv_h * arith.constant(query_group_size, type=T.i32) + qhi_pos
                    _po_row_lse = _po_row_base + qi_val
                    pl_off = _po_row_lse * stride_pl_ql + qhead_lse
                    lse_as_i32 = arith.bitcast(T.i32, lse_val)
                    buffer_ops.buffer_store(lse_as_i32, pl_rsrc,
                        pl_off * arith.constant(4, type=T.i32), offset_is_bytes=True)
            with _if_else(_if_valid):
                # ── Early skip path: write zero partial_out and -inf LSE ──
                _neg_inf_i32 = arith.bitcast(T.i32, NEG_INF)
                _zero_vec = arith.constant_vector(0.0, T.f32x4)
                for _mtp_g in range_constexpr(_mtp_groups):
                    _g_off = _mtp_g * 16
                    # Recompute qi_val, qhi_pos (same logic as the if-branch)
                    _lane_pair_raw = lane16id + arith.constant(_g_off, type=T.i32)
                    _c_total_pairs = arith.constant(_total_pairs, type=T.i32)
                    _c_pair_max = arith.constant(_total_pairs - 1, type=T.i32)
                    _c_ql_m1 = arith.constant(query_length - 1, type=T.i32)
                    _lane_pair = arith.select(_lane_pair_raw < _c_total_pairs, _lane_pair_raw, _c_pair_max)
                    _qi_raw = _lane_pair // arith.constant(query_group_size, type=T.i32)
                    qi_val = arith.select(_qi_raw < _c_ql_m1, _qi_raw, _c_ql_m1)
                    qhi_pos = _lane_pair % arith.constant(query_group_size, type=T.i32)

                    # Write zero partial_out for each vhe
                    for vhe in range_constexpr(VHELOOP):
                        hs_base = (arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                                   + warp_id * arith.constant(MFMA_N, type=T.i32)
                                   + rowid * arith.constant(4, type=T.i32))
                        qhead = kv_h * arith.constant(query_group_size, type=T.i32) + qhi_pos
                        _po_row = _po_row_base + qi_val
                        po_off = (_po_row * stride_po_ql
                                  + qhead * arith.constant(HEAD_SIZE, type=T.i32)
                                  + hs_base)
                        buffer_ops.buffer_store(_zero_vec, po_rsrc,
                            po_off * arith.constant(4, type=T.i32), offset_is_bytes=True)

                    # Write -inf LSE
                    qhead_lse = kv_h * arith.constant(query_group_size, type=T.i32) + qhi_pos
                    _po_row_lse = _po_row_base + qi_val
                    pl_off = _po_row_lse * stride_pl_ql + qhead_lse
                    buffer_ops.buffer_store(_neg_inf_i32, pl_rsrc,
                        pl_off * arith.constant(4, type=T.i32), offset_is_bytes=True)

    # ── @flyc.jit launch wrapper ─────────────────────────────────────
    @flyc.jit
    def launch_pa_decode_ps(out, po, pl, q, kc, vc, cl,
                            qs, ks, vs,
                            work_indptr, work_info, kv_page_indices, kv_indptr,
                            s_q_seq, s_q_head,
                            s_k_block, s_k_head,
                            s_v_block, s_v_head,
                            s_out_seq, s_out_head,
                            s_po_partial, s_pl_partial,
                            s_ks_block, s_ks_head,
                            s_po_ql, s_pl_ql,
                            num_sm,
                            stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        pa_decode_ps_kernel(
            out, po, pl, q, kc, vc, cl, qs, ks, vs,
            work_indptr, work_info, kv_page_indices, kv_indptr,
            s_q_seq, s_q_head, s_k_block, s_k_head,
            s_v_block, s_v_head,
            s_out_seq, s_out_head,
            s_po_partial, s_pl_partial,
            s_ks_block, s_ks_head,
            s_po_ql, s_pl_ql,
        ).launch(
            grid=(num_sm, 1, TILES_PER_BLOCK),
            block=(BLOCK_THREADS, 1, 1), stream=stream)

    return {
        'launch': launch_pa_decode_ps,
        'kernel': pa_decode_ps_kernel,
        'allocator': allocator,
    }


# =====================================================================
# Launch API — Persistent Scheduling mode
# =====================================================================

def get_pa_metadata(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_indptr: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
):
    """Compute PA metadata (worklist, reduce maps) via get_pa_metadata_v1.

    Then expand each 1024-token work tile into 4 block-split partials so the PS
    kernel can launch with `grid=(num_sm, 1, 4)` and still reuse `pa_reduce_v1`.

    Returns a dict with: work_indptr, work_info_flat, reduce_indptr,
    reduce_final_map, reduce_partial_map, num_sm, partial_output,
    partial_lse, stride_po_partial, stride_pl_partial.
    """
    import aiter

    dev = query.device
    batch_size = context_lengths.shape[0]
    query_length = query.shape[0] // batch_size
    head_size = query.shape[-1]

    props = torch.cuda.get_device_properties(dev)
    num_sm = props.multi_processor_count

    seqlens_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=dev) * query_length

    block_size = key_cache.shape[-2] if len(key_cache.shape) == 5 else key_cache.shape[-2]

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_pa_metadata_info_v1(batch_size, num_kv_heads)

    work_metadata_ptrs = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=dev)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=dev)
    work_info = torch.empty(work_info_set_size, dtype=work_info_set_type, device=dev)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=dev)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=dev)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device=dev)

    aiter.get_pa_metadata_v1(
        seqlens_qo_indptr, kv_indptr, context_lengths,
        num_query_heads // num_kv_heads, num_kv_heads, True,
        work_metadata_ptrs, work_indptr, work_info,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        kv_granularity=max(block_size, 16), block_size=block_size,
        max_seqlen_qo=query_length, uni_seqlen_qo=query_length,
        fast_mode=True, max_split_per_batch=-1,
    )

    work_info, reduce_indptr, reduce_final_map, reduce_partial_map = (
        _expand_pa_metadata_for_block_splits(
            work_indptr, work_info, query_length, block_split_factor=TILES_PER_BLOCK
        )
    )
    work_info_flat = work_info.reshape(-1).contiguous()

    num_partials = reduce_partial_map.size(0)
    max_qlen = query_length
    partial_output = torch.empty(
        ((num_partials + 1) * max_qlen, 1, num_query_heads, head_size),
        dtype=torch.float32, device=dev)
    partial_lse = torch.empty(
        ((num_partials + 1) * max_qlen, 1, num_query_heads, 1),
        dtype=torch.float32, device=dev)

    stride_po_partial = query_length * num_query_heads * head_size
    stride_pl_partial = query_length * num_query_heads
    stride_po_ql = num_query_heads * head_size
    stride_pl_ql = num_query_heads

    return {
        'work_indptr': work_indptr,
        'work_info_flat': work_info_flat,
        'reduce_indptr': reduce_indptr,
        'reduce_final_map': reduce_final_map,
        'reduce_partial_map': reduce_partial_map,
        'num_sm': num_sm,
        'partial_output': partial_output,
        'partial_lse': partial_lse,
        'stride_po_partial': stride_po_partial,
        'stride_pl_partial': stride_pl_partial,
        'stride_po_ql': stride_po_ql,
        'stride_pl_ql': stride_pl_ql,
        'query_length': query_length,
    }



def _sw_reduce_partitions(output, exp_sums, max_logits, temporary_output,
                           context_lengths, query_length, query_group_size,
                           max_context_partition_num, sliding_window, context_partition_size):
    """Reduce partitioned attention outputs into final output.

    Implements online softmax reduction across context partitions:
    output = sum(exp(ml_i - ml_max) * tmp_out_i) / sum(exp(ml_i - ml_max) * es_i)
    """
    batch_size = context_lengths.shape[0]
    num_kv_heads = exp_sums.shape[1]
    head_size = temporary_output.shape[-1]
    eqgs = query_length * query_group_size

    # Process on GPU via PyTorch ops
    # max_logits: [batch, kv_heads, parts, eqgs]
    # exp_sums:   [batch, kv_heads, parts, eqgs]
    # tmp_out:    [batch, kv_heads, parts, eqgs, head_size]

    # Global max across partitions
    global_max = max_logits.max(dim=2, keepdim=True).values  # [B, H, 1, E]

    # Rescale factors: exp(ml - global_max)
    rescale = torch.exp(max_logits - global_max)  # [B, H, P, E]
    rescale = rescale.nan_to_num(0.0)  # handle -inf - (-inf) = NaN

    # Weight = exp_sums * rescale (total probability mass per partition)
    weights = exp_sums * rescale  # [B, H, P, E]

    # Total weight across partitions
    total_weight = weights.sum(dim=2, keepdim=True)  # [B, H, 1, E]
    total_weight = total_weight.clamp(min=1e-30)

    # Weighted sum of normalized outputs: sum(weight_p * tmp_out_p)
    weighted_out = (temporary_output.float() * weights.unsqueeze(-1)).sum(dim=2)  # [B, H, E, D]

    # Final normalized output
    result = weighted_out / total_weight.squeeze(2).unsqueeze(-1)  # [B, H, E, D]

    # Reshape to output format: [batch*ql, num_q_heads, head_size]
    # result is [B, num_kv_heads, ql*qgs, head_size]
    # output is [B*ql, num_kv_heads*qgs, head_size]
    result_reshaped = result.reshape(batch_size, num_kv_heads, query_length, query_group_size, head_size)
    result_reshaped = result_reshaped.permute(0, 2, 1, 3, 4).reshape(
        batch_size * query_length, num_kv_heads * query_group_size, head_size)
    output.copy_(result_reshaped.to(output.dtype))


def _is_unit_query_scale(query_scale) -> bool:
    if query_scale is None:
        return True
    if isinstance(query_scale, torch.Tensor):
        return query_scale.numel() == 1 and math.isclose(float(query_scale.item()), 1.0)
    return math.isclose(float(query_scale), 1.0)


def _get_query_input_dtype(query: torch.Tensor) -> str:
    if query.dtype in _PACKED_FP8_QUERY_DTYPES:
        return "packed_fp8"
    if query.dtype == torch.bfloat16:
        return "bf16"
    if query.dtype == torch.float16:
        return "f16"
    raise ValueError(
        f"Unsupported query dtype for pa_decode_ps_launch: {query.dtype}. "
        "Expected packed FP8/uint8, bf16, or f16."
    )


def pa_decode_ps_launch(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_page_indices: torch.Tensor,    # [total_pages] int32
    kv_indptr: torch.Tensor,          # [num_seqs + 1] int32
    softmax_scale: float,
    query_scale: torch.Tensor = None,
    key_scale: torch.Tensor = None,
    value_scale: torch.Tensor = None,
    *,
    sliding_window: int = 0,
    metadata: dict = None,
    block_tables: torch.Tensor = None,  # [num_seqs, max_blocks_per_seq] i32
    max_context_partition_num: int = 0,
    exp_sums: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    temporary_output: torch.Tensor = None,
    stream=None,
) -> str:
    """Launch PA decode with persistent scheduling.

    Args:
        metadata: Pre-computed metadata dict from get_pa_metadata().
                  If None, calls get_pa_metadata() internally.
    """
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    trans_v = len(value_cache.shape) == 5
    query_input_dtype = _get_query_input_dtype(query)

    dev = query.device
    if query_input_dtype != "packed_fp8" and not _is_unit_query_scale(query_scale):
        raise ValueError(
            "Non-packed query inputs require query_scale == 1.0. "
            "For scaled query quantization, pass a pre-quantized packed FP8 query."
        )
    if not isinstance(query_scale, torch.Tensor):
        query_scale = torch.tensor([float(query_scale or 1.0)], device=dev, dtype=torch.float32)
    if not isinstance(key_scale, torch.Tensor):
        key_scale = torch.tensor([float(key_scale or 1.0)], device=dev, dtype=torch.float32)
    if not isinstance(value_scale, torch.Tensor):
        value_scale = torch.tensor([float(value_scale or 1.0)], device=dev, dtype=torch.float32)

    # Detect per-token vs per-tensor quantization from scale tensor dimensionality
    per_token_kv = key_scale.ndim > 1  # per-tensor: shape [1]; per-token: shape [blocks, heads, block_size, 1]

    if metadata is None:
        metadata = get_pa_metadata(
            query, key_cache, context_lengths, kv_indptr,
            num_query_heads, num_kv_heads)

    work_indptr = metadata['work_indptr']
    work_info_flat = metadata['work_info_flat']
    partial_output = metadata['partial_output']
    partial_lse = metadata['partial_lse']
    stride_po_partial = metadata['stride_po_partial']
    stride_pl_partial = metadata['stride_pl_partial']
    num_sm = metadata['num_sm']

    query_length = query.shape[0] // context_lengths.shape[0]
    query_group_size = num_query_heads // num_kv_heads

    # Strides for key_scale/value_scale
    if per_token_kv:
        stride_ks_block = key_scale.stride(0)
        stride_ks_head = key_scale.stride(1)
    else:
        stride_ks_block = 0
        stride_ks_head = 0

    s = stream or torch.cuda.current_stream()

    if sliding_window > 0:
        # Launch one CTA per 256-token tile inside each 1024-token physical block:
        # grid = (batch, kv_heads, max_context_partition_num * 4).
        batch_size = context_lengths.shape[0]
        head_size = query.shape[-1]
        eqgs = query_length * query_group_size
        physical_partition_size = KV_BLOCK_SIZE
        context_partition_size = KV_COMPUTE_BLOCK

        if max_context_partition_num == 0:
            max_context_partition_num = (
                (sliding_window + physical_partition_size - 1) // physical_partition_size
            ) + 1
        total_context_partition_num = max_context_partition_num * TILES_PER_BLOCK

        if exp_sums is None:
            exp_sums = torch.empty(batch_size, num_kv_heads, total_context_partition_num, eqgs,
                                    device=dev, dtype=torch.float32)
        if max_logits is None:
            max_logits = torch.full((batch_size, num_kv_heads, total_context_partition_num, eqgs),
                                     float('-inf'), device=dev, dtype=torch.float32)
        if temporary_output is None:
            temporary_output = torch.zeros(batch_size, num_kv_heads, total_context_partition_num,
                                            eqgs, head_size, device=dev, dtype=torch.bfloat16)

        compiled_sw = compile_pa_decode_ps_sw(
            sliding_window=sliding_window,
            softmax_scale=softmax_scale, trans_v=trans_v, query_group_size=query_group_size,
            per_token_kv=per_token_kv, query_length=query_length,
            query_input_dtype=query_input_dtype)

        compiled_sw['launch'](
            exp_sums, max_logits, temporary_output,
            query, key_cache, value_cache,
            block_tables, context_lengths,
            query_scale, key_scale, value_scale,
            query.stride(0), query.stride(1),
            key_cache.stride(0), key_cache.stride(1),
            value_cache.stride(0), value_cache.stride(1),
            exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
            temporary_output.stride(0), temporary_output.stride(1),
            temporary_output.stride(2), temporary_output.stride(3),
            block_tables.stride(0),
            stride_ks_block, stride_ks_head,
            batch_size, num_kv_heads, max_context_partition_num, s)

        # Fix NaN from fastmath in fully-masked partitions
        temporary_output.nan_to_num_(0.0)
        exp_sums.nan_to_num_(0.0)
        max_logits.nan_to_num_(nan=float('-inf'))

        # Reduce: use Gluon reduce kernel for merging partitions
        from aiter.ops.triton.gluon.pa_decode_gluon import (
            _paged_attention_decode_v2_reduce_kernel_wrapper,
        )
        head_size = query.shape[-1]
        output_5d = output.reshape(
            batch_size, query_length, num_kv_heads, query_group_size, head_size)
        reduce_grid = (batch_size, num_kv_heads, 1)
        _paged_attention_decode_v2_reduce_kernel_wrapper(
            reduce_grid,
            output_5d,
            exp_sums,
            max_logits,
            temporary_output,
            context_lengths,
            None,  # sinks
            output_5d.stride(0),
            output_5d.stride(1),
            output_5d.stride(2),
            output_5d.stride(3),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            query_seq_len=query_length,
            query_group_size=query_group_size,
            head_size=head_size,
            CONTEXT_PARTITION_SIZE=context_partition_size,
            PS=True,
            context_partition_num=total_context_partition_num,
        )
        return "ps_sw_partitioned"
    else:
        compiled = compile_pa_decode_ps(
            softmax_scale=softmax_scale, trans_v=trans_v, query_group_size=query_group_size,
            per_token_kv=per_token_kv, query_length=query_length,
            query_input_dtype=query_input_dtype)

    stride_po_ql = metadata.get('stride_po_ql', num_query_heads * query.shape[-1])
    stride_pl_ql = metadata.get('stride_pl_ql', num_query_heads)

    compiled['launch'](
        output, partial_output, partial_lse,
        query, key_cache, value_cache,
        context_lengths, query_scale, key_scale, value_scale,
        work_indptr, work_info_flat, kv_page_indices, kv_indptr,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1),
        value_cache.stride(0), value_cache.stride(1),
        output.stride(0), output.stride(1),
        stride_po_partial, stride_pl_partial,
        stride_ks_block, stride_ks_head,
        stride_po_ql, stride_pl_ql,
        num_sm, s)

    # Fix NaN from fastmath in fully-masked partitions (sliding window):
    # exp2(-inf - (-inf)) produces NaN with fastmath ninf flag.
    # Also clamp -inf LSE (pa_reduce_v1 does not handle -inf correctly).
    output.nan_to_num_(0.0)
    partial_output.nan_to_num_(0.0)
    partial_lse.clamp_(min=-1e30)

    from aiter.ops.attention import pa_reduce_v1
    pa_reduce_v1(
        partial_output[query_length:],
        partial_lse[query_length:],
        metadata['reduce_indptr'],
        metadata['reduce_final_map'],
        metadata['reduce_partial_map'],
        query_length,  # max_qlen
        output,
        None,
    )

    return "ps_split_reduce"


# =====================================================================
# =====================================================================
# compile_pa_decode_ps_sw — Sliding Window kernel with split grid-z
# Grid = (batch_size, num_kv_heads, max_context_partition_num * 4)
# Each block handles one (batch, kv_head, physical_block, 256-token sub-tile).
# Each physical KV block contributes 4 partial slots to the reduce stage.
# Uses block_tables for physical block lookup instead of kv_page_indices.
# Output: exp_sums, max_logits, temporary_output -> reduced by a separate kernel.
# =====================================================================
@functools.lru_cache(maxsize=256)
def compile_pa_decode_ps_sw(
    sliding_window: int,   # required > 0 -- baked as compile-time constant
    softmax_scale=None,
    trans_v=False,
    query_group_size=QUERY_GROUP_SIZE,
    per_token_kv=False,
    query_length: int = 1,
    query_input_dtype: str = "packed_fp8",
):
    """Compile a Gluon-style partitioned PA decode kernel for sliding window.

    Grid = (batch_size, num_kv_heads, max_context_partition_num * 4).
    Each GPU block processes one 256-token sub-partition inside a 1024-token KV block.
    sliding_window is a compile-time constant.
    """
    assert sliding_window > 0, "Use compile_pa_decode_ps for sliding_window=0"
    arch = get_hip_arch()
    query_packed_fp8 = query_input_dtype == "packed_fp8"
    query_load_is_bf16 = query_input_dtype == "bf16"
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)
    _softmax_scale = float(softmax_scale)
    _bs = KV_BLOCK_SIZE  # 1024

    CompilationContext._compile_hints.data = {}

    LDS_VMAX_BYTES = NUM_WARPS * MFMA_N * 4 if per_token_kv else 0
    LDS_SOFTMAX_TOTAL = LDS_SOFTMAX_BYTES + LDS_VMAX_BYTES
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_ps_sw_smem")
    logits_off = 0
    allocator.ptr = LDS_LOGITS_BYTES
    softmax_off = LDS_LOGITS_BYTES
    allocator.ptr += LDS_SOFTMAX_TOTAL

    @flyc.kernel
    def pa_decode_ps_sw_kernel(
        exp_sums_ptr: fx.Tensor,      # [batch, kv_heads, max_parts, eqgs] f32
        max_logits_ptr: fx.Tensor,    # [batch, kv_heads, max_parts, eqgs] f32
        tmp_out_ptr: fx.Tensor,       # [batch, kv_heads, max_parts, eqgs, head_size] bf16
        query_ptr: fx.Tensor,
        key_cache_ptr: fx.Tensor,
        value_cache_ptr: fx.Tensor,
        block_tables_ptr: fx.Tensor,  # [batch, max_blocks_per_seq] i32
        context_lengths_ptr: fx.Tensor,
        query_scale_ptr: fx.Tensor,
        key_scale_ptr: fx.Tensor,
        value_scale_ptr: fx.Tensor,
        stride_q_seq: Int32,
        stride_q_head: Int32,
        stride_k_block: Int32,
        stride_k_head: Int32,
        stride_v_block: Int32,
        stride_v_head: Int32,
        stride_es_seq: Int32,
        stride_es_head: Int32,
        stride_es_part: Int32,
        stride_to_seq: Int32,
        stride_to_head: Int32,
        stride_to_part: Int32,
        stride_to_group: Int32,
        stride_bt_seq: Int32,
        stride_ks_block: Int32,
        stride_ks_head: Int32,
    ):
        tid = gpu.thread_idx.x
        batch_idx = gpu.block_idx.x
        kv_h = gpu.block_idx.y
        partition_idx = gpu.block_idx.z

        lane16id = tid & arith.constant(15, type=T.i32)
        lane4id = tid & arith.constant(3, type=T.i32)
        rowid = (tid >> arith.constant(4, type=T.i32)) & arith.constant(3, type=T.i32)
        warp_id = tid >> arith.constant(6, type=T.i32)

        q_rsrc = buffer_ops.create_buffer_resource(query_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(key_cache_ptr, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(value_cache_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)
        to_rsrc = buffer_ops.create_buffer_resource(tmp_out_ptr, max_size=True)
        cl_rsrc = buffer_ops.create_buffer_resource(context_lengths_ptr, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(block_tables_ptr, max_size=True)

        qs_rsrc = buffer_ops.create_buffer_resource(query_scale_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
        vs_rsrc = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
        q_scale_val = buffer_ops.buffer_load(qs_rsrc, arith.constant(0, type=T.i32), vec_width=1)
        k_scale_val = buffer_ops.buffer_load(ks_rsrc, arith.constant(0, type=T.i32), vec_width=1)
        v_scale_val = buffer_ops.buffer_load(vs_rsrc, arith.constant(0, type=T.i32), vec_width=1)

        smem_base = allocator.get_base()
        logits_lds_i32 = SmemPtr(smem_base, logits_off, T.i32, shape=(LDS_LOGITS_BYTES // 4,)).get()
        softmax_lds_f32 = SmemPtr(smem_base, softmax_off, T.f32, shape=(LDS_SOFTMAX_TOTAL // 4,)).get()
        logits_lds_i64 = SmemPtr(smem_base, logits_off, T.i64, shape=(LDS_LOGITS_BYTES // 8,)).get()

        c_kb = stride_k_block
        c_kh = stride_k_head
        c_vb = stride_v_block
        c_vh = stride_v_head

        _softmax_q_scale = arith.constant(_softmax_scale, type=T.f32) * q_scale_val
        _scale = _softmax_q_scale * k_scale_val
        c_w = arith.constant(WARP_SIZE, type=T.i32)
        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = arith.constant(0.0, type=T.f32)
        c_cps = arith.constant(KV_COMPUTE_BLOCK, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_bs = arith.constant(_bs, type=T.i32)
        c_four = arith.constant(4, type=T.i32)
        c_tpb = arith.constant(TILES_PER_BLOCK, type=T.i32)

        local_qhead_idx = warp_id * arith.constant(4, type=T.i32) + rowid

        # ── Context length and partition mapping ──
        context_len = buffer_ops.buffer_load(cl_rsrc, batch_idx, vec_width=1, dtype=T.i32)
        sequence_partition_idx = partition_idx // c_tpb
        block_split_idx = partition_idx % c_tpb
        tile_token_offset = block_split_idx * c_cps
        _c_sw = arith.constant(sliding_window, type=T.i32)
        seq_start_global = context_len - _c_sw
        partition_offset = seq_start_global // c_bs
        partition_offset = arith.select(partition_offset > arith.constant(0, type=T.i32),
                                         partition_offset, arith.constant(0, type=T.i32))
        seq_partition_idx_raw = partition_offset + sequence_partition_idx
        # Check if this partition is valid (within context range)
        num_blocks_for_seq = (context_len + c_bs - c_one) // c_bs
        _is_valid = seq_partition_idx_raw < num_blocks_for_seq
        # Clamp for safe memory access (invalid partitions will be skipped below)
        seq_partition_idx = arith.select(_is_valid, seq_partition_idx_raw,
                                          arith.constant(0, type=T.i32))
        kv_seq_start = seq_partition_idx * c_bs + tile_token_offset
        # For invalid partitions, set context_len to 0 so all tokens get masked
        context_len = arith.select(_is_valid, context_len, arith.constant(0, type=T.i32))

        # Look up physical block (clamped index is always safe)
        bt_off = batch_idx * stride_bt_seq + seq_partition_idx
        phys_block = buffer_ops.buffer_load(bt_rsrc, bt_off, vec_width=1, dtype=T.i32)

        _k_head_off = kv_h * c_kh
        _v_head_off = kv_h * c_vh

        # ── Pre-computed thread-invariant offsets ──
        _k_tok_thread_base = warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32) + lane16id
        _c_tok_stride_dw = arith.constant(FP8_ELEMS_16B // 4, type=T.i32)
        _c_he_stride_dw = arith.constant(_bs * FP8_ELEMS_16B // 4, type=T.i32)
        _k_he_off_dw = [rowid * _c_he_stride_dw + arith.constant(qkhe * 4, type=T.i32) * _c_he_stride_dw
                        for qkhe in range(QKHELOOP)]
        _vhead_elems = [arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                        + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id
                        for vhe in range(VHELOOP)]
        _v_tok_thread_off = [arith.constant(vt * TOKENS_PER_WARP, type=T.i32)
                             + rowid * arith.constant(MFMA_N, type=T.i32)
                             for vt in range(VTLOOP)]
        if trans_v:
            _vhead_elem_dw = [_vhead_elems[vhe] * arith.constant(FP8_ELEMS_16B // 4, type=T.i32)
                              for vhe in range(VHELOOP)]
        else:
            _vhead_elem_dw = [_vhead_elems[vhe] * arith.constant(_bs // 4, type=T.i32)
                              for vhe in range(VHELOOP)]
        _kv_tok_thread_base = warp_id * arith.constant(TOKENS_PER_WARP, type=T.i32) + rowid * c_four
        _rowid_8x8 = rowid // arith.constant(2, type=T.i32)
        _offset_in_slot = rowid % arith.constant(2, type=T.i32)
        _prob_wr_thread_base = (warp_id * arith.constant(4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                + lane16id * arith.constant(PROB_ROW_STRIDE_BYTES, type=T.i32)
                                + _rowid_8x8 * arith.constant(8, type=T.i32)
                                + _offset_in_slot * c_four)
        _pv_prob_read_base = (rowid * arith.constant(MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                              + lane16id * arith.constant(PROB_ROW_STRIDE_BYTES, type=T.i32))
        _sm_max_off = arith.index_cast(T.index,
            warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
        _sm_sum_off = arith.index_cast(T.index,
            arith.constant(NUM_WARPS * MFMA_N, type=T.i32) + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
        _sm_rd_max_offs = [arith.index_cast(T.index,
            arith.constant(w * MFMA_N, type=T.i32) + lane16id) for w in range(NUM_WARPS)]
        _sm_rd_sum_offs = [arith.index_cast(T.index,
            arith.constant(NUM_WARPS * MFMA_N + w * MFMA_N, type=T.i32) + lane16id) for w in range(NUM_WARPS)]
        if per_token_kv:
            _sm_vmax_wr_off = arith.index_cast(T.index,
                arith.constant(2 * NUM_WARPS * MFMA_N, type=T.i32)
                + warp_id * arith.constant(MFMA_N, type=T.i32) + lane16id)
            _sm_vmax_rd_offs = [arith.index_cast(T.index,
                arith.constant(2 * NUM_WARPS * MFMA_N + w * MFMA_N, type=T.i32) + lane16id)
                for w in range(NUM_WARPS)]

        # ════════════════════════════════════════════════════════
        # Helper functions (K load, V load, QK+softmax, PV MFMA)
        # ════════════════════════════════════════════════════════
        def _load_k_flat_ps(k_block_base_dw, tile_token_offset_i32):
            k_flat = []
            tile_tok_base = tile_token_offset_i32 + _k_tok_thread_base
            for td in range_constexpr(TLOOP):
                kbo = tile_tok_base + arith.constant(td * MFMA_N, type=T.i32)
                kbo_dw = k_block_base_dw + kbo * _c_tok_stride_dw
                for qkhe in range_constexpr(QKHELOOP):
                    ka_dw = kbo_dw + _k_he_off_dw[qkhe]
                    k4 = buffer_ops.buffer_load(k_rsrc, ka_dw, vec_width=4, dtype=T.i32)
                    k_flat.append(_pack_i32_pair_to_i64(
                        vector.extract(k4, static_position=[0]),
                        vector.extract(k4, static_position=[1])))
                    k_flat.append(_pack_i32_pair_to_i64(
                        vector.extract(k4, static_position=[2]),
                        vector.extract(k4, static_position=[3])))
            return k_flat

        def _load_q_words(q_base_elem):
            if query_packed_fp8:
                q_off = q_base_elem + lane16id * arith.constant(FP8_ELEMS_16B, type=T.i32)
                q_vec = buffer_ops.buffer_load(
                    q_rsrc,
                    q_off // arith.constant(4, type=T.i32),
                    vec_width=4,
                    dtype=T.i32,
                )
                return [
                    vector.extract(q_vec, static_position=[0], dynamic_position=[]),
                    vector.extract(q_vec, static_position=[1], dynamic_position=[]),
                    vector.extract(q_vec, static_position=[2], dynamic_position=[]),
                    vector.extract(q_vec, static_position=[3], dynamic_position=[]),
                ]

            q_elem = q_base_elem + lane16id * arith.constant(FP8_ELEMS_16B, type=T.i32)
            q_words = []
            for qwi in range_constexpr(4):
                q_src = buffer_ops.buffer_load(
                    q_rsrc,
                    q_elem + arith.constant(qwi * 4, type=T.i32),
                    vec_width=4,
                    dtype=T.bf16 if query_load_is_bf16 else T.f16,
                )
                q_f32 = _mlir_arith.ExtFOp(T.f32x4, q_src).result
                p0 = vector.extract(q_f32, static_position=[0], dynamic_position=[])
                p1 = vector.extract(q_f32, static_position=[1], dynamic_position=[])
                p2 = vector.extract(q_f32, static_position=[2], dynamic_position=[])
                p3 = vector.extract(q_f32, static_position=[3], dynamic_position=[])
                lo = rocdl.cvt_pk_fp8_f32(
                    T.i32, p0, p1, arith.constant(0, type=T.i32), False
                )
                pk = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)
                q_words.append(pk)
            return q_words

        def _unflatten_k(k_flat):
            return [[k_flat[td * (QKHELOOP * 2) + j]
                      for j in range(QKHELOOP * 2)]
                     for td in range(TLOOP)]

        def _qk_and_intra_softmax(k_ops, partition_start, v_block_base_dw, tile_token_offset_i32, phys_block_arg=None):
            va_dws = []
            for vt in range_constexpr(VTLOOP):
                vhe_data = []
                for vhe in range_constexpr(VHELOOP):
                    v_token_in_block = tile_token_offset_i32 + _v_tok_thread_off[vt]
                    if trans_v:
                        vt_group = v_token_in_block // arith.constant(FP8_ELEMS_16B, type=T.i32)
                        va_dw = v_block_base_dw + vt_group * arith.constant(HEAD_SIZE * FP8_ELEMS_16B // 4, type=T.i32) + _vhead_elem_dw[vhe]
                    else:
                        va_dw = v_block_base_dw + _vhead_elem_dw[vhe] + v_token_in_block // c_four
                    vhe_data.append(va_dw)
                va_dws.append(vhe_data)
            if per_token_kv:
                scale_block_base = phys_block_arg * stride_ks_block + kv_h * stride_ks_head
                _scale_tok_base_pt = tile_token_offset_i32 + _k_tok_thread_base
                _scale_src_lane_base = rowid * arith.constant(20, type=T.i32)
                k_scale_vecs = []
                v_scale_vecs = []
                for td in range_constexpr(TLOOP):
                    tok_off = _scale_tok_base_pt + arith.constant(td * MFMA_N, type=T.i32)
                    k_scale_lane = buffer_ops.buffer_load(
                        ks_rsrc, scale_block_base + tok_off, vec_width=1, dtype=T.f32
                    )
                    v_scale_lane = buffer_ops.buffer_load(
                        vs_rsrc, scale_block_base + tok_off, vec_width=1, dtype=T.f32
                    )
                    k_scale_i32 = arith.bitcast(T.i32, k_scale_lane)
                    v_scale_i32 = arith.bitcast(T.i32, v_scale_lane)
                    k_scale_vals = []
                    v_scale_vals = []
                    for i in range_constexpr(4):
                        bcast_addr = (_scale_src_lane_base + arith.constant(i, type=T.i32)) * c_four
                        sk_i32 = rocdl.ds_bpermute(
                            T.i32, arith.unwrap(bcast_addr), arith.unwrap(k_scale_i32)
                        )
                        sv_i32 = rocdl.ds_bpermute(
                            T.i32, arith.unwrap(bcast_addr), arith.unwrap(v_scale_i32)
                        )
                        k_scale_vals.append(arith.bitcast(T.f32, sk_i32))
                        v_scale_vals.append(arith.bitcast(T.f32, sv_i32))
                    k_scale_vecs.append(vector.from_elements(T.f32x4, k_scale_vals))
                    v_scale_vecs.append(vector.from_elements(T.f32x4, v_scale_vals))
            else:
                v_scale_vecs = None
            v_results = []
            d_out = []
            for td in range_constexpr(TLOOP):
                vhe_data = []
                acc = arith.constant_vector(0.0, T.f32x4)
                for k_step in range_constexpr(QKHELOOP * 2):
                    if k_step % 2 == 0:
                        v_4xi32 = buffer_ops.buffer_load(v_rsrc, va_dws[td][k_step // 2], vec_width=4, dtype=T.i32)
                        vhe_data.append(v_4xi32)
                    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [k_ops[td][k_step], q_frags[k_step], acc, 0, 0, 0])
                v_results.append(vhe_data)
                if per_token_kv:
                    d_out.append(
                        acc
                        * (
                            k_scale_vecs[td]
                            * vector.broadcast(T.f32x4, _softmax_q_scale)
                        )
                    )
                else:
                    d_out.append(acc * vector.broadcast(T.f32x4, _scale))
            kv_tok_base = partition_start + _kv_tok_thread_base
            qk_max = NEG_INF
            for td in range_constexpr(TLOOP):
                for i in range_constexpr(4):
                    s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                    kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                    s = arith.select(kv_tok < causal_bound, s, NEG_INF)
                    s = arith.select(kv_tok > seq_start, s, NEG_INF)
                    qk_max = qk_max.maximumf(s)
            for sh in [32, 16]:
                qk_max = qk_max.maximumf(qk_max.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
            vector.store(vector.from_elements(T.vec(1, T.f32), [qk_max]), softmax_lds_f32, [_sm_max_off])
            exp_sum = ZERO_F
            for td in range_constexpr(TLOOP):
                for i in range_constexpr(4):
                    s = vector.extract(d_out[td], static_position=[i], dynamic_position=[])
                    diff = s - qk_max
                    p = (diff * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                    kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                    p = arith.select(kv_tok < causal_bound, p, ZERO_F)
                    p = arith.select(kv_tok > seq_start, p, ZERO_F)
                    exp_sum = exp_sum + p
                    d_out[td] = vector.insert(p, d_out[td], static_position=[i], dynamic_position=[])
            for sh in [32, 16]:
                exp_sum = exp_sum + exp_sum.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
            vector.store(vector.from_elements(T.vec(1, T.f32), [exp_sum]), softmax_lds_f32, [_sm_sum_off])
            if per_token_kv:
                v_max_warp = ZERO_F
                for td in range_constexpr(TLOOP):
                    vs = v_scale_vecs[td]
                    for i in range_constexpr(4):
                        kv_tok = kv_tok_base + arith.constant(td * MFMA_N + i, type=T.i32)
                        vs_i = vector.extract(vs, static_position=[i], dynamic_position=[])
                        vs_i = arith.select(kv_tok < causal_bound, vs_i, ZERO_F)
                        vs_i = arith.select(kv_tok > seq_start, vs_i, ZERO_F)
                        vs = vector.insert(vs_i, vs, static_position=[i], dynamic_position=[])
                    v_max_warp = v_max_warp.maximumf(vector.reduction(T.f32, "maxnumf", vs))
                for sh in [32, 16]:
                    v_max_warp = v_max_warp.maximumf(
                        v_max_warp.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
                vector.store(vector.from_elements(T.vec(1, T.f32), [v_max_warp]),
                             softmax_lds_f32, [_sm_vmax_wr_off])
            return d_out, v_results, v_scale_vecs

        def _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, o0, o1, v_scale_vecs=None):
            partition_max = NEG_INF
            partition_sum = ZERO_F
            warp_rescale_factors = []
            for w in range_constexpr(NUM_WARPS):
                w_max = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [_sm_rd_max_offs[w]]), static_position=[0])
                partition_max = partition_max.maximumf(w_max)
                warp_rescale_factors.append(w_max)
            for w in range_constexpr(NUM_WARPS):
                diff_w = warp_rescale_factors[w] - partition_max
                diff_w = arith.select(partition_max > NEG_INF, diff_w, ZERO_F)
                wf = (diff_w * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast)
                w_sum = vector.extract(vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [_sm_rd_sum_offs[w]]), static_position=[0])
                partition_sum = partition_sum + w_sum * wf
                warp_rescale_factors[w] = wf
            my_warp_rescale = warp_rescale_factors[0]
            for w in range_constexpr(1, NUM_WARPS):
                my_warp_rescale = arith.select(
                    warp_id == arith.constant(w, type=T.i32),
                    warp_rescale_factors[w], my_warp_rescale)
            new_rmax = rmax.maximumf(partition_max)
            accum_scale = arith.select(rmax > NEG_INF,
                ((rmax - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast),
                ZERO_F)
            part_to_new = arith.select(partition_max > NEG_INF,
                ((partition_max - new_rmax) * arith.constant(LOG2E, type=T.f32)).exp2(fastmath=arith.FastMathFlags.fast),
                ZERO_F)
            rsum = accum_scale * rsum + partition_sum * part_to_new
            rmax = new_rmax
            o0 = o0 * vector.broadcast(T.f32x4, accum_scale)
            o1 = o1 * vector.broadcast(T.f32x4, accum_scale)
            if per_token_kv and v_scale_vecs is not None:
                v_max_global = ZERO_F
                for w in range_constexpr(NUM_WARPS):
                    w_vmax = vector.extract(
                        vector.load_op(T.vec(1, T.f32), softmax_lds_f32, [_sm_vmax_rd_offs[w]]),
                        static_position=[0])
                    v_max_global = v_max_global.maximumf(w_vmax)
                v_max_safe = v_max_global + arith.constant(1e-8, type=T.f32)
                c_fp8_max = arith.constant(FP8_MAX, type=T.f32)
                norm_factor = c_fp8_max / v_max_safe
                prob_scale = my_warp_rescale
                v_correction = v_max_global / c_fp8_max * part_to_new
                for td in range_constexpr(TLOOP):
                    d_out[td] = d_out[td] * (
                        v_scale_vecs[td]
                        * vector.broadcast(T.f32x4, prob_scale * norm_factor)
                    )
            else:
                prob_scale = my_warp_rescale * part_to_new
                v_correction = v_scale_val
                for td in range_constexpr(TLOOP):
                    d_out[td] = d_out[td] * vector.broadcast(T.f32x4, prob_scale)
            for td in range_constexpr(TLOOP):
                p0 = vector.extract(d_out[td], static_position=[0], dynamic_position=[])
                p1 = vector.extract(d_out[td], static_position=[1], dynamic_position=[])
                p2 = vector.extract(d_out[td], static_position=[2], dynamic_position=[])
                p3 = vector.extract(d_out[td], static_position=[3], dynamic_position=[])
                lo = rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, arith.constant(0, type=T.i32), False)
                pk = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)
                byte_base = _prob_wr_thread_base + arith.constant(td * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                i32_off = byte_base // c_four
                pk_vec = vector.from_elements(T.vec(1, T.i32), [pk])
                vector.store(pk_vec, logits_lds_i32, [arith.index_cast(T.index, i32_off)])
            return rmax, rsum, o0, o1, v_correction

        def _pv_mfma(v_ops, o0, o1, v_correction):
            pv_results = [arith.constant_vector(0.0, T.f32x4) for _ in range_constexpr(VHELOOP)]
            v_i64s = []
            p_i64s = []
            for vhe in range_constexpr(VHELOOP):
                for vt in range_constexpr(VTLOOP):
                    v_4xi32 = v_ops[vt][vhe]
                    for j in range_constexpr(2):
                        v_i64 = _pack_i32_pair_to_i64(
                            vector.extract(v_4xi32, static_position=[j * 2]),
                            vector.extract(v_4xi32, static_position=[j * 2 + 1]))
                        v_i64s.append(v_i64)
                        p_byte = (arith.constant(vt * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES, type=T.i32)
                                  + _pv_prob_read_base
                                  + arith.constant(j * 8, type=T.i32))
                        p_i32_idx = p_byte // c_four
                        pw0 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                            [arith.index_cast(T.index, p_i32_idx)]), static_position=[0])
                        pw1 = vector.extract(vector.load_op(T.vec(1, T.i32), logits_lds_i32,
                            [arith.index_cast(T.index, p_i32_idx + c_one)]), static_position=[0])
                        p_i64 = _pack_i32_pair_to_i64(pw0, pw1)
                        p_i64s.append(p_i64)
            for vhe in range_constexpr(VHELOOP):
                tmp_out = arith.constant_vector(0.0, T.f32x4)
                for vt in range_constexpr(VTLOOP):
                    for j in range_constexpr(2):
                        tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [v_i64s[vhe * VTLOOP * 2 + vt * 2 + j], p_i64s[vhe * VTLOOP * 2 + vt * 2 + j], tmp_out, 0, 0, 0])
                        pv_results[vhe] = tmp_out
            o0 = o0 + pv_results[0] * vector.broadcast(T.f32x4, v_correction)
            o1 = o1 + pv_results[1] * vector.broadcast(T.f32x4, v_correction)
            return o0, o1

        def _process_block_split(rmax, rsum, o0, o1, tile_token_offset_i32, k_ops):
            """Process one 256-token tile inside the selected physical block."""
            v_base = (phys_block * c_vb + _v_head_off) // c_four
            d_out_0, v0_ops, vs0 = _qk_and_intra_softmax(
                k_ops, kv_seq_start, v_base, tile_token_offset_i32, phys_block
            )
            gpu.barrier()
            rmax, rsum, o0, o1, vc0 = _cross_warp_softmax_and_prob_pack(d_out_0, rmax, rsum, o0, o1, vs0)
            gpu.barrier()
            o0, o1 = _pv_mfma(v0_ops, o0, o1, vc0)
            return rmax, rsum, o0, o1

        # ── MTP groups ──
        _mtp_groups = math.ceil(query_length * query_group_size / 16)
        _total_pairs = query_length * query_group_size
        for _mtp_g in range_constexpr(_mtp_groups):
            _g_off = _mtp_g * 16
            _lane_pair_raw = lane16id + arith.constant(_g_off, type=T.i32)
            _c_total_pairs = arith.constant(_total_pairs, type=T.i32)
            _c_pair_max = arith.constant(_total_pairs - 1, type=T.i32)
            _c_ql_m1 = arith.constant(query_length - 1, type=T.i32)
            _lane_pair = arith.select(_lane_pair_raw < _c_total_pairs, _lane_pair_raw, _c_pair_max)
            _qi_raw = _lane_pair // arith.constant(query_group_size, type=T.i32)
            qi_val = arith.select(_qi_raw < _c_ql_m1, _qi_raw, _c_ql_m1)
            qhi_pos = _lane_pair % arith.constant(query_group_size, type=T.i32)
            causal_bound = context_len + arith.constant(1 - query_length, type=T.i32) + qi_val
            seq_start = context_len - arith.constant(sliding_window, type=T.i32) + qi_val

            _lqh_pair_raw = local_qhead_idx + arith.constant(_g_off, type=T.i32)
            _lqh_pair = arith.select(_lqh_pair_raw < _c_total_pairs, _lqh_pair_raw, _c_pair_max)
            _lqi_raw = _lqh_pair // arith.constant(query_group_size, type=T.i32)
            qi_for_q = arith.select(_lqi_raw < _c_ql_m1, _lqi_raw, _c_ql_m1)
            local_qhead_idx_for_q = _lqh_pair % arith.constant(query_group_size, type=T.i32)

            if _mtp_g > 0:
                gpu.barrier()
                SmemPtr._view_cache = None
            q_row = batch_idx * arith.constant(query_length, type=T.i32) + qi_for_q
            q_base = q_row * stride_q_seq + (kv_h * arith.constant(query_group_size, type=T.i32) + local_qhead_idx_for_q) * stride_q_head
            offset1 = lane16id // arith.constant(4, type=T.i32)
            lds_q_base = (offset1 * arith.constant(2048, type=T.i32)
                          + lane4id * arith.constant(512, type=T.i32)
                          + local_qhead_idx * arith.constant(32, type=T.i32))
            q_w0, q_w1, q_w2, q_w3 = _load_q_words(q_base)
            v01 = vector.from_elements(T.vec(2, T.i32), [q_w0, q_w1])
            v23 = vector.from_elements(T.vec(2, T.i32), [q_w2, q_w3])
            lds_q_i32 = lds_q_base // arith.constant(4, type=T.i32)
            vector.store(v01, logits_lds_i32, [arith.index_cast(T.index, lds_q_i32)])
            vector.store(v23, logits_lds_i32,
                [arith.index_cast(T.index, lds_q_i32 + arith.constant(2, type=T.i32))])
            q_frags = []
            gpu.barrier()
            for qkhe in range_constexpr(QKHELOOP):
                for qkr in range_constexpr(2):
                    lds_rd_byte = (arith.constant(qkhe * 2048, type=T.i32)
                                   + rowid * arith.constant(512, type=T.i32)
                                   + lane16id * arith.constant(32, type=T.i32)
                                   + arith.constant(qkr * 8, type=T.i32))
                    lds_rd_base = lds_rd_byte // arith.constant(8, type=T.i32)
                    q_v1 = vector.load_op(T.vec(1, T.i64), logits_lds_i64,
                        [arith.index_cast(T.index, lds_rd_base)])
                    q_frags.append(vector.extract(q_v1, static_position=[0]))
            SmemPtr._view_cache = None

            # ── Load K for the single block ──
            k_base = (phys_block * c_kb + _k_head_off) // c_four
            k0_flat = _load_k_flat_ps(k_base, tile_token_offset)
            k0_ops = _unflatten_k(k0_flat)

            # ── Process this CTA's 256-token tile ──
            running_max, running_sum, out0, out1 = _process_block_split(
                NEG_INF, ZERO_F,
                arith.constant_vector(0.0, T.f32x4),
                arith.constant_vector(0.0, T.f32x4),
                tile_token_offset,
                k0_ops,
            )
            SmemPtr._view_cache = None

            # ── Normalize and write output ──
            safe_sum = arith.select(running_sum > ZERO_F, running_sum, arith.constant(1.0, type=T.f32))
            inv_sum = arith.constant(1.0, type=T.f32) / safe_sum
            out0_norm = out0 * vector.broadcast(T.f32x4, inv_sum)
            out1_norm = out1 * vector.broadcast(T.f32x4, inv_sum)
            outelems_norm = [out0_norm, out1_norm]

            eqgs_lane = qi_val * arith.constant(query_group_size, type=T.i32) + qhi_pos

            for vhe in range_constexpr(VHELOOP):
                hs_base = (arith.constant(vhe * NUM_WARPS * MFMA_N, type=T.i32)
                           + warp_id * arith.constant(MFMA_N, type=T.i32)
                           + rowid * arith.constant(4, type=T.i32))
                to_off = (batch_idx * stride_to_seq
                          + kv_h * stride_to_head
                          + partition_idx * stride_to_part
                          + eqgs_lane * stride_to_group
                          + hs_base)
                out_bf16 = arith.trunc_f(T.vec(4, T.bf16), outelems_norm[vhe])
                out_i32 = vector.bitcast(T.vec(2, T.i32), out_bf16)
                buffer_ops.buffer_store(out_i32, to_rsrc,
                    to_off * arith.constant(2, type=T.i32), offset_is_bytes=True)

            es_off = (batch_idx * stride_es_seq
                      + kv_h * stride_es_head
                      + partition_idx * stride_es_part
                      + eqgs_lane)
            es_i32 = arith.bitcast(T.i32, running_sum)
            ml_i32 = arith.bitcast(T.i32, running_max)
            buffer_ops.buffer_store(es_i32, es_rsrc,
                es_off * arith.constant(4, type=T.i32), offset_is_bytes=True)
            buffer_ops.buffer_store(ml_i32, ml_rsrc,
                es_off * arith.constant(4, type=T.i32), offset_is_bytes=True)

    @flyc.jit
    def launch_pa_decode_ps_sw(es, ml, to, q, kc, vc, bt, cl,
                               qs, ks, vs,
                               s_q_seq, s_q_head,
                               s_k_block, s_k_head,
                               s_v_block, s_v_head,
                               s_es_seq, s_es_head, s_es_part,
                               s_to_seq, s_to_head, s_to_part, s_to_group,
                               s_bt_seq,
                               s_ks_block, s_ks_head,
                               gx, gy, gz,
                               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        pa_decode_ps_sw_kernel(
            es, ml, to, q, kc, vc, bt, cl, qs, ks, vs,
            s_q_seq, s_q_head, s_k_block, s_k_head,
            s_v_block, s_v_head,
            s_es_seq, s_es_head, s_es_part,
            s_to_seq, s_to_head, s_to_part, s_to_group,
            s_bt_seq,
            s_ks_block, s_ks_head,
        ).launch(
            grid=(gx, gy, gz * TILES_PER_BLOCK),
            block=(BLOCK_THREADS, 1, 1), stream=stream)

    return {
        'launch': launch_pa_decode_ps_sw,
        'kernel': pa_decode_ps_sw_kernel,
        'allocator': allocator,
    }

