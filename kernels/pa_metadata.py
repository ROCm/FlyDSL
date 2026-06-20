# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""FlyDSL implementation of aiter's ``get_pa_metadata_v1`` worklist scheduler.

This replaces the ``aiter.ops.attention.get_pa_metadata_v1`` C++/CUDA dependency
(``module_pa_metadata.so``) with a FlyDSL device kernel, so paged-attention
decode (``kernels/pa_decode_fp8.py``) can build its CU worklist without aiter.

Scope — PA-decode-specialized port of
``aiter/csrc/kernels/mla/metadata/v1_2_pa_device.cuh::kn_get_pa_metadata_v1_2``.
The following invariants always hold for the PA decode use and are baked in:

* ``kQoSplits == False`` — ``packed_qo_len = query_length * gqa`` is small
  (<= ~32) so it never exceeds ``kPackedQoLenPerWg=128`` ⇒ ``num_qo_tiles = 1``,
  ``qo_tile_size = query_length``.
* uniform qo length across batches (``uni_seqlen_qo = query_length``).
* causal, non-sparse (``topk = -1``), ``qk_batch_ratio = 1``,
  ``num_splits = num_cu`` (``max_split_per_batch = -1``).

All six outputs are produced as a faithful drop-in for the C++ kernel:
``work_metadata_ptrs``, ``work_indptr``, ``work_info`` (8 fields),
``reduce_indptr``, ``reduce_final_map`` and ``reduce_partial_map`` — each
verified element-for-element against aiter. The caller consumes them directly
(no post-hoc expansion).

work_info layout (8 x int32 per work), matching ``PaWorkInfo``:
  [0] batch_idx  [1] partial_qo_loc(-1 if no split)  [2] qo_start  [3] qo_end
  [4] kv_start   [5] kv_end                          [6] kv_offset(=0)
  [7] q_head_range = (qhead_end << 16) | (qhead_start & 0xFFFF)

The kernel is launched single-thread (grid=block=(1,1,1)); the scheduler is a
serial algorithm (warp reductions / lane-parallel fills in the original collapse
to serial loops). It runs once per shape and the result is cached, so single-
thread is the correct, simplest model.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch
from kernels.kernels_common import get_warp_size

_WORK_INFO_FIELDS = 8


def get_pa_metadata_info_v1(batch_size: int, num_head_k: int = 1, num_cu: int = None):
    """Buffer sizes/dtypes, matching ``aiter.get_pa_metadata_info_v1``.

    Returns (shape, dtype) tuples for:
      work_metadata_ptrs, work_indptr, work_info_set,
      reduce_indptr, reduce_final_map, reduce_partial_map.

    ``num_cu`` overrides the worklist bin count (default = device CU count);
    pass a multiple of the CU count to oversubscribe the persistent grid.
    """
    if num_cu is None:
        gpu = torch.cuda.current_device()
        num_cu = torch.cuda.get_device_properties(gpu).multi_processor_count
    cu_num = num_cu

    tile_cnt = batch_size
    max_work = (tile_cnt + cu_num - 1) * num_head_k
    max_split_tiles = min(batch_size + cu_num - 1, (cu_num - 1) * 2)

    return (
        ((2,), torch.uint64),  # work_metadata_ptrs
        ((cu_num + 1,), torch.int32),  # work_indptr
        ((max_work, _WORK_INFO_FIELDS), torch.int32),  # work_info_set
        ((tile_cnt + 1,), torch.int32),  # reduce_indptr
        ((tile_cnt, 2), torch.int32),  # reduce_final_map
        ((max_split_tiles,), torch.int32),  # reduce_partial_map
    )


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


@functools.lru_cache(maxsize=256)
def compile_pa_metadata_v1(
    *,
    num_cu: int,
    num_heads_k: int,
    gqa: int,
    kv_granularity: int,
    query_length: int,
    warp_size: int,
):
    """Compile the FlyDSL worklist scheduler for a fixed device/shape config.

    ``num_batches`` stays a runtime kernel argument; everything else is baked.

    Launched as a single warp (``grid=block=(warp_size,1,1)``), matching aiter's
    ``<<<grid, warp_size, ...>>>``: Phase-1 (per-batch block counts + sum) is
    warp-parallel (lane-strided + warp reduce), while the serial CU x batch
    scheduler runs uniformly on all lanes — exactly as the original, whose
    work_indptr / work_info writes are not lane-guarded (benign same-value
    races). The original's lane-divided / lane-0-guarded work is only for the
    reduce_* maps, which the caller recomputes and we therefore omit.
    """
    assert _is_pow2(kv_granularity), "kv_granularity must be power of 2"
    assert num_cu % num_heads_k == 0, "num_cu must be divisible by num_heads_k"
    num_splits_per_khead = num_cu // num_heads_k
    # warp-reduce shuffle offsets: warp_size/2, ..., 1
    _shuffle_offsets = []
    _o = warp_size // 2
    while _o >= 1:
        _shuffle_offsets.append(_o)
        _o //= 2

    @flyc.kernel(known_block_size=(warp_size, 1, 1))
    def pa_metadata_v1_kernel(
        seqlens_qo_indptr_ptr: fx.Tensor,  # [num_batches + 1] i32 (cumulative qo seqlens)
        pages_kv_indptr_ptr: fx.Tensor,  # [num_batches + 1] i32 (cumulative pages)
        context_lens_ptr: fx.Tensor,  # [num_batches] i32
        work_indptr_ptr: fx.Tensor,  # [num_cu + 1] i32   (output)
        work_info_ptr: fx.Tensor,  # [max_work * 8] i32 (output, flattened)
        reduce_indptr_ptr: fx.Tensor,  # [num_batches + 1] i32 (output)
        reduce_final_map_ptr: fx.Tensor,  # [num_batches * 2] i32 (output, flattened)
        reduce_partial_map_ptr: fx.Tensor,  # [max_split_tiles] i32 (output)
        num_batches: Int32,
    ):
        i32 = T.i32
        sq_rsrc = buffer_ops.create_buffer_resource(seqlens_qo_indptr_ptr, max_size=True)
        ctx_rsrc = buffer_ops.create_buffer_resource(context_lens_ptr, max_size=True)
        wi_rsrc = buffer_ops.create_buffer_resource(work_indptr_ptr, max_size=True)
        winfo_rsrc = buffer_ops.create_buffer_resource(work_info_ptr, max_size=True)
        rip_rsrc = buffer_ops.create_buffer_resource(reduce_indptr_ptr, max_size=True)
        rfm_rsrc = buffer_ops.create_buffer_resource(reduce_final_map_ptr, max_size=True)
        rpm_rsrc = buffer_ops.create_buffer_resource(reduce_partial_map_ptr, max_size=True)
        # pages_kv_indptr_ptr is accepted for signature compat but unused: the
        # work unit is a `partition_size`-token partition, so both the load
        # balance and the kv ranges are counted as ceil(context_len/kv_gran)
        # partitions (kv_granularity == partition_size). This matches the
        # original kernel's kLdsBatchInfo=true path where curr_kv_pages is the
        # per-batch num_blocks, not the pages_kv_indptr delta.

        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        c_qlen = fx.Int32(query_length)
        c_nb = num_batches  # Int32 runtime
        c_nspk = fx.Int32(num_splits_per_khead)
        c_numcu = fx.Int32(num_cu)
        c_ws = fx.Int32(warp_size)
        c_kvg = fx.Int32(kv_granularity)
        lane = fx.Int32(gpu.thread_id("x"))

        def _load(rsrc, off):
            return fx.Int32(buffer_ops.buffer_load(rsrc, fx.Int32(off).ir_value(), vec_width=1, dtype=i32))

        def _num_part(batch_idx):
            # number of partition_size-token partitions for this batch =
            # ceil(context_len[batch_idx] / kv_granularity)
            ctxv = _load(ctx_rsrc, batch_idx)
            return fx.Int32(arith.ceildivui(ctxv.ir_value(), c_kvg.ir_value()))

        def _store(rsrc, off, val):
            # NOTE: no masked stores — masked buffer_store sets OOB offset
            # (0x7FFFFFFF) expecting HW bounds-check to drop it, but our
            # max_size resources disable bounds-checking, so a masked store
            # faults. All stores here are unconditional + overwrite-safe.
            buffer_ops.buffer_store(fx.Int32(val).ir_value(), rsrc, fx.Int32(off).ir_value())

        def _sel(cond_b, a, b):
            return fx.Int32(arith.select(cond_b.ir_value(), fx.Int32(a).ir_value(), fx.Int32(b).ir_value()))

        # work_indptr[0] = 0 ; reduce_indptr[0] = 0
        _store(wi_rsrc, 0, 0)
        _store(rip_rsrc, 0, 0)

        # ---- Phase 1: sum_blocks = Sum_b ceil(context_lens[b] / kv_granularity) ----
        # (causal + uniform + tiny qo  =>  effective_kv == context_lens[b])
        # warp-parallel: each lane sums batches {lane, lane+ws, lane+2ws, ...},
        # then a warp reduce-add gives the total in every lane (matches aiter).
        wl1 = scf.WhileOp([i32, i32], [lane.ir_value(), c0.ir_value()])
        cb1 = ir.Block.create_at_start(wl1.before, [i32, i32])
        bb1 = ir.Block.create_at_start(wl1.after, [i32, i32])
        with ir.InsertionPoint(cb1):
            b = fx.Int32(cb1.arguments[0])
            scf.ConditionOp((b < c_nb).ir_value(), list(cb1.arguments))
        with ir.InsertionPoint(bb1):
            b = fx.Int32(bb1.arguments[0])
            s = fx.Int32(bb1.arguments[1])
            nblk = _num_part(b)
            scf.YieldOp([(b + c_ws).ir_value(), (s + nblk).ir_value()])
        sum_blocks = fx.Int32(wl1.results[1])
        for sh in _shuffle_offsets:
            sum_blocks = sum_blocks + sum_blocks.shuffle_xor(arith.constant(sh, type=i32), c_ws.ir_value())

        average = fx.Int32(arith.divui(sum_blocks.ir_value(), c_nspk.ir_value()))
        reminder = fx.Int32(arith.remui(sum_blocks.ir_value(), c_nspk.ir_value()))

        def _remain_for_cid(cid_val):
            # remain = average + (1 if (cid % num_splits_per_khead) < reminder else 0)
            mod = fx.Int32(arith.remui(cid_val.ir_value(), c_nspk.ir_value()))
            return average + _sel(mod < reminder, 1, 0)

        # ---- Phase 2: per khead, flattened CU x batch scheduler ----
        # cid and num_works persist across kheads; the rest reset per khead.
        cid = c0
        num_works = c0

        for khead in range_constexpr(num_heads_k):
            qh_start = khead * gqa
            qh_end = (khead + 1) * gqa
            qhr_const = (qh_end << 16) | (qh_start & 0xFFFF)  # python int constant

            kvend0 = _num_part(c0)  # partitions in batch 0 (cumulative kv end)
            remain0 = _remain_for_cid(cid)

            # State vector (11 i32):
            #  0 cid, 1 batch, 2 kvblk, 3 nsplit, 4 num_works,
            #  5 pidx, 6 kvbeg, 7 kvend, 8 remain,
            #  9 last_reduce_indptr (lri), 10 global_reduce_tile_idx (grt)
            # cid + num_works persist across kheads; lri + grt reset per khead
            # (matches the original, which overwrites reduce_* per khead).
            init = [
                cid.ir_value(),
                c0.ir_value(),
                c0.ir_value(),
                c0.ir_value(),
                num_works.ir_value(),
                c0.ir_value(),
                c0.ir_value(),
                kvend0.ir_value(),
                remain0.ir_value(),
                c0.ir_value(),
                c0.ir_value(),
            ]
            wl = scf.WhileOp([i32] * 11, init)
            cbk = ir.Block.create_at_start(wl.before, [i32] * 11)
            bbk = ir.Block.create_at_start(wl.after, [i32] * 11)

            with ir.InsertionPoint(cbk):
                s_cid = fx.Int32(cbk.arguments[0])
                s_batch = fx.Int32(cbk.arguments[1])
                cond = (s_cid < c_numcu) & (s_batch < c_nb)
                scf.ConditionOp(cond.ir_value(), list(cbk.arguments))

            with ir.InsertionPoint(bbk):
                a = bbk.arguments
                cid_ = fx.Int32(a[0])
                batch_ = fx.Int32(a[1])
                kvblk_ = fx.Int32(a[2])
                nsplit_ = fx.Int32(a[3])
                nworks_ = fx.Int32(a[4])
                pidx_ = fx.Int32(a[5])
                kvbeg_ = fx.Int32(a[6])
                kvend_ = fx.Int32(a[7])
                remain_ = fx.Int32(a[8])
                lri_ = fx.Int32(a[9])
                grt_ = fx.Int32(a[10])

                pages = kvend_ - kvbeg_
                remain_kv = pages - kvblk_
                do_finish = remain_ >= remain_kv  # fx bool

                # qo_start/qo_end from seqlens_qo_indptr (the QoState array path,
                # matching C++). batch_ < num_batches in the loop, so batch_+1 <=
                # num_batches indexes the valid last element (no OOB). For uniform
                # qo (sq = arange*query_length) this equals query_length*batch.
                qo_start = _load(sq_rsrc, batch_)
                qo_end = _load(sq_rsrc, batch_ + c1)
                kv_start = kvbeg_ + kvblk_  # same for both branches

                # ---- finish branch (CU completes this batch) ----
                f_kv_end = kvend_  # min(kv_start + remain_kv, kvend_) == kvend_
                nsplit_pos = nsplit_ > c0
                f_ploc = _sel(nsplit_pos, pidx_, -1)
                f_pidx2 = _sel(nsplit_pos, pidx_ + c_qlen, pidx_)
                f_nworks2 = nworks_ + c1
                f_remain2 = remain_ - remain_kv
                f_batch2 = batch_ + c1
                # next batch kv window (in partition units). max_size buffer rsrc
                # disables HW bounds checking, so clamp the index before loading
                # context_lens (f_batch2 can equal num_batches → OOB on last batch;
                # the result is unused after the loop exits anyway).
                nb_in_range = f_batch2 < c_nb
                safe_idx = _sel(nb_in_range, f_batch2, 0)
                f_new_pages = _sel(nb_in_range, _num_part(safe_idx), 0)
                f_kvbeg2 = kvend_
                f_kvend2 = kvend_ + f_new_pages

                # ---- split branch (CU does a partial; close cid) ----
                s_emit = remain_ > c0
                s_kv_end_raw = kv_start + remain_
                s_kv_end = _sel(s_kv_end_raw < kvend_, s_kv_end_raw, kvend_)
                s_nworks2 = _sel(s_emit, nworks_ + c1, nworks_)
                s_pidx2 = _sel(s_emit, pidx_ + c_qlen, pidx_)
                s_kvblk2 = _sel(s_emit, kvblk_ + remain_, kvblk_)
                s_nsplit2 = _sel(s_emit, nsplit_ + c1, nsplit_)
                s_cid2 = cid_ + c1
                s_remain2 = _remain_for_cid(s_cid2)

                # ---- emit work entry at slot nworks_ ----
                # Overwrite-safe: if this step does not emit, nworks_ is unchanged
                # so the slot is reused by the next emit (or lies beyond valid_work).
                w_ploc = _sel(do_finish, f_ploc, pidx_)
                w_kv_end = _sel(do_finish, f_kv_end, s_kv_end)
                base = nworks_ * fx.Int32(_WORK_INFO_FIELDS)
                _store(winfo_rsrc, base + fx.Int32(0), batch_)
                _store(winfo_rsrc, base + fx.Int32(1), w_ploc)
                _store(winfo_rsrc, base + fx.Int32(2), qo_start)
                _store(winfo_rsrc, base + fx.Int32(3), qo_end)
                _store(winfo_rsrc, base + fx.Int32(4), kv_start)
                _store(winfo_rsrc, base + fx.Int32(5), w_kv_end)
                _store(winfo_rsrc, base + fx.Int32(6), c0)
                _store(winfo_rsrc, base + fx.Int32(7), fx.Int32(qhr_const))

                # ---- reduce maps: only when finishing a SPLIT batch (nsplit>0) ----
                # This batch was split across (nsplit_+1) CUs and this CU finishes
                # it, forming one reduce group. Faithful to the C++ kernel
                # (kQoSplits=False path).
                do_reduce = do_finish & nsplit_pos
                num_splits = nsplit_ + c1
                # reduce_indptr[grt+1] = lri + num_splits ; reduce_final_map[grt] =
                # (qo_start, qo_end). Unconditional + overwrite-safe (same argument
                # as work_indptr: grt only advances on do_reduce, so non-do_reduce
                # writes to grt+1 are overwritten by the next do_reduce or the tail;
                # reduce_final_map[grt*2..] beyond the final grt is never read).
                _store(rip_rsrc, grt_ + c1, lri_ + num_splits)
                _store(rfm_rsrc, grt_ * fx.Int32(2), qo_start)
                _store(rfm_rsrc, grt_ * fx.Int32(2) + c1, qo_end)
                # reduce_partial_map[lri + s] = pidx - (nsplit - s)*qlen, s in [0,num_splits)
                # nested loop runs num_splits times when do_reduce, else 0 times.
                rcount = _sel(do_reduce, num_splits, 0)
                wlr = scf.WhileOp([i32], [c0.ir_value()])
                cbr = ir.Block.create_at_start(wlr.before, [i32])
                bbr = ir.Block.create_at_start(wlr.after, [i32])
                with ir.InsertionPoint(cbr):
                    sidx = fx.Int32(cbr.arguments[0])
                    scf.ConditionOp((sidx < rcount).ir_value(), list(cbr.arguments))
                with ir.InsertionPoint(bbr):
                    sidx = fx.Int32(bbr.arguments[0])
                    val = pidx_ - (nsplit_ - sidx) * c_qlen
                    _store(rpm_rsrc, lri_ + sidx, val)
                    scf.YieldOp([(sidx + c1).ir_value()])
                n_lri = lri_ + _sel(do_reduce, num_splits, 0)
                n_grt = grt_ + _sel(do_reduce, 1, 0)

                # ---- new state via select(do_finish, finish, split) ----
                n_cid = _sel(do_finish, cid_, s_cid2)
                n_batch = _sel(do_finish, f_batch2, batch_)
                n_kvblk = _sel(do_finish, 0, s_kvblk2)
                n_nsplit = _sel(do_finish, 0, s_nsplit2)
                n_nworks = _sel(do_finish, f_nworks2, s_nworks2)
                n_pidx = _sel(do_finish, f_pidx2, s_pidx2)
                n_kvbeg = _sel(do_finish, f_kvbeg2, kvbeg_)
                n_kvend = _sel(do_finish, f_kvend2, kvend_)
                n_remain = _sel(do_finish, f_remain2, s_remain2)

                # ---- work_indptr[cid_+1] = n_nworks (unconditional, overwrite-safe) ----
                # In the finish branch cid_ does not advance, so repeated writes to
                # work_indptr[cid_+1] keep updating until the cid closes; the last
                # write (before cid advances in the split branch, or loop exit) holds
                # the correct running num_works for that cid. cid_+1 <= num_cu (loop
                # guard cid_ < num_cu) so the index is always in-bounds.
                _store(wi_rsrc, cid_ + c1, n_nworks)

                scf.YieldOp(
                    [
                        n_cid.ir_value(),
                        n_batch.ir_value(),
                        n_kvblk.ir_value(),
                        n_nsplit.ir_value(),
                        n_nworks.ir_value(),
                        n_pidx.ir_value(),
                        n_kvbeg.ir_value(),
                        n_kvend.ir_value(),
                        n_remain.ir_value(),
                        n_lri.ir_value(),
                        n_grt.ir_value(),
                    ]
                )

            cid = fx.Int32(wl.results[0])
            num_works = fx.Int32(wl.results[4])
            last_reduce_indptr = fx.Int32(wl.results[9])
            global_reduce_tile_idx = fx.Int32(wl.results[10])

            # ---- post-khead close: advance cid past the last processed cid so
            # the next khead (and the tail) start fresh. The loop already wrote
            # work_indptr[last_cid+1]=num_works on its final iteration, so no
            # store is needed here — only the cid advance.
            in_range = cid < c_numcu
            cid = _sel(in_range, cid + c1, cid)

        # ---- tail: work_indptr[i] = num_works for i in [cid, num_cu] ----
        wlt = scf.WhileOp([i32], [cid.ir_value()])
        cbt = ir.Block.create_at_start(wlt.before, [i32])
        bbt = ir.Block.create_at_start(wlt.after, [i32])
        with ir.InsertionPoint(cbt):
            it = fx.Int32(cbt.arguments[0])
            scf.ConditionOp((it <= c_numcu).ir_value(), list(cbt.arguments))
        with ir.InsertionPoint(bbt):
            it = fx.Int32(bbt.arguments[0])
            _store(wi_rsrc, it, num_works)
            scf.YieldOp([(it + c1).ir_value()])

        # ---- tail: reduce_indptr[i] = last_reduce_indptr for i in [grt, num_batches] ----
        # (reduce_indptr has num_batches+1 entries; uses the final khead's grt/lri,
        # matching the original which resets grt per khead and fills the tail once.)
        c_rip_size = c_nb + c1  # reduce_indptr length = num_batches + 1
        wlr2 = scf.WhileOp([i32], [global_reduce_tile_idx.ir_value()])
        cbr2 = ir.Block.create_at_start(wlr2.before, [i32])
        bbr2 = ir.Block.create_at_start(wlr2.after, [i32])
        with ir.InsertionPoint(cbr2):
            it = fx.Int32(cbr2.arguments[0])
            scf.ConditionOp((it < c_rip_size).ir_value(), list(cbr2.arguments))
        with ir.InsertionPoint(bbr2):
            it = fx.Int32(bbr2.arguments[0])
            _store(rip_rsrc, it, last_reduce_indptr)
            scf.YieldOp([(it + c1).ir_value()])

    @flyc.jit
    def launch_pa_metadata_v1(
        seqlens_qo_indptr: fx.Tensor,
        pages_kv_indptr: fx.Tensor,
        context_lens: fx.Tensor,
        work_indptr: fx.Tensor,
        work_info: fx.Tensor,
        reduce_indptr: fx.Tensor,
        reduce_final_map: fx.Tensor,
        reduce_partial_map: fx.Tensor,
        num_batches: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        pa_metadata_v1_kernel(
            seqlens_qo_indptr,
            pages_kv_indptr,
            context_lens,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            num_batches,
        ).launch(grid=(1, 1, 1), block=(warp_size, 1, 1), stream=stream)

    return {"kernel": pa_metadata_v1_kernel, "launch": launch_pa_metadata_v1}


def get_pa_metadata_v1(
    seqlens_qo_indptr: torch.Tensor,
    pages_kv_indptr: torch.Tensor,
    context_lens: torch.Tensor,
    num_heads_per_head_k: int,
    num_heads_k: int,
    is_causal: bool,
    work_metadata_ptrs: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    kv_granularity: int = 16,
    block_size: int = 16,
    max_seqlen_qo: int = -1,
    uni_seqlen_qo: int = -1,
    fast_mode: bool = True,
    topk: int = -1,
    max_split_per_batch: int = -1,
    num_cu: int = None,
) -> None:
    """Drop-in replacement for ``aiter.ops.attention.get_pa_metadata_v1``.

    PA-decode-specialized: requires causal, non-sparse, uniform qo. Fills
    ``work_indptr`` and ``work_info`` in-place; ``reduce_*`` are left untouched
    (the caller recomputes them from work_indptr/work_info).

    ``num_cu`` overrides the worklist bin count (default = device CU count);
    pass a multiple of the CU count to oversubscribe the persistent grid.
    """
    assert is_causal, "FlyDSL pa_metadata only supports causal"
    assert topk == -1, "FlyDSL pa_metadata does not support sparse (topk)"
    assert uni_seqlen_qo >= 1, "FlyDSL pa_metadata requires uniform qo length"

    dev = pages_kv_indptr.device
    if num_cu is None:
        num_cu = torch.cuda.get_device_properties(dev).multi_processor_count
    num_batches = context_lens.shape[0]
    query_length = uni_seqlen_qo
    warp_size = get_warp_size(get_rocm_arch())

    compiled = compile_pa_metadata_v1(
        num_cu=num_cu,
        num_heads_k=num_heads_k,
        gqa=num_heads_per_head_k,
        kv_granularity=kv_granularity,
        query_length=query_length,
        warp_size=warp_size,
    )

    # work_metadata_ptrs[0/1] = device addresses of work_indptr / work_info,
    # matching the C++ kernel (which writes them in-kernel via reinterpret_cast).
    # These are exactly the tensors' data_ptr() values, so writing them host-side
    # produces identical bytes.
    if work_metadata_ptrs is not None and work_metadata_ptrs.numel() >= 2:
        work_metadata_ptrs[0] = work_indptr.data_ptr()
        work_metadata_ptrs[1] = work_info.data_ptr()

    # work_info [max_work, 8] and reduce_final_map [num_batches, 2] are written
    # flattened by the kernel.
    work_info_flat = work_info.view(-1)
    reduce_final_map_flat = reduce_final_map.view(-1)

    compiled["launch"](
        seqlens_qo_indptr,
        pages_kv_indptr,
        context_lens,
        work_indptr,
        work_info_flat,
        reduce_indptr,
        reduce_final_map_flat,
        reduce_partial_map,
        num_batches,
    )
