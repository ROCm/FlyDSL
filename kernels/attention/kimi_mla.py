# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 MLA decode and prefill kernels for gfx950 (MI350/MI355X).

The MFMA32 layout, online softmax and PV rearrangement are derived from the
FlyDSL MFMA32 forward in pytorch/pytorch#194309.

Decode (``KimiMLADecodePlan``) is the compressed, absorbed MLA used by Kimi-K3
with eight speculative-verify query tokens and DCP8 round-robin KV sharding:

* Q is FP8 E4M3FN ``[8, 96, 576]`` (96 heads after the DCP all-gather).
* KV is FP8 E4M3FN ``[slots, 576]``; the first 512 channels are also V.
  A contiguous int32 index vector maps rank-local logical tokens to slots.
* ``global_seq_len`` is a device int32 ``[1]`` that includes the verify block.
  Query ``t`` on DCP rank ``r`` sees ``max(0, (len - 8 + t + 8 - r) // 8)``
  local tokens.
* O is BF16 ``[8, 96, 512]``; LSE is natural-log FP32 ``[8, 96]``.

Unlike ``mla_fwd_decode_m16x8_fp8_fp8`` (128 heads, one query token per
sequence, host-built ``work_indptr`` metadata), this kernel computes the 96
heads without padding, applies the multi-token causal verify mask and the DCP
length formula itself, and selects its split count on the device, so the
sequence length can change inside a captured graph.

Prefill (``KimiMLAPrefillPlan``) is dense causal attention after
``kv_b_proj`` with 12 heads per rank: Q/K ``[S, 12, 192]``, V ``[S, 12, 128]``,
returning BF16 O ``[S, 12, 128]`` and natural-log FP32 LSE ``[S, 12]``.
"""

import functools
import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

DECODE_QLEN = 8
DECODE_HEADS = 96
DECODE_QK_DIM = 576
DECODE_V_DIM = 512
DECODE_CP_WORLD = 8
DECODE_MAX_SPLITS = 42

PREFILL_HEADS = 12
PREFILL_QK_DIM = 192
PREFILL_V_DIM = 128
PREFILL_MIN_SEQ = 128
PREFILL_MAX_SEQ = 262144

# Softmax scale shared by decode and prefill (qk_nope 128 + qk_rope 64).
SOFTMAX_SCALE = 192 ** (-0.5)

# E8M0 scale words of 1.0 for both MFMA operands.
_UNIT_SCALE = 0x7F7F7F7F


def _global(tensor, size):
    return fx.rocdl.make_buffer_tensor(fx.make_view(fx.get_iter(tensor), fx.make_layout(size, 1)))


def _value_layout(rows, cols):
    return fx.make_layout(((8, rows // 8), (32, cols // 32)), ((32, 8 * cols), (1, 256)))


def _exp(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(x).ir_value()))


def _mfma_fp8(a, b, c):
    """32x32x64 FP8 MFMA with unit block scales."""
    return fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
        T.vec(16, T.f32),
        [
            a.ir_value(),
            b.ir_value(),
            Vec(c).ir_value(),
            0,
            0,
            0,
            fx.Int32(_UNIT_SCALE).ir_value(),
            0,
            fx.Int32(_UNIT_SCALE).ir_value(),
        ],
    )


def _swap_halves(lo, hi):
    pair = fx.rocdl.permlane32_swap(
        ir.Type.parse("!llvm.struct<(i32, i32)>"), fx.Int32(lo).ir_value(), fx.Int32(hi).ir_value(), False, True
    )
    return fx.Int32(llvm.extractvalue(T.i32, pair, [0])), fx.Int32(llvm.extractvalue(T.i32, pair, [1]))


def _store_accumulator_bf16(acc, buf, offset):
    """Store a lane's 32-column MFMA accumulator slice as two BF16x8 groups.

    ``offset`` is the element offset in ``buf`` of this lane's first group;
    the second group starts 16 columns later.
    """
    words = Vec(acc).bitcast(fx.Int32)
    for group in fx.range_constexpr(2):
        lower, upper = ([], [])
        for i in fx.range_constexpr(4):
            lo, hi = _swap_halves(words[group * 8 + i], words[group * 8 + i + 4])
            lower.append(lo)
            upper.append(hi)
        frag = fx.make_rmem_tensor(8, fx.BFloat16)
        frag.store(Vec.from_elements(lower + upper, fx.Int32).bitcast(fx.Float32).to(fx.BFloat16))
        dst = fx.make_view(fx.get_iter(buf) + (offset + group * 16), fx.make_layout(8, 1))
        fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16), frag, dst)


def _require_gfx950(device, what):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"{what} requires a gfx950 GPU")
    if device.index is None:
        device = torch.device("cuda", torch.accelerator.current_device_index())
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch != "gfx950":
        raise ValueError(f"{what} requires gfx950, got {arch}")
    return device


def _check_tensor(device, name, tensor, shape, dtype):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {tuple(shape)} and dtype {dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _build_decode(*, max_local, cache_size, cp_rank, return_lse=True):
    heads = DECODE_HEADS
    qlen = DECODE_QLEN
    splits = DECODE_MAX_SPLITS
    cp_world = DECODE_CP_WORLD
    waves = 4
    value_tile = 512
    # FP8 probabilities are scaled by 256 before conversion; undone in the merge.
    prob_scale = 256.0
    rows = heads * qlen
    threads = 64 * waves
    chunks = value_tile // 32
    kv_type = fx.Float8E4M3FN
    query_blocks = (rows + waves * 32 - 1) // (waves * 32)
    vblocks = DECODE_V_DIM // value_tile
    vtype = fx.Float8E4M3FN
    tile_kv = 64
    # Two FP8 KV tiles (double buffer), stored in a BF16-typed array.
    shared_elements = tile_kv * DECODE_QK_DIM // 2 * 2

    @fx.struct
    class Shared:
        kv: fx.Array[fx.BFloat16, shared_elements, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def kimi_mla_decode_split(
        q: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        seq_len: fx.Tensor,
        q_scale: fx.Tensor,
        kv_scale: fx.Tensor,
        partial: fx.Tensor,
        stats: fx.Tensor,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % 64
        wave = tid // 64
        split = fx.Int32(fx.block_idx.y)
        vbase = fx.Int32(fx.block_idx.z) * value_tile
        lds = fx.SharedAllocator().allocate(Shared).peek()
        qbuf = _global(q, rows * 576)
        kvbuf = _global(kv, cache_size * 576)
        ibuf = _global(indices, max_local)
        sbuf = _global(seq_len, 1)
        pbuf = _global(partial, splits * rows * 512)
        stbuf = _global(stats, splits * rows * 2)
        qs = fx.Float32(_global(q_scale, 1)[0])
        ks = fx.Float32(_global(kv_scale, 1)[0])
        score_scale = qs * ks * fx.Float32(SOFTMAX_SCALE * math.log2(math.e))

        # Device-side split selection: up to 42 splits of whole 64-token tiles.
        glen = fx.Int32(sbuf[0])
        raw_len = (glen + cp_world - 1 - cp_rank) // cp_world
        local_len = (raw_len > 0).select(raw_len, fx.Int32(0))
        tile_count = (local_len + tile_kv - 1) // tile_kv
        active_splits = (tile_count < splits).select(tile_count, fx.Int32(splits))
        active_splits = (active_splits > 0).select(active_splits, fx.Int32(1))
        split_tiles = (local_len + active_splits * tile_kv - 1) // (active_splits * tile_kv)
        begin = split * split_tiles
        last = (local_len + tile_kv - 1) // tile_kv
        end = (begin + split_tiles < last).select(begin + split_tiles, last)

        mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16)),
            fx.make_layout((1, 1, 1), (1, 1, 1)),
        )
        thr = mma.get_slice(lane)
        ccoord = thr.partition_C(fx.make_view(0, fx.make_layout((32, 32), (1, 0))))
        rcoord = thr.partition_C(fx.make_view(0, fx.make_layout((32, 32), (0, 1))))
        row = fx.Int32(fx.block_idx.x) * (waves * 32) + wave * 32 + fx.Int32(fx.get_scalar(rcoord[0]))
        safe_row = (row < rows).select(row, fx.Int32(0))
        # Causal verify mask with DCP round-robin: rows are (token, head).
        raw_limit = (glen - qlen + safe_row // heads + cp_world - cp_rank) // cp_world
        limit = (raw_limit > 0).select(raw_limit, fx.Int32(0))

        klayout = _value_layout(tile_kv, 576)
        vlayout = _value_layout(tile_kv, 576)
        v_mma_layout = fx.composition(
            fx.select(vlayout, [1, 0]),
            fx.make_tile(
                fx.make_layout(576, 1),
                fx.make_layout((4, 2, 2, tile_kv // 16), (1, 8, 4, 16)),
            ),
        )
        transpose = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(8, 64), vtype)
        copy_v = fx.make_tiled_copy_A(transpose, mma).get_slice(lane)

        # Q stays in registers for the whole split: 9 x 64-channel MFMA steps.
        qregs = []
        cp8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float8E4M3FN)
        for step in fx.range_constexpr(9):
            words = []
            for pack in fx.range_constexpr(2):
                frag = fx.make_rmem_tensor(16, fx.Float8E4M3FN)
                col = step * 64 + lane // 32 * 32 + pack * 16
                source = fx.make_view(fx.get_iter(qbuf) + safe_row * 576 + col, fx.make_layout(16, 1))
                fx.copy(cp8, source, frag)
                packed = Vec(frag.load()).bitcast(fx.Int32)
                words.extend([packed[i] for i in fx.range_constexpr(4)])
            qregs.append(Vec.from_elements(words, fx.Int32))

        def prefetch_key(base):
            coords = fx.right_inverse(klayout)
            slots, columns, keeps, frags = ([], [], [], [])
            for it in fx.range_constexpr(tile_kv * 576 // (threads * 16)):
                logical = fx.Int32(fx.get_scalar(fx.crd2idx((it * threads + tid) * 16, coords)))
                token = base + logical % tile_kv
                keep = token < local_len
                slots.append(fx.Int32(ibuf[keep.select(token, fx.Int32(0))]))
                columns.append(logical // tile_kv)
                keeps.append(keep)
            fx.rocdl.sched_barrier(0)
            for it in fx.range_constexpr(tile_kv * 576 // (threads * 16)):
                frag = fx.make_rmem_tensor(16, kv_type)
                src = fx.make_view(fx.get_iter(kvbuf) + slots[it] * 576 + columns[it], fx.make_layout(16, 1))
                fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), kv_type), src, frag)
                frags.append(frag)
            fx.rocdl.sched_barrier(0)
            return (frags, keeps)

        def store_prefetched(frags, keeps, ptr):
            # Tokens past the local length are zeroed so stale/NaN slots cannot leak into PV.
            dst = fx.logical_divide(fx.make_view(ptr, fx.make_layout(tile_kv * 576, 1)), fx.make_layout(16, 1))
            for it in fx.range_constexpr(tile_kv * 576 // (threads * 16)):
                words = Vec(frags[it].load()).bitcast(fx.Int32)
                clean = Vec.from_elements(
                    [keeps[it].select(fx.Int32(words[j]), fx.Int32(0)) for j in fx.range_constexpr(4)],
                    fx.Int32,
                )
                frags[it].store(clean.bitcast(kv_type))
                fx.copy(
                    fx.make_copy_atom(fx.UniversalCopy128b(), kv_type),
                    frags[it],
                    fx.slice(dst, (None, it * threads + tid)),
                )

        zero = fx.full(16, 0.0, fx.Float32)
        initial = [fx.Float32(-1e30), fx.Float32(0.0)] + [zero for _ in range(chunks)]
        final = initial
        kptr = fx.recast_iter(fx.Float8E4M3FN, lds.kv.ptr)
        preload, keep_preload = prefetch_key(begin * tile_kv)
        store_prefetched(preload, keep_preload, kptr)
        fx.gpu.barrier()
        for tile, args in range(begin, end, fx.Int32(1), init=initial):
            tile = fx.Int32(tile)
            oldmax, oldsum = (fx.Float32(args[0]), fx.Float32(args[1]))
            accum = [args[2 + d] for d in fx.range_constexpr(chunks)]
            stage = (tile - begin) % 2
            baseptr = fx.recast_iter(fx.Float8E4M3FN, lds.kv.ptr)
            kptr = baseptr + stage * (tile_kv * 576)
            nextptr = baseptr + (1 - stage) * (tile_kv * 576)
            value_tiles = fx.flat_divide(fx.make_view(kptr, v_mma_layout), (32, 16))
            prefetched, keep_next = prefetch_key((tile + 1) * tile_kv)

            # S^T = K Q^T for 64 tokens x 32 rows per wave.
            tile_scores = [zero, zero]
            cp8lds = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float8E4M3FN)
            for step in fx.range_constexpr(9):
                qr = Vec(qregs[step])
                for half in fx.range_constexpr(2):
                    shared_row = fx.logical_divide(
                        fx.slice(fx.make_view(kptr, klayout), (lane % 32 + half * 32, None)),
                        fx.make_layout(16, 1),
                    )
                    words = []
                    for pack in fx.range_constexpr(2):
                        frag = fx.make_rmem_tensor(16, fx.Float8E4M3FN)
                        fx.copy(cp8lds, fx.slice(shared_row, (None, step * 4 + lane // 32 * 2 + pack)), frag)
                        packed = Vec(frag.load()).bitcast(fx.Int32)
                        words.extend([packed[i] for i in fx.range_constexpr(4)])
                    a = Vec.from_elements(words, fx.Int32)
                    tile_scores[half] = Vec(_mfma_fp8(a, qr, tile_scores[half]))
            fx.rocdl.sched_dsrd(8)
            for step in fx.range_constexpr(9):
                fx.rocdl.sched_mfma(2)
                if const_expr(step + 2 < 9):
                    fx.rocdl.sched_dsrd(4)
            fx.rocdl.sched_barrier(0)

            # Online softmax over the 64-token tile.
            values = []
            for half in fx.range_constexpr(2):
                for e in fx.range_constexpr(16):
                    keep = (row < rows) & (tile * tile_kv + half * 32 + fx.Int32(fx.get_scalar(ccoord[e])) < limit)
                    values.append(
                        keep.select(fx.Float32(Vec(tile_scores[half])[e]) * score_scale, fx.Float32(float("-inf")))
                    )
            levels = values
            for depth in fx.range_constexpr(5):
                levels = [levels[2 * i].maximumf(levels[2 * i + 1]) for i in fx.range_constexpr(16 >> depth)]
            m = levels[0].maximumf(fx.Float32(fx.gpu.shuffle_xor(levels[0], 32, 64))).maximumf(oldmax)
            correction = _exp(oldmax - m)
            probs = [_exp(values[i] - m) for i in fx.range_constexpr(32)]
            levels = probs
            for depth in fx.range_constexpr(5):
                levels = [levels[2 * i] + levels[2 * i + 1] for i in fx.range_constexpr(16 >> depth)]
            total = oldsum * correction + levels[0] + fx.Float32(fx.gpu.shuffle_xor(levels[0], 32, 64))
            corr = Vec.from_elements([correction], fx.Float32).broadcast_to(16)
            words = []
            for w in fx.range_constexpr(8):
                off = w * 4
                word = fx.rocdl.cvt_pk_fp8_f32(
                    T.i32, probs[off] * prob_scale, probs[off + 1] * prob_scale, fx.Int32(0), 0
                )
                word = fx.rocdl.cvt_pk_fp8_f32(T.i32, probs[off + 2] * prob_scale, probs[off + 3] * prob_scale, word, 1)
                words.append(fx.Int32(word))
            bp = Vec.from_elements(words, fx.Int32)

            # O^T += V^T P^T across all 512 latent channels; V read transposed from the K tile.
            for d in fx.range_constexpr(chunks):
                vwords = []
                for pack in fx.range_constexpr(4):
                    vt = fx.slice(value_tiles, (None, None, d + vbase // 32, pack))
                    a = fx.make_fragment_like(thr.partition_A(vt), vtype)
                    fx.copy(transpose, copy_v.partition_S(vt), copy_v.retile(a))
                    packed = Vec(a.load()).bitcast(fx.Int32)
                    vwords.extend([packed[0], packed[1]])
                ap = Vec.from_elements(vwords, fx.Int32)
                accum[d] = _mfma_fp8(ap, bp, Vec(accum[d]) * corr)
                fx.rocdl.sched_barrier(0)
            fx.rocdl.sched_dsrd(8)
            for d in fx.range_constexpr(chunks):
                fx.rocdl.sched_mfma(1)
                if const_expr(d + 2 < chunks):
                    fx.rocdl.sched_dsrd(4)
            fx.rocdl.sched_barrier(0)
            store_prefetched(prefetched, keep_next, nextptr)
            fx.gpu.barrier()
            final = yield [m, total, *accum]

        if row < rows:
            for d in fx.range_constexpr(chunks):
                col = vbase + d * 32 + lane // 32 * 8
                _store_accumulator_bf16(final[2 + d], pbuf, (split * rows + row) * 512 + col)
            if (lane < 32) & (fx.block_idx.z == 0):
                stbuf[(split * rows + row) * 2] = final[0]
                stbuf[(split * rows + row) * 2 + 1] = final[1]

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kimi_mla_decode_merge(
        partial: fx.Tensor,
        stats: fx.Tensor,
        kv_scale: fx.Tensor,
        seq_len: fx.Tensor,
        output: fx.Tensor,
        lse: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        lane = fx.Int32(fx.thread_idx.x)
        pb = _global(partial, splits * rows * 512)
        sb = _global(stats, splits * rows * 2)
        ob = _global(output, rows * 512)
        if const_expr(return_lse):
            lb = _global(lse, rows)
        ks = fx.Float32(_global(kv_scale, 1)[0])

        def combine(count):
            maxima, sums = ([], [])
            m = fx.Float32(-1e30)
            for s in fx.range_constexpr(count):
                mx = fx.Float32(sb[(s * rows + row) * 2])
                sm = fx.Float32(sb[(s * rows + row) * 2 + 1])
                maxima.append(mx)
                sums.append(sm)
                m = m.maximumf(mx)
            weights = [_exp(maxima[s] - m) for s in fx.range_constexpr(count)]
            total = fx.Float32(0.0)
            acc = fx.full(8, 0.0, fx.Float32)
            for s in fx.range_constexpr(count):
                total = total + sums[s] * weights[s]
                part = fx.make_rmem_tensor(8, fx.BFloat16)
                src = fx.make_view(fx.get_iter(pb) + (s * rows + row) * 512 + lane * 8, fx.make_layout(8, 1))
                fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16), src, part)
                pv = Vec(part.load()).to(fx.Float32)
                acc = acc + pv * Vec.from_elements([weights[s]], fx.Float32).broadcast_to(8)
            return (acc, m, total)

        # Only read the split slots the split kernel can have filled at this length.
        acc = fx.full(8, 0.0, fx.Float32)
        m, total = (fx.Float32(-1e30), fx.Float32(0.0))
        glen = fx.Int32(_global(seq_len, 1)[0])
        if glen <= 8192 + cp_rank:
            acc, m, total = combine(16)
        elif glen <= 16384 + cp_rank:
            acc, m, total = combine(32)
        else:
            acc, m, total = combine(splits)
        inv = (total > 0.0).select(ks / (total * prob_scale), fx.Float32(0.0))
        frag = fx.make_rmem_tensor(8, fx.BFloat16)
        frag.store((acc * Vec.from_elements([inv], fx.Float32).broadcast_to(8)).to(fx.BFloat16))
        dst = fx.make_view(fx.get_iter(ob) + row * 512 + lane * 8, fx.make_layout(8, 1))
        fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16), frag, dst)
        if const_expr(return_lse):
            if lane == 0:
                lb[row] = (total > 0.0).select((m + fx.math.log2(total)) * math.log(2.0), fx.Float32(float("-inf")))

    @flyc.jit
    def launch(
        q: fx.Tensor,
        kv: fx.Tensor,
        indices: fx.Tensor,
        seq_len: fx.Tensor,
        q_scale: fx.Tensor,
        kv_scale: fx.Tensor,
        partial: fx.Tensor,
        stats: fx.Tensor,
        output: fx.Tensor,
        lse: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kimi_mla_decode_split(q, kv, indices, seq_len, q_scale, kv_scale, partial, stats).launch(
            grid=(query_blocks, splits, vblocks), block=(threads, 1, 1), stream=stream
        )
        kimi_mla_decode_merge(partial, stats, kv_scale, seq_len, output, lse).launch(
            grid=(rows, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.compile_hints = {"waves_per_eu": 1}
    return launch


class KimiMLADecodePlan:
    """Kimi-K3 MLA decode for one DCP rank; reuse one plan per stream.

    Q is FP8 E4M3FN [8, 96, 576]. KV is FP8 E4M3FN [cache_slots, 576].
    The int32 index vector maps logical rank-local tokens to physical KV slots.
    global_seq_len is a device int32 [1], including the eight verify tokens.
    Its value must be >= 8 and its rank-local length must fit max_local_tokens;
    the indices covering that length must be valid slots.
    q_scale and kv_scale are device FP32 dequantization multipliers, either
    zero-dimensional or one-element vectors.

    O is BF16 [8, 96, 512]; LSE is FP32 [8, 96], in natural-log units, for the
    cross-rank DCP merge. No max-score output is produced.
    Calls allocate nothing, read no device scalars on the host, and select the
    split count on the device, so the length may change between replays of a
    captured graph. Warm up once before capture. Calls sharing a plan or
    workspace must be ordered; each call overwrites the plan's outputs.
    """

    def __init__(
        self,
        *,
        max_local_tokens: int,
        cache_slots: int,
        cp_rank: int,
        device="cuda",
        return_lse: bool = True,
        workspace=None,
    ):
        if not 0 <= cp_rank < DECODE_CP_WORLD:
            raise ValueError(f"cp_rank must be in [0, {DECODE_CP_WORLD})")
        if max_local_tokens < 1 or cache_slots < 1:
            raise ValueError("KV capacities must be positive")
        if cache_slots * DECODE_QK_DIM >= 2**31:
            raise ValueError("KV storage must fit the signed 32-bit element offsets")
        self.device = _require_gfx950(device, "Kimi MLA decode")
        self.max_local_tokens = max_local_tokens
        self.cache_slots = cache_slots
        self.return_lse = return_lse
        out_shape = (DECODE_QLEN, DECODE_HEADS, DECODE_V_DIM)
        self.output = torch.empty(out_shape, dtype=torch.bfloat16, device=self.device)
        self.lse = torch.empty(out_shape[:2], dtype=torch.float32, device=self.device) if return_lse else None
        self._unused_lse = None if return_lse else torch.empty((1,), dtype=torch.float32, device=self.device)
        partial_shape = (DECODE_MAX_SPLITS, *out_shape)
        stats_shape = (DECODE_MAX_SPLITS, DECODE_QLEN, DECODE_HEADS, 2)
        if workspace is None:
            self.partial = torch.empty(partial_shape, dtype=torch.bfloat16, device=self.device)
            self.stats = torch.empty(stats_shape, dtype=torch.float32, device=self.device)
        else:
            self.partial, self.stats = workspace
            _check_tensor(self.device, "partial workspace", self.partial, partial_shape, torch.bfloat16)
            _check_tensor(self.device, "stats workspace", self.stats, stats_shape, torch.float32)
        self._launch = _build_decode(
            max_local=max_local_tokens, cache_size=cache_slots, cp_rank=cp_rank, return_lse=return_lse
        )

    def __call__(self, q, kv, indices, global_seq_len, q_scale, kv_scale, *, output=None, lse=None, stream=None):
        _check_tensor(self.device, "q", q, (DECODE_QLEN, DECODE_HEADS, DECODE_QK_DIM), torch.float8_e4m3fn)
        _check_tensor(self.device, "kv", kv, (self.cache_slots, DECODE_QK_DIM), torch.float8_e4m3fn)
        _check_tensor(self.device, "indices", indices, (self.max_local_tokens,), torch.int32)
        _check_tensor(self.device, "global_seq_len", global_seq_len, (1,), torch.int32)
        for name, scale in (("q_scale", q_scale), ("kv_scale", kv_scale)):
            if not isinstance(scale, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if scale.numel() != 1 or scale.ndim > 1:
                raise ValueError(f"{name} must be a scalar or one-element vector")
            _check_tensor(self.device, name, scale, scale.shape, torch.float32)
        # FlyDSL tensor arguments need a stride-1 axis; these views are metadata-only.
        if q_scale.ndim == 0:
            q_scale = q_scale.view(1)
        if kv_scale.ndim == 0:
            kv_scale = kv_scale.view(1)
        output = self.output if output is None else output
        _check_tensor(self.device, "output", output, (DECODE_QLEN, DECODE_HEADS, DECODE_V_DIM), torch.bfloat16)
        if self.return_lse:
            lse = self.lse if lse is None else lse
            _check_tensor(self.device, "lse", lse, (DECODE_QLEN, DECODE_HEADS), torch.float32)
        elif lse is not None:
            raise ValueError("This plan was built with return_lse=False")
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        self._launch(
            q,
            kv,
            indices,
            global_seq_len,
            q_scale,
            kv_scale,
            self.partial,
            self.stats,
            output,
            lse if self.return_lse else self._unused_lse,
            stream=stream,
        )
        return output, lse


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


def _prefill_pack8(vec):
    a = Vec(vec).to(fx.Float32)
    words = []
    for i in fx.range_constexpr(2):
        w = fx.rocdl.cvt_pk_fp8_f32(T.i32, a[i * 4], a[i * 4 + 1], fx.Int32(0), 0)
        w = fx.rocdl.cvt_pk_fp8_f32(T.i32, a[i * 4 + 2], a[i * 4 + 3], w, 1)
        words.append(fx.Int32(w))
    return Vec.from_elements(words, fx.Int32).bitcast(fx.Float8E4M3FN)


@functools.lru_cache(maxsize=None)
def _build_prefill(s, *, bf16_input, waves, kv_tile):
    threads = waves * 64
    # Skip accumulator rescaling when no row max grows; pays off on long rows.
    conditional_rescale = s >= 32768
    tm, tn, dk, dv = (waves * 32, kv_tile, PREFILL_QK_DIM, PREFILL_V_DIM)
    heads = PREFILL_HEADS
    reduction_depth = int(math.log2(tn // 2))
    nblocks = (s + tm - 1) // tm
    tile_bytes = tn * (dk + dv)
    input_type = fx.BFloat16 if bf16_input else fx.Float8E4M3FN
    sm_scale = math.log2(math.e) / math.sqrt(dk)

    @fx.struct
    class Shared:
        kv: fx.Array[fx.BFloat16, tile_bytes, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def kimi_mla_prefill(q: fx.Tensor, k: fx.Tensor, v: fx.Tensor, o: fx.Tensor, lse: fx.Tensor):
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = (tid % 64, tid // 64)
        head = fx.Int32(fx.block_idx.x)
        # Longest causal query blocks are dispatched first.
        block = fx.Int32(nblocks - 1) - fx.Int32(fx.block_idx.y)
        qb = _global(q, s * heads * dk)
        kb = _global(k, s * heads * dk)
        vb = _global(v, s * heads * dv)
        ob = _global(o, s * heads * dv)
        lb = _global(lse, s * heads)
        lds = fx.SharedAllocator().allocate(Shared).peek()
        ptr = fx.recast_iter(fx.Float8E4M3FN, lds.kv.ptr)
        mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(32, 32, 16, fx.BFloat16)),
            fx.make_layout((1, 1, 1), (1, 1, 1)),
        )
        thr = mma.get_slice(lane)
        cc = thr.partition_C(fx.make_view(0, fx.make_layout((32, 32), (1, 0))))
        rc = thr.partition_C(fx.make_view(0, fx.make_layout((32, 32), (0, 1))))
        row = block * tm + wave * 32 + fx.Int32(fx.get_scalar(rc[0]))
        safe_row = (row < s).select(row, fx.Int32(0))
        limit = row + 1
        tile_end = ((block + 1) * tm + tn - 1) // tn
        tile_end = (tile_end < (s + tn - 1) // tn).select(tile_end, fx.Int32((s + tn - 1) // tn))
        kl = fx.make_layout(((8, tn // 8), (16, dk // 16)), ((16, 8 * dk), (1, 8 * 16)))
        vl = _value_layout(tn, dv)
        vml = fx.composition(
            fx.select(vl, [1, 0]),
            fx.make_tile(
                fx.make_layout(dv, 1),
                fx.make_layout((4, 2, 2, tn // 16), (1, 8, 4, 16)),
            ),
        )
        trans = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(8, 64), fx.Float8E4M3FN)
        cv = fx.make_tiled_copy_A(trans, mma).get_slice(lane)

        def load8(buf, offset):
            frag = fx.make_rmem_tensor(8, input_type)
            fx.copy(
                fx.make_copy_atom(fx.rocdl.BufferCopy128b() if bf16_input else fx.rocdl.BufferCopy64b(), input_type),
                fx.make_view(fx.get_iter(buf) + offset, fx.make_layout(8, 1)),
                frag,
            )
            if const_expr(bf16_input):
                return _prefill_pack8(frag.load())
            return Vec(frag.load())

        qr = []
        for step in fx.range_constexpr(3):
            words = []
            for pack in fx.range_constexpr(4):
                vals = load8(qb, (safe_row * heads + head) * dk + step * 64 + lane // 32 * 32 + pack * 8).bitcast(
                    fx.Int32
                )
                words.extend([vals[0], vals[1]])
            qr.append(Vec.from_elements(words, fx.Int32))

        def prefetch(buf, base, dim, layout):
            inv = fx.right_inverse(layout)
            # The final speculative prefetch must stay inside the allocation.
            base = (base < s).select(base, fx.Int32(0))
            frags = []
            for it in fx.range_constexpr(tn * dim // (threads * 8)):
                logical = fx.Int32(fx.get_scalar(fx.crd2idx((it * threads + tid) * 8, inv)))
                token = base + logical % tn
                frags.append(load8(buf, (token * heads + head) * dim + logical // tn))
            return frags

        def store(frags, pointer, dim):
            dst = fx.logical_divide(fx.make_view(pointer, fx.make_layout(tn * dim, 1)), fx.make_layout(8, 1))
            for it in fx.range_constexpr(tn * dim // (threads * 8)):
                frag = fx.make_rmem_tensor(8, fx.Float8E4M3FN)
                frag.store(Vec(frags[it]))
                fx.copy(
                    fx.make_copy_atom(fx.UniversalCopy64b(), fx.Float8E4M3FN),
                    frag,
                    fx.slice(dst, (None, it * threads + tid)),
                )

        kk = prefetch(kb, fx.Int32(0), dk, kl)
        vv = prefetch(vb, fx.Int32(0), dv, vl)
        store(kk, ptr, dk)
        store(vv, ptr + tn * dk, dv)
        fx.gpu.barrier()
        zero = fx.full(16, 0.0, fx.Float32)
        initial = [fx.Float32(-1e30), fx.Float32(0.0)] + [zero for _ in range(4)]

        def process(tile, args, stage):
            tile = fx.Int32(tile)
            mx, sm = (fx.Float32(args[0]), fx.Float32(args[1]))
            acc = [args[2 + d] for d in fx.range_constexpr(4)]
            kp = ptr + stage * tile_bytes
            vp = kp + tn * dk
            nxt = ptr + (1 - stage) * tile_bytes
            nk = prefetch(kb, (tile + 1) * tn, dk, kl)
            nv = prefetch(vb, (tile + 1) * tn, dv, vl)
            scores = [zero for _ in range(tn // 32)]
            for step in fx.range_constexpr(3):
                for half in fx.range_constexpr(tn // 32):
                    shared = fx.logical_divide(
                        fx.slice(fx.make_view(kp, kl), (lane % 32 + half * 32, None)),
                        fx.make_layout(16, 1),
                    )
                    words = []
                    for pack in fx.range_constexpr(2):
                        frag = fx.make_rmem_tensor(16, fx.Float8E4M3FN)
                        fx.copy(
                            fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float8E4M3FN),
                            fx.slice(shared, (None, step * 4 + lane // 32 * 2 + pack)),
                            frag,
                        )
                        w = Vec(frag.load()).bitcast(fx.Int32)
                        words.extend([w[j] for j in fx.range_constexpr(4)])
                    kr = Vec.from_elements(words, fx.Int32)
                    scores[half] = Vec(_mfma_fp8(kr, Vec(qr[step]), scores[half]))
            fx.rocdl.sched_barrier(0)
            # Write the next tile to LDS after QK to shorten register lifetimes.
            store(nk, nxt, dk)
            store(nv, nxt + tn * dk, dv)
            vals = [fx.Float32(Vec(scores[i // 16])[i % 16]) * sm_scale for i in fx.range_constexpr(tn // 2)]
            available = limit - tile * tn - fx.Int32(fx.get_scalar(cc[0]))
            # Fully visible tiles skip per-element causal comparisons.
            if tile * tn + tn > block * tm:
                for half in fx.range_constexpr(tn // 32):
                    for j in fx.range_constexpr(16):
                        visible = available > half * 32 + j // 4 * 8 + j % 4
                        vals[half * 16 + j] = visible.select(vals[half * 16 + j], fx.Float32(float("-inf")))
            levels = vals
            for d in fx.range_constexpr(reduction_depth):
                levels = [levels[i * 2].maximumf(levels[i * 2 + 1]) for i in fx.range_constexpr(tn // 4 >> d)]
            m = levels[0].maximumf(fx.Float32(fx.gpu.shuffle_xor(levels[0], 32, 64))).maximumf(mx)
            if const_expr(conditional_rescale):
                # Rescale only when a row's running maximum increases.
                rescale = fx.Uint64(fx.rocdl.ballot(T.i64, (m > mx).ir_value())) != fx.Uint64(0)
                corr = fx.Float32(1.0)
                if rescale:
                    corr = _exp(mx - m)
                    for d in fx.range_constexpr(4):
                        acc[d] = Vec(acc[d]) * Vec.from_elements([corr], fx.Float32).broadcast_to(16)
            else:
                corr = _exp(mx - m)
            # exp2(x - (m - 8)) folds the FP8 probability scale of 256 into the exponent.
            exponent_max = m - 8.0
            probs = [_exp(vals[i] - exponent_max) for i in fx.range_constexpr(tn // 2)]
            levels = probs
            for d in fx.range_constexpr(reduction_depth):
                levels = [levels[i * 2] + levels[i * 2 + 1] for i in fx.range_constexpr(tn // 4 >> d)]
            total = sm * corr + levels[0] + fx.Float32(fx.gpu.shuffle_xor(levels[0], 32, 64))
            prs = []
            for part in fx.range_constexpr(tn // 64):
                pwords = []
                for pack in fx.range_constexpr(4):
                    pv = Vec.from_elements(
                        [probs[part * 32 + pack * 8 + j] for j in fx.range_constexpr(8)],
                        fx.Float32,
                    )
                    w = _prefill_pack8(pv).bitcast(fx.Int32)
                    pwords.extend([w[0], w[1]])
                prs.append(Vec.from_elements(pwords, fx.Int32))
            value_tiles = fx.flat_divide(fx.make_view(vp, vml), (32, 16))
            cvv = Vec.from_elements([corr], fx.Float32).broadcast_to(16)
            for d in fx.range_constexpr(4):
                if const_expr(conditional_rescale):
                    scaled = Vec(acc[d])
                else:
                    scaled = Vec(acc[d]) * cvv
                for part in fx.range_constexpr(tn // 64):
                    words = []
                    for pack in fx.range_constexpr(4):
                        vt = fx.slice(value_tiles, (None, None, d, part * 4 + pack))
                        frag = fx.make_fragment_like(thr.partition_A(vt), fx.Float8E4M3FN)
                        fx.copy(trans, cv.partition_S(vt), cv.retile(frag))
                        w = Vec(frag.load()).bitcast(fx.Int32)
                        words.extend([w[0], w[1]])
                    vr = Vec.from_elements(words, fx.Int32)
                    scaled = Vec(_mfma_fp8(vr, Vec(prs[part]), scaled))
                acc[d] = scaled
            fx.gpu.barrier()
            return [m, total, *acc]

        final = initial
        for tile, args in range(fx.Int32(0), tile_end, fx.Int32(1), init=initial):
            result = process(tile, args, tile % 2)
            final = yield result
        if row < s:
            inv = fx.Float32(1.0) / (fx.Float32(final[1]))
            inv_vec = Vec.from_elements([inv], fx.Float32).broadcast_to(16)
            for d in fx.range_constexpr(4):
                _store_accumulator_bf16(
                    Vec(final[2 + d]) * inv_vec, ob, (row * heads + head) * dv + d * 32 + lane // 32 * 8
                )
            if lane < 32:
                lb[row * heads + head] = (fx.Float32(final[0]) + fx.math.log2(fx.Float32(final[1])) - 8.0) * math.log(
                    2.0
                )

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        o: fx.Tensor,
        lse: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kimi_mla_prefill(q, k, v, o, lse).launch(grid=(heads, nblocks, 1), block=(threads, 1, 1), stream=stream)

    launch.compile_hints = {"waves_per_eu": 2}
    return launch


@functools.lru_cache(maxsize=None)
def _build_prefill_convert(s):
    nq, nk, nv = (
        s * PREFILL_HEADS * PREFILL_QK_DIM,
        s * PREFILL_HEADS * PREFILL_QK_DIM,
        s * PREFILL_HEADS * PREFILL_V_DIM,
    )
    qt, kt, vt = ((nq + 2047) // 2048, (nk + 2047) // 2048, (nv + 2047) // 2048)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def kimi_mla_prefill_convert(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        qo: fx.Tensor,
        ko: fx.Tensor,
        vo: fx.Tensor,
    ):
        block = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)

        def convert(src, dst, n, b):
            offset = b * 2048 + tid * 8
            if offset < n:
                source = fx.make_view(fx.get_iter(_global(src, n)) + offset, fx.make_layout(8, 1))
                target = fx.make_view(fx.get_iter(_global(dst, n)) + offset, fx.make_layout(8, 1))
                inp = fx.make_rmem_tensor(8, fx.BFloat16)
                out = fx.make_rmem_tensor(8, fx.Float8E4M3FN)
                fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16), source, inp)
                out.store(_prefill_pack8(inp.load()))
                fx.copy(fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Float8E4M3FN), out, target)

        if block < qt:
            convert(q, qo, nq, block)
        elif block < qt + kt:
            convert(k, ko, nk, block - qt)
        else:
            convert(v, vo, nv, block - qt - kt)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        qo: fx.Tensor,
        ko: fx.Tensor,
        vo: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        kimi_mla_prefill_convert(q, k, v, qo, ko, vo).launch(
            grid=(qt + kt + vt, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


class KimiMLAPrefillPlan:
    """Dense causal Kimi-K3 prefill after KV decompression, 12 heads per rank.

    Q/K are [S, 12, 192] and V is [S, 12, 128], contiguous BF16 or FP8 E4M3FN
    as selected by input_dtype. BF16 inputs use scale-1 FP8 conversion (the
    precision path of K3's AITER FP8 prefill); BF16 calls with S <= 512
    convert inside the attention kernel, longer ones use one fused Q/K/V
    conversion launch into a preallocated (or caller-shared) FP8 workspace.
    O is BF16 [S, 12, 128] and LSE is natural-log FP32 [S, 12].

    Q length must equal KV length, a multiple of 128 in [128, 262144].
    The plan does not perform kv_b_proj, paged-cache lookup or chunked-context
    merging. Calls sharing a plan or workspace must be ordered. Warm up before
    graph capture; each call overwrites the plan's output and LSE buffers.
    """

    def __init__(self, *, seq_len: int, device="cuda", input_dtype=torch.bfloat16, workspace=None):
        if (
            not isinstance(seq_len, int)
            or not PREFILL_MIN_SEQ <= seq_len <= PREFILL_MAX_SEQ
            or seq_len % PREFILL_MIN_SEQ
        ):
            raise ValueError(f"seq_len must be a multiple of 128 in [{PREFILL_MIN_SEQ}, {PREFILL_MAX_SEQ}]")
        if input_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError("input_dtype must be BF16 or FP8 E4M3FN")
        self.device = _require_gfx950(device, "Kimi MLA prefill")
        self.seq_len = seq_len
        self.input_dtype = input_dtype
        self.output = torch.empty((seq_len, PREFILL_HEADS, PREFILL_V_DIM), device=self.device, dtype=torch.bfloat16)
        self.lse = torch.empty((seq_len, PREFILL_HEADS), device=self.device, dtype=torch.float32)
        direct_bf16 = input_dtype == torch.bfloat16 and seq_len <= 512
        self._convert = None
        self.workspace = None
        if input_dtype == torch.bfloat16 and not direct_bf16:
            shapes = self._input_shapes()
            if workspace is None:
                workspace = tuple(torch.empty(shape, device=self.device, dtype=torch.float8_e4m3fn) for shape in shapes)
            if len(workspace) != 3:
                raise ValueError("workspace must contain FP8 Q, K and V buffers")
            for name, tensor, shape in zip(("Q workspace", "K workspace", "V workspace"), workspace, shapes):
                _check_tensor(self.device, name, tensor, shape, torch.float8_e4m3fn)
            self.workspace = tuple(workspace)
            self._convert = _build_prefill_convert(seq_len)
        elif workspace is not None:
            raise ValueError("This plan does not need an FP8 conversion workspace")
        self._launch = _build_prefill(
            seq_len,
            bf16_input=direct_bf16,
            waves=4 if seq_len < 32768 else 8,
            kv_tile=64 if direct_bf16 else 128,
        )

    def _input_shapes(self):
        s = self.seq_len
        return (
            (s, PREFILL_HEADS, PREFILL_QK_DIM),
            (s, PREFILL_HEADS, PREFILL_QK_DIM),
            (s, PREFILL_HEADS, PREFILL_V_DIM),
        )

    def __call__(self, q, k, v, *, output=None, lse=None, stream=None):
        s = self.seq_len
        for name, tensor, shape in zip(("q", "k", "v"), (q, k, v), self._input_shapes()):
            _check_tensor(self.device, name, tensor, shape, self.input_dtype)
        output = self.output if output is None else output
        lse = self.lse if lse is None else lse
        _check_tensor(self.device, "output", output, (s, PREFILL_HEADS, PREFILL_V_DIM), torch.bfloat16)
        _check_tensor(self.device, "lse", lse, (s, PREFILL_HEADS), torch.float32)
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        if self._convert is not None:
            self._convert(q, k, v, *self.workspace, stream=stream)
            q, k, v = self.workspace
        self._launch(q, k, v, output, lse, stream=stream)
        return output, lse
