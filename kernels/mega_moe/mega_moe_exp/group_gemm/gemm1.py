# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Standalone grouped-MoE GEMM1 (a8w4, gate/up-interleave): out[slot]=quant_fp8(silu(gate)*up). CDNA4-only."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as _buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.fp8_gemm_utils import ceildiv

from .gemm1_util import (
    _PACK,
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    SiluQuantEpilogue,
    TileScheduler,
    wait_lds_barrier,
)


class _LdsF32View:
    """Float32 LDS view (.ptr) over the Int8 A_buf pool, for the epilogue cshuffle staging."""

    def __init__(self, ptr):
        self.ptr = ptr


def compile_moe_group_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int = 1,
    tile_m: int = 64,
    tile_n: int = 256,
    tile_k: int = 256,
    sort_block_m: int = 64,
    sched_nmajor: bool = False,
    pipe_weights: bool = True,
    mfma_amajor: bool = False,
    swizzle_a: bool = False,
    a_dtype: str = "fp8",
    out_dtype: str = "fp8",
    debug_gate: bool = False,
    use_xcd: bool = True,
    num_waves: int = 4,
    num_cu: int = 256,
    grid_mult: int = 2,
    wgm: int = 1,
):
    is_f8_a = a_dtype == "fp8"
    a_row_bytes = model_dim if is_f8_a else model_dim // 2
    assert tile_m % 16 == 0 and tile_n % 16 == 0
    assert model_dim % tile_k == 0
    NUM_WAVES = int(num_waves)
    assert tile_n % NUM_WAVES == 0
    n_per_wave = tile_n // NUM_WAVES
    assert (2 * inter_dim) % tile_n == 0, "2*inter_dim must tile evenly by tile_n"
    N_TILES = (2 * inter_dim) // tile_n
    assert grid_mult in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32), "grid_mult out of range"
    grid_x = num_cu * grid_mult
    NUM_XCDS = 8
    use_xcd_eff = use_xcd and (grid_x % NUM_XCDS == 0)  # XCD grouping needs grid.x % 8 == 0
    GROUP_SIZE = wgm * N_TILES
    BLOCKS_PER_XCD = grid_x // NUM_XCDS
    M_REPEAT = tile_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0

    TILE_K_BYTES = tile_k // 2  # fp4 packed K-step bytes (one LDS row must be %128==0)
    assert TILE_K_BYTES % 128 == 0
    A_K_STEP_BYTES = tile_k if is_f8_a else tile_k // 2
    SUB_BYTES = 256 if is_f8_a else 128  # A bytes per 256-K MFMA sub-step
    assert A_K_STEP_BYTES % SUB_BYTES == 0
    STAGE = A_K_STEP_BYTES // SUB_BYTES  # 256-K sub-steps per A refill/barrier
    SUB_I32 = SUB_BYTES // 4
    K_ITERS = model_dim // tile_k
    TOTAL_THREADS = NUM_WAVES * 64

    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    cs_size = sort_block_m * cs_tile_n
    lds_pool_bytes = max(2 * a_lds_size, cs_size * 4)  # A_buf (K-loop) aliased with C_tile (epilogue)
    n_scale_bytes = sort_block_m * (model_dim // 32)  # per-tile e8m0 A-scale staged to LDS once

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    @flyc.kernel(known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        out: fx.Tensor,
        x: fx.Tensor,
        w: fx.Tensor,
        scale_x: fx.Tensor,
        scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        out_scale: fx.Tensor,
        tokens: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        size_expert_ids: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))

        wave_id = fx.thread_idx.x // 64

        x_rsrc = _buffer_ops.create_buffer_resource(x, max_size=True)
        w_rsrc = _buffer_ops.create_buffer_resource(w, max_size=True)
        sx_rsrc = _buffer_ops.create_buffer_resource(scale_x, max_size=True)
        sw_rsrc = _buffer_ops.create_buffer_resource(scale_w, max_size=True)
        sorted_rsrc = _buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        expert_rsrc = _buffer_ops.create_buffer_resource(expert_ids, max_size=True)
        nv_rsrc = _buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
        # out/out_scale MUST be byte-bounded: padding rows store to a sentinel relying on HW OOB drop.
        _rows = size_expert_ids * fx.Int32(sort_block_m)
        out_elem_bytes = 2 if out_dtype == "bf16" else 1
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        out_nbytes = _rows * fx.Int32(inter_dim * out_elem_bytes)
        os_nbytes = _rows * fx.Int32(scale_cols) + fx.Int32(8192)
        out_rsrc = _buffer_ops.create_buffer_resource(out, max_size=False, num_records_bytes=out_nbytes)
        os_rsrc = _buffer_ops.create_buffer_resource(out_scale, max_size=False, num_records_bytes=os_nbytes)

        num_valid = _buffer_ops.buffer_load(nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32)
        num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
        total_work = num_m_tiles * fx.Int32(N_TILES)

        sched = TileScheduler(expert_rsrc=expert_rsrc, inter_dim=inter_dim, use_xcd=use_xcd_eff)
        n_wave_base = wave_id * fx.Int32(n_per_wave)

        a_gather = ATileLoader(
            x_rsrc=x_rsrc,
            row_bytes=a_row_bytes,
            sort_block_m=sort_block_m,
            k_step_bytes=A_K_STEP_BYTES,
            total_threads=TOTAL_THREADS,
            swizzle=swizzle_a,
        )
        a_s2r = AS2RLoader(is_f8_a=is_f8_a, m_repeat=M_REPEAT, k_step_bytes=A_K_STEP_BYTES, swizzle=swizzle_a)
        b_loader = BWeightLoader(w_rsrc=w_rsrc, num_acc_n=NUM_ACC_N, k_step_bytes=TILE_K_BYTES, model_dim=model_dim)
        b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=NUM_ACC_N, model_dim=model_dim)
        a_scale = AScaleLoader(
            scale_rsrc=sx_rsrc,
            m_repeat=M_REPEAT,
            model_dim=model_dim,
            sort_block_m=sort_block_m,
            total_threads=TOTAL_THREADS,
        )
        mfma = MfmaScaleGU(is_f8_a=is_f8_a, m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N)
        epi = SiluQuantEpilogue(
            out_rsrc=out_rsrc,
            out_scale_rsrc=os_rsrc,
            sorted_rsrc=sorted_rsrc,
            tokens=tokens,
            inter_dim=inter_dim,
            m_repeat=M_REPEAT,
            num_acc_n=NUM_ACC_N,
            sort_block_m=sort_block_m,
            tile_m=sort_block_m,
            tile_n=tile_n,
            num_waves=NUM_WAVES,
            lds_out=c_tile,
            out_dtype=out_dtype,
            debug_gate=debug_gate,
        )

        # Flat 1-D persistent loop over (m_tile, n_tile); XCD mode keeps a wgm*N_TILES super-group on one XCD.
        if const_expr(use_xcd_eff):
            it0 = fx.Int32(fx.block_idx.x) // fx.Int32(NUM_XCDS)
            it_stride = fx.Int32(BLOCKS_PER_XCD)
            xcd_id = fx.Int32(fx.block_idx.x) % fx.Int32(NUM_XCDS)
        else:
            it0 = fx.Int32(fx.block_idx.x)
            it_stride = fx.Int32(grid_x)
            xcd_id = fx.Int32(0)

        def _it_to_flat(it_v):
            if const_expr(use_xcd_eff):
                gsz = fx.Int32(GROUP_SIZE)
                gi = it_v // gsz
                within = it_v - gi * gsz
                sg = xcd_id + gi * fx.Int32(NUM_XCDS)
                return sg * gsz + within
            return it_v

        # One (m_tile, n_tile) unit.  Kept as a nested fn CALLED BY NAME from the while body so the atom
        # objects pass as plain args (not method-call receivers) -> the while's carried-var analysis only
        # threads `itv`, not the atoms.  range(init=) K-loop works here (nested body is AST-rewritten).
        def _do_tile(
            m_tile,
            n_tile_base,
            sched,
            a_gather,
            a_s2r,
            b_loader,
            b_scale,
            a_scale,
            mfma,
            epi,
            a_buf,
            a_scale_lds,
            a_lds_i32,
            K_ITERS,
            STAGE,
            SUB_I32,
            M_REPEAT,
            NUM_ACC_N,
            A_K_STEP_BYTES,
            sort_block_m,
            pipe_weights,
            mfma_amajor,
        ):
            N_ACC = M_REPEAT * NUM_ACC_N
            last = fx.Int32(K_ITERS - 1)
            tile_row_base = m_tile * fx.Int32(sort_block_m)
            expert = sched.expert_of(m_tile)
            b_row = sched.gate_base_row(expert) + n_tile_base
            a_gather.for_tile(tile_row_base)
            a_gather.store(a_buf, a_gather.load_regs(fx.Int32(0)), fx.Int32(0))
            a_scale.stage(a_scale_lds, tile_row_base)
            wait_lds_barrier()
            if pipe_weights:
                # Pipelined weights: carry b_prev + acc; issue next-step B loads into this step's MFMA.
                assert STAGE == 1, "pipelined weight prefetch assumes STAGE==1 (fp8-A, tile_k=256)"
                b0 = b_loader.load_step(b_row, fx.Int32(0))
                init = [mfma.zero_value for _ in range(N_ACC)]
                init += [h for ni_list in b0 for h in ni_list]
                for sp_i, state in range(0, K_ITERS, 1, init=init):
                    sp = fx.Int32(sp_i)
                    acc = [Vec(a) for a in state[:N_ACC]]
                    b_prev = [[Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)]
                    cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
                    nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
                    spn = (sp + fx.Int32(1) < last).select(sp + fx.Int32(1), last)
                    a_regs = a_gather.load_regs(spn * fx.Int32(A_K_STEP_BYTES))

                    def a_load(mi, ks, _base=cur_off):
                        return a_s2r.load_operand(a_buf, mi, ks, _base)

                    sa = a_scale.load_step(a_scale_lds, sp)
                    sb = b_scale.load_step(b_row, sp)

                    def load_next(ni, _kn=spn):
                        return b_loader.load_ni(b_row, ni, _kn)

                    call_pipe = mfma.call_pipe_am if mfma_amajor else mfma.call_pipe
                    acc, b_next = call_pipe(a_load, b_prev, acc, sa, sb, load_next)
                    a_gather.store(a_buf, a_regs, nxt_off)
                    wait_lds_barrier()
                    yv = list(acc) + [h for ni_list in b_next for h in ni_list]
                    state = yield yv
                acc = [Vec(r) for r in state[:N_ACC]]
                epi.store(acc, m_tile, tile_row_base, n_tile_base)
            else:
                # Plain rolled K-loop: weights loaded in-body, consumed immediately (lower B-reg pressure).
                init = [mfma.zero_value for _ in range(N_ACC)]
                for sp_i, state in range(0, K_ITERS, 1, init=init):
                    sp = fx.Int32(sp_i)
                    acc = [Vec(a) for a in state]
                    cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
                    nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
                    spn = (sp + fx.Int32(1) < last).select(sp + fx.Int32(1), last)
                    a_regs = a_gather.load_regs(spn * fx.Int32(A_K_STEP_BYTES))
                    for s in range_constexpr(STAGE):
                        kk = sp * fx.Int32(STAGE) + fx.Int32(s)
                        a_base = cur_off + fx.Int32(s * SUB_I32)

                        def a_load(mi, ks, _base=a_base):
                            return a_s2r.load_operand(a_buf, mi, ks, _base)

                        b = b_loader.load_step(b_row, kk)
                        sa = a_scale.load_step(a_scale_lds, kk)
                        sb = b_scale.load_step(b_row, kk)
                        acc = mfma.call(a_load, b, acc, sa, sb)
                    a_gather.store(a_buf, a_regs, nxt_off)
                    wait_lds_barrier()
                    state = yield list(acc)
                acc = [Vec(r) for r in state[:N_ACC]]
                epi.store(acc, m_tile, tile_row_base, n_tile_base)

        # Persistent grid-stride loop over dynamic (runtime) work count (high-level device while loop).
        itv = it0
        while _it_to_flat(itv) < total_work:
            flat = _it_to_flat(itv)
            if const_expr(sched_nmajor):  # n_tile-major: consecutive flat ids share the B column (L2-hot)
                n_tile = flat // num_m_tiles
                m_tile = flat - n_tile * num_m_tiles
            else:
                m_tile = flat // fx.Int32(N_TILES)
                n_tile = flat - m_tile * fx.Int32(N_TILES)
            n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
            _do_tile(
                m_tile,
                n_tile_base,
                sched,
                a_gather,
                a_s2r,
                b_loader,
                b_scale,
                a_scale,
                mfma,
                epi,
                a_buf,
                a_scale_lds,
                a_lds_i32,
                K_ITERS,
                STAGE,
                SUB_I32,
                M_REPEAT,
                NUM_ACC_N,
                A_K_STEP_BYTES,
                sort_block_m,
                pipe_weights,
                mfma_amajor,
            )
            itv = itv + it_stride

    @flyc.jit
    def launch(
        out: fx.Tensor,
        x: fx.Tensor,
        w: fx.Tensor,
        scale_x: fx.Tensor,
        scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        out_scale: fx.Tensor,
        tokens: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        size_expert_ids: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            out,
            x,
            w,
            scale_x,
            scale_w,
            sorted_token_ids,
            expert_ids,
            num_valid_ids,
            out_scale,
            tokens,
            n,
            k,
            size_expert_ids,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}"},
        ).launch(grid=(grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch
