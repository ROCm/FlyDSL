# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.common import buffer_ops
from kernels.common.layout_utils import crd2idx

from .gemm1 import _a16w4_swizzle_xor16, _e8m0_byte_to_f32
from .mxfp4_gemm_common import (
    _a_lds_swz_block_idx,
    _a_lds_swz_block_layout,
    _buffer_rsrc,
    _e8m0_from_amax,
    _fabs_f32,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    _inline_dpp_quad_amax,
    _lds_ptr3,
    _lds_swizzle_mask,
    _raw,
    _scale_mma_atoms,
    _udiv,
    _umod,
    bq_bytes_for,
    bscale_bytes_for,
    k_half_for,
    k_tiles_total_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kBS_stride_k0_dw,
    kbs_stride_n0_dw_for,
    kmchunks_for,
    kStages,
    kunroll_for,
    lds_acc_bytes_for,
    num_n_blocks_for,
)

NUM_CU = 256


def aq_bytes_for(max_m, k):
    return max_m * k_half_for(k)


def saq_slot_bytes(BM, KH_TILE):
    return BM * KH_TILE


def tiling(BM):
    n_load_waves = min(4, BM // 8)
    rows_per_wave = BM // n_load_waves
    return n_load_waves, rows_per_wave, rows_per_wave // 8


def _issue_a_load_lds(aq_dma_tiles4, s_aq_i32x4_tiles, slot, kt, car, lane, slot_bytes, lds_row, KH_TILE, k_half):
    # A global->LDS async DMA (no register fragment), via BufferCopyLDS128b. Mirrors
    # gemm1's issue_a_load_lds: the BufferCopyLDS atom's soffset is an element count.
    lane_mod_8 = lane % fx.Int32(8)
    mask = _lds_swizzle_mask(lds_row + (lane // fx.Int32(8)))
    voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + car * fx.Int32(k_half)
    off_i32 = fx.Int32(slot * slot_bytes) + lds_row * fx.Int32(KH_TILE)
    aq_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
    fx.copy(
        aq_dma_atom,
        fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
        fx.slice(s_aq_i32x4_tiles, (None, off_i32 // fx.Int32(16))),
        soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
    )


def compile_gemm2_a4w4_port(
    BM=32,
    use_nt=False,
    *,
    NE,
    N_OUT,
    epilog="atomic",
    D_INTER,
    D_INTER_REAL=None,
    BN=256,
    BK=256,
    xcd_swizzle=0,
):
    assert BN == 256 and BK == 256, f"only BN==BK==256 supported, got BN={BN} BK={BK}"
    KH_TILE = BK // 2
    _K = D_INTER
    _K_REAL = D_INTER if D_INTER_REAL is None else D_INTER_REAL
    assert _K % BK == 0, (
        f"D_INTER (gemm2 contraction K = inter_dim) must be a multiple of {BK}, "
        f"got {_K}; inter_dim not divisible by {BK} (e.g. 384/192) is not "
        f"supported by this BK={BK} kernel"
    )
    assert (
        _K_REAL % 128 == 0 and 0 < _K_REAL <= _K
    ), f"D_INTER_REAL={_K_REAL} must be a multiple of 128 and in (0, {_K}]"
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _persistent = epilog in ("nonatomic", "nonatomic_mxfp4")
    _slot_bytes = saq_slot_bytes(BM, KH_TILE)
    _aStages = kStages if _K_TILES_TOTAL <= kStages else 3
    _acc_rows = min(BM, 64) if epilog == "nonatomic_cshuffle" else BM
    _lds_bytes = (
        lds_acc_bytes_for(_acc_rows, BN) + _aStages * _slot_bytes if epilog != "nonatomic" else _aStages * _slot_bytes
    )
    _num_n_blocks = num_n_blocks_for(N_OUT, BN)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(BM)
    _epi_tag = {
        "atomic": "atomic",
        "nonatomic": "nonatomic",
        "nonatomic_mxfp4": "nonatomic_mxfp4",
        "nonatomic_cshuffle": "nonatomic_cshuffle",
    }[epilog]
    _rtag = "" if _K_REAL == _K else f"r{_K_REAL}"
    _tag = f"ne{NE}_h{N_OUT}_i{_K}{_rtag}_bm{BM}{'_nt' if use_nt else ''}_{_epi_tag}"
    if xcd_swizzle > 0:
        _tag += f"_xcd{xcd_swizzle}"
    _name = f"gemm2_a4w4_port_{_tag}"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, _lds_bytes, 16]

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)

        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        _aq_num_bytes = fx.Int64(i32_max_m_blocks) * fx.Int64(BM * _K_HALF)
        aq_dma_tiles4 = _global_i32_buffer_tiles(arg_aq, _aq_num_bytes, 4)
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        # s_aq as flat i32, divided into 4-element (128-bit) tiles for the LDS DMA dst.
        s_aq_i32_flat = fx.make_view(
            fx.recast_iter(fx.Int32, lds_raw_ptr),
            fx.make_layout(kStages * _slot_bytes // 4, 1),
        )
        s_aq_i32x4_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(4, 1))

        def _issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):
                for sub in range_constexpr(_kSubBlocks):
                    lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
                    car = m_row0 + lds_row + (lane // fx.Int32(8))
                    _issue_a_load_lds(
                        aq_dma_tiles4,
                        s_aq_i32x4_tiles,
                        slot,
                        slot,
                        car,
                        lane,
                        _slot_bytes,
                        lds_row,
                        KH_TILE=KH_TILE,
                        k_half=_K_HALF,
                    )

        def _run_tile(tile_i32):
            _gemm2_body(
                lds_raw_ptr,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                i32_M,
                i32_max_m_blocks,
                arg_out,
                arg_out_scale,
                tile_i32,
                lane,
                wave,
                BM,
                use_nt,
                NE,
                N_OUT,
                epilog,
                D_INTER=_K,
                D_INTER_REAL=_K_REAL,
                aStages=_aStages,
                BN=BN,
                BK=BK,
                KH_TILE=KH_TILE,
            )

        if const_expr(_persistent):
            cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
            total_m_blocks = _udiv(cumsum0, BM)
            bound = total_m_blocks * fx.Int32(_num_n_blocks)
            grid_nb = fx.Int32(gpu.grid_dim.x)

            _NXCD = 8
            _xq = _udiv(bound, _NXCD)
            _xr = _umod(bound, _NXCD)
            _SW = xcd_swizzle

            def _xcd(pid):
                xc = _umod(pid, _NXCD)
                wgid = xc * _xq + fx.Int32(arith.minsi(_raw(xc), _raw(_xr))) + _udiv(pid, _NXCD)
                if const_expr(_SW <= 0):
                    return wgid
                _ng = fx.Int32(_SW * _num_n_blocks)
                group_id = wgid // _ng
                first_pid_m = group_id * fx.Int32(_SW)
                remaining_m = total_m_blocks - first_pid_m
                group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
                wig = wgid % _ng
                m_block = first_pid_m + (wig % group_size_m)
                n_block = wig // group_size_m
                return m_block * fx.Int32(_num_n_blocks) + n_block

            if bx_i32 < bound:
                tile = _xcd(bx_i32)
                _issue_all_a_loads(_udiv(tile, _num_n_blocks) * fx.Int32(BM))
                rocdl.sched_barrier(0)
                _run_tile(tile)

            for iv in range(bx_i32 + grid_nb, bound, gpu.grid_dim.x):
                wu = fx.Int32(iv)
                gpu.barrier()
                tile = _xcd(wu)
                _issue_all_a_loads(_udiv(tile, _num_n_blocks) * fx.Int32(BM))
                _run_tile(tile)
        else:
            cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
            total_m_blocks = _udiv(cumsum0, BM)
            bound = total_m_blocks * fx.Int32(_num_n_blocks)

            # Non-persistent atomic path is HBM-bandwidth-bound (down-proj reads the
            # full fp4 weight column-block per tile, ~4% L2 reuse). A plain m-major
            # linear grid clusters consecutive tiles onto the same XCD/HBM channels;
            # round-robin the launch index across the 8 XCDs (bijective over [0,bound))
            # to balance channel utilization. Optional group swizzle (xcd_swizzle>0)
            # further improves per-XCD L2 locality along M.
            _NXCD = 8
            _xq = _udiv(bound, _NXCD)
            _xr = _umod(bound, _NXCD)
            _SW = xcd_swizzle

            def _xcd_np(pid):
                xc = _umod(pid, _NXCD)
                wgid = xc * _xq + fx.Int32(arith.minsi(_raw(xc), _raw(_xr))) + _udiv(pid, _NXCD)
                if const_expr(_SW <= 0):
                    return wgid
                _ng = fx.Int32(_SW * _num_n_blocks)
                group_id = wgid // _ng
                first_pid_m = group_id * fx.Int32(_SW)
                remaining_m = total_m_blocks - first_pid_m
                group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
                wig = wgid % _ng
                m_block = first_pid_m + (wig % group_size_m)
                n_block = wig // group_size_m
                return m_block * fx.Int32(_num_n_blocks) + n_block

            if bx_i32 < bound:
                tile = _xcd_np(bx_i32)
                m_row0 = _udiv(tile, _num_n_blocks) * fx.Int32(BM)
                if const_expr(_n_load_waves < 4):
                    if wave < fx.Int32(_n_load_waves):
                        _issue_all_a_loads(m_row0)
                else:
                    _issue_all_a_loads(m_row0)
                rocdl.sched_barrier(0)
                _run_tile(tile)

    @flyc.jit
    def launch_gemm2(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        if const_expr(_persistent):
            tw = i32_max_m_blocks * fx.Int32(_num_n_blocks)
            persist = _raw(tw > fx.Int32(NUM_CU * 4))
            grid_i32 = arith.select(persist, _raw(fx.Int32(NUM_CU)), _raw(tw))
            grid_x = arith.index_cast(T.index, grid_i32)
        else:
            grid_x = arith.index_cast(T.index, i32_max_m_blocks) * fx.Index(_num_n_blocks)
        gemm2_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            arg_out_scale,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    if BM == 16:
        launch_gemm2.compile_hints["llvm_options"] = {"enable-post-misched": False}

    return launch_gemm2


@flyc.jit
def _gemm2_body(
    lds_raw_ptr,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    i32_M,
    i32_max_m_blocks,
    arg_out,
    arg_out_scale,
    bx_i32,
    lane,
    wave,
    BM,
    use_nt,
    NE,
    N_OUT,
    epilog,
    *,
    D_INTER,
    D_INTER_REAL=None,
    aStages=kStages,
    BN,
    BK,
    KH_TILE,
):
    _aStages = aStages
    _kMChunks = kmchunks_for(BM)
    _slot_bytes = saq_slot_bytes(BM, KH_TILE)
    _K = D_INTER
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _K_REAL = D_INTER if D_INTER_REAL is None else D_INTER_REAL
    _n_real_half = (_K_REAL + 127) // 128
    _kUnroll = kunroll_for(_K, BK)
    _kAS_per_chunk_dw = kas_per_chunk_dw_for(_K)
    _kBS_stride_n0_dw = kbs_stride_n0_dw_for(_K)
    _asc_chunk_div = 16 if const_expr(BM == 16) else 32
    _asc_per_mb = (BM // _asc_chunk_div) * _kAS_per_chunk_dw * 4
    _bq_bytes = bq_bytes_for(NE, N_OUT, _K)
    _bscale_bytes = bscale_bytes_for(NE, N_OUT, _K)
    _kbs_per_expert_dw = kbs_per_expert_dw_for(N_OUT, _K)
    _num_n_blocks = num_n_blocks_for(N_OUT, BN)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(BM)
    b_aux = 2 if use_nt else 0

    m_block_idx = _udiv(bx_i32, _num_n_blocks)
    n_block_idx = bx_i32 - m_block_idx * fx.Int32(_num_n_blocks)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)

    _asc_num_bytes = fx.Int64(i32_max_m_blocks) * fx.Int64(_asc_per_mb)
    ascale_tiles = _global_i32_buffer_tiles(arg_ascale, _asc_num_bytes, 1)
    bq_tiles = _global_i32_buffer_tiles(arg_bq, _bq_bytes, 4)
    bscale_tiles = _global_i32_buffer_tiles(arg_bscale, _bscale_bytes, 1)
    ascale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    bscale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    bq_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_aux), fx.Int32)
    bq_reg_lay = fx.make_layout(4, 1)
    scalar_reg_lay = fx.make_layout(1, 1)

    # Sequential LDS layout: saq bytes at offset 0, f32 accumulator after them.
    saq_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
    lds_acc_base_i32 = saq_base_i32 + fx.Int32(_aStages * _slot_bytes)

    # A global->LDS DMA source (buffer tensor) + s_aq LDS dst tiles (flat i32, 128-bit).
    _aq_num_bytes = fx.Int64(i32_max_m_blocks) * fx.Int64(BM * _K_HALF)
    aq_dma_tiles4 = _global_i32_buffer_tiles(arg_aq, _aq_num_bytes, 4)
    s_aq_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(_aStages * _slot_bytes // 4, 1),
    )
    s_aq_i32x4_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(4, 1))
    lds_a_read_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    lds_a_read_lay = fx.make_layout(4, 1)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    b_load_s_base = []
    for j in range_constexpr(4):
        v = (e * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)) * fx.Int32(
            _K_HALF
        )
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    mni_base = n_block_idx * fx.Int32(BN // 16 // 2) + wave * fx.Int32(BN // 64 // 2)
    b_scale_s_base = []
    for mw in range_constexpr(2):
        v = (e * fx.Int32(_kbs_per_expert_dw) + (mni_base + fx.Int32(mw)) * fx.Int32(_kBS_stride_n0_dw)) * fx.Int32(4)
        b_scale_s_base.append(rocdl.readfirstlane(T.i32, v))

    chunk_base = m_row // fx.Int32(16 if const_expr(BM == 16) else 32)
    a_scale_s_base = [
        rocdl.readfirstlane(
            T.i32,
            (chunk_base + fx.Int32(sub)) * fx.Int32(_kAS_per_chunk_dw) * fx.Int32(4),
        )
        for sub in range_constexpr(_kSubBlocks)
    ]

    v_voff_scale = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)

    def load_a_scale_tile(kt):
        out = [None] * _kSubBlocks
        for sub in range_constexpr(_kSubBlocks):
            idx = (v_voff_scale + fx.Int32(kt * 256)) // fx.Int32(4)
            r = fx.make_rmem_tensor(scalar_reg_lay, fx.Int32)
            fx.copy(
                ascale_copy_atom,
                fx.slice(ascale_tiles, (None, idx)),
                r,
                soffset=a_scale_s_base[sub] // fx.Int32(4),
            )
            out[sub] = r.load()[0]
        return out

    def load_b_scale_tile(kt):
        imm = kt * (kBS_stride_k0_dw * 4)
        out = [None, None]
        for mw in range_constexpr(2):
            idx = (v_voff_scale + fx.Int32(imm)) // fx.Int32(4)
            r = fx.make_rmem_tensor(scalar_reg_lay, fx.Int32)
            fx.copy(
                bscale_copy_atom,
                fx.slice(bscale_tiles, (None, idx)),
                r,
                soffset=b_scale_s_base[mw] // fx.Int32(4),
            )
            out[mw] = r.load()[0]
        return out

    def load_b_tile(kt):
        v_voff_b = (lane_div_16 * fx.Int32(256)) + (lane_mod_16 * fx.Int32(16)) + fx.Int32(kt * 2048)
        out = [[None, None] for _ in range(4)]
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                if const_expr(kt * 2 + half >= _n_real_half):
                    continue
                idx = (v_voff_b + fx.Int32(half * 1024)) // fx.Int32(16)
                r = fx.make_rmem_tensor(bq_reg_lay, fx.Int32)
                fx.copy(
                    bq_copy_atom,
                    fx.slice(bq_tiles, (None, idx)),
                    r,
                    soffset=b_load_s_base[j] // fx.Int32(4),
                )
                out[j][half] = r
        return out

    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(_kSubBlocks):
            lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
            car = m_row + lds_row + (lane // fx.Int32(8))
            _issue_a_load_lds(
                aq_dma_tiles4,
                s_aq_i32x4_tiles,
                slot,
                kt,
                car,
                lane,
                _slot_bytes,
                lds_row,
                KH_TILE=KH_TILE,
                k_half=_K_HALF,
            )

    # A-LDS read block-swizzle via layout algebra (crd2idx over composed swizzle).
    _a_lds_swz = _a_lds_swz_block_layout(_aStages * BM)

    def issue_a_ds_read(slot):
        lane_row = lane_mod_16
        lane_col_block = lane_div_16  # 16-byte block column within the 128-byte row
        a = [[None, None] for _ in range(_kMChunks)]
        for k in range_constexpr(2):
            block_col = lane_col_block + fx.Int32(k * 4)  # +64 bytes == +4 blocks
            for i in range_constexpr(_kMChunks):
                global_row = fx.Int32(slot * BM) + lane_row + fx.Int32(i * 16)
                block_idx = _a_lds_swz_block_idx(_a_lds_swz, global_row, block_col)
                r = fx.make_rmem_tensor(lds_a_read_lay, fx.Int32)
                fx.copy_atom_call(lds_a_read_atom, fx.slice(s_aq_i32x4_tiles, (None, block_idx)), r)
                a[i][k] = r
        return a

    zero4 = Vec.filled(4, 0.0, fx.Float32)
    # Scaled down-proj MMA via fx.gemm + CDNA4 MFMA_Scale atoms (fp4 x fp4, e8m0
    # scales). opsel_a/opsel_b select the active 128-K half of the shared operand;
    # scale_a/scale_b carry the e8m0 words. Perf-neutral vs the raw
    # mfma_scale_f32_16x16x128_f8f6f4 intrinsic on gfx950.
    scale_atoms = _scale_mma_atoms("fp4")
    # accm[i][J] holds the running f32[4] accumulator as an rmem tensor.
    accm = [[None, None, None, None] for _ in range(_kMChunks)]

    def _mma(atom, cf, a_frag, b_frag, sa, sb):
        fx.gemm(atom, cf, a_frag, b_frag, cf, scale_a=sa, scale_b=sb)

    def mfma_cluster(b_tile, a, a_scale_sub, b_scale_slot, init, kt=0):
        _skip_h1 = (kt * 2 + 1) >= _n_real_half
        for J in range_constexpr(4):
            mni = J // 2
            in_b = J % 2
            sb = b_scale_slot[mni]
            b_J0 = b_tile[J][0]
            b_J1 = None if const_expr(_skip_h1) else b_tile[J][1]
            for sub in range_constexpr(_kSubBlocks):
                sa = a_scale_sub[sub]
                i0 = sub * 2
                i1 = sub * 2 + 1
                if const_expr(init):
                    accm[i0][J] = fx.make_rmem_tensor(4, fx.Float32)
                    accm[i0][J].store(zero4)
                    if const_expr(_kMChunks > 1):
                        accm[i1][J] = fx.make_rmem_tensor(4, fx.Float32)
                        accm[i1][J].store(zero4)
                _mma(scale_atoms[(0, 0 + in_b)], accm[i0][J], a[i0][0], b_J0, sa, sb)
                if const_expr(_kMChunks > 1):
                    _mma(scale_atoms[(1, 0 + in_b)], accm[i1][J], a[i1][0], b_J0, sa, sb)
                if const_expr(not _skip_h1):
                    _mma(scale_atoms[(2, 2 + in_b)], accm[i0][J], a[i0][1], b_J1, sa, sb)
                    if const_expr(_kMChunks > 1):
                        _mma(scale_atoms[(3, 2 + in_b)], accm[i1][J], a[i1][1], b_J1, sa, sb)

    def _kloop_fence():
        gpu.barrier()

    if const_expr(_K_TILES_TOTAL <= kStages):
        a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b_scale_v = [load_b_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b = [load_b_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        for S in range_constexpr(_K_TILES_TOTAL):
            kt = S
            slot = kt % kStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b[slot], a, a_scale_sub, b_scale_v[slot], init=(S == 0), kt=kt)
    else:
        a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b_scale_v = [load_b_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        # Software-pipeline the B tiles instead of preloading all _K_TILES_TOTAL of
        # them: preloading all K tiles keeps every B fragment live across the whole
        # k-loop (>=384 VGPR for K=3072/BK=256), which forces the f32 accumulators
        # into AGPRs and drops occupancy to 1 wave/SIMD. Keeping only _bPF B tiles
        # resident (one-ahead prefetch) lets the accumulators stay in ArchVGPRs and
        # restores 2 waves/SIMD. A stays LDS-double-buffered as before.
        _bPF = 2
        b_pf = [load_b_tile(kt) for kt in range_constexpr(_bPF)]

        for OFFSET in range_constexpr(_kUnroll):
            kt = OFFSET
            slot = kt % _aStages
            next_kt = kStages + OFFSET
            write_slot = next_kt % _aStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            issue_a_load_lds(write_slot, next_kt)
            # Prefetch the B tile _bPF iterations ahead so at most _bPF B tiles are
            # live at once.
            b_cur = b_pf[kt % _bPF]
            b_next_kt = kt + _bPF
            if const_expr(b_next_kt < _K_TILES_TOTAL):
                b_pf[kt % _bPF] = load_b_tile(b_next_kt)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b_cur, a, a_scale_sub, b_scale_v[kt], init=(OFFSET == 0))

        for S in range_constexpr(kStages):
            kt = _K_TILES_TOTAL - kStages + S
            slot = kt % _aStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            b_cur = b_pf[kt % _bPF]
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b_cur, a, a_scale_sub, b_scale_v[kt], init=False)

    # Materialize the f32[4] accumulators as raw vector values for the (raw) epilogs.
    accm = [[accm[i][J].load().ir_value() for J in range(4)] for i in range(_kMChunks)]

    if epilog == "nonatomic":
        out_base = _global_base_ptr1(arg_out)
        _flat_bf16_epilog(accm, out_base, m_row, n_block_idx, wave, lane, N_OUT, BN, _kMChunks)
    elif epilog == "nonatomic_cshuffle":
        _cshuffle_flat_bf16_epilog(
            lds_acc_base_i32,
            accm,
            arg_out,
            m_row,
            n_block_idx,
            wave,
            lane,
            BM,
            N_OUT,
            BN,
        )
    elif epilog == "nonatomic_mxfp4":
        out_q_base = _global_base_ptr1(arg_out)
        out_scale_base = _global_base_ptr1(arg_out_scale)
        tid_i32 = fx.Int32(gpu.thread_id("x"))
        _flat_mxfp4_epilog(
            accm,
            out_q_base,
            out_scale_base,
            m_row,
            n_block_idx,
            wave,
            lane,
            tid_i32,
            N_OUT,
            BN,
            lds_acc_base_i32,
            _kMChunks,
        )
    else:
        _atomic_bf16_epilog(
            lds_acc_base_i32,
            accm,
            arg_out,
            arg_stids,
            arg_sweights,
            m_row,
            n_block_idx,
            wave,
            lane,
            i32_M,
            BM,
            N_OUT,
            BN,
        )


def _flat_bf16_epilog(accm, out_base, m_row, n_block_idx, wave, lane, N_OUT, BN, kMChunks):
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    row_base = m_row + lane_div_16 * fx.Int32(4)
    gn_base = n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + lane_mod_16
    byte_base = (fx.Int64(row_base) * fx.Int64(N_OUT) + fx.Int64(gn_base)) * fx.Int64(2)
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(4):
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                const_off = ((i * 16 + v) * N_OUT + J * 16) * 2
                bf = Vec.from_elements([vec[v]], fx.Float32).to(fx.BFloat16)
                llvm.StoreOp(_raw(bf), _gep1(out_base, byte_base + fx.Int64(const_off)))


def _cshuffle_flat_bf16_epilog(lds_acc_base_i32, accm, arg_out, m_row, n_block_idx, wave, lane, BM, N_OUT, BN):
    _iC = BM // 16
    _REPS = BM // 8
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    out_base = _global_base_ptr1(arg_out)

    for i in range_constexpr(_iC):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            bf4 = Vec(accm[i][J]).to(fx.BFloat16)
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(bf4[v]), _gep3(lds_base, idx * fx.Int32(2)))
    gpu.barrier()
    for mr in range_constexpr(_REPS):
        row_local = fx.Int32(mr * 8) + m_lane
        sorted_row = m_row + row_local
        for s in range_constexpr(4):
            idx0 = row_local * fx.Int32(BN) + col_start + fx.Int32(s * 64)
            pk = Vec(llvm.load(T.vec(2, T.bf16), _gep3(lds_base, idx0 * fx.Int32(2))))
            n_col = n_block_idx * fx.Int32(BN) + col_start + fx.Int32(s * 64)
            elem = fx.Int64(sorted_row) * fx.Int64(N_OUT) + fx.Int64(n_col)
            llvm.StoreOp(_raw(pk), _gep1(out_base, elem * fx.Int64(2)))


@flyc.jit
def _flat_mxfp4_epilog(
    accm,
    out_q_base,
    out_scale_base,
    m_row,
    n_block_idx,
    wave,
    lane,
    tid_i32,
    N_OUT,
    BN,
    lds_acc_base_i32,
    kMChunks,
):
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(BN // 4) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))
    gpu.barrier()

    NBLK = BN // 32
    m_lane = tid_i32 // fx.Int32(16)
    n_lane = tid_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)
    _m_base = m_row + m_lane
    _q_row0 = fx.Int64(_m_base) * fx.Int64(N_OUT // 2)
    _s_row0 = fx.Int64(_m_base) * fx.Int64(N_OUT // 32)
    _blocks = [(mr, half) for mr in range(kMChunks) for half in range(NBLK // 4)]

    def _issue_load(mr, half):
        row_local = fx.Int32(mr * 16) + m_lane
        group = wave_grp + fx.Int32(half * 4)
        col0 = group * fx.Int32(32) + kk * fx.Int32(8)
        base_idx = row_local * fx.Int32(BN) + col0
        v0 = Vec(llvm.load(T.vec(4, T.f32), _gep3(lds_base, base_idx * fx.Int32(4))))
        v1 = Vec(
            llvm.load(
                T.vec(4, T.f32),
                _gep3(lds_base, (base_idx + fx.Int32(4)) * fx.Int32(4)),
            )
        )
        return [v0[0], v0[1], v0[2], v0[3], v1[0], v1[1], v1[2], v1[3]], group, col0

    _r_next, _grp_next, _col0_next = _issue_load(*_blocks[0])
    for _bi in range_constexpr(len(_blocks)):
        mr, half = _blocks[_bi]
        r, group, col0 = _r_next, _grp_next, _col0_next
        if _bi + 1 < len(_blocks):
            _r_next, _grp_next, _col0_next = _issue_load(*_blocks[_bi + 1])
        if True:
            amax_f = _raw(_fabs_f32(r[0]))
            for e in range_constexpr(1, 8):
                abs_e = _raw(_fabs_f32(r[e]))
                amax_f = arith.maxnumf(amax_f, abs_e)
            amax = arith.shrui(arith.bitcast(T.i32, amax_f), _raw(fx.Int32(16)))
            amax_dpp = _raw(_inline_dpp_quad_amax(amax))
            f32b = arith.shli(amax_dpp, _raw(fx.Int32(16)))
            e8m0, qscale_f = _e8m0_from_amax(fx.Float32(arith.bitcast(T.f32, f32b)))
            e8 = _raw(e8m0)
            qscale = _raw(qscale_f)
            packed = _raw(fx.Int32(0))
            packed = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, packed, _raw(r[0]), _raw(r[1]), qscale, 0)
            packed = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, packed, _raw(r[2]), _raw(r[3]), qscale, 1)
            packed = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, packed, _raw(r[4]), _raw(r[5]), qscale, 2)
            packed = rocdl.cvt_scalef32_pk_fp4_f32(T.i32, packed, _raw(r[6]), _raw(r[7]), qscale, 3)
            global_col = n_block_idx * fx.Int32(BN) + col0
            blk = n_block_idx * fx.Int32(NBLK) + group
            q_byte = _q_row0 + fx.Int64(mr * 16 * (N_OUT // 2)) + fx.Int64(global_col // fx.Int32(2))
            s_byte = _s_row0 + fx.Int64(mr * 16 * (N_OUT // 32)) + fx.Int64(blk)
            llvm.StoreOp(packed, _gep1(out_q_base, q_byte), nontemporal=True)
            if kk == fx.Int32(0):
                llvm.StoreOp(arith.trunci(T.i8, e8), _gep1(out_scale_base, s_byte))


@flyc.jit
def _atomic_bf16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    BN,
):
    _kMChunks = kmchunks_for(BM)
    M_REPS = BM // 8
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(llvm.load(T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True))
        weight.append(llvm.load(T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True))

    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            for s in range_constexpr(4):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4))))
                pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
                out_ptr = _gep1(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )


# =============================================================================
# a16w4 (bf16 intermediate A x mxfp4 W2 -> bf16 out) down-projection stage2.
# Dedicated path (separate from the a4w4/a8w4 scaled body above; that code is
# byte-identical). Reuses the M1-proven a16w4 primitives: bf16 A -> LDS, mxfp4 W
# dwordx4 load + cvt_scalef32_pk_bf16_fp4 upconvert, non-scaled MFMA(16,16,32,
# bf16) via fx.gemm, and the existing atomic bf16 scatter epilogue.
# =============================================================================


def _gemm2_body_a16w4(
    lds_raw_ptr,
    arg_a,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    arg_out,
    bx_i32,
    lane,
    wave,
    i32_M,
    *,
    BM,
    TILE_N,
    TILE_K,
    N_OUT,
    INTER,
    NE,
):
    """a16w4 stage2 body. K=inter_dim (contraction), N=model_dim (N_OUT).

    A = bf16 stage1 intermediate indexed by SORTED position (like the a4w4 path):
    A[sorted_row, k]. W2 = mxfp4, per-1x32 e8m0 scale. Output = bf16 scattered to
    token rows (atomic-fadd, routing-weighted) at [tokens, model_dim].
    """
    elem_bytes = 2
    KH_TILE_BYTES = TILE_K * elem_bytes
    LDS_STRIDE = TILE_K
    K = INTER
    K_HALF = K // 2
    K_TILES_TOTAL = K // TILE_K
    m_repeat = BM // 16
    k_unroll = KH_TILE_BYTES // 64
    _k0_count = TILE_K // 128
    # 4 waves split the TILE_N=256 tile: each wave owns 64 N-cols (num_acc_n=4).
    # This matches the atomic-epilog's wave-split (col = wave*64 + J*16 + lane%16).
    _n_per_wave = TILE_N // 4
    num_acc_n = _n_per_wave // 16
    k_blocks16 = KH_TILE_BYTES // 16
    _num_n_blocks = N_OUT // TILE_N

    # W2 (mxfp4) preshuffle layout (make_preshuffle_b_layout, N-major, fp4).
    bl_k0 = K_HALF // 64
    bl_stride_klane = 256
    bl_stride_k0 = 1024
    bl_stride_n0 = bl_k0 * bl_stride_k0
    layout_b = fx.make_layout(
        (N_OUT // 16, bl_k0, 4, 16, 16),
        (bl_stride_n0, bl_stride_k0, bl_stride_klane, 16, 1),
    )
    scale_k_padded = ((K + 255) // 256) * 256
    sc_k1 = ((scale_k_padded // 32) // 4) // 2
    sc_stride_klane = 16
    sc_stride_k0 = 64
    sc_stride_n0 = sc_k1 * sc_stride_k0

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    m_block_idx = bx_i32 // fx.Int32(_num_n_blocks)
    n_block_idx = bx_i32 % fx.Int32(_num_n_blocks)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)  # first sorted row of this m-block
    by_n = n_block_idx * fx.Int32(TILE_N)
    expert_off = e * fx.Int32(N_OUT)

    w_rsrc = _buffer_rsrc(arg_bq, num_records_bytes=min(NE * N_OUT * K_HALF, 0xFFFFFFFF))
    sw_rsrc = _buffer_rsrc(arg_bscale, num_records_bytes=min(NE * N_OUT * (scale_k_padded // 32), 0xFFFFFFFF))

    # ---- A gather (per-thread) -> LDS. A row = SORTED position m_row + row_local.
    total_threads = 256
    bytes_per_thread = (BM * TILE_K * elem_bytes) // total_threads
    x_load_bytes = 16
    num_x_loads = bytes_per_thread // x_load_bytes
    tile_k_dwords = (TILE_K * elem_bytes) // 4
    c_k_div4 = (K * elem_bytes) // 4
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    chunk_i32 = x_load_bytes // 4
    tx_base = tx_i32 * fx.Int32(chunk_i32)

    x_row_local = []
    x_col_dw = []
    x_row_base_div4 = []
    for i in range_constexpr(num_x_loads):
        tile_idx = tx_base + fx.Int32(i * total_threads * chunk_i32)
        row_local = tile_idx // fx.Int32(tile_k_dwords)
        col_dw = tile_idx % fx.Int32(tile_k_dwords)
        x_row_local.append(row_local)
        x_col_dw.append(col_dw)
        sorted_row = m_row + row_local
        x_row_base_div4.append(sorted_row * fx.Int32(c_k_div4))

    x_buf = _global_i32_buffer_view(arg_a, fx.Int64(0xFFFFFFFF))
    x_dma_tiles4 = fx.logical_divide(x_buf, fx.make_layout(4, 1))
    x_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)

    s_x_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(BM * LDS_STRIDE // 2, 1),
    )
    s_x_i32x4_tiles = fx.logical_divide(s_x_i32_flat, fx.make_layout(4, 1))
    a_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

    def dma_a_tile_to_lds(base_k):
        base_k_div4 = (base_k * fx.Int32(elem_bytes)) // fx.Int32(4)
        for i in range_constexpr(num_x_loads):
            col_bytes = x_col_dw[i] * fx.Int32(4)
            col_sw = _a16w4_swizzle_xor16(x_row_local[i], col_bytes, fx.Int32(k_blocks16))
            row_k_dw = x_row_base_div4[i] + base_k_div4
            global_byte = row_k_dw * fx.Int32(4) + col_bytes
            lds_byte = x_row_local[i] * fx.Int32(KH_TILE_BYTES) + col_sw
            fx.copy(
                x_dma_atom,
                fx.slice(x_dma_tiles4, (None, global_byte // fx.Int32(16))),
                fx.slice(s_x_i32x4_tiles, (None, lds_byte // fx.Int32(16))),
            )

    row_a_lds = lane_mod_16
    col_base_bytes_L = lane_div_16 * fx.Int32(64)

    def _a_col_bytes_for_ku(ku):
        _k0_blk = ku // 4
        _ku_in = ku % 4
        return col_base_bytes_L + fx.Int32(_ku_in * 16 + _k0_blk * 256)

    def lds_load_a(mi, ku):
        row = row_a_lds + fx.Int32(mi * 16)
        col_swz_bytes = _a16w4_swizzle_xor16(row, _a_col_bytes_for_ku(ku), fx.Int32(k_blocks16))
        byte_off = row * fx.Int32(KH_TILE_BYTES) + col_swz_bytes
        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
        fx.copy_atom_call(a_copy_atom, fx.slice(s_x_i32x4_tiles, (None, byte_off // fx.Int32(16))), r)
        return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.BFloat16)

    def load_b_raw(base_k, n_blk, n_intra):
        raw = []
        for k0i in range_constexpr(_k0_count):
            k0 = (base_k + fx.Int32(k0i * 128)) // fx.Int32(128)
            idx_pack = fx.Int32(
                crd2idx(
                    [fx.Int64(n_blk), fx.Int64(k0), fx.Int64(lane_div_16), fx.Int64(n_intra), fx.Int64(0)],
                    layout_b,
                )
            )
            v4 = buffer_ops.buffer_load(
                _raw(w_rsrc), _raw(idx_pack // fx.Int32(4)), vec_width=4, dtype=T.i32, cache_modifier=2
            )
            v4 = fx.Vector(v4)
            raw.append([fx.Int32(v4[j]) for j in range(4)])
        return raw

    def load_b_scale(base_k, mni, n_pack):
        scales = []
        cache = {}
        for ku in range_constexpr(k_unroll):
            _k0_blk = ku // 4
            adj_ku = base_k // fx.Int32(32) + fx.Int32(_k0_blk * 4) + lane_div_16
            k_pack_sub = (adj_ku // fx.Int32(4)) % fx.Int32(2)
            s_ku = adj_ku // fx.Int32(8)
            if _k0_blk not in cache:
                idx = (
                    mni * fx.Int32(sc_stride_n0)
                    + s_ku * fx.Int32(sc_stride_k0)
                    + lane_div_16 * fx.Int32(sc_stride_klane)
                    + lane_mod_16
                )
                cache[_k0_blk] = fx.Int32(buffer_ops.buffer_load(_raw(sw_rsrc), _raw(idx), vec_width=1, dtype=T.i32))
            packed = cache[_k0_blk]
            byte_even = k_pack_sub * fx.Int32(2)
            byte_odd = byte_even + fx.Int32(1)
            se = _e8m0_byte_to_f32(packed, byte_even)
            so = _e8m0_byte_to_f32(packed, byte_odd)
            is_even = arith.cmpi(arith.CmpIPredicate.eq, _raw(n_pack), _raw(fx.Int32(0)))
            scales.append(fx.Float32(arith.select(is_even, _raw(se), _raw(so))))
        return scales

    vec2_bf16 = ir.Type.parse("vector<2xbf16>")

    def upconvert_b(raw, ku, scale_f32):
        i32_val = _raw(raw[ku // 4][ku % 4])
        s_raw = _raw(scale_f32)
        i32s = []
        for sel in range_constexpr(4):
            pp = rocdl.cvt_scalef32_pk_bf16_fp4(vec2_bf16, i32_val, s_raw, sel)
            i32s.append(fx.Int32(fx.Vector(pp).bitcast(fx.Int32)[0]))
        v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
        return v4i32.bitcast(fx.BFloat16)

    # ---- N-column addressing (W2 cols of model_dim; wave owns _n_per_wave) ------
    n_tile_base = wave * fx.Int32(_n_per_wave)
    col_g_list = []
    n_blk_list, n_intra_list, scale_mni_list, scale_np_list = [], [], [], []
    for ni in range_constexpr(num_acc_n):
        col_g = by_n + n_tile_base + fx.Int32(ni * 16) + lane_mod_16
        col_g_list.append(col_g)
        row_w = expert_off + col_g
        n_blk_list.append(row_w // fx.Int32(16))
        n_intra_list.append(row_w % fx.Int32(16))
        ng = expert_off + by_n + n_tile_base + fx.Int32(ni * 16)
        scale_mni_list.append(ng // fx.Int32(32))
        scale_np_list.append((ng // fx.Int32(16)) % fx.Int32(2))

    # ---- accumulators: accm[mi][ni] f32[4] (layout the atomic epilog expects) --
    acc_layout = fx.make_layout(4, 1)
    accm = [[fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(num_acc_n)] for _ in range(m_repeat)]
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    for mi in range_constexpr(m_repeat):
        for ni in range_constexpr(num_acc_n):
            accm[mi][ni].store(zero4)

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))

    def _bf16_frag(v8):
        t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        t.store(v8)
        return t

    def _mma(acc, a8, b8):
        fx.gemm(mma_atom, acc, _bf16_frag(a8), _bf16_frag(b8), acc)

    for kt in range_constexpr(K_TILES_TOTAL):
        base_k = fx.Int32(kt * TILE_K)
        dma_a_tile_to_lds(base_k)
        b_raw = [load_b_raw(base_k, n_blk_list[ni], n_intra_list[ni]) for ni in range_constexpr(num_acc_n)]
        b_sc = [load_b_scale(base_k, scale_mni_list[ni], scale_np_list[ni]) for ni in range_constexpr(num_acc_n)]
        gpu.barrier()
        for ni in range_constexpr(num_acc_n):
            for ku in range_constexpr(k_unroll):
                bb = upconvert_b(b_raw[ni], ku, b_sc[ni][ku])
                for mi in range_constexpr(m_repeat):
                    a8 = lds_load_a(mi, ku)
                    _mma(accm[mi][ni], a8, bb)
        gpu.barrier()

    # ---- epilogue: reuse the existing atomic bf16 scatter (routing-weighted) ---
    # accm[mi][ni] maps to (mchunk=mi, J=ni); the atomic epilog reads accm[i][J].
    # The K-loop is done, so the A-LDS region (offset 0) is free for the f32 acc
    # staging the atomic epilog needs (BM*TILE_N*4 bytes, sized into SharedStorage).
    gpu.barrier()
    lds_acc_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))
    accm_v = [[accm[i][J].load().ir_value() for J in range(num_acc_n)] for i in range(m_repeat)]
    _atomic_bf16_epilog(
        lds_acc_base_i32,
        accm_v,
        arg_out,
        arg_stids,
        arg_sweights,
        m_row,
        n_block_idx,
        wave,
        lane,
        i32_M,
        BM,
        N_OUT,
        TILE_N,
    )


def gemm2_a16w4_grid(BM, *, N_OUT, TILE_N, max_m_blocks):
    """Flattened grid for a16w4 gemm2: (m-blocks) x (model_dim/tile_n) n-blocks."""
    return int(max_m_blocks) * (N_OUT // TILE_N)


def compile_gemm2_a16w4_port(
    BM=32,
    *,
    NE,
    N_OUT,
    D_INTER,
    TILE_N=256,
    TILE_K=256,
    xcd_swizzle=1,
):
    """a16w4 (bf16 intermediate A x mxfp4 W2) stage2 builder.

    N_OUT = model_dim (down-proj output width). D_INTER = inter_dim (contraction).
    Output is bf16 [tokens, model_dim] via the atomic (routing-weighted) scatter.

    ``xcd_swizzle`` (>0) round-robins the launch index bijectively across the 8
    XCDs (like the a4w4 gemm2 ``_xcd_np`` grid). a16w4 gemm2 is HBM-bandwidth-bound
    (flyprof: L2 hit 14.9%, 2705/8000 GB/s, MFMA 4.6%); the plain m-major grid
    clusters consecutive tiles onto the same XCD/HBM channels, so remapping balances
    per-channel utilization. ``xcd_swizzle`` also enables an optional M-group swizzle
    for per-XCD L2 locality (group size = xcd_swizzle m-blocks).
    """
    _K = D_INTER
    assert _K % TILE_K == 0, f"D_INTER (K) must be a multiple of {TILE_K}, got {_K}"
    assert N_OUT % TILE_N == 0, f"model_dim (N_OUT) must be a multiple of {TILE_N}, got {N_OUT}"
    assert BM % 16 == 0, f"BM must be a multiple of 16, got {BM}"
    _num_n_blocks = N_OUT // TILE_N
    KH_TILE_BYTES = TILE_K * 2

    # LDS: A tile (BM x TILE_K bf16) then f32 accumulator region (BM x TILE_N f32).
    _a_bytes = BM * KH_TILE_BYTES
    _acc_bytes = lds_acc_bytes_for(BM, TILE_N)
    _lds_bytes = _a_bytes + _acc_bytes

    _name = f"gemm2_a16w4_port_ne{NE}_h{N_OUT}_i{_K}_bm{BM}_tn{TILE_N}"
    if xcd_swizzle > 0:
        _name += f"_xcd{xcd_swizzle}"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, _lds_bytes, 16]

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_num_n_blocks)

        # Bijective XCD round-robin over the valid tiles [0, bound) to balance the
        # per-XCD/HBM-channel weight-load traffic (a16w4 gemm2 is HBM-bound). With
        # xcd_swizzle>0, additionally group-swizzle along M for per-XCD L2 locality.
        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd_np(pid):
            xc = _umod(pid, _NXCD)
            wgid = xc * _xq + fx.Int32(arith.minsi(_raw(xc), _raw(_xr))) + _udiv(pid, _NXCD)
            if const_expr(_SW <= 0):
                return wgid
            _ng = fx.Int32(_SW * _num_n_blocks)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_num_n_blocks) + n_block

        if bx_i32 < bound:
            tile = _xcd_np(bx_i32)
            _gemm2_body_a16w4(
                lds_raw_ptr,
                arg_a,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_out,
                tile,
                lane,
                wave,
                i32_M,
                BM=BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                N_OUT=N_OUT,
                INTER=_K,
                NE=NE,
            )

    @flyc.jit
    def launch_gemm2(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid: fx.Int32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        gemm2_kernel(
            arg_a,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2
