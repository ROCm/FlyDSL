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

from .common import (
    _buffer_rsrc,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_view,
    _lds_ptr3,
    _raw,
    _udiv,
    _umod,
    kmchunks_for,
    lds_acc_bytes_for,
)
from .gemm1 import _a16w4_swizzle_xor16, _e8m0_byte_to_f32


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
# Dedicated path. Reuses the M1-proven a16w4 primitives: bf16 A -> LDS, mxfp4 W
# dwordx4 load + cvt_scalef32_pk_bf16_fp4 upconvert, non-scaled MFMA(16,16,32,
# bf16) via fx.gemm, and the atomic bf16 scatter epilogue above.
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
    b_cache_mod=2,
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
            # A-LDS bank-conflict XOR swizzle: LDS dest stays LINEAR (the DMA
            # buffer_load_lds hardware does not honor an arbitrary swizzled per-lane
            # LDS dest -- the M1 NaN); instead swizzle the GMEM source column so
            # linear LDS slot [row][col] holds A[row][swz(row,col)]. The LDS read
            # (lds_load_a) applies the SAME swizzle to its offset, so it fetches the
            # right logical K. Same convention as kernels/gemm/mxfp4_preshuffle.py.
            col_sw = _a16w4_swizzle_xor16(x_row_local[i], col_bytes, fx.Int32(k_blocks16), enable=True)
            row_k_dw = x_row_base_div4[i] + base_k_div4
            global_byte = row_k_dw * fx.Int32(4) + col_sw
            lds_byte = x_row_local[i] * fx.Int32(KH_TILE_BYTES) + col_bytes
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
        # Same XOR swizzle as the write above (enable=True) -> read fetches logical
        # (row, col) from its swizzled physical byte. Read cols are 16-byte-aligned
        # multiples, and the mask XORs by multiples of 16, so alignment is preserved.
        col_swz_bytes = _a16w4_swizzle_xor16(row, _a_col_bytes_for_ku(ku), fx.Int32(k_blocks16), enable=True)
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
                _raw(w_rsrc), _raw(idx_pack // fx.Int32(4)), vec_width=4, dtype=T.i32, cache_modifier=b_cache_mod
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
    b_cache_mod=2,
    waves_per_eu=None,
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
    if b_cache_mod != 2:
        _name += f"_bcm{b_cache_mod}"
    if xcd_swizzle > 0:
        _name += f"_xcd{xcd_swizzle}"
    if waves_per_eu:
        _name += f"_w{waves_per_eu}"

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
                b_cache_mod=b_cache_mod,
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
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2
