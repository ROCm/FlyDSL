# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4/MXFP6/MXFP8 A x MXFP4/MXFP8 B preshuffle GEMM (gfx950): per-32 E8M0 scales folded
into a scaled 16x16x128 fx.gemm; A streams global->LDS via double-buffered async DMA. Layout
matches the host preshuffle (shuffle_weight_w4(.,16) + shuffle_scale_w4).

Two scale modes: ``blockscale="ab"`` puts both operands on a coarse E8M0 grid (A per
1x128, B per 128x128); ``"none"`` keeps the per-1x32 MX scales. The separate ``_bs_a`` /
``_bs_b`` flags below exist because the A and B scale readers differ in lane broadcasting,
not to offer mixed modes. ``k_batch`` > 1 enables split-K, reduced by
``launch_splitk_reduce``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Constexpr,
    Float4E2M1FN,
    Float6E2M3FN,
    Float8E4M3FN,
    Float16,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from kernels.common.mma.mfma_preshuffle_pipeline import xcd_remap_bx_by

_A_ELEM = {"fp4": Float4E2M1FN, "fp6": Float6E2M3FN, "fp8": Float8E4M3FN}
_B_ELEM = {"fp4": Float4E2M1FN, "fp8": Float8E4M3FN}


def _scale_mma_atoms(a_dtype, b_dtype):
    """16 (opsel_a, opsel_b) scaled-MFMA atoms; A elem is fp4/fp6/fp8, B is fp4/fp8."""
    elem_a = _A_ELEM[a_dtype]
    elem_b = _B_ELEM[b_dtype]
    return {
        (osa, osb): fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, elem_a, elem_b, opsel_a=osa, opsel_b=osb))
        for osa in range(4)
        for osb in range(4)
    }


def _bq_view(arg_bq_addr, row_elems, KH4, k_tiles, k_halves, pair):
    """Preshuffled B view for one N-row tile; index [l//16, l%16, kt, half, p, None] -> i32[4].

    `pair` = K0 blocks per 128-K MFMA: 1 for fp4 B (one i32[4]), 2 for fp8 B (lo/hi halves
    packed into i32[8] by load_b). K0 blocks (256 i32 each) run contiguously along K as
    ((kt*k_halves + kh)*pair + p); the fp4 case keeps a size-1 `p` dim (byte-identical view).
    """
    col_base = rocdl.readfirstlane(T.i32, row_elems * KH4)
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, arg_bq_addr + off_i64 * 4)
    shape = (4, 16, k_tiles, k_halves, pair, 4)
    strides = (64, 4, k_halves * pair * 256, pair * 256, 256, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, strides)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


# K-tiles one row-layout scale panel row holds (see ``row_scale_panel``).
ROW_SCALE_CHUNK = 4
# gfx950 LDS: bounds the prefetch pipeline's A ring and scale panels.
LDS_LIMIT_BYTES = 160 * 1024


def row_scale_panel(rows, tile_k, num_threads):
    """(rows per slice, DMA passes per slice, bytes) of one row-layout LDS scale panel.

    A panel row holds ROW_SCALE_CHUNK K-tiles of one scale row; slice j of the next
    chunk is DMA'd a dword per lane in iteration j, and a slice's spare lanes run on
    past it, so the panel is padded to cover the last slice's tail.
    """
    crow = ROW_SCALE_CHUNK * tile_k // 32
    rs = -(-rows // ROW_SCALE_CHUNK)
    passes = -(-(rs * crow // 4) // num_threads)
    size = max(rows * crow, (ROW_SCALE_CHUNK - 1) * rs * crow + passes * num_threads * 4)
    return rs, passes, (size + 15) // 16 * 16


# prefetch_depth's per-tile caps, tuned on gfx950 (ROCm/aiter#7309).
_PF_MAX_DEPTH = 8
_PF_MID_TILE_M = 48  # tiles of this many rows or more cap at _PF_MID_MAX_DEPTH
_PF_MID_MAX_DEPTH = 3
# A 192-row tile's ring is twice a 96-row one's: P = 2 beat 3 by 4-6% on
# 160-192 x 25600 x 6144 (192x128x128).
_PF_BIG_TILE_M = 192
_PF_BIG_MAX_DEPTH = 2


def prefetch_depth(tile_m, tile_n, tile_k, N, K, k_batch=1, num_cus=256):
    """Deepest legal ``prefetch`` for an a8w8 row-scale launch, or 0 when none is.

    A depth must fit the vmcnt counter ((P - 1) * loads <= 63) and the LDS (an A ring
    of P + 1 tiles plus the scale panels), and leave enough LDS for the workgroups
    each CU gets (the N tiles x splits over ``num_cus``) to run together. The per-tile
    caps follow aiter's gfx950 tuning.
    """
    k_tiles = K // tile_k // max(k_batch, 1)
    num_threads = min(4, tile_n // 16) * 64
    _, pass_a, panel_a = row_scale_panel(tile_m, tile_k, num_threads)
    _, pass_b, panel_b = row_scale_panel(tile_n, tile_k, num_threads)
    b_loads = (tile_n // (num_threads // 64) // 16) * (tile_k // 128) * 2  # lo, hi
    a_loads = tile_m * tile_k // (num_threads * 16)
    loads = pass_a + pass_b + b_loads + a_loads  # the kernel's PF_LOADS
    want = min(_PF_MAX_DEPTH, -(-(N // tile_n * max(k_batch, 1)) // num_cus))
    max_depth = _PF_MID_MAX_DEPTH if tile_m >= _PF_MID_TILE_M else _PF_MAX_DEPTH
    if tile_m >= _PF_BIG_TILE_M:
        max_depth = _PF_BIG_MAX_DEPTH
    best = 0
    for p in range(2, min(max_depth, k_tiles) + 1):
        lds = tile_m * tile_k * (p + 1) + (2 + -(-(p - 1) // ROW_SCALE_CHUNK)) * (panel_a + panel_b)
        if (p - 1) * loads <= 63 and lds <= LDS_LIMIT_BYTES and LDS_LIMIT_BYTES // lds >= want:
            best = p
    return best


@flyc.jit
def launch_gemm(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    a_dtype: Constexpr[str],
    out_dtype: Constexpr[str],
    b_dtype: Constexpr[str],
    batch: Constexpr[int],
    a_row_stride: Constexpr[int],
    a_batch_stride: Constexpr[int],
    sca_row_stride: Constexpr[int],
    sca_batch_stride: Constexpr[int],
    c_row_stride: Constexpr[int],
    c_batch_stride: Constexpr[int],
    waves_per_eu: Constexpr[int],
    xcd_swizzle: Constexpr[int] = 0,
    k_batch: Constexpr[int] = 1,
    blockscale: Constexpr[str] = "none",
    a_scale_layout: Constexpr[str] = "mfma_tile",
    b_scale_layout: Constexpr[str] = "mfma_tile",
    prefetch: Constexpr[int] = 1,
):
    """Direct @flyc.jit launcher. Operands are fx.Pointer (pass ptr_arg(t): raw data_ptr, no
    per-launch DLPack). Compile once with flyc.compile, then cf(*runtime). a_dtype fp4/fp6/fp8
    A x preshuffled b_dtype (fp4/fp8) B, e8m0 scales (a8w8 = a_dtype=fp8, b_dtype=fp8).
    batch>1 = strided-batched over grid.z. The a_/sca_/c_ row/batch strides make A/scale_a/C
    addressing caller-controlled; each <0 keeps the contiguous [B,M,*] bmn default, all set =
    the [M,B,*] mbn layout. waves_per_eu<=0 = unset.

    ``*_scale_layout``: "mfma_tile" takes that operand's e8m0 scale pre-shuffled
    (shuffle_scale_w4), read from global; "row" the unshuffled row-major
    ``[rows, K // 32]`` tensor as the quantizer emits it, staged through LDS by the
    kernel (1x32 scales, a8w8, batch 1).

    ``prefetch`` P > 1 (both scales "row") runs the prefetch pipeline: B and the
    scales are loaded P - 1 K-tiles ahead of their MFMAs, B into P register sets,
    A into a ring of P + 1 LDS tiles by async DMA; the last P tiles are peeled.
    """
    BM, BN, BK = tile_m, tile_n, tile_k
    assert blockscale in ("none", "ab"), f"blockscale must be 'none' or 'ab', got {blockscale!r}"
    assert BK in (128, 256), f"tile_k must be 128 or 256 (tiles_per_chunk = 256 // tile_k), got {BK}"
    assert BM % 32 == 0, f"tile_m must be a multiple of 32 (the A e8m0 scale is 32-row granular), got {BM}"
    assert BN % 16 == 0, f"tile_n must be a multiple of 16, got {BN}"
    for _layout in (a_scale_layout, b_scale_layout):
        assert _layout in ("mfma_tile", "row"), f"scale layout {_layout}"
    _row_a = a_scale_layout == "row"
    _row_b = b_scale_layout == "row"
    _row_any = _row_a or _row_b
    if const_expr(out_dtype == "bf16"):
        out_elem = BFloat16
    else:
        out_elem = Float16

    # Row sizes + read_a fragment layout (i32 units): fp6/fp8 read two b128 halves -> i32[A_NDW], fp4 one -> i32[4].
    if const_expr(a_dtype == "fp4"):  # 2 codes/byte
        a_row_bytes, A_ROW_B = K // 2, BK // 2
        A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 16, 0, 4
    else:
        a_row_bytes, A_ROW_B = K, BK
        if const_expr(a_dtype == "fp8"):
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 32, 16, 8
        else:  # fp6
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 8, 32, 4, 6

    A_LDS_B = BM * A_ROW_B  # LDS A buffer bytes (row-major [m][col], shared by the N-waves)
    A_ROW_I32 = A_ROW_B // 4
    swz_lds = a_dtype in ("fp4", "fp8")
    k_blk16 = A_ROW_B // 16
    # B fragment layout: fp8 B needs i32[8] (two K0 blocks / 128-K MFMA); fp4 B i32[4] (one).
    if const_expr(b_dtype == "fp8"):
        b_row_bytes, B_NDW, B_BLK_PER_MMA = K, 8, 2
    else:  # fp4
        b_row_bytes, B_NDW, B_BLK_PER_MMA = K // 2, 4, 1
    KH4 = b_row_bytes // 4  # i32 per N-row in preshuffled B (== (K//2)//4 for fp4)
    K_TILES = K // BK
    assert K_TILES % k_batch == 0, "K_TILES must be divisible by k_batch"
    k_tiles_local = K_TILES // k_batch
    k_halves = BK // 128  # 16x16x128 MFMA k-steps per K-tile
    # e8m0 scales are 256-K granular, B 128-K: tiles_per_chunk K-tiles share a word (hi/lo 16b = 128-K half).
    tiles_per_chunk = 256 // BK  # 1 for tile_k=256, 2 for tile_k=128
    m_chunks = BM // 16
    num_waves = min(4, BN // 16)
    num_threads = num_waves * 64
    num_acc_n = (BN // num_waves) // 16  # 16-col n-subblocks per wave
    _bs_a = blockscale in ("a", "ab")
    _bs_b = blockscale in ("b", "ab")
    _sc_k0_a = 16 if _bs_a else 64  # per-256K-chunk dword stride (A: drop K_Lane=4)
    _sc_k0_b = 1 if _bs_b else 64  # (B: drop K_Lane*N_Lane=64)
    _scale_chunk_dw_a = ((K + 255) // 256) * _sc_k0_a
    _scale_chunk_dw_b = ((K + 255) // 256) * _sc_k0_b
    _b_sc_rows = (N // 128) if _bs_b else (N // 32)  # B scale super-rows
    a_copy_granularity = num_threads * 16  # bytes one cooperative round covers
    assert A_LDS_B % a_copy_granularity == 0, (
        f"A_LDS_B ({A_LDS_B}B) must be divisible by num_threads*16 ({a_copy_granularity}B), else part "
        "of the A tile is never DMA'd; adjust tile_m / tile_k"
    )
    n_coop = A_LDS_B // a_copy_granularity  # 16B cooperative loads per thread
    assert k_batch == 1 or (c_row_stride < 0 and c_batch_stride < 0), (
        "split-K writes tmp[batch*k_batch, M, N] as a contiguous fp32 slab, so k_batch > 1 "
        "requires the default c_row_stride / c_batch_stride (-1)"
    )
    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)
    if const_expr(_row_any):
        # Row-major scales: a scale row's bytes for one K-tile are BK//32 apart from the
        # next row's, so per-tile loads touch a cache line per row. Instead an LDS panel
        # row holds SC_C K-tiles (CROW_B contiguous bytes), filled by dword LDS-DMA:
        # slice j (rows [j*rs, (j+1)*rs)) of chunk c+1 rides iteration j of chunk c,
        # double-buffered by chunk parity. A slice's spare lanes run on into the
        # following rows (the data those will hold) or the panel tail. Only a "row"
        # operand gets a panel (A first, then B).
        assert batch == 1, "row scale layout: batch 1"
        assert a_dtype == "fp8" and b_dtype == "fp8", "row scale layout: a8w8"
        assert blockscale == "none", "row scale layout: 1x32 scales"
        SC_ROW_B = BK // 32  # bytes per scale row per K-tile
        SC_WPR = SC_ROW_B // 4  # one dword per 128-K MFMA step
        SC_C = ROW_SCALE_CHUNK  # K-tiles per chunk
        CROW_B = SC_C * SC_ROW_B
        SC_RS_A, SC_PASS_A, SC_PANEL_A = row_scale_panel(BM, BK, num_threads) if _row_a else (0, 0, 0)
        SC_RS_B, SC_PASS_B, SC_PANEL_B = row_scale_panel(BN, BK, num_threads) if _row_b else (0, 0, 0)
        SC_LDS_B = SC_PANEL_A + SC_PANEL_B

    PF = int(prefetch)
    _pf = PF > 1
    if const_expr(_pf):
        assert _row_a and _row_b, "prefetch: row-layout 1x32 A and B scales"
        # Chunk c's slices are issued while chunk c - SC_D is read; its last one retires
        # P - 1 iterations later, before chunk c's first tile, when
        # SC_D * SC_C >= SC_C + P - 1.
        SC_D = 1 + -(-(PF - 1) // SC_C)
        SC_NB = SC_D + 1
        # Loads in flight at an iteration's end: P - 1 iterations' scale slices, B (lo,
        # hi per 128-K step and N subtile) and A DMA. vmcnt counts to 63.
        PF_LOADS = SC_PASS_A + SC_PASS_B + num_acc_n * k_halves * 2 + n_coop
        assert (PF - 1) * PF_LOADS <= 63, f"prefetch {PF}: vmcnt {(PF - 1) * PF_LOADS}"
        PF_LDS_B = A_LDS_B * (PF + 1) + SC_LDS_B * SC_NB
        assert PF_LDS_B <= LDS_LIMIT_BYTES, f"prefetch {PF}: LDS {PF_LDS_B}B > {LDS_LIMIT_BYTES}B"

    # Scheduler counts per loop iter: MFMAs, A LDS reads/thread (fp6/fp8 2 per (mi,kh)), gmem loads.
    sched_mfma_total = k_halves * m_chunks * num_acc_n
    if const_expr(a_dtype == "fp4"):
        a_ds_per = 1
    else:
        a_ds_per = 2
    sched_num_ds_load = m_chunks * k_halves * a_ds_per
    sched_num_gmem = n_coop + num_acc_n * k_halves * B_BLK_PER_MMA + m_pairs + n_pairs
    if const_expr(_row_any):
        # A row operand's scale rows come from LDS (one read per 16-row tile); its DMA
        # replaces that operand's per-pair global scale loads.
        sched_num_sc_ds = (m_chunks if _row_a else 0) + (num_acc_n if _row_b else 0)
        # Also counted in the A-read group: the slack keeps every A ds_read ahead of the
        # MFMAs (an exact count measured up to 1.4x slower on some tiles).
        sched_num_ds_load += sched_num_sc_ds
        sched_num_gmem += (SC_PASS_A - m_pairs if _row_a else 0) + (SC_PASS_B - n_pairs if _row_b else 0)

    if const_expr(_pf):

        @fx.struct
        class SharedA:
            # One array each: the A ring of P + 1 tiles A_LDS_B apart, the SC_NB scale
            # panels SC_LDS_B apart.
            a0: fx.Array[Int8, A_LDS_B * (PF + 1), 16]
            s0: fx.Array[Int8, SC_LDS_B * SC_NB, 16]

    elif const_expr(_row_any):

        @fx.struct
        class SharedA:
            a0: fx.Array[Int8, A_LDS_B, 16]
            a1: fx.Array[Int8, A_LDS_B, 16]
            s0: fx.Array[Int8, SC_LDS_B, 16]
            s1: fx.Array[Int8, SC_LDS_B, 16]

    else:

        @fx.struct
        class SharedA:
            a0: fx.Array[Int8, A_LDS_B, 16]
            a1: fx.Array[Int8, A_LDS_B, 16]

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Int64,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        scale_atoms = _scale_mma_atoms(a_dtype, b_dtype)

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx

        if const_expr(k_batch > 1):
            bz_batch = bid_z // k_batch
            kt0 = fx.Int32(bid_z % k_batch) * fx.Int32(k_tiles_local)
        else:
            bz_batch = bid_z
            kt0 = fx.Int32(0)
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        lane = tid % 64
        lane_div_16 = lane // 16
        lane_mod_16 = lane % 16
        # XCD swizzle: remap (bid_x, bid_y) for L2-cache reuse (no-op when xcd_swizzle<=0).
        if const_expr(xcd_swizzle > 0):
            bid_x, bid_y = xcd_remap_bx_by(
                bid_x,
                bid_y,
                i32_m,
                tile_m=BM,
                tile_n=BN,
                N=N,
                xcd_swizzle=xcd_swizzle,
            )
        bx_m = bid_x * BM
        by_n = bid_y * BN

        # Strided-batched: shift each base to batch bid_z (A/scale_a via explicit strides or
        # the contiguous default; B/scale_b stay batch-contiguous). batch==1 emits no batch math.
        if const_expr(batch > 1):
            a_rstride = fx.Int32(a_row_bytes if a_row_stride < 0 else a_row_stride)
            sca_rstride = fx.Int32(_scale_chunk_dw_a if sca_row_stride < 0 else sca_row_stride)
            bz = fx.Int64(bz_batch)
            if const_expr(a_batch_stride < 0):
                arg_a = arg_a + bz * (fx.Int64(i32_m) * fx.Int64(a_row_bytes))
            else:
                arg_a = arg_a + bz * fx.Int64(a_batch_stride)
            arg_b = arg_b + bz * fx.Int64(N * b_row_bytes)
            if const_expr(sca_batch_stride < 0):
                sc_bstride = fx.Int64((i32_m + 31) // 32) * fx.Int64(_scale_chunk_dw_a) * 4
                arg_scale_a = arg_scale_a + bz * sc_bstride
            else:
                arg_scale_a = arg_scale_a + bz * fx.Int64(sca_batch_stride)
            arg_scale_b = arg_scale_b + bz * fx.Int64(_b_sc_rows * _scale_chunk_dw_b * 4)
        else:
            a_rstride = fx.Int32(a_row_bytes)
            sca_rstride = fx.Int32(_scale_chunk_dw_a)

        # A source view, bound to the last valid M row (ragged M OOB -> 0).
        _i8g = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=16)
        if const_expr(batch > 1 and a_row_stride >= 0):
            a_nrec = fx.Int64(i32_m - 1) * fx.Int64(a_rstride) + fx.Int64(a_row_bytes)
        else:
            a_nrec = fx.Int64(i32_m) * fx.Int64(a_row_bytes)
        a_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(
                    fx.inttoptr(_i8g, arg_a),
                    fx.make_layout(65536 * a_row_bytes, 1),
                )
            ),
            max_size=False,
            num_records_bytes=a_nrec,
        )
        a_flat_div = fx.logical_divide(a_flat, fx.make_layout(1, 1))
        lds = fx.SharedAllocator().allocate(SharedA).peek()
        # A-LDS modeled as i32 (16B = 4 i32): fx.copy is dtype-agnostic, only the MMA cares.
        sA0_i32 = fx.recast_iter(Int32, lds.a0.ptr)
        if const_expr(_pf):
            lds_db = fx.Int32(A_LDS_B)  # ring slot byte stride
        else:
            lds_db = fx.Int32(fx.ptrtoint(lds.a1.ptr)) - fx.Int32(fx.ptrtoint(lds.a0.ptr))  # ping/pong byte stride
        lds_db_i32 = lds_db // 4
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        # The prefetch pipeline's LDS DMAs are async: they retire by asyncmark group, so
        # an LDS read is not held behind every copy in flight.
        dma_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.BufferLoadAsyncLDS128b() if _pf else fx.rocdl.BufferCopyLDS128b(), 128
        )
        _i8s = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
        sA0_i8 = fx.recast_iter(_i8s, lds.a0.ptr)

        def _iter_of(parity):  # parity in {0,1} (runtime) -> i32 LDS iterator
            return fx.add_offset(sA0_i32, parity * lds_db_i32)

        def _lds_view(base_iter, off_i32):
            return fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(4, 1))

        # Async A: gmem->LDS DMA (buffer_load_lds); issued after B/scale loads to overlap the MFMAs.
        def dma_a_to_lds(kt, parity):
            base_off = rocdl.readfirstlane(T.i32, parity * lds_db + wave * (64 * 16))
            lds_ptr = fx.add_offset(sA0_i8, base_off)
            base_k_byte = kt * A_ROW_B
            for i in range_constexpr(n_coop):
                if const_expr(i > 0):
                    lds_ptr = fx.add_offset(lds_ptr, fx.Int32(num_threads * 16))
                lin = (i * num_threads + tid) * 16
                row = lin // A_ROW_B
                col = lin % A_ROW_B
                if const_expr(swz_lds):
                    col = col ^ ((row % k_blk16) * 16)
                gmem_byte = (bx_m + row) * a_rstride + base_k_byte + col
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(a_flat_div, (None, gmem_byte))
                fx.copy(dma_atom, src, dst)

        def _read16(base_iter, off_i32):
            # ds_read_b128 straight into an i32[4] register fragment.
            t = fx.make_rmem_tensor(4, Int32)
            fx.copy(lds_copy, _lds_view(base_iter, off_i32), t)
            return t

        def read_a(parity):
            base_iter = _iter_of(parity)
            av = []
            for mi in range_constexpr(m_chunks):
                for kh in range_constexpr(k_halves):
                    row = mi * 16 + lane_mod_16
                    row_base = row * A_ROW_I32
                    lo_blk = kh * (A_KH_I32 // 4) + lane_div_16 * (A_GK_I32 // 4)
                    if const_expr(swz_lds):
                        off = row_base + (lo_blk ^ (row % k_blk16)) * 4
                    else:
                        off = row_base + kh * A_KH_I32 + lane_div_16 * A_GK_I32
                    if const_expr(a_dtype == "fp4"):
                        av.append(_read16(base_iter, off))
                    else:
                        # fp6/fp8: pack two halves (64 K apart, f8f6f4 ABI) into i32[A_NDW].
                        if const_expr(swz_lds):
                            hi_off = row_base + ((lo_blk + A_HI_OFF // 4) ^ (row % k_blk16)) * 4
                        else:
                            hi_off = off + A_HI_OFF
                        lo = Vec(fx.memref_load_vec(_read16(base_iter, off)))
                        hi = Vec(fx.memref_load_vec(_read16(base_iter, hi_off)))
                        t = fx.make_rmem_tensor(A_NDW, Int32)
                        t.store(lo.shuffle(hi, list(range(A_NDW))))
                        av.append(t)
            return av

        n_col_base = by_n + wave * (BN // num_waves)
        bq_views = [
            _bq_view(arg_b, n_col_base + ni * 16, KH4, K_TILES, k_halves, B_BLK_PER_MMA)
            for ni in range_constexpr(num_acc_n)
        ]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        bs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        # e8m0 scale buffers bounded to real size (OOB rows read 0); scale_a to the last 32-row chunk.
        _i32g = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        _sc_layout = fx.make_layout(1 << 28, 1)
        _a_sc_chunks = (i32_m + 31) // 32
        if const_expr(batch > 1 and sca_row_stride >= 0):
            a_sc_nrec = (fx.Int64(_a_sc_chunks - 1) * fx.Int64(sca_rstride) + fx.Int64(_scale_chunk_dw_a)) * 4
        else:
            a_sc_nrec = fx.Int64(_a_sc_chunks) * fx.Int64(_scale_chunk_dw_a) * 4
        b_sc_nrec = fx.Int64(_b_sc_rows * _scale_chunk_dw_b * 4)
        sa_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, arg_scale_a), _sc_layout)),
                max_size=False,
                num_records_bytes=a_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        sb_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, arg_scale_b), _sc_layout)),
                max_size=False,
                num_records_bytes=b_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        a_sc_base = [(bx_m // 32 + mp) * sca_rstride for mp in range_constexpr(m_pairs)]
        nsb = (by_n + wave * (BN // num_waves)) // 32
        # B blockscale: 4 consecutive 32-N super-rows share one 128-N block scale.
        b_sc_base = [((nsb + np) // 4 if _bs_b else (nsb + np)) * _scale_chunk_dw_b for np in range_constexpr(n_pairs)]
        sc_lane = lane_div_16 * 16 + lane_mod_16
        # blockscale drops broadcast dims from the per-lane offset: A drops K_Lane
        # (lane_div_16); B drops K_Lane + N_Lane (all lanes read one word).
        sc_lane_a = lane_mod_16 if _bs_a else sc_lane
        sc_lane_b = fx.Int32(0) if _bs_b else sc_lane

        n_acc = m_chunks * num_acc_n

        if const_expr(_row_any):
            _i8g4 = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=4)
            sc_row_g = K // 32  # bytes per global scale row

            def _sc_src(addr, nrec):
                buf = fx.rocdl.make_buffer_tensor(
                    fx.Tensor(fx.make_view(fx.inttoptr(_i8g4, addr), fx.make_layout(1 << 30, 1))),
                    max_size=False,
                    num_records_bytes=nrec,
                )
                return fx.logical_divide(buf, fx.make_layout(1, 1))

            # Bounded to the tensors: rows past M read 0 (and are never stored).
            # (source, first row, rows per slice, passes, panel byte)
            sc_panels = []
            if const_expr(_row_a):
                sa_row = _sc_src(arg_scale_a, fx.Int64(i32_m) * fx.Int64(sc_row_g))
                sc_panels.append((sa_row, bx_m, SC_RS_A, SC_PASS_A, 0))
            if const_expr(_row_b):
                sb_row = _sc_src(arg_scale_b, fx.Int64(N * sc_row_g))
                sc_panels.append((sb_row, by_n, SC_RS_B, SC_PASS_B, SC_PANEL_A))
            sc_dma = fx.make_copy_atom(
                fx.rocdl.cdna4.BufferLoadAsyncLDS32b() if _pf else fx.rocdl.BufferCopyLDS32b(), 32
            )
            sc_row_copy = fx.make_copy_atom({1: fx.UniversalCopy32b, 2: fx.UniversalCopy64b}[SC_WPR](), Int32)
            sS0_i8 = fx.recast_iter(_i8s, lds.s0.ptr)
            sS0_i32 = fx.recast_iter(Int32, lds.s0.ptr)
            if const_expr(_pf):
                sc_db = fx.Int32(SC_LDS_B)  # panel buffer byte stride
            else:
                sc_db = fx.Int32(fx.ptrtoint(lds.s1.ptr)) - fx.Int32(fx.ptrtoint(lds.s0.ptr))
            sc_db_dw = sc_db // 4
            b_wave_n = wave * (BN // num_waves)
            # v_perm pool = {hi:lo}; selector byte b+4 picks hi.byte(b), b picks lo.byte(b),
            # 0x0C gives 0. Lane l reads 32-K group b = l // 16.
            sc_sel = 0x0C0C0000 + lane_div_16 * 256 + lane_div_16 + 4

            # CROW_B // 4 (dwords per panel row) is tile_k // 32: a power of two.
            _SC_DW_SHIFT = (CROW_B // 4).bit_length() - 1
            assert 1 << _SC_DW_SHIFT == CROW_B // 4

            def dma_scale_slice(kt_c, j, buf):
                """Slice j of the chunk starting at K-tile kt_c into buffer buf."""
                for src, row0, rs, passes, off in sc_panels:
                    for i in range_constexpr(passes):
                        # Lane l of a wave lands at the wave-uniform base + l * 4.
                        q = i * num_threads + tid
                        if const_expr(_pf):
                            # Non-negative operands and power-of-two divisors: shifts and
                            # masks, not signed division fix-ups.
                            row = j * rs + (q >> _SC_DW_SHIFT)
                            col_dw = q & (CROW_B // 4 - 1)
                        else:
                            row = j * rs + q // (CROW_B // 4)
                            col_dw = q % (CROW_B // 4)
                        gbyte = (row0 + row) * sc_row_g + kt_c * SC_ROW_B + col_dw * 4
                        lds_off = rocdl.readfirstlane(
                            T.i32, buf * sc_db + off + j * (rs * CROW_B) + i * num_threads * 4 + wave * 256
                        )
                        dst = fx.make_view(fx.add_offset(sS0_i8, lds_off), fx.make_layout(1, 1))
                        fx.copy(sc_dma, fx.slice(src, (None, gbyte)), dst)

            def _sc_row(buf, j, lrow, crow, base):
                # Panel row lrow + crow, K-tile j of the chunk: SC_WPR dwords, one per kh.
                # lrow is the per-lane part; the compile-time crow and base stay out of it
                # so they fold into the ds_read immediate offset.
                t = fx.make_rmem_tensor(SC_WPR, Int32)
                off = buf * sc_db_dw + lrow * (CROW_B // 4) + j * SC_WPR + (crow * CROW_B + base) // 4
                fx.copy(sc_row_copy, fx.make_view(fx.add_offset(sS0_i32, off), fx.make_layout(SC_WPR, 1)), t)
                return Vec(fx.memref_load_vec(t))

            def _sc_pack(r0, r1):
                # The shuffled-layout word: byte kh*2 + i = row (r + 16*i)'s scale for MFMA
                # step kh, so the opsel atoms and one VGPR per row pair are kept.
                w = fx.Int32(rocdl.perm_b32(r0[0], r1[0], sc_sel))
                if const_expr(k_halves > 1):
                    w = w | (fx.Int32(rocdl.perm_b32(r0[1], r1[1], sc_sel)) << 16)
                return w

            def _sc_pairs(buf, j, lrow, n_tiles, base):
                words = []
                for p_ in range_constexpr(max(1, n_tiles // 2)):
                    r0 = _sc_row(buf, j, lrow, p_ * 32, base)
                    r1 = _sc_row(buf, j, lrow, p_ * 32 + 16, base) if 2 * p_ + 1 < n_tiles else r0
                    words.append(_sc_pack(r0, r1))
                return words

            def load_sc_row(buf, j):
                # (sa, sb) of the row operands from LDS; None for a mfma_tile one.
                sa = _sc_pairs(buf, j, lane_mod_16, m_chunks, 0) if _row_a else None
                sb = _sc_pairs(buf, j, b_wave_n + lane_mod_16, num_acc_n, SC_PANEL_A) if _row_b else None
                return sa, sb

        def load_b(kt):
            # buffer_load_dwordx4 into i32[4] frags; fp8 B packs two K0 blocks (lo/hi, 64 K
            # apart, f8f6f4 ABI) into i32[8] — same shuffle as read_a's fp6/fp8 A path.
            ops = []
            for ni in range_constexpr(num_acc_n):
                for kh in range_constexpr(k_halves):
                    lo = fx.make_rmem_tensor(4, Int32)
                    fx.copy(b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, 0, None], lo)
                    if const_expr(b_dtype == "fp4"):
                        ops.append(lo)
                    else:  # fp8: lo ++ hi -> i32[B_NDW]
                        hi = fx.make_rmem_tensor(4, Int32)
                        fx.copy(b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, 1, None], hi)
                        t = fx.make_rmem_tensor(B_NDW, Int32)
                        t.store(fx.memref_load_vec(lo).shuffle(fx.memref_load_vec(hi), list(range(B_NDW))))
                        ops.append(t)
            return ops

        def load_sc(chunk_kt, want_a=True, want_b=True):
            # (sa, sb) e8m0 words per m-/n-pair for one 256-K chunk (uniform base -> SGPR soffset).
            # blockscale uses smaller per-chunk strides (compact buffer) + broadcast lanes.
            # A row-layout operand comes from load_sc_row instead (want_* False -> None).
            koff_a = chunk_kt * _sc_k0_a
            koff_b = chunk_kt * _sc_k0_b
            sa = None
            if const_expr(want_a):
                sa = [
                    Vec(
                        fly.copy_atom_call_ssa(
                            [T.vec(1, T.i32)],
                            bs_copy,
                            sa_flat[
                                None,
                                rocdl.readfirstlane(T.i32, a_sc_base[mp] + koff_a) + sc_lane_a,
                            ],
                        )
                    )[0]
                    for mp in range_constexpr(m_pairs)
                ]
            sb = None
            if const_expr(want_b):
                _bs_b_dedup = _bs_b and (BN // num_waves) <= 128 and (128 % (BN // num_waves) == 0)
                if const_expr(_bs_b_dedup):
                    _sb0 = Vec(
                        fly.copy_atom_call_ssa(
                            [T.vec(1, T.i32)],
                            bs_copy,
                            sb_flat[
                                None,
                                rocdl.readfirstlane(T.i32, b_sc_base[0] + koff_b) + sc_lane_b,
                            ],
                        )
                    )[0]
                    sb = [_sb0 for _ in range_constexpr(n_pairs)]
                else:
                    sb = [
                        Vec(
                            fly.copy_atom_call_ssa(
                                [T.vec(1, T.i32)],
                                bs_copy,
                                sb_flat[
                                    None,
                                    rocdl.readfirstlane(T.i32, b_sc_base[np] + koff_b) + sc_lane_b,
                                ],
                            )
                        )[0]
                        for np in range_constexpr(n_pairs)
                    ]
            return sa, sb

        def compute(accs, av, bv, sa_v, sb_v, scale_shift=None):
            # tile_k=128: shift the active 128-K half of the shared 256-K word into the opsel's low bytes.
            # A row operand's packed word holds only this K-tile: nothing to shift.
            if const_expr(scale_shift is not None):
                if const_expr(not _row_a):
                    sa_v = [v.shrui(scale_shift) for v in sa_v]
                if const_expr(not _row_b):
                    sb_v = [v.shrui(scale_shift) for v in sb_v]
            if const_expr(BN < 128 and not _row_b):
                _bnsh = ((by_n + wave * (BN // num_waves)) % 32) // 16 * 8
                sb_v = [v.shrui(_bnsh) for v in sb_v]
            # kh OUTERMOST: consecutive MFMAs hit distinct accumulators (dense issue). Each
            # scaled MFMA = fx.gemm over rank-1 i32[4] A/B frags, e8m0 word on scale_a=/scale_b=.
            c_frags = [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(n_acc)]
            for idx in range_constexpr(n_acc):
                c_frags[idx].store(Vec(accs[idx]))
            for kh in range_constexpr(k_halves):
                for ni in range_constexpr(num_acc_n):
                    np_i, in_b = ni // 2, ni % 2
                    for mi in range_constexpr(m_chunks):
                        mp_i, im = mi // 2, mi % 2
                        cf = c_frags[mi * num_acc_n + ni]
                        fx.gemm(
                            scale_atoms[(kh * 2 + im, kh * 2 + in_b)],
                            cf,
                            av[mi * k_halves + kh],
                            bv[ni * k_halves + kh],
                            cf,
                            scale_a=sa_v[mp_i],
                            scale_b=sb_v[np_i],
                        )
            for idx in range_constexpr(n_acc):
                accs[idx] = c_frags[idx].load().ir_value()
            return accs

        def hot_loop_scheduler():
            # Interleave the MFMAs with the tile's vmem + A-LDS loads: preload all hints, then issue MFMAs 1-by-1.
            if const_expr(_row_any):
                # Scale LDS reads ahead of every DMA: an LDS read after an in-flight
                # buffer_load...lds may alias it, so the backend would wait vmcnt.
                rocdl.sched_dsrd(sched_num_sc_ds)
            rocdl.sched_vmem(sched_num_gmem)
            rocdl.sched_dsrd(sched_num_ds_load)
            for _ in range_constexpr(sched_mfma_total):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        accs_init = [fx.full(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)]

        def pf_main_loop():
            """The prefetch pipeline (see ``prefetch``): returns the accumulators."""
            k_loc = k_tiles_local
            steady = max(0, k_loc - PF)  # iterations that fetch tile + PF
            acc_t = [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(n_acc)]
            for t in acc_t:
                t.store(fx.full(4, 0.0, Float32))
            # B register set s holds the tiles kt0 + s (mod PF); static indices keep each
            # set in its registers across the unrolled loop.
            b_regs = [
                [fx.make_rmem_tensor(B_NDW, Int32) for _ in range_constexpr(num_acc_n * k_halves)]
                for _ in range_constexpr(PF)
            ]

            def prefetch_b(kt_abs, s):
                for d, o in zip(b_regs[s], load_b(kt_abs)):
                    d.store(o.load())

            # asyncmark group k holds tile k's LDS copies, here and in the loop (steady
            # iteration i fetches tile i + PF). A prologue chunk c joins the group of tile
            # min(c * SC_C, n_pro - 1): it lands before its first tile is read. Only tile
            # 0's group is waited for here.
            n_pro = min(PF, k_loc)
            for t in range_constexpr(n_pro):
                dma_a_to_lds(kt0 + t, fx.Int32(t))
                prefetch_b(kt0 + t, t)
                for c in range_constexpr(SC_D):
                    if const_expr(c * SC_C < k_loc and min(c * SC_C, n_pro - 1) == t):
                        for j in range_constexpr(SC_C):
                            dma_scale_slice(kt0 + c * SC_C, fx.Int32(j), fx.Int32(c))
                rocdl.asyncmark()
            rocdl.wait_asyncmark(n_pro - 1)
            gpu.barrier()

            def step(ivi, ph, fetch, pending):
                """Local K tile ivi, B set ph (static). fetch issues tile ivi + PF; pending
                is how many iterations' LDS copies (asyncmark groups) may stay in flight at
                its end."""
                kt = kt0 + ivi
                c_idx = ivi // SC_C
                c_j = ivi % SC_C
                # Every LDS read of this tile before this iteration's DMAs: an LDS read
                # issued behind an in-flight LDS DMA waits for it.
                sa_v, sb_v = load_sc_row(c_idx % SC_NB, c_j)
                av = read_a(ivi % (PF + 1))
                rocdl.sched_barrier(0)
                if const_expr(fetch):
                    dma_scale_slice(kt0 + (c_idx + SC_D) * SC_C, c_j, (c_idx + SC_D) % SC_NB)
                    dma_a_to_lds(kt + PF, (ivi + PF) % (PF + 1))
                    rocdl.asyncmark()
                    rocdl.sched_barrier(0)
                # The MFMAs read B set ph in place (a copy would wait for its loads); the
                # set is refilled with tile + PF once they have.
                accs = compute([t.load().ir_value() for t in acc_t], av, b_regs[ph], sa_v, sb_v)
                for t, v in zip(acc_t, accs):
                    t.store(Vec(v))
                if const_expr(fetch):
                    prefetch_b(kt + PF, ph)
                # Tile ivi + 1's copies have landed (B registers are waited for at their
                # MFMAs by the backend), and this tile's LDS reads are done before the
                # barrier lets a copy overwrite its slot.
                rocdl.wait_asyncmark(pending)
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()

            for g in range(steady // PF):
                for ph in range_constexpr(PF):
                    step(fx.Int32(g) * PF + ph, ph, True, PF - 1)
            for t in range_constexpr(steady - steady % PF, steady):
                step(fx.Int32(t), t % PF, True, PF - 1)
            # The last PF tiles, peeled: no fetch. Groups are tiles: tile t + 1's has
            # landed, the later ones may stay in flight.
            for t in range_constexpr(steady, k_loc):
                step(fx.Int32(t), t % PF, False, max(0, k_loc - t - 2))
            return [t.load().ir_value() for t in acc_t]

        if const_expr(_pf):
            accs = pf_main_loop()
        else:
            dma_a_to_lds(kt0, fx.Int32(0))
            if const_expr(_row_any):
                for j in range_constexpr(SC_C):
                    dma_scale_slice(kt0, fx.Int32(j), fx.Int32(0))
            rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            gpu.barrier()
            for iv, state in range(0, k_tiles_local, 1, init=accs_init):
                accs = list(state)
                ivi = fx.Int32(iv)
                cur = ivi % 2
                nxt = (ivi + 1) % 2
                kt = kt0 + ivi  # absolute K-tile for A/B/scale addressing
                nkt = ivi + 1
                # clamp last-iter prefetch to the local last tile, then rebase to absolute
                pf_kt = kt0 + (nkt - nkt // k_tiles_local)
                chunk_kt = kt if tiles_per_chunk == 1 else kt // tiles_per_chunk
                scale_shift = None if tiles_per_chunk == 1 else (kt % tiles_per_chunk) * 16
                if const_expr(_row_any):
                    # Next chunk's slice first (its scattered rows are the slowest load the
                    # end-of-iteration drain waits for), then this tile's scale reads ahead
                    # of A's: LDS returns in order.
                    c_idx = ivi // SC_C
                    c_j = ivi % SC_C
                    dma_scale_slice(kt0 + (c_idx + 1) * SC_C, c_j, (c_idx + 1) % 2)
                    sa_r, sb_r = load_sc_row(c_idx % 2, c_j)
                av = read_a(cur)
                bv = load_b(kt)
                if const_expr(_row_a and _row_b):
                    sa_v, sb_v = sa_r, sb_r
                else:
                    sa_v, sb_v = load_sc(chunk_kt, not _row_a, not _row_b)
                    if const_expr(_row_a):
                        sa_v = sa_r
                    if const_expr(_row_b):
                        sb_v = sb_r
                dma_a_to_lds(pf_kt, nxt)  # A DMA after B/scale loads -> overlaps the MFMAs
                accs = compute(accs, av, bv, sa_v, sb_v, scale_shift)
                hot_loop_scheduler()
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)  # drain the A DMA before the barrier
                gpu.barrier()
                results = yield accs
            accs = results

        # Epilogue via fx.copy: a lane owns 4 rows per (mi,ni) accm (row m*16+(l//16)*4+ii, col
        # base+l%16), c_stride apart.
        c_stride = N if c_row_stride < 0 else c_row_stride
        # split-K writes an fp32 partial slab (no cast); no-split writes bf16/fp16 out.
        if const_expr(k_batch > 1):
            store_elem = Float32
            _ebytes = 4
            # arg_c is tmp[batch*k_batch, M, N] fp32; this WG's slab index == bid_z.
            c_addr = arg_c + fx.Int64(bid_z) * fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(_ebytes)
        else:
            store_elem = out_elem
            _ebytes = 2
            c_addr = arg_c
            if const_expr(batch > 1):
                c_bstride = fx.Int64(i32_m) * fx.Int64(N) * 2 if c_batch_stride < 0 else fx.Int64(c_batch_stride)
                c_addr = c_addr + fx.Int64(bz_batch) * c_bstride

        c_tile_addr = c_addr + fx.Int64(bx_m) * fx.Int64(c_stride) * fx.Int64(_ebytes)
        _rows_rem = i32_m - fx.Int32(bx_m)
        _rows_wg = (_rows_rem < fx.Int32(BM)).select(_rows_rem, fx.Int32(BM))
        c_nrec = fx.Int64(_rows_wg) * fx.Int64(c_stride) * fx.Int64(_ebytes)
        c_ptr_ty = fx.PointerType.get(store_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=_ebytes)
        c_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(c_ptr_ty, c_tile_addr), fx.make_layout(1 << 28, 1))),
                max_size=False,
                num_records_bytes=c_nrec,
            ),
            fx.make_layout(1, 1),
        )
        if const_expr(k_batch > 1):
            c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), store_elem)
        else:
            c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), store_elem)
        c_rstride = fx.Int32(c_stride)
        col_w = by_n + wave * (BN // num_waves) + lane_mod_16
        for mi in range_constexpr(m_chunks):
            row_local = mi * 16 + lane_div_16 * 4  # relative to this WG (base folded into c_tile_addr)
            for ni in range_constexpr(num_acc_n):
                col = col_w + ni * 16
                acc = Vec(accs[mi * num_acc_n + ni]).to(store_elem)
                for ii in range_constexpr(4):
                    cf = fx.make_rmem_tensor(1, store_elem)
                    cf.store(fx.full(1, acc[ii], store_elem))
                    off = (row_local + ii) * c_rstride + col
                    fx.copy(c_copy, cf, c_flat[None, off])

    c_addr = fx.Int64(fx.ptrtoint(arg_c))
    a_addr = fx.Int64(fx.ptrtoint(arg_a))
    b_addr = fx.Int64(fx.ptrtoint(arg_b))
    sa_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
    sb_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
    if const_expr(waves_per_eu > 0):
        wpe = waves_per_eu
    else:
        wpe = None
    gx = (i32_m + (BM - 1)) // BM
    gy = i32_n // BN
    gz = batch * k_batch  # split-K: k_batch splits per (real) batch on grid.z
    kernel_gemm(
        c_addr,
        a_addr,
        b_addr,
        sa_addr,
        sb_addr,
        i32_m,
        i32_n,
        value_attrs={"rocdl.waves_per_eu": wpe},
    ).launch(grid=(gx, gy, gz), block=(num_threads, 1, 1), stream=stream)


# ── split-K reduce ────────────────────────────────────────────────────────────

_REDUCE_BLOCK = 256
_REDUCE_VEC = 2  # out elems per thread == one dword (2 fp32 in per slab)


@flyc.jit
def launch_splitk_reduce(
    arg_tmp: fx.Pointer,
    arg_out: fx.Pointer,
    n_out_dw: fx.Int32,  # output dwords = M*N // 2 (2 out elems per dword)
    slab_stride_dw: fx.Int32,  # dwords per split slab = M*N (fp32: 1 dword/elem)
    row_dw: fx.Int32,  # N // 2: output dwords per row
    ldc_dw: fx.Int32,  # Out's row stride // 2 (== row_dw unless strided)
    stream: fx.Stream,
    split_k: Constexpr[int],
    out_dtype: Constexpr[str],
    strided: Constexpr[bool] = False,
):
    """Sum ``split_k`` fp32 slabs of ``arg_tmp`` -> bf16/fp16 ``arg_out``.

    arg_tmp: (split_k, M, N) fp32 contiguous. arg_out: (M, N) out_dtype, row
    stride ``2 * ldc_dw`` elements when ``strided`` (a column slice of a wider
    output), else contiguous. One output dword (= 2 out elems = 2 fp32 inputs
    per slab) per thread; grid.x covers all.
    """
    if const_expr(out_dtype == "bf16"):
        out_elem = BFloat16
    else:
        out_elem = Float16

    @flyc.kernel
    def reduce_kernel(
        tmp: fx.Pointer,
        out: fx.Pointer,
        n_out_dw_i: fx.Int32,
        slab_dw_i: fx.Int32,
        row_dw_i: fx.Int32,
        ldc_dw_i: fx.Int32,
    ):
        frag_layout = fx.make_layout(_REDUCE_VEC, 1)
        vt = fx.Int32(fx.block_idx.x) * fx.Int32(_REDUCE_BLOCK) + fx.Int32(fx.thread_idx.x)
        slab_frags = slab_dw_i // fx.Int32(_REDUCE_VEC)

        def _tiled(ptr, elem, num_records_bytes):
            typed = fx.PointerType.get(
                elem.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=_REDUCE_VEC * elem.width // 8,
            )
            buf = fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(typed, fx.Int64(fx.ptrtoint(ptr))),
                        fx.make_layout(1 << 30, 1),
                    )
                ),
                max_size=False,
                num_records_bytes=num_records_bytes,
            )
            return fx.logical_divide(buf, frag_layout)

        in_t = _tiled(tmp, Float32, fx.Int64(slab_dw_i) * fx.Int64(split_k) * 4)
        if const_expr(strided):
            # Dwords from the first output element to one past the last.
            out_span_dw = (n_out_dw_i // row_dw_i - 1) * ldc_dw_i + row_dw_i
        else:
            out_span_dw = n_out_dw_i
        out_t = _tiled(out, out_elem, fx.Int64(out_span_dw) * 4)
        in_copy = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), Float32)  # 2 x f32
        out_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), out_elem)  # 2 x 16b

        if vt < n_out_dw_i:
            acc = fx.full(_REDUCE_VEC, 0.0, Float32)
            for sk in range_constexpr(split_k):
                frag = fx.make_rmem_tensor(frag_layout, Float32)
                fx.copy(
                    in_copy,
                    fx.slice(in_t, (None, vt + fx.Int32(sk) * slab_frags)),
                    frag,
                )
                acc = acc + frag.load()
            out_frag = fx.make_rmem_tensor(frag_layout, out_elem)
            out_frag.store(acc.to(out_elem))
            if const_expr(strided):
                out_vt = vt + (vt // row_dw_i) * (ldc_dw_i - row_dw_i)
            else:
                out_vt = vt
            fx.copy(out_copy, out_frag, fx.slice(out_t, (None, out_vt)))

    gx = (n_out_dw + (_REDUCE_BLOCK - 1)) // _REDUCE_BLOCK
    reduce_kernel(arg_tmp, arg_out, n_out_dw, slab_stride_dw, row_dw, ldc_dw).launch(
        grid=(gx, 1, 1), block=(_REDUCE_BLOCK, 1, 1), stream=stream
    )
