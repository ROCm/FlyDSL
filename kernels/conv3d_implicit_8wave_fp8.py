"""8-wave double-buffered implicit-GEMM conv3d (FP8, CDNA4 only).

x: (N, C, D, H, W) bf16 NCDHW, weight: (K, C, T, R, S) bf16 KCTRS.
Returns (N, K, Do, Ho, Wo) bf16. Requires gfx95x; C%128==0, CRS%128==0, NPQ%128==0.
"""

import functools
import weakref

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr
from flydsl.expr.typing import T
from kernels.fp8_gemm_utils import Mfma16x16x128, make_fp8_buffer_tensor, pack_i32x4_i32x8

TILE_M = 128
TILE_N = 128
TILE_K = 128
STAGES = 2

WAVE_M = 2
WAVE_N = 4
WARP_SIZE = 64
BLOCK_THREADS = WAVE_M * WAVE_N * WARP_SIZE

MFMA_M = 16
MFMA_N = 16
MFMA_C_VALUES = 4

HALF_M = TILE_M // 2
HALF_N = TILE_N // 2
QM_STEPS = HALF_M // WAVE_M // MFMA_M
QN_STEPS = HALF_N // WAVE_N // MFMA_N
N_SUB = QM_STEPS * QN_STEPS

assert QM_STEPS == 2 and QN_STEPS == 1

LDG_VEC = 16
HALF_TILE_VECS = HALF_M * TILE_K // (LDG_VEC * BLOCK_THREADS)
assert HALF_TILE_VECS == 1

LDS_A_SIZE = STAGES * TILE_M * TILE_K
LDS_B_SIZE = STAGES * TILE_N * TILE_K
PACK_BLOCK_THREADS = 256

PACK_TR_TILE = 64
PACK_TR_VEC = 8
PACK_TR_THREADS = 256
PACK_TR_VPL = PACK_TR_TILE // PACK_TR_VEC
PACK_TR_ITERS = (PACK_TR_TILE * PACK_TR_TILE) // (PACK_TR_VEC * PACK_TR_THREADS)
PACK_TR_PAD = 8
PACK_TR_LDS_S = PACK_TR_TILE + PACK_TR_PAD

_WEIGHT_FP8_CACHE = {}


def _run_compiled(exe, *args):
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)


@functools.lru_cache(maxsize=64)
def compile_pack_activation_ncdhw_bf16_to_ndhwc_fp8(n, c, d, h, width):
    """Pack activation BF16 NCDHW -> FP8 bytes in NDHWC order (transpose + cast)."""
    assert c % PACK_TR_VEC == 0, f"tiled FP8 pack needs C % {PACK_TR_VEC} == 0, got C={c}"
    dhw = d * h * width
    assert dhw % PACK_TR_VEC == 0, f"tiled FP8 pack needs DHW % {PACK_TR_VEC} == 0, got DHW={dhw}"
    total_bytes = n * c * dhw
    grid_s = (dhw + PACK_TR_TILE - 1) // PACK_TR_TILE
    grid_c = (c + PACK_TR_TILE - 1) // PACK_TR_TILE
    elem_ty = fx.BFloat16

    @flyc.kernel(known_block_size=[PACK_TR_THREADS, 1, 1])
    def pack_x_kernel(out: fx.Tensor, x: fx.Tensor):
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=False, num_records_bytes=total_bytes)
        x_rsrc = buffer_ops.create_buffer_resource(x, max_size=False, num_records_bytes=total_bytes * 2)
        lds_alloc = fx.SharedAllocator(static=False)
        lds = lds_alloc.allocate(fx.Array[elem_ty, PACK_TR_TILE * PACK_TR_LDS_S, 16]).peek()

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(PACK_TR_VEC, elem_ty)

        class BF16Ty:
            ir_type = elem_ty.ir_type

        tid = fx.thread_idx.x
        s0 = fx.block_idx.x * PACK_TR_TILE
        c0 = fx.block_idx.y * PACK_TR_TILE
        nb = fx.block_idx.z
        in_base = nb * c * dhw
        out_base = nb * dhw * c

        def lds_store_vec8(elem_offset, value):
            base = fx.Int64(fx.ptrtoint(lds.ptr)) + fx.Int64(elem_offset * 2)
            ptr = buffer_ops.create_llvm_ptr(base, address_space=3)
            llvm.StoreOp(value, ptr, alignment=16)

        def lds_load_scalar(elem_offset):
            u8 = fx.recast_iter(fx.Uint8, lds.ptr)
            return fx.ptr_load(u8 + fx.Int32(elem_offset * 2), result_type=BF16Ty)

        # Read coalesced along contiguous S from NCDHW into LDS[c_local, s_local].
        for i in range_constexpr(PACK_TR_ITERS):
            lin = tid + i * PACK_TR_THREADS
            rc = lin // PACK_TR_VPL
            sv = (lin % PACK_TR_VPL) * PACK_TR_VEC
            cc = c0 + rc
            ss = s0 + sv
            valid = (cc < c) & (ss < dhw)
            g = arith.index_cast(T.i32, in_base + cc * dhw + ss)
            safe = arith.select(valid, g, arith.constant(0, type=T.i32))
            v = buffer_ops.buffer_load(x_rsrc, safe, vec_width=PACK_TR_VEC, dtype=elem_ty)
            lds_store_vec8(rc * PACK_TR_LDS_S + sv, v)

        llvm.InlineAsmOp(None, [], "s_waitcnt lgkmcnt(0)\n\ts_barrier", "", has_side_effects=True)

        # Read LDS transposed and store FP8-packed dwords along contiguous C.
        for i in range_constexpr(PACK_TR_ITERS):
            lin = tid + i * PACK_TR_THREADS
            rs = lin // PACK_TR_VPL
            cv = (lin % PACK_TR_VPL) * PACK_TR_VEC
            ss = s0 + rs
            cc = c0 + cv
            valid = arith.andi(ss < dhw, cc < c)
            store_if = scf.IfOp(valid, results_=[], has_else=False)
            with ir.InsertionPoint(store_if.then_block):
                scalars = [
                    lds_load_scalar((cv + j) * PACK_TR_LDS_S + rs).to(fx.Float32) for j in range_constexpr(PACK_TR_VEC)
                ]
                lo0 = fx.rocdl.cvt_pk_fp8_f32(T.i32, scalars[0], scalars[1], fx.Int32(0), False)
                p0 = fx.rocdl.cvt_pk_fp8_f32(T.i32, scalars[2], scalars[3], lo0, True)
                lo1 = fx.rocdl.cvt_pk_fp8_f32(T.i32, scalars[4], scalars[5], fx.Int32(0), False)
                p1 = fx.rocdl.cvt_pk_fp8_f32(T.i32, scalars[6], scalars[7], lo1, True)
                packed = Vec.from_elements([p0, p1], fx.Int32)
                byte_off = out_base + ss * c + cc
                buffer_ops.buffer_store(packed, out_rsrc, byte_off, offset_is_bytes=True)
                scf.YieldOp([])

    @flyc.jit
    def launch(out: fx.Tensor, x: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        pack_x_kernel(out, x).launch(
            grid=(grid_s, grid_c, n),
            block=(PACK_TR_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=64)
def compile_pack_weight_kctrs_bf16_to_ktrsc_fp8(k, c, kt, kh, kw):
    """Pack weight BF16 KCTRS -> FP8 bytes in KTRSC order (transpose + cast)."""
    assert c % 4 == 0, f"FP8 pack stores 4 channels per dword, got C={c}"
    trs = kt * kh * kw
    total_bytes = k * c * trs
    total_packs = total_bytes // 4
    grid_x = (total_packs + PACK_BLOCK_THREADS - 1) // PACK_BLOCK_THREADS

    @flyc.kernel(known_block_size=[PACK_BLOCK_THREADS, 1, 1])
    def pack_w_kernel(out: fx.Tensor, weight: fx.Tensor):
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=False, num_records_bytes=total_bytes)
        w_rsrc = buffer_ops.create_buffer_resource(weight, max_size=False, num_records_bytes=total_bytes * 2)

        pack_idx = fx.block_idx.x * PACK_BLOCK_THREADS + fx.thread_idx.x
        if pack_idx < fx.Index(total_packs):
            c_pack = pack_idx % (c // 4)
            rest = pack_idx // (c // 4)
            c_base = c_pack * 4
            k_idx = rest // trs
            trs_idx = rest % trs

            src_base = (k_idx * c + c_base) * trs + trs_idx
            v0 = buffer_ops.buffer_load(w_rsrc, src_base, vec_width=1, dtype=fx.BFloat16).extf(T.f32)
            v1 = buffer_ops.buffer_load(w_rsrc, src_base + fx.Index(trs), vec_width=1, dtype=fx.BFloat16).extf(T.f32)
            v2 = buffer_ops.buffer_load(w_rsrc, src_base + fx.Index(2 * trs), vec_width=1, dtype=fx.BFloat16).extf(
                T.f32
            )
            v3 = buffer_ops.buffer_load(w_rsrc, src_base + fx.Index(3 * trs), vec_width=1, dtype=fx.BFloat16).extf(
                T.f32
            )
            lo = fx.rocdl.cvt_pk_fp8_f32(T.i32, v0, v1, fx.Int32(0), False)
            packed = fx.rocdl.cvt_pk_fp8_f32(T.i32, v2, v3, lo, True)
            buffer_ops.buffer_store(packed, out_rsrc, pack_idx * 4, offset_is_bytes=True)

    @flyc.jit
    def launch(out: fx.Tensor, weight: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        pack_w_kernel(out, weight).launch(
            grid=(grid_x, 1, 1),
            block=(PACK_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=64)
def compile_conv3d_implicit_8wave_fp8(
    n, c, d, h, width, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias=False, splitk=1
):
    """Compile the FP8 conv: x is NDHWC FP8 bytes, weight is KTRSC FP8 bytes."""
    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (width + 2 * pw - kw) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = c * kt * kh * kw
    k_tiles = crs // TILE_K

    assert c % LDG_VEC == 0, f"FP8 vector load needs C % {LDG_VEC} == 0, got C={c}"
    assert crs % TILE_K == 0, f"FP8 MFMA path needs CRS % {TILE_K} == 0, got CRS={crs}"
    assert npq % TILE_M == 0, f"FP8 path needs NPQ % {TILE_M} == 0, got NPQ={npq}"
    assert k % TILE_N == 0, f"FP8 path needs K % {TILE_N} == 0, got K={k}"
    assert k_tiles >= 1

    splitk = max(1, min(splitk, k_tiles))
    while k_tiles % splitk != 0:
        splitk -= 1
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1

    grid_m = npq // TILE_M
    grid_n = k // TILE_N
    elem_ty = fx.Float8E4M3FN

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_8wave_fp8_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
        x_num_records = n * d * h * width * c
        y_rsrc = buffer_ops.create_buffer_resource(
            y, max_size=False, num_records_bytes=npq * k * (4 if const_expr(use_splitk) else 2)
        )
        if const_expr(has_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(bias, max_size=False, num_records_bytes=k * 4)

        f8_ir_t = elem_ty.ir_type
        x_buf = make_fp8_buffer_tensor(x, f8_ir_t)
        x_div = fx.logical_divide(x_buf, fx.make_layout(1, 1))
        w_buf = make_fp8_buffer_tensor(weight, f8_ir_t)
        w_div = fx.logical_divide(w_buf, fx.make_layout(1, 1))

        lds_alloc = fx.SharedAllocator(static=False)
        a_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_A_SIZE, 16]).peek()
        b_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_B_SIZE, 16]).peek()

        tid = fx.thread_idx.x
        m_offset = fx.block_idx.x * TILE_M
        n_offset = fx.block_idx.y * TILE_N
        if const_expr(use_splitk):
            k_off = fx.block_idx.z * (tiles_per_split * TILE_K)
        else:
            k_off = fx.Index(0)

        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        wave_m = wid // WAVE_N
        wave_n = wid % WAVE_N
        lane_div_16 = lane // MFMA_N
        lane_mod_16 = lane % MFMA_N
        c_m_vec = lane_div_16 * MFMA_C_VALUES
        c_n = lane_mod_16

        mfma = Mfma16x16x128(QM_STEPS, QN_STEPS)
        acc00 = [mfma.zero_value for _ in range_constexpr(N_SUB)]
        acc01 = [mfma.zero_value for _ in range_constexpr(N_SUB)]
        acc10 = [mfma.zero_value for _ in range_constexpr(N_SUB)]
        acc11 = [mfma.zero_value for _ in range_constexpr(N_SUB)]

        Vec = fx.Vector

        class Vec16U8Ty:
            ir_type = Vec.make_type(16, fx.Uint8)

        def barrier():
            # Wait for the in-flight global->LDS copies (vmcnt) and LDS reads
            # (lgkmcnt) of this stage before the next stage reuses the buffers.
            llvm.InlineAsmOp(None, [], "s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier", "", has_side_effects=True)

        def a_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_M + row) * TILE_K + col

        def b_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_N + row) * TILE_K + col

        def in_range(v, hi):
            return (v >= 0) & (v < fx.Index(hi))

        g2s_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        LdsPtrTy = fx.PointerType.get(f8_ir_t, 2, 512)

        def copy_g2s(src_div, lds_array, elem_offset, src_elem):
            lds_byte_addr = fx.Int32(fx.ptrtoint(lds_array.ptr)) + fx.Int32(elem_offset)
            lds_ptr = fx.inttoptr(LdsPtrTy, lds_byte_addr)
            dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
            src = fx.slice(src_div, (None, fx.Int32(src_elem)))
            fx.copy(g2s_atom, src, dst)

        # ---- 3D im2col gather: global FP8 -> LDS (direct async copy) ----
        def g2s_a_half(stage, m_half, k_base):
            linear = tid * LDG_VEC
            local_m = linear // TILE_K
            local_k = linear % TILE_K
            row = m_offset + m_half * HALF_M + local_m
            row_valid = row < fx.Index(npq)
            n_idx = row // dhw
            rem = row % dhw
            ot = rem // hw_o
            rem2 = rem % hw_o
            oh = rem2 // wo
            ow = rem2 % wo
            lds_elem = a_lds_off(stage, fx.Index(m_half * HALF_M) + local_m, local_k)
            k_abs = fx.Index(k_base) + fx.Index(local_k)
            cc = k_abs % c
            ckk = k_abs // c
            kw_i = ckk % kw
            ckk2 = ckk // kw
            kh_i = ckk2 % kh
            kt_i = ckk2 // kh
            in_t = ot * st + kt_i - pt
            in_h = oh * sh + kh_i - ph
            in_w = ow * sw + kw_i - pw
            valid_data = row_valid & in_range(in_t, d) & in_range(in_h, h) & in_range(in_w, width)
            g_elem = (((n_idx * d + in_t) * h + in_h) * width + in_w) * c + cc
            g_elem_i = arith.index_cast(T.i32, g_elem)
            safe_elem = arith.select(valid_data, g_elem_i, arith.constant(x_num_records, type=T.i32))
            copy_g2s(x_div, a_lds, lds_elem, safe_elem)

        def g2s_b_half(stage, n_half, k_base):
            linear = tid * LDG_VEC
            local_n = linear // TILE_K
            local_k = linear % TILE_K
            col = n_offset + fx.Index(n_half * HALF_N) + local_n
            lds_elem = b_lds_off(stage, fx.Index(n_half * HALF_N) + local_n, local_k)
            g_elem = col * crs + (fx.Index(k_base) + fx.Index(local_k))
            g_elem_i = arith.index_cast(T.i32, g_elem)
            copy_g2s(w_div, b_lds, lds_elem, g_elem_i)

        def g2s_full_tile(stage, k_base):
            g2s_a_half(stage, 0, k_base)
            g2s_a_half(stage, 1, k_base)
            g2s_b_half(stage, 0, k_base)
            g2s_b_half(stage, 1, k_base)

        def lds_load_vec16(lds_array, elem_offset):
            u8_ptr = fx.recast_iter(fx.Uint8, lds_array.ptr)
            return fx.ptr_load(u8_ptr + fx.Int32(elem_offset), result_type=Vec16U8Ty)

        def lds_load_pack(lds_array, elem_offset):
            lo = lds_load_vec16(lds_array, elem_offset).bitcast(fx.Int32)
            hi = lds_load_vec16(lds_array, elem_offset + fx.Index(64)).bitcast(fx.Int32)
            return pack_i32x4_i32x8(lo, hi)

        def read_a_vec(stage, m_half, wm):
            a_row = m_half * HALF_M + wave_m * (HALF_M // WAVE_M) + wm * MFMA_M + lane_mod_16
            a_col = lane_div_16 * 16
            return lds_load_pack(a_lds, a_lds_off(stage, fx.Index(a_row), fx.Index(a_col)))

        def read_b_vec(stage, n_half, wn):
            b_row = n_half * HALF_N + wave_n * (HALF_N // WAVE_N) + wn * MFMA_N + lane_mod_16
            b_col = lane_div_16 * 16
            return lds_load_pack(b_lds, b_lds_off(stage, fx.Index(b_row), fx.Index(b_col)))

        def setprio(level):
            llvm.InlineAsmOp(None, [], f"s_setprio {level}", "", has_side_effects=True)

        def mfma_one(a, b, c_acc):
            out = mfma._do_mma(a, b, c_acc)
            fx.rocdl.sched_mfma(1)
            return out

        # ---- software-pipelined main loop ----
        stage = 0
        next_stage = 1
        g2s_full_tile(stage, k_off)
        barrier()
        a0_0 = read_a_vec(stage, 0, 0)
        a0_1 = read_a_vec(stage, 0, 1)
        b0_0 = read_b_vec(stage, 0, 0)
        fx.rocdl.sched_dsrd(3)

        for kt_idx in range_constexpr(tiles_per_split):
            # prefetch next tile: global -> LDS (async)
            if const_expr(kt_idx + 1 < tiles_per_split):
                g2s_full_tile(next_stage, k_off + (kt_idx + 1) * TILE_K)

            setprio(1)
            # acc00 = a0 . b0
            acc00[0] = mfma_one(a0_0, b0_0, acc00[0])
            b1_0 = read_b_vec(stage, 1, 0)
            fx.rocdl.sched_dsrd(1)
            acc00[1] = mfma_one(a0_1, b0_0, acc00[1])

            # acc01 = a0 . b1
            acc01[0] = mfma_one(a0_0, b1_0, acc01[0])
            a1_0 = read_a_vec(stage, 1, 0)
            fx.rocdl.sched_dsrd(1)
            acc01[1] = mfma_one(a0_1, b1_0, acc01[1])
            a1_1 = read_a_vec(stage, 1, 1)
            fx.rocdl.sched_dsrd(1)

            # acc10 = a1 . b0
            acc10[0] = mfma_one(a1_0, b0_0, acc10[0])
            acc10[1] = mfma_one(a1_1, b0_0, acc10[1])

            # acc11 = a1 . b1
            acc11[0] = mfma_one(a1_0, b1_0, acc11[0])
            acc11[1] = mfma_one(a1_1, b1_0, acc11[1])
            setprio(0)

            if const_expr(kt_idx + 1 < tiles_per_split):
                barrier()
                stage = next_stage
                next_stage = (stage + 1) % STAGES
                a0_0 = read_a_vec(stage, 0, 0)
                a0_1 = read_a_vec(stage, 0, 1)
                b0_0 = read_b_vec(stage, 0, 0)
                fx.rocdl.sched_dsrd(3)

        def store_half_pair(acc0, acc1, m_half):
            for wm in range_constexpr(QM_STEPS):
                row_base = m_offset + m_half * HALF_M + wave_m * (HALF_M // WAVE_M) + wm * MFMA_M + c_m_vec
                for n_half in range_constexpr(2):
                    acc = acc0 if const_expr(n_half == 0) else acc1
                    for wn in range_constexpr(QN_STEPS):
                        col = n_offset + fx.Index(n_half * HALF_N + wave_n * (HALF_N // WAVE_N) + wn * MFMA_N) + c_n
                        # Under split-K the partial sums accumulate atomically into
                        # FP32; bias is a single per-output add left to the host
                        # post-pass (adding it per z-slice would scale it by splitk).
                        if const_expr(has_bias and not use_splitk):
                            bias_val = fx.Float32(buffer_ops.buffer_load(bias_rsrc, col, vec_width=1, dtype=fx.Float32))
                        acc_vec = Vec(acc[wm * QN_STEPS + wn])
                        for i in range_constexpr(MFMA_C_VALUES):
                            row = row_base + i
                            out = acc_vec[i]
                            if const_expr(use_splitk):
                                off_b = arith.index_cast(T.i32, (row * k + col) * 4)
                                z0 = arith.constant(0, type=T.i32)
                                fx.rocdl.raw_ptr_buffer_atomic_fadd(out, y_rsrc, off_b, z0, z0)
                            else:
                                if const_expr(has_bias):
                                    out = out + bias_val
                                buffer_ops.buffer_store(out.to(fx.BFloat16), y_rsrc, row * k + col)

        store_half_pair(acc00, acc01, 0)
        store_half_pair(acc10, acc11, 1)

    @flyc.jit
    def launch(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv3d_8wave_fp8_kernel(
            y,
            x,
            weight,
            bias,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_m, grid_n, splitk), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def _normalize_3(v):
    if isinstance(v, int):
        return (v, v, v)
    assert len(v) == 3, f"expected int or length-3 tuple, got {v!r}"
    return tuple(v)


def _choose_splitk(npq, crs, k, device):
    if npq % TILE_M != 0 or k % TILE_N != 0:  # atomic accumulation needs clean tiles
        return 1
    base = (npq // TILE_M) * (k // TILE_N)
    k_tiles = crs // TILE_K
    if npq < 4096 or k_tiles < 16:  # too small: atomic/convert overhead dominates
        return 1
    try:
        num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        num_cu = 256
    if base >= (3 * num_cu) // 4:  # base grid already (nearly) fills the machine
        return 1
    sk = min(4, max(1, num_cu // base), k_tiles)  # aim to roughly fill the CUs
    while sk > 1 and k_tiles % sk != 0:  # prefer a divisor (no overhang)
        sk -= 1
    return sk


def _resolve_splitk(splitk, npq, crs, k, device):
    sk = _choose_splitk(npq, crs, k, device) if splitk is None else max(1, int(splitk))
    k_tiles = crs // TILE_K
    sk = max(1, min(sk, k_tiles))
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    return sk


def pack_activation_ncdhw_bf16_to_ndhwc_fp8(x: torch.Tensor, stream=None) -> torch.Tensor:
    """BF16 NCDHW activation -> int8 storage of FP8 E4M3FN in NDHWC order."""
    assert x.dtype == torch.bfloat16, f"expected BF16 activation, got {x.dtype}"
    n, c, d, h, width = x.shape
    s = d * h * width
    out_numel = n * d * h * width * c
    if not (x.is_contiguous() and c % PACK_TR_VEC == 0 and s % PACK_TR_VEC == 0):
        return x.to(torch.float8_e4m3fn).permute(0, 2, 3, 4, 1).contiguous().view(torch.int8).view(-1)
    out = torch.empty((out_numel,), device=x.device, dtype=torch.int8)
    exe = compile_pack_activation_ncdhw_bf16_to_ndhwc_fp8(n, c, d, h, width)
    _run_compiled(
        exe,
        flyc.from_torch_tensor(out),
        flyc.from_torch_tensor(x.contiguous()),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out


def pack_weight_kctrs_bf16_to_ktrsc_fp8(weight: torch.Tensor, stream=None) -> torch.Tensor:
    """BF16 KCTRS weight -> int8 storage of FP8 E4M3FN in KTRSC order."""
    assert weight.dtype == torch.bfloat16, f"expected BF16 weight, got {weight.dtype}"
    k, c, kt, kh, kw = weight.shape
    assert c % 4 == 0, f"FP8 pack stores 4 channels per dword, got C={c}"
    out_numel = k * c * kt * kh * kw
    out = torch.empty((out_numel,), device=weight.device, dtype=torch.int8)
    exe = compile_pack_weight_kctrs_bf16_to_ktrsc_fp8(k, c, kt, kh, kw)
    _run_compiled(
        exe,
        flyc.from_torch_tensor(out),
        flyc.from_torch_tensor(weight.contiguous()),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out


def _prep_weight_fp8(weight: torch.Tensor, stream=None) -> torch.Tensor:
    """Pack + cache the FP8 weight by source tensor identity (weights are reused)."""
    key = id(weight)
    ent = _WEIGHT_FP8_CACHE.get(key)
    if ent is not None and ent[0]() is weight:
        return ent[1]
    out = pack_weight_kctrs_bf16_to_ktrsc_fp8(weight, stream=stream)
    _WEIGHT_FP8_CACHE[key] = (weakref.ref(weight), out)
    return out


def conv3d_implicit_8wave_fp8(x, weight, bias=None, stride=1, padding=0, splitk=None, stream=None):
    """FP8 (E4M3FN) implicit conv3d. Same interface as the BF16 v6mb kernel.

    x: (N, C, D, H, W) bf16, weight: (K, C, T, R, S) bf16. Inputs are packed once
    to FP8 (NDHWC activation / cached KTRSC weight), then run through the CDNA4
    16x16x128 MFMA conv with a software-pipelined loop and optional split-K.
    Returns bf16 (N, K, Do, Ho, Wo). splitk=None auto-dispatches."""
    n, c, d, h, width = x.shape
    k, wc, kt, kh, kw = weight.shape
    assert c == wc, f"in-channel mismatch: x has {c}, weight has {wc}"
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    st, sh, sw = _normalize_3(stride)
    pt, ph, pw = _normalize_3(padding)
    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (width + 2 * pw - kw) // sw + 1
    npq = n * do * ho * wo
    crs = c * kt * kh * kw

    assert c % LDG_VEC == 0, f"FP8 vector load needs C % {LDG_VEC} == 0, got C={c}"
    assert crs % TILE_K == 0, f"FP8 MFMA path needs CRS % {TILE_K} == 0, got CRS={crs}"
    assert npq % TILE_M == 0, f"FP8 path needs NPQ % {TILE_M} == 0, got NPQ={npq}"
    assert k % TILE_N == 0, f"FP8 path needs K % {TILE_N} == 0, got K={k}"

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    x_arg = pack_activation_ncdhw_bf16_to_ndhwc_fp8(x, stream=launch_stream)
    w_arg = _prep_weight_fp8(weight, stream=launch_stream)

    has_bias = bias is not None
    bias_arg = (
        bias.to(device=x.device, dtype=torch.float32).contiguous().view(-1)
        if has_bias
        else torch.empty(1, device=x.device, dtype=torch.float32)
    )
    if has_bias:
        assert bias_arg.numel() == k, f"bias must have {k} elements, got {bias_arg.numel()}"

    sk = _resolve_splitk(splitk, npq, crs, k, x.device)
    use_splitk = sk > 1
    y = (
        torch.zeros((npq, k), device=x.device, dtype=torch.float32)
        if use_splitk
        else torch.empty((npq, k), device=x.device, dtype=torch.bfloat16)
    )
    exe = compile_conv3d_implicit_8wave_fp8(n, c, d, h, width, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias, sk)
    _run_compiled(
        exe,
        flyc.from_torch_tensor(y.view(-1)),
        flyc.from_torch_tensor(x_arg),
        flyc.from_torch_tensor(w_arg),
        flyc.from_torch_tensor(bias_arg),
        launch_stream,
    )
    if use_splitk:
        if has_bias:
            y = y + bias_arg.view(1, k)
        y = y.to(torch.bfloat16)
    return y.view(n, do, ho, wo, k).permute(0, 4, 1, 2, 3)


__all__ = ["conv3d_implicit_8wave_fp8"]
