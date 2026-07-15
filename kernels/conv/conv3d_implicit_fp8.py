"""Double-buffered implicit-GEMM conv3d (FP8, CDNA4 only).

x: (N, C, D, H, W) fp8 (E4M3FN) NCDHW, weight: (K, C, T, R, S) fp8 KCTRS.
Returns (N, K, Do, Ho, Wo) bf16. Requires gfx95x; C%128==0, CRS%128==0, NPQ%128==0.

Inputs are consumed natively as FP8 (no internal bf16->fp8 cast); the host entry
only reorders them into the kernel-native NDHWC / KTRSC layout (a memory-bound
layout transpose, weight cached by identity).
"""

import functools
import os
import weakref

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace as _TAS
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr
from flydsl.expr.utils.arith import ArithValue as _ArithValue
from kernels.gemm.fp8_gemm_utils import Mfma16x16x128, make_fp8_buffer_tensor, pack_i32x4_i32x8


def _make_fp8_buffer_tensor_from_addr(addr_i64, fp8_ir_t, ref_buf_tensor):
    """Create a rebased FP8 buffer tensor from a raw i64 byte address.

    Mirrors make_buffer_tensor() in rocdl/universal.py: converts the i64 base
    address to an llvm ptr, wraps it in a BufferDesc pointer via make_ptr, and
    returns a Tensor view over the same layout as ref_buf_tensor. A 2 GB
    num_records bound (BIG_ASYNC_NR) ensures OOB-routed padding taps zero.
    """
    from flydsl.expr.rocdl.universal import make_ptr

    BIG_ASYNC_NR = 0x80000000  # 2 GB
    alignment = fx.PointerType(fx.get_iter(ref_buf_tensor).type).alignment
    f8_ptr_ty = fx.PointerType.get(
        elem_ty=fp8_ir_t,
        address_space=_TAS.BufferDesc,
        alignment=alignment,
    )
    # Convert i64 address -> llvm.ptr (the base pointer for the buffer resource)
    llvm_ptr_ty = _ir.Type.parse("!llvm.ptr")
    addr_val = addr_i64.ir_value() if hasattr(addr_i64, "ir_value") else addr_i64
    base_ptr = _llvm.IntToPtrOp(llvm_ptr_ty, addr_val).result
    # Wrap in a BufferDesc pointer using the same make_ptr path as make_buffer_tensor
    buf_ptr = make_ptr(
        f8_ptr_ty,
        [
            _ArithValue(base_ptr),
            fx.Int16(0).ir_value(),
            fx.Int64(BIG_ASYNC_NR).ir_value(),
            fx.Int32(buffer_ops._get_buffer_flags()).ir_value(),
        ],
    )
    return fx.Tensor(fx.make_view(buf_ptr, fx.get_layout(ref_buf_tensor)))


# TILE_K is pinned to the FP8 MFMA k-dim (mfma_f32_16x16x128 -> 128). The tile
# size (TILE_M/TILE_N) and wave layout (WAVE_M/WAVE_N) are compile-time
# parameters of compile_conv3d_implicit_fp8 (autotuned per shape).
TILE_K = 128
STAGES = 2
WARP_SIZE = 64

MFMA_M = 16
MFMA_N = 16
MFMA_C_VALUES = 4

LDG_VEC = 16

# gfx950 (CDNA4) LDS capacity (this kernel is CDNA4-only).
LDS_CAPACITY_BYTES = 163840
FP8_BYTES = 1

# Default tile config = the original hand-tuned 128x128 shape (2x4 = 8 waves);
# the autotuner picks larger tiles (e.g. 256x256x4x4 = 16 waves) per shape.
DEFAULT_TILE = (128, 128, 2, 4)


def _autotune_enabled():
    return os.environ.get("FLYDSL_CONV3D_AUTOTUNE", "0").lower() in ("1", "true", "yes")


_WEIGHT_FP8_CACHE = {}


@functools.lru_cache(maxsize=256)
def compile_conv3d_implicit_fp8(
    n, c, d, h, width, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias=False, splitk=1, tile=DEFAULT_TILE
):
    """Compile the FP8 conv: x is NDHWC FP8 bytes, weight is KTRSC FP8 bytes."""
    TILE_M, TILE_N, WAVE_M, WAVE_N = tile
    BLOCK_THREADS = WAVE_M * WAVE_N * WARP_SIZE
    MI_M = TILE_M // WAVE_M // MFMA_M
    MI_N = TILE_N // WAVE_N // MFMA_N
    N_ACC = MI_M * MI_N
    WARP_M = MI_M * MFMA_M
    WARP_N = MI_N * MFMA_N
    LDS_A_SIZE = STAGES * TILE_M * TILE_K
    LDS_B_SIZE = STAGES * TILE_N * TILE_K
    # Rows of a tile written per full pass of the block (each thread stores one
    # LDG_VEC-wide chunk along K); TILE_K == vec chunks per row here.
    A_ROW_BLKS = TILE_M // (BLOCK_THREADS * LDG_VEC // TILE_K)
    B_ROW_BLKS = TILE_N // (BLOCK_THREADS * LDG_VEC // TILE_K)

    assert TILE_K == 128
    assert TILE_M % (WAVE_M * MFMA_M) == 0, f"TILE_M={TILE_M} not divisible by WAVE_M*16"
    assert TILE_N % (WAVE_N * MFMA_N) == 0, f"TILE_N={TILE_N} not divisible by WAVE_N*16"
    assert (TILE_M * TILE_K) % (BLOCK_THREADS * LDG_VEC) == 0, "A tile not a whole multiple of block loads"
    assert (TILE_N * TILE_K) % (BLOCK_THREADS * LDG_VEC) == 0, "B tile not a whole multiple of block loads"
    assert A_ROW_BLKS >= 1 and B_ROW_BLKS >= 1
    assert c % LDG_VEC == 0, f"FP8 vector load needs C % {LDG_VEC} == 0, got C={c}"
    assert BLOCK_THREADS <= 1024, f"BLOCK_THREADS={BLOCK_THREADS} exceeds 1024"
    lds_bytes = STAGES * (TILE_M + TILE_N) * TILE_K * FP8_BYTES
    assert lds_bytes <= LDS_CAPACITY_BYTES, f"LDS {lds_bytes} exceeds {LDS_CAPACITY_BYTES}"

    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (width + 2 * pw - kw) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = c * kt * kh * kw
    k_tiles = (crs + TILE_K - 1) // TILE_K
    BIG_IN = (n * c * d * h * width) > 0x7FFFFFFF
    BIG_OUT = (n * k * do * ho * wo * 2) > 0x7FFFFFFF
    temporal_only_fast = (
        kh == 1
        and kw == 1
        and st == 1
        and sh == 1
        and sw == 1
        and ph == 0
        and pw == 0
        and do == d
        and ho == h
        and wo == width
    )

    assert k_tiles >= 1

    splitk = max(1, min(splitk, k_tiles))
    while k_tiles % splitk != 0:
        splitk -= 1
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1

    grid_m = (npq + TILE_M - 1) // TILE_M
    grid_n = (k + TILE_N - 1) // TILE_N
    elem_ty = fx.Float8E4M3FN

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_implicit_fp8_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
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

        if const_expr(BIG_IN):
            nbase = m_offset // dhw
            ot_base0 = (m_offset % dhw) // hw_o
            base_t = ot_base0 - fx.Index(pt)
            base_t = arith.select(base_t < fx.Index(0), fx.Index(0), base_t)
            x_base_byte = ((nbase * fx.Index(d) + base_t) * fx.Index(h)) * fx.Index(width) * fx.Index(c)
            x_addr = fx.Int64(buffer_ops.extract_base_index(x)) + fx.Int64(x_base_byte)
            x_div = fx.logical_divide(
                _make_fp8_buffer_tensor_from_addr(x_addr, f8_ir_t, x_buf),
                fx.make_layout(1, 1),
            )

        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        wave_m = wid // WAVE_N
        wave_n = wid % WAVE_N
        lane_div_16 = lane // MFMA_N
        lane_mod_16 = lane % MFMA_N
        c_m_vec = lane_div_16 * MFMA_C_VALUES
        c_n = lane_mod_16

        mfma = Mfma16x16x128(MI_M, MI_N)
        acc = [mfma.zero_value for _ in range_constexpr(N_ACC)]

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

        # Rows of a tile written per full pass of the block (each thread copies
        # one LDG_VEC-wide chunk along K).
        ROWS_PER_PASS = BLOCK_THREADS * LDG_VEC // TILE_K

        # ---- 3D im2col gather: global FP8 -> LDS (direct async copy) ----
        def g2s_a_block(stage, blk, k_base):
            linear = tid * LDG_VEC
            local_m = linear // TILE_K
            local_k = linear % TILE_K
            row = m_offset + blk * ROWS_PER_PASS + local_m
            row_valid = row < fx.Index(npq)
            lds_elem = a_lds_off(stage, fx.Index(blk * ROWS_PER_PASS) + local_m, local_k)
            k_abs = fx.Index(k_base) + fx.Index(local_k)
            cc = k_abs % c
            k_valid = k_abs < fx.Index(crs)
            if const_expr(temporal_only_fast):
                kt_i = k_abs // c
                temporal_delta = kt_i - pt
                out_t = (row // hw_o) % d
                in_t = out_t + temporal_delta
                valid_data = row_valid & k_valid & in_range(in_t, d)
                if const_expr(BIG_IN):
                    # Offset relative to x_div's rebased origin so the i32 element
                    # offset does not overflow.
                    g_elem = ((row + temporal_delta * hw_o) - (nbase * dhw + base_t * hw_o)) * c + cc
                else:
                    g_elem = (row + temporal_delta * hw_o) * c + cc
            else:
                n_idx = row // dhw
                rem = row % dhw
                ot = rem // hw_o
                rem2 = rem % hw_o
                oh = rem2 // wo
                ow = rem2 % wo
                ckk = k_abs // c
                kw_i = ckk % kw
                ckk2 = ckk // kw
                kh_i = ckk2 % kh
                kt_i = ckk2 // kh
                in_t = ot * st + kt_i - pt
                in_h = oh * sh + kh_i - ph
                in_w = ow * sw + kw_i - pw
                valid_data = row_valid & k_valid & in_range(in_t, d) & in_range(in_h, h) & in_range(in_w, width)
                if const_expr(BIG_IN):
                    di = n_idx - nbase
                    g_elem = (((di * d + (in_t - base_t)) * h + in_h) * width + in_w) * c + cc
                else:
                    g_elem = (((n_idx * d + in_t) * h + in_h) * width + in_w) * c + cc
            g_elem_i = fx.Int32(g_elem)
            safe_elem = arith.select(valid_data, g_elem_i, fx.Int32(x_num_records))
            copy_g2s(x_div, a_lds, lds_elem, safe_elem)

        def g2s_b_block(stage, blk, k_base):
            linear = tid * LDG_VEC
            local_n = linear // TILE_K
            local_k = linear % TILE_K
            col = n_offset + fx.Index(blk * ROWS_PER_PASS) + local_n
            lds_elem = b_lds_off(stage, fx.Index(blk * ROWS_PER_PASS) + local_n, local_k)
            g_elem = col * crs + (fx.Index(k_base) + fx.Index(local_k))
            g_elem_i = fx.Int32(g_elem)
            copy_g2s(w_div, b_lds, lds_elem, g_elem_i)

        def g2s_full_tile(stage, k_base):
            for blk in range_constexpr(A_ROW_BLKS):
                g2s_a_block(stage, blk, k_base)
            for blk in range_constexpr(B_ROW_BLKS):
                g2s_b_block(stage, blk, k_base)

        def lds_load_vec16(lds_array, elem_offset):
            u8_ptr = fx.recast_iter(fx.Uint8, lds_array.ptr)
            return fx.ptr_load(u8_ptr + fx.Int32(elem_offset), result_type=Vec16U8Ty)

        def lds_load_pack(lds_array, elem_offset):
            lo = lds_load_vec16(lds_array, elem_offset).bitcast(fx.Int32)
            hi = lds_load_vec16(lds_array, elem_offset + fx.Index(64)).bitcast(fx.Int32)
            return pack_i32x4_i32x8(lo, hi)

        def read_a_vec(stage, mi):
            a_row = wave_m * WARP_M + mi * MFMA_M + lane_mod_16
            a_col = lane_div_16 * 16
            return lds_load_pack(a_lds, a_lds_off(stage, fx.Index(a_row), fx.Index(a_col)))

        def read_b_vec(stage, ni):
            b_row = wave_n * WARP_N + ni * MFMA_N + lane_mod_16
            b_col = lane_div_16 * 16
            return lds_load_pack(b_lds, b_lds_off(stage, fx.Index(b_row), fx.Index(b_col)))

        def setprio(level):
            llvm.InlineAsmOp(None, [], f"s_setprio {level}", "", has_side_effects=True)

        def mfma_one(a, b, c_acc):
            out = mfma._do_mma(a, b, c_acc)
            fx.rocdl.sched_mfma(1)
            return out

        def read_a_frags(stage):
            frags = [read_a_vec(stage, mi) for mi in range_constexpr(MI_M)]
            fx.rocdl.sched_dsrd(MI_M)
            return frags

        def read_b_frags(stage):
            frags = [read_b_vec(stage, ni) for ni in range_constexpr(MI_N)]
            fx.rocdl.sched_dsrd(MI_N)
            return frags

        # ---- software-pipelined main loop ----
        stage = 0
        next_stage = 1
        g2s_full_tile(stage, k_off)
        barrier()
        a_frags = read_a_frags(stage)
        b_frags = read_b_frags(stage)

        for kt_idx in range_constexpr(tiles_per_split):
            # prefetch next tile: global -> LDS (async)
            if const_expr(kt_idx + 1 < tiles_per_split):
                g2s_full_tile(next_stage, k_off + (kt_idx + 1) * TILE_K)

            setprio(1)
            for mi in range_constexpr(MI_M):
                for ni in range_constexpr(MI_N):
                    idx = mi * MI_N + ni
                    acc[idx] = mfma_one(a_frags[mi], b_frags[ni], acc[idx])
            setprio(0)

            if const_expr(kt_idx + 1 < tiles_per_split):
                barrier()
                stage = next_stage
                next_stage = (stage + 1) % STAGES
                a_frags = read_a_frags(stage)
                b_frags = read_b_frags(stage)

        _vec_store = (n == 1) and (not use_splitk) and (dhw % MFMA_C_VALUES == 0) and (not BIG_OUT)

        if const_expr(BIG_OUT):
            y_elem_base = fx.Int64(buffer_ops.extract_base_index(y))

        def _big_store(off_elem, value):
            addr = y_elem_base + fx.Int64(off_elem) * fx.Int64(2)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            v = value.ir_value() if hasattr(value, "ir_value") else value
            llvm.StoreOp(v, ptr, alignment=2)

        def store_acc():
            for mi in range_constexpr(MI_M):
                row_base = m_offset + wave_m * WARP_M + mi * MFMA_M + c_m_vec
                for ni in range_constexpr(MI_N):
                    col = n_offset + fx.Index(wave_n * WARP_N + ni * MFMA_N) + c_n
                    col_valid = col < fx.Index(k)
                    if const_expr(has_bias and not use_splitk):
                        bias_val = fx.Float32(buffer_ops.buffer_load(bias_rsrc, col, vec_width=1, dtype=fx.Float32))
                    acc_vec = Vec(acc[mi * MI_N + ni])

                    if const_expr(_vec_store):
                        off0 = col * dhw + fx.Index(row_base)

                        def _emit_vec4():
                            vals = []
                            for i in range_constexpr(MFMA_C_VALUES):
                                o = acc_vec[i] + bias_val if const_expr(has_bias) else acc_vec[i]
                                vals.append(o.to(fx.BFloat16))
                            v4 = fx.Vector.from_elements(vals, dtype=fx.BFloat16)
                            buffer_ops.buffer_store(v4, y_rsrc, off0)

                        if col_valid:
                            _emit_vec4()
                        continue

                    for i in range_constexpr(MFMA_C_VALUES):
                        row = row_base + i
                        out = acc_vec[i]
                        if const_expr(use_splitk):
                            # Atomics ignore hardware OOB suppression; guard explicitly.
                            valid = (col < fx.Index(k)) & (row < fx.Index(npq))
                            if valid:
                                off_b = fx.Int32((row * k + col) * 4)
                                z0 = fx.Int32(0)
                                fx.rocdl.raw_ptr_buffer_atomic_fadd(out, y_rsrc, off_b, z0, z0)
                        else:
                            if const_expr(has_bias):
                                out = out + bias_val
                            # NCDHW output[n_idx, col, sp]: n_idx*(k*dhw) + col*dhw + sp.
                            # n==1 fast path: n_idx=0, sp=row, no integer division.
                            if const_expr(n == 1):
                                off_ncdhw = col * dhw + row
                            else:
                                n_idx = row // dhw
                                sp = row % dhw
                                off_ncdhw = n_idx * (k * dhw) + col * dhw + sp
                            if const_expr(BIG_OUT):
                                if col_valid:
                                    _big_store(off_ncdhw, out.to(fx.BFloat16))
                            else:
                                buffer_ops.buffer_store(out.to(fx.BFloat16), y_rsrc, off_ncdhw, mask=col_valid)

        store_acc()

    @flyc.jit
    def launch(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv3d_implicit_fp8_kernel(
            y,
            x,
            weight,
            bias,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(grid=(grid_m, grid_n, splitk), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def _normalize_3(v):
    if isinstance(v, int):
        return (v, v, v)
    assert len(v) == 3, f"expected int or length-3 tuple, got {v!r}"
    return tuple(v)


def _resolve_splitk(splitk, npq, crs, k, device, tile=DEFAULT_TILE):
    k_tiles = (crs + TILE_K - 1) // TILE_K
    if splitk is None:
        tile_m, tile_n = tile[0], tile[1]
        base = (npq // tile_m) * (k // tile_n) if (npq % tile_m == 0 and k % tile_n == 0) else 0
        if (
            npq % tile_m != 0
            or k % tile_n != 0
            or crs % TILE_K != 0
            or npq * k * 4 > 0x7FFFFFFF  # split-K fp32 output atomic uses an i32 byte offset
            or npq < 4096
            or k_tiles < 16
        ):
            sk = 1
        else:
            try:
                num_cu = torch.cuda.get_device_properties(device).multi_processor_count
            except Exception:
                num_cu = 256
            sk = 1 if base >= (3 * num_cu) // 4 else min(4, max(1, num_cu // base), k_tiles)
    else:
        sk = max(1, int(splitk))
    sk = max(1, min(sk, k_tiles))
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    MAX_TILES_PER_SPLIT = 54
    tiles_per_split = k_tiles // sk
    if tiles_per_split > MAX_TILES_PER_SPLIT:
        min_sk = (k_tiles + MAX_TILES_PER_SPLIT - 1) // MAX_TILES_PER_SPLIT
        for candidate in range(min_sk, k_tiles + 1):
            if k_tiles % candidate == 0 and k_tiles // candidate <= MAX_TILES_PER_SPLIT:
                sk = candidate
                break
    return sk


def layout_activation_ncdhw_to_ndhwc_fp8(x: torch.Tensor) -> torch.Tensor:
    """FP8 NCDHW activation -> int8 storage of FP8 in NDHWC order (layout only).

    Input is already FP8 (E4M3FN); this is a pure memory reorder (C moved innermost),
    no dtype conversion.
    """
    assert x.dtype == torch.float8_e4m3fn, f"expected FP8 E4M3FN activation, got {x.dtype}"
    return x.permute(0, 2, 3, 4, 1).contiguous().view(torch.int8).view(-1)


def _prep_weight_fp8(weight: torch.Tensor) -> torch.Tensor:
    """Reorder + cache the FP8 weight (KCTRS -> KTRSC) by source identity (weights reused).

    Input is already FP8 (E4M3FN); the transpose is a pure memory reorder.
    """
    assert weight.dtype == torch.float8_e4m3fn, f"expected FP8 E4M3FN weight, got {weight.dtype}"
    key = id(weight)
    ent = _WEIGHT_FP8_CACHE.get(key)
    if ent is not None and ent[0]() is weight:
        return ent[1]
    out = weight.permute(0, 2, 3, 4, 1).contiguous().view(torch.int8).view(-1)
    _WEIGHT_FP8_CACHE[key] = (weakref.ref(weight), out)
    return out


def _conv3d_impl_fp8(x, weight, bias=None, stride=1, padding=0, splitk=None, stream=None, tile=None, autotune=None):
    """FP8 (E4M3FN) 3D implicit-GEMM implementation; the public
    conv3d_implicit_fp8 entry dispatches 1D/2D/3D by filter rank and
    forwards true 3D calls here.

    x: (N, C, D, H, W) fp8 (E4M3FN), weight: (K, C, T, R, S) fp8. Inputs are consumed
    natively as FP8 (no bf16->fp8 cast); the host only reorders them into the
    kernel-native NDHWC activation / cached KTRSC weight layout, then runs the CDNA4
    16x16x128 MFMA conv with a software-pipelined loop and optional split-K.
    Returns bf16 (N, K, Do, Ho, Wo). splitk=None auto-dispatches.

    tile=(TILE_M,TILE_N,WAVE_M,WAVE_N) forces a config; autotune=True picks the
    best tile per shape (also enabled via FLYDSL_CONV3D_AUTOTUNE=1)."""
    n, c, d, h, width = x.shape
    k, wc, kt, kh, kw = weight.shape
    assert c == wc, f"in-channel mismatch: x has {c}, weight has {wc}"
    assert (
        x.dtype == torch.float8_e4m3fn and weight.dtype == torch.float8_e4m3fn
    ), f"expected FP8 E4M3FN x/weight, got x={x.dtype}, weight={weight.dtype}"
    st, sh, sw = _normalize_3(stride)
    pt, ph, pw = _normalize_3(padding)
    do = (d + 2 * pt - kt) // st + 1
    ho = (h + 2 * ph - kh) // sh + 1
    wo = (width + 2 * pw - kw) // sw + 1
    npq = n * do * ho * wo
    crs = c * kt * kh * kw

    assert c % LDG_VEC == 0, f"FP8 vector load needs C % {LDG_VEC} == 0, got C={c}"

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    x_arg = layout_activation_ncdhw_to_ndhwc_fp8(x)
    w_arg = _prep_weight_fp8(weight)

    has_bias = bias is not None
    bias_arg = (
        bias.to(device=x.device, dtype=torch.float32).contiguous().view(-1)
        if has_bias
        else torch.empty(1, device=x.device, dtype=torch.float32)
    )
    if has_bias:
        assert bias_arg.numel() == k, f"bias must have {k} elements, got {bias_arg.numel()}"

    x_arg_t = flyc.from_torch_tensor(x_arg)
    w_arg_t = flyc.from_torch_tensor(w_arg)
    bias_t = flyc.from_torch_tensor(bias_arg)

    def _run(the_tile):
        sk = _resolve_splitk(splitk, npq, crs, k, x.device, the_tile)
        if sk > 1:
            y = torch.zeros((npq, k), device=x.device, dtype=torch.float32)
        else:
            y = torch.empty((n, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)
        exe = compile_conv3d_implicit_fp8(
            n, c, d, h, width, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias, sk, the_tile
        )
        exe(flyc.from_torch_tensor(y.view(-1)), x_arg_t, w_arg_t, bias_t, launch_stream)
        return y, sk

    shape = (n, c, d, h, width, k, kt, kh, kw, st, sh, sw, pt, ph, pw, has_bias)
    if tile is not None:
        chosen = tuple(tile)
    elif autotune or (autotune is None and _autotune_enabled()):
        from kernels.conv.conv3d_autotune import FP8_CANDIDATES, autotune_conv3d

        chosen = autotune_conv3d("fp8", shape, "fp8", FP8_CANDIDATES, x.device, lambda t: _run(t)[0])
    else:
        chosen = DEFAULT_TILE

    y, sk = _run(chosen)
    if sk > 1:
        if has_bias:
            y = y + bias_arg.view(1, k)
        y = y.to(torch.bfloat16)
        return y.view(n, do, ho, wo, k).permute(0, 4, 1, 2, 3)
    return y


def _conv2d_impl_fp8(x, weight, bias=None, stride=1, padding=0, **kwargs):
    """2D FP8 conv via the 3D kernel (depth-1 degenerate case).

    x: (N, C, H, W) fp8 (E4M3FN), weight: (K, C, R, S) fp8. stride/padding int or 2-tuple.
    Returns (N, K, Ho, Wo) bf16. Reshapes to the D=T=1 case and squeezes depth back.
    """
    assert x.dim() == 4 and weight.dim() == 4, "conv2d fp8 expects (N,C,H,W) / (K,C,R,S)"
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    n, c, h, w = x.shape
    k, wc, r, s = weight.shape
    x5 = x.reshape(n, c, 1, h, w)
    w5 = weight.reshape(k, wc, 1, r, s)
    y5 = _conv3d_impl_fp8(x5, w5, bias=bias, stride=(1, sh, sw), padding=(0, ph, pw), **kwargs)
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[3], y5.shape[4])


def _conv1d_impl_fp8(x, weight, bias=None, stride=1, padding=0, **kwargs):
    """1D FP8 conv via the 3D kernel (depth/height-1 degenerate case).

    x: (N, C, W) fp8 (E4M3FN), weight: (K, C, S) fp8. stride/padding int or 1-tuple.
    Returns (N, K, Wo) bf16. Reshapes to the D=H=T=R=1 case and squeezes back.
    """
    assert x.dim() == 3 and weight.dim() == 3, "conv1d fp8 expects (N,C,W) / (K,C,S)"
    sw = stride if isinstance(stride, int) else stride[0]
    pw = padding if isinstance(padding, int) else padding[0]
    n, c, w = x.shape
    k, wc, s = weight.shape
    x5 = x.reshape(n, c, 1, 1, w)
    w5 = weight.reshape(k, wc, 1, 1, s)
    y5 = _conv3d_impl_fp8(x5, w5, bias=bias, stride=(1, 1, sw), padding=(0, 0, pw), **kwargs)
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[4])


def conv3d_implicit_fp8(x, weight, bias=None, stride=1, padding=0, **kwargs):
    """Main FP8 implicit-GEMM conv entry; dispatches 1D/2D/3D by filter rank.

    Rank is taken from the filter (weight.dim() - 2): 3 -> 3D (N,C,D,H,W)/(K,C,T,R,S),
    2 -> 2D (N,C,H,W)/(K,C,R,S), 1 -> 1D (N,C,W)/(K,C,S); x and weight must match.
    True 3D calls run the implementation directly; 2D/1D reshape to the degenerate
    5D case. stride/padding/bias and extra kwargs (splitk, tile, autotune, stream)
    forward to the chosen path.
    """
    assert x.dim() == weight.dim(), f"x rank {x.dim()} != weight rank {weight.dim()}"
    spatial_rank = weight.dim() - 2
    if spatial_rank == 3:
        return _conv3d_impl_fp8(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    if spatial_rank == 2:
        return _conv2d_impl_fp8(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    if spatial_rank == 1:
        return _conv1d_impl_fp8(x, weight, bias=bias, stride=stride, padding=padding, **kwargs)
    raise ValueError(f"conv3d_implicit_fp8 supports 1D/2D/3D; got filter rank {weight.dim()}")


__all__ = ["conv3d_implicit_fp8"]
