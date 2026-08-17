# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Double-buffered implicit-GEMM conv3d (BF16).

x: (N, C, D, H, W) bf16 NCDHW by default, weight: (K, C/groups, T, R, S) bf16 KCTRS.
Returns (N, K, Do, Ho, Wo) bf16 by default. ``input_layout`` / ``output_layout`` select
NCDHW or NDHWC independently; the GEMM itself is channels-last, so NDHWC input skips the
pre-transpose and NDHWC output is the raw row-major (npq, K) the epilogue produces.
Supports stride, padding (int, per-axis tuple, or torch's "same" / "valid"),
padding_mode, dilation, bias, groups, and split-K.
"""

import functools
import os
import weakref

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops
from kernels.common.mem_ops import buffer_atomic_add

TILE_K = 32
STAGES = 2
WARP_SIZE = 64

MFMA_M = 16
MFMA_N = 16
MFMA_A_VALUES = 8
MFMA_B_VALUES = 8
MFMA_C_VALUES = 4

LDG_VEC = 8

BF16_BYTES = 2

DEFAULT_TILE = (128, 128, 2, 4)

# Same set torch's nn.ConvNd accepts. All four are resolved inside the im2col gather:
# "zeros" masks an out-of-range tap, the rest remap it onto a real input coordinate.
PADDING_MODES = ("zeros", "reflect", "replicate", "circular")

# Applied around both tracing and flyc.compile so the hinted and the fast-dispatch
# paths lower identically. Keys are the ones rocm.py reads: waves_per_eu, maxnreg,
# fast_fp_math, unsafe_fp_math, llvm_options.
#
# Deliberately empty. Swept on gfx950 over five shapes (g in {1,2,4,8}, small and
# large): fast_fp_math is a wash -- MFMA does the arithmetic and the epilogue is a
# single f32->bf16 convert, so there is nothing to reassociate -- and waves_per_eu
# of 1/2 measured 3%/11% slower, 4 within noise. Left as the hook the launchers
# already thread through, not as a tuning knob that currently pays.
CONV_COMPILE_HINTS = {}


def _as_stream(stream):
    return stream if hasattr(stream, "_is_stream_param") else fx.Stream(stream)


def _dispatch(exe, *args, stream=None):
    """Run a builder's launcher, pre-compiling on first use.

    ``exe.compile(...)`` both compiles and executes, and hands back a
    ``CompiledFunction`` whose call path skips signature binding and cache lookup
    (~5 us vs ~35 us for the @flyc.jit wrapper). Cached on the launcher, which is
    itself memoized per problem shape by the builder's lru_cache.
    """
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = exe.compile(*args, stream=stream)
        return
    cf(*args, _as_stream(stream))


def _autotune_enabled():
    return os.environ.get("FLYDSL_CONV3D_AUTOTUNE", "0").lower() in ("1", "true", "yes")


_WEIGHT_CACHE = {}


def _pad_channels(c):
    return (c + LDG_VEC - 1) // LDG_VEC * LDG_VEC


# Layout names accepted per spatial rank. The 1D/2D entries translate their own names to
# the 3D pair before delegating, so only these two ever reach the builder.
LAYOUTS = {
    3: ("NCDHW", "NDHWC"),
    2: ("NCHW", "NHWC"),
    1: ("NCW", "NWC"),
}


def _check_layouts(rank, input_layout, output_layout):
    names = LAYOUTS[rank]
    for what, v in (("input_layout", input_layout), ("output_layout", output_layout)):
        assert v in names, f"{what} must be one of {names}, got {v!r}"


def _shape_ncdhw(x, ndhwc):
    """Unpack a 5-D input in either layout to (n, c, d, h, w)."""
    if ndhwc:
        n, d, h, w, c = x.shape
    else:
        n, c, d, h, w = x.shape
    return n, c, d, h, w


def _pad_spatial(x, ndhwc, pads, mode="constant"):
    """Pad (D, H, W) with torch's (w_lo, w_hi, h_lo, h_hi, d_lo, d_hi) ordering.

    Returns the padded tensor and its layout flag. Constant padding is layout-agnostic --
    the channel axis just picks up a (0, 0) pair when it is trailing. The other three modes
    are implemented only for a channels-first 5-D input, so a channels-last one is relabeled
    with a permute and comes back channels-first; the caller's later transpose to NDHWC then
    does the copy that permute deferred. These are the paths that already materialize a
    padded input, so the extra pass rides along with one torch has to make anyway.
    """
    if mode == "constant":
        return torch.nn.functional.pad(x, ((0, 0) + pads) if ndhwc else pads), ndhwc
    if ndhwc:
        x = x.permute(0, 4, 1, 2, 3)
    return torch.nn.functional.pad(x, pads, mode=mode), False


def _big_in(n, c, groups, d, h, w, pt, ph, pw):
    """Whether the kernel's 64-bit BIG_IN address path would engage for this input.

    Mirrors the kernel's own test, but on the padded channel count and the worst-case
    (pre-padded) spatial extents, so a "no" here holds for either lowering.
    """
    cp = _pad_channels(c // groups) * groups
    return n * cp * (d + 2 * pt) * (h + 2 * ph) * (w + 2 * pw) > 0x7FFFFFFF


def _evict_weight(key, _ref):
    """weakref callback: drop the entry the dead weight was pinning."""
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is None:
        del _WEIGHT_CACHE[key]


def _prep_weight(w, k, kt, kh, kw, c):
    """Pack (K, C, T, R, S) -> (K, T*R*S*Cpad), memoized on the source weight.

    The memo has to notice an in-place update: an optimizer step or a load_state_dict
    into existing storage leaves both ``id(w)`` and ``data_ptr()`` unchanged, so a key
    built from either alone would keep returning the previous step's packed weights.
    The stamp therefore carries torch's version counter, which is the same signal
    autograd uses to detect mutation -- this is exactly as sensitive as PyTorch's own
    checks, no more and no less. (Mutation through ``w.data`` escapes the version
    counter by design, which is why torch documents it as unsafe; it is invisible here
    for the same reason it is invisible to autograd.)

    Entries do not outlive their weight: the weakref carries a callback that removes
    its own key, so neither the dict nor the packed GPU tensors it pins accumulate.
    """
    key = w.data_ptr()
    stamp = (w._version, tuple(w.shape), w.stride(), w.dtype)
    ent = _WEIGHT_CACHE.get(key)
    if ent is not None and ent[0]() is w and ent[2] == stamp:
        return ent[1]
    cp = _pad_channels(c)
    wsrc = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, 0, 0, cp - c)) if cp != c else w
    wk = wsrc.permute(0, 2, 3, 4, 1).contiguous().reshape(k, kt * kh * kw * cp)
    _WEIGHT_CACHE[key] = (weakref.ref(w, functools.partial(_evict_weight, key)), wk, stamp)
    return wk


TR_TILE = 64
TR_VEC = 8
TR_THREADS = 256
_TR_VPL = TR_TILE // TR_VEC
_TR_ITERS = (TR_TILE * TR_TILE) // (TR_VEC * TR_THREADS)
_TR_PAD = 8
_TR_LDS_S = TR_TILE + _TR_PAD

# Largest S = D*H*W the transpose kernel can address on its 64-bit (BIG) path. That path
# rebases the descriptor per (channel, spatial) tile, but the per-thread read offset
# `rc * s + sv` still carries the full row stride in 32 bits, with rc up to TR_TILE-1 and
# sv up to TR_TILE-TR_VEC. Beyond this the product wraps and the gather silently reads
# the wrong rows, so _ncdhw_to_ndhwc hands those copies to torch instead. Folding the row
# term into the 64-bit base is not possible here: rc varies per lane and a buffer
# descriptor has to be wave-uniform.
TR_MAX_BIG_S = (0x7FFFFFFF - (TR_TILE - TR_VEC)) // (TR_TILE - 1)


@functools.lru_cache(maxsize=64)
def compile_transpose_ncdhw_ndhwc(n, c, s):
    """Transpose flat (N, C, S) -> (N, S, C) (S == T*H*W). Requires c%8==0."""
    grid_s = (s + TR_TILE - 1) // TR_TILE
    grid_c = (c + TR_TILE - 1) // TR_TILE
    elem_ty = fx.BFloat16
    BIG = (n * c * s) > 0x7FFFFFFF

    @flyc.kernel(known_block_size=[TR_THREADS, 1, 1])
    def transpose_kernel(out: fx.Tensor, inp: fx.Tensor):
        # max_size: an exact num_records would zero the whole straddling tail read.
        in_rsrc = buffer_ops.create_buffer_resource(inp)
        out_rsrc = buffer_ops.create_buffer_resource(out)
        lds_alloc = fx.SharedAllocator(static=False)
        lds = lds_alloc.allocate(fx.Array[elem_ty, TR_TILE * _TR_LDS_S, 16]).peek()

        class BF16Ty:
            ir_type = elem_ty.ir_type

        tid = fx.thread_idx.x
        s0 = fx.block_idx.x * TR_TILE
        c0 = fx.block_idx.y * TR_TILE
        nb = fx.block_idx.z
        if const_expr(BIG):
            in_base_elem = fx.Index(nb) * fx.Index(c) * fx.Index(s) + fx.Index(c0) * fx.Index(s) + fx.Index(s0)
            in_addr = fx.Int64(buffer_ops.extract_base_index(inp)) + fx.Int64(in_base_elem) * fx.Int64(2)
            in_rsrc = buffer_ops.create_buffer_resource_from_addr(in_addr)
            out_base_elem = fx.Index(nb) * fx.Index(s) * fx.Index(c) + fx.Index(s0) * fx.Index(c) + fx.Index(c0)
            out_addr = fx.Int64(buffer_ops.extract_base_index(out)) + fx.Int64(out_base_elem) * fx.Int64(2)
            out_rsrc = buffer_ops.create_buffer_resource_from_addr(out_addr)
        else:
            in_base = nb * c * s
            out_base = nb * s * c

        def lds_store_vec8(elem_offset, value):
            base = fx.Int64(fx.ptrtoint(lds.ptr)) + fx.Int64(elem_offset * 2)
            ptr = buffer_ops.create_llvm_ptr(base, address_space=3)
            llvm.StoreOp(value, ptr, alignment=16)

        def lds_load_scalar(elem_offset):
            u8 = fx.recast_iter(fx.Uint8, lds.ptr)
            return fx.ptr_load(u8 + fx.Int32(elem_offset * 2), result_type=BF16Ty)

        # Read: coalesced vec8 along contiguous S -> LDS[c_local][s_local].
        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rc = lin // _TR_VPL
            sv = (lin % _TR_VPL) * TR_VEC
            cc = c0 + rc
            ss = s0 + sv
            valid = (cc < c) & (ss < s)
            if const_expr(BIG):
                g = fx.Int32(rc * s + sv)
            else:
                g = fx.Int32(in_base + cc * s + ss)
            safe = arith.select(valid, g, fx.Int32(0))
            v = buffer_ops.buffer_load(in_rsrc, safe, vec_width=TR_VEC, dtype=elem_ty)
            lds_store_vec8(rc * _TR_LDS_S + sv, v)

        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.s_barrier()

        for i in range_constexpr(_TR_ITERS):
            lin = tid + i * TR_THREADS
            rs = lin // _TR_VPL
            cv = (lin % _TR_VPL) * TR_VEC
            ss = s0 + rs
            cc = c0 + cv
            scalars = [lds_load_scalar((cv + j) * _TR_LDS_S + rs) for j in range_constexpr(TR_VEC)]
            vv = fx.Vector.from_elements(scalars, dtype=elem_ty)
            valid = (ss < s) & (cc < c)
            if valid:
                if const_expr(BIG):
                    go = fx.Int32(rs * c + cv)
                else:
                    go = fx.Int32(out_base + ss * c + cc)
                buffer_ops.buffer_store(vv, out_rsrc, go)

    @flyc.jit
    def launch_transpose(out: fx.Tensor, inp: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        transpose_kernel(out, inp).launch(
            grid=(grid_s, grid_c, n),
            block=(TR_THREADS, 1, 1),
            stream=stream,
        )

    def _launch(out, inp, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch_transpose(out, inp, stream=_as_stream(stream))

    def _compile(out, inp, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch_transpose, out, inp, _as_stream(stream))

    _launch.compile = _compile
    return _launch


def _ncdhw_to_ndhwc(x, stream):
    """Fast NCDHW->NDHWC via the tiled transpose kernel; falls back to torch."""
    n, c, t, h, w = x.shape
    s = t * h * w
    big = n * c * s > 0x7FFFFFFF
    if not (x.is_contiguous() and x.dtype == torch.bfloat16 and c % TR_VEC == 0):
        return x.permute(0, 2, 3, 4, 1).contiguous()
    if big and s > TR_MAX_BIG_S:
        return x.permute(0, 2, 3, 4, 1).contiguous()
    out = torch.empty((n, t, h, w, c), device=x.device, dtype=x.dtype)
    exe = compile_transpose_ncdhw_ndhwc(n, c, s)
    _dispatch(exe, out, x, stream=torch.cuda.current_stream() if stream is None else stream)
    return out


@functools.lru_cache(maxsize=256)
def compile_conv3d_implicit(
    n,
    c,
    d,
    h,
    w,
    k,
    kt,
    kh,
    kw,
    st,
    sh,
    sw,
    pt,
    ph,
    pw,
    dt=1,
    dh=1,
    dw=1,
    pad_mode="zeros",
    has_bias=False,
    splitk=1,
    tile=DEFAULT_TILE,
    wgm=1,
    groups=1,
    out_ndhwc=False,
):
    TILE_M, TILE_N, WAVE_M, WAVE_N = tile
    BLOCK_THREADS = WAVE_M * WAVE_N * WARP_SIZE
    # Per-wave MFMA grid (flat acc[mi * MI_N + ni]); WARP_M/N is the per-wave tile span.
    MI_M = TILE_M // WAVE_M // MFMA_M
    MI_N = TILE_N // WAVE_N // MFMA_N
    N_ACC = MI_M * MI_N
    WARP_M = MI_M * MFMA_M
    WARP_N = MI_N * MFMA_N
    BLOCK_VECS = LDG_VEC * BLOCK_THREADS
    LDG_A_COUNT = TILE_M * TILE_K // BLOCK_VECS
    LDG_B_COUNT = TILE_N * TILE_K // BLOCK_VECS

    # `c` is the padded TOTAL channel count and stays the NDHWC row stride. CGP is the
    # per-group channel count and is what the GEMM K axis decomposes against; the two
    # coincide only when groups == 1.
    CGP = c // groups
    KG = k // groups

    assert TILE_K == 32
    assert TILE_M % (WAVE_M * MFMA_M) == 0, f"TILE_M={TILE_M} not divisible by WAVE_M*16"
    assert TILE_N % (WAVE_N * MFMA_N) == 0, f"TILE_N={TILE_N} not divisible by WAVE_N*16"
    assert (TILE_M * TILE_K) % BLOCK_VECS == 0, f"A tile {TILE_M}x{TILE_K} not a multiple of {BLOCK_VECS} vecs"
    assert (TILE_N * TILE_K) % BLOCK_VECS == 0, f"B tile {TILE_N}x{TILE_K} not a multiple of {BLOCK_VECS} vecs"
    assert LDG_A_COUNT >= 1 and LDG_B_COUNT >= 1
    assert c % groups == 0, f"c={c} not divisible by groups={groups}"
    assert k % groups == 0, f"k={k} not divisible by groups={groups}"
    assert CGP % LDG_VEC == 0, f"c/groups={CGP} must be a multiple of LDG_VEC={LDG_VEC}; use _conv3d_impl to pad"
    assert BLOCK_THREADS <= 1024, f"BLOCK_THREADS={BLOCK_THREADS} exceeds 1024"

    # Dilation only stretches the filter's footprint; the K axis (CRS) is unchanged.
    do = (d + 2 * pt - (dt * (kt - 1) + 1)) // st + 1
    ho = (h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    dhw = do * ho * wo
    hw_o = ho * wo
    npq = n * dhw
    crs = CGP * kt * kh * kw
    k_tiles = (crs + TILE_K - 1) // TILE_K

    BIG_IN = (n * c * d * h * w) > 0x7FFFFFFF
    BIG_OUT = (n * k * do * ho * wo * BF16_BYTES) > 0x7FFFFFFF

    X_BYTES = n * c * d * h * w * BF16_BYTES
    W_BYTES = k * crs * BF16_BYTES
    OOB_SENTINEL_ELEM = 0x7FFFFF80  # *2 = 0xFFFFFF00 bytes (~4.2950 GB), just under 2^32
    OOB_SENTINEL_BYTES = OOB_SENTINEL_ELEM * BF16_BYTES
    BIG_IN_NR = 0x80000000  # 2 GB num_records for the rebased BIG_IN resource
    assert W_BYTES < OOB_SENTINEL_BYTES, f"weight {W_BYTES}B exceeds limit {OOB_SENTINEL_BYTES}B"
    assert X_BYTES < OOB_SENTINEL_BYTES or BIG_IN, f"input {X_BYTES}B exceeds limit"
    BIG_IN_N1 = BIG_IN and n == 1
    BIG_IN_NM = BIG_IN and n > 1

    # BIG_IN_N1 rebases the descriptor to the block's own origin and addresses everything
    # from there in 32 bits, so the block's reachable footprint has to fit the window.
    # Rebasing on D alone leaves a whole H*W*C slice in the offset, which is itself past
    # 2 GiB on a large 2D input, so the origin drops to H as well. Rows run
    # (n, ot, oh, ow), so oh is monotone within a block and only restarts when the block
    # crosses an ot boundary -- which cannot happen when the ot windows are a whole number
    # of tiles. The two bounds below are the worst case over every block: sound in both
    # cases, and tight in the aligned one.
    _t_aligned = BIG_IN_N1 and hw_o % TILE_M == 0
    if BIG_IN_N1:
        _ot_span = (TILE_M - 1) // hw_o + (1 if _t_aligned else 2)
        _t_span = min(d - 1, (_ot_span - 1) * st + dt * (kt - 1))
        _h_span = min(h - 1, ((TILE_M - 1) // wo + 1) * sh + dh * (kh - 1)) if _t_aligned else h - 1
        _span = (((_t_span * h + _h_span) * w + (w - 1)) * c + c) * BF16_BYTES
        assert _span <= BIG_IN_NR, (
            f"input sample too large for the 32-bit gather: a {TILE_M}-row tile reaches "
            f"{_span / 2**30:.2f} GiB from its rebased origin, past the "
            f"{BIG_IN_NR / 2**30:.0f} GiB the buffer descriptor addresses. Split the batch "
            f"over N, or pass a narrower tile=(TILE_M, ...)."
        )
    assert pad_mode in PADDING_MODES, f"pad_mode must be one of {PADDING_MODES}, got {pad_mode!r}"
    # BIG_IN_N1 rebases the buffer to the block's first input row, and a reflected tap can
    # resolve below that base; _conv3d_impl keeps non-zero modes off the BIG_IN path.
    assert pad_mode == "zeros" or not BIG_IN, "non-zero pad_mode requires the non-BIG_IN address path"
    X_SAMPLE_BYTES = c * d * h * w * BF16_BYTES

    # A tile must never straddle a group boundary -- every column in it shares one A tile in
    # LDS, and different groups need different input channels. So the N grid is
    # over-provisioned per group and the per-group tail is masked.
    tiles_per_group = (KG + TILE_N - 1) // TILE_N
    n_tail = KG % TILE_N != 0
    grid_n = groups * tiles_per_group

    splitk = max(1, min(splitk, k_tiles))
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1
    # The bound _resolve_splitk applies, restated where the unsafe arithmetic actually
    # lives so a caller reaching this builder directly fails loudly instead of silently
    # wrapping the epilogue's 32-bit atomic offset.
    assert (
        not use_splitk or npq * k * 4 <= SPLITK_MAX_STAGING_BYTES
    ), f"split-K staging {npq * k * 4}B exceeds the {SPLITK_MAX_STAGING_BYTES}B buffer window"

    # Software-pipeline depth. 4 stages is optimal across all shapes on gfx950 --
    # even short-K, memory-bound 3x1x1 depends more (not less) on deep prefetch to
    # hide DMA latency; a shallower pipeline measured slower (2/3/4-stage A/B).
    PIPE_STAGES = 4

    LDS_A_SIZE = PIPE_STAGES * TILE_M * TILE_K
    LDS_B_SIZE = PIPE_STAGES * TILE_N * TILE_K

    grid_m = (npq + TILE_M - 1) // TILE_M

    # HSA's dispatch packet carries grid_size_x as a uint32 *work-item* count, not a block
    # count, so grid.x * block.x has to stay under 2^32 -- about 2^30 output rows at any
    # tile size. Past that hipModuleLaunchKernel rejects the launch, and FlyDSL's wrapper
    # only prints that to stderr, so the kernel would hand back its uninitialised output
    # as if it had run. The surplus M tiles therefore spill onto grid.z, which is a plain
    # block count; grid.z already carries split-K, so the two are packed into it together.
    MAX_GRID_X = 0xFFFFFFFF // BLOCK_THREADS
    MAX_GRID_YZ = 65535
    grid_x = min(grid_m, MAX_GRID_X)
    m_chunks = (grid_m + grid_x - 1) // grid_x
    assert grid_n <= MAX_GRID_YZ, f"grid.y = {grid_n} exceeds the {MAX_GRID_YZ}-block limit"
    assert (
        m_chunks * splitk <= MAX_GRID_YZ
    ), f"grid.z = {m_chunks} M-chunks x {splitk} splits exceeds the {MAX_GRID_YZ}-block limit"
    # The block swizzle mixes grid.x and grid.y and has no meaning once M is split across
    # two axes; it is a locality tweak, so drop it rather than complicate the mapping.
    WGM = 1 if m_chunks > 1 else max(1, int(wgm))
    elem_ty = fx.BFloat16
    mfma_fn = rocdl.mfma_f32_16x16x32_bf16
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
        and wo == w
    )

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def conv3d_implicit_kernel(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor):
        w_rsrc = buffer_ops.create_buffer_resource(weight, num_records_bytes=W_BYTES)
        if const_expr(not BIG_IN):
            x_rsrc = buffer_ops.create_buffer_resource(x, num_records_bytes=X_BYTES)
        y_rsrc = buffer_ops.create_buffer_resource(y)
        if const_expr(has_bias):
            bias_rsrc = buffer_ops.create_buffer_resource(bias)

        lds_alloc = fx.SharedAllocator(static=False)
        a_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_A_SIZE, 16]).peek()
        b_lds = lds_alloc.allocate(fx.Array[elem_ty, LDS_B_SIZE, 16]).peek()

        tid = fx.thread_idx.x
        if const_expr(m_chunks > 1):
            # grid.z packs (split, m_chunk); the M tiles that did not fit grid.x live here.
            m_chunk = fx.Index(fx.block_idx.z) % fx.Index(m_chunks)
            m_offset = (fx.Index(fx.block_idx.x) + m_chunk * fx.Index(grid_x)) * TILE_M
            n_tile = fx.block_idx.y
        elif const_expr(WGM > 1):
            pid = fx.Index(fx.block_idx.x) + fx.Index(fx.block_idx.y) * fx.Index(grid_m)
            blocks_per_swizzle = fx.Index(WGM * grid_n)
            swizzle_id = pid // blocks_per_swizzle
            first_m = swizzle_id * fx.Index(WGM)
            swizzle_rows = fx.Index(grid_m) - first_m
            swizzle_rows = fx.Index(arith.select(swizzle_rows < fx.Index(WGM), swizzle_rows, fx.Index(WGM)))
            local = pid % blocks_per_swizzle
            m_offset = fx.Index(first_m + (local % swizzle_rows)) * TILE_M
            n_tile = fx.Index(local // swizzle_rows)
        else:
            m_offset = fx.block_idx.x * TILE_M
            n_tile = fx.block_idx.y
        # n_offset is the GLOBAL output-channel base (drives the B row and the store),
        # n_local is the base within this group (drives every tail check), and ch_base is
        # this group's first input channel. All three are block-uniform.
        if const_expr(groups > 1):
            gi = n_tile // tiles_per_group
            n_local = (n_tile % tiles_per_group) * TILE_N
            n_offset = gi * KG + n_local
            ch_base = gi * CGP
        else:
            n_offset = n_tile * TILE_N
            n_local = n_offset
        if const_expr(use_splitk):
            if const_expr(m_chunks > 1):
                split_idx = fx.Index(fx.block_idx.z) // fx.Index(m_chunks)
            else:
                split_idx = fx.Index(fx.block_idx.z)
            k_off = split_idx * (tiles_per_split * TILE_K)
        else:
            k_off = 0

        if const_expr(BIG_IN_N1):
            nbase = m_offset // dhw
            rem0 = m_offset % dhw
            ot_base0 = rem0 // hw_o
            # ot*st - pt is the first input row this ot can read; every later row in the
            # block has ot >= ot_base0, so this is a lower bound for the whole block.
            base_t = ot_base0 * fx.Index(st) - fx.Index(pt)
            base_t = arith.select(base_t < fx.Index(0), fx.Index(0), base_t)
            if const_expr(_t_aligned):
                # An ot window is a whole number of tiles, so oh cannot restart mid-block
                # and the same argument carries to H.
                oh_base0 = (rem0 % hw_o) // wo
                base_h = oh_base0 * fx.Index(sh) - fx.Index(ph)
                base_h = arith.select(base_h < fx.Index(0), fx.Index(0), base_h)
            else:
                base_h = fx.Index(0)
            x_base_elem = ((nbase * fx.Index(d) + base_t) * fx.Index(h) + base_h) * fx.Index(w) * fx.Index(c)
            x_addr = fx.Int64(buffer_ops.extract_base_index(x)) + fx.Int64(x_base_elem) * fx.Int64(2)
            x_rsrc = buffer_ops.create_buffer_resource_from_addr(x_addr, num_records_bytes=BIG_IN_NR)
        if const_expr(BIG_IN_NM):
            x_base_addr = fx.Int64(buffer_ops.extract_base_index(x))

        wid = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        wave_m = wid // WAVE_N
        wave_n = wid % WAVE_N

        lane_m = lane % MFMA_M
        lane_n = lane % MFMA_N
        lane_k_a = lane // MFMA_M * MFMA_A_VALUES
        lane_k_b = lane // MFMA_N * MFMA_B_VALUES
        c_m_vec = lane // MFMA_N * MFMA_C_VALUES
        c_n = lane % MFMA_N

        Vec = fx.Vector

        class Vec8Ty:
            ir_type = Vec.make_type(8, elem_ty)

        acc0 = Vec.filled(MFMA_C_VALUES, 0.0, fx.Float32)
        acc = [acc0 for _ in range_constexpr(N_ACC)]

        def barrier(vmcnt=0, lgkmcnt=None):
            # None means "do not wait on this counter" -- s_waitcnt encodes it as the max.
            rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt)
            rocdl.s_barrier()

        def lds_load_vec8(lds_array, elem_offset):
            u8_ptr = fx.recast_iter(fx.Uint8, lds_array.ptr)
            return fx.ptr_load(u8_ptr + fx.Int32(elem_offset * 2), result_type=Vec8Ty)

        def a_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_M + row) * TILE_K + col

        def b_lds_off(stage, row, col):
            return (fx.Index(stage) * TILE_N + row) * TILE_K + col

        def in_range(v, hi):
            return (v >= 0) & (v < fx.Index(hi))

        def dil(tap, factor):
            # Filter tap -> input offset. Kept off the multiply when undilated.
            scaled = tap * factor if const_expr(factor != 1) else tap
            return scaled

        def pad_coord(v, ext, pad):
            """Tap coordinate -> in-bounds input coordinate; returns (coord, mask).

            "zeros" leaves the coordinate alone and returns a range mask, which the
            caller folds into the OOB-sentinel routing so the load reads as zero. Every
            other mode resolves the coordinate into [0, ext) instead and returns no mask
            -- the three range checks the zeros path needs disappear, which offsets most
            of what the remap costs. One step is enough because torch caps reflect at
            pad < ext and circular at pad <= ext (both asserted host-side), so a
            coordinate can never wrap past the far edge.

            Index compares here lower to UNSIGNED predicates, so a negative coordinate
            reads as a huge value and `v < 0` would fold to false. Everything below is
            therefore expressed on u = v + pad, which is >= 0 by construction (v is
            ot*stride - pad + tap, so u is ot*stride + tap). Both the tests and every
            branch value stay non-negative, which makes the unsigned semantics correct
            rather than merely lucky. The +pad cancels against the -pad already inside
            v, so it costs nothing once folded.
            """
            if const_expr(pad_mode == "zeros"):
                return v, in_range(v, ext)
            u = v + fx.Index(pad)
            low = u < fx.Index(pad)  # v < 0
            high = u >= fx.Index(pad + ext)  # v >= ext
            mid = u - fx.Index(pad)  # v, where in range
            if const_expr(pad_mode == "replicate"):
                r = arith.select(high, fx.Index(ext - 1), mid)
                r = arith.select(low, fx.Index(0), r)
            elif const_expr(pad_mode == "reflect"):
                # [a b c d e] pad 2 -> [c b a b c d e d c]: -v near, 2*(ext-1) - v far.
                r = arith.select(high, fx.Index(2 * (ext - 1) + pad) - u, mid)
                r = arith.select(low, fx.Index(pad) - u, r)
            else:  # circular: v + ext near, v - ext far
                r = arith.select(high, u - fx.Index(pad + ext), mid)
                r = arith.select(low, u + fx.Index(ext - pad), r)
            return fx.Index(r), None

        def gather_valid(base, *masks):
            for m in masks:
                if const_expr(m is not None):
                    base = base & m
            return base

        # ---- Per-thread row decomposition (loop-invariant across K) ----
        _row_dec = []  # per-i tuple of precomputed row terms
        for i in range_constexpr(LDG_A_COUNT):
            linear = (tid + i * BLOCK_THREADS) * LDG_VEC
            local_m = linear // TILE_K
            local_k = linear % TILE_K
            row = m_offset + local_m
            row_valid = row < fx.Index(npq)
            if const_expr(temporal_only_fast):
                out_t = (row // hw_o) % d
                _row_dec.append((local_k, row, row_valid, out_t))
            else:
                n_idx = row // dhw
                rem = row % dhw
                ot = rem // hw_o
                rem2 = rem % hw_o
                oh = rem2 // wo
                ow = rem2 % wo
                in_t0 = ot * st - pt
                in_h0 = oh * sh - ph
                in_w0 = ow * sw - pw
                if const_expr(BIG_IN_N1):
                    di = n_idx - nbase
                    _row_dec.append((local_k, row_valid, di, in_t0, in_h0, in_w0))
                elif const_expr(BIG_IN_NM):
                    _row_dec.append((local_k, row_valid, n_idx, in_t0, in_h0, in_w0))
                else:
                    _row_dec.append((local_k, row_valid, n_idx, in_t0, in_h0, in_w0))

        SCALAR_K = CGP % TILE_K == 0

        # ---- 3D im2col address math ----
        # The K axis decomposes against CGP (per-group channels) while every g_off below
        # keeps `c` (padded total channels) as the NDHWC row stride. `cc` is the absolute
        # input channel: the group base plus the offset within the group.
        def _a_addr(i, kbase_i, cc_base, ckk_base):
            dec = _row_dec[i]
            local_k = dec[0]
            k_abs = kbase_i + fx.Index(local_k)
            if const_expr(SCALAR_K):
                cc = cc_base + fx.Index(local_k)  # cc_base already carries ch_base
            else:
                cc = k_abs % CGP
                if const_expr(groups > 1):
                    cc = ch_base + cc
            k_valid = k_abs < fx.Index(crs)
            if const_expr(temporal_only_fast):
                _, row, row_valid, out_t = dec
                kt_i = ckk_base if const_expr(SCALAR_K) else k_abs // CGP
                temporal_delta = dil(kt_i, dt) - pt
                in_t, m_t = pad_coord(out_t + temporal_delta, d, pt)
                valid = gather_valid(row_valid & k_valid, m_t)
                # `row` already encodes out_t, so the gather shifts it by the resolved
                # delta; under a remap that is no longer the raw tap offset.
                delta = temporal_delta if const_expr(pad_mode == "zeros") else (in_t - out_t)
                if const_expr(BIG_IN_N1):
                    g_off = ((row + delta * hw_o) - (fx.Index(nbase) * dhw + base_t * hw_o)) * c + cc
                else:
                    g_off = (row + delta * hw_o) * c + cc
            else:
                ckk = ckk_base if const_expr(SCALAR_K) else k_abs // CGP
                kw_i = ckk % kw
                ckk2 = ckk // kw
                kh_i = ckk2 % kh
                kt_i = ckk2 // kh
                if const_expr(BIG_IN_N1):
                    _, row_valid, di, in_t0, in_h0, in_w0 = dec
                    in_t, m_t = pad_coord(in_t0 + dil(kt_i, dt), d, pt)
                    in_h, m_h = pad_coord(in_h0 + dil(kh_i, dh), h, ph)
                    in_w, m_w = pad_coord(in_w0 + dil(kw_i, dw), w, pw)
                    valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
                    g_off = (((di * d + (in_t - base_t)) * h + (in_h - base_h)) * w + in_w) * c + cc
                elif const_expr(BIG_IN_NM):
                    _, row_valid, n_idx, in_t0, in_h0, in_w0 = dec
                    in_t, m_t = pad_coord(in_t0 + dil(kt_i, dt), d, pt)
                    in_h, m_h = pad_coord(in_h0 + dil(kh_i, dh), h, ph)
                    in_w, m_w = pad_coord(in_w0 + dil(kw_i, dw), w, pw)
                    valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
                    g_off = ((in_t * h + in_h) * w + in_w) * c + cc
                    return fx.Int32(g_off), valid, n_idx
                else:
                    _, row_valid, n_idx, in_t0, in_h0, in_w0 = dec
                    in_t, m_t = pad_coord(in_t0 + dil(kt_i, dt), d, pt)
                    in_h, m_h = pad_coord(in_h0 + dil(kh_i, dh), h, ph)
                    in_w, m_w = pad_coord(in_w0 + dil(kw_i, dw), w, pw)
                    valid = gather_valid(row_valid & k_valid, m_t, m_h, m_w)
                    g_off = (((n_idx * d + in_t) * h + in_h) * w + in_w) * c + cc
            return fx.Int32(g_off), valid

        def _b_addr(i, k_base):
            linear = (tid + i * BLOCK_THREADS) * LDG_VEC
            local_n = linear // TILE_K
            local_k = linear % TILE_K
            col = n_offset + fx.Index(local_n)
            g_off = fx.Int32(col * crs + (fx.Index(k_base) + fx.Index(local_k)))
            # Tail is per group: the N grid is over-provisioned to groups*tiles_per_group.
            col_valid = ((n_local + fx.Index(local_n)) < fx.Index(KG)) if const_expr(n_tail) else None
            return g_off, col_valid

        # ---- global -> LDS DMA copy, masking via OOB routing ----
        DMA_BYTES = LDG_VEC * BF16_BYTES  # 16
        OOB_ELEM = fx.Int32(OOB_SENTINEL_ELEM)

        def _lds_dma_ptr(lds_array, stage_tile, i):
            off_elems = fx.Index(stage_tile) + (fx.Index(tid) + fx.Index(i * BLOCK_THREADS)) * fx.Index(LDG_VEC)
            base_bytes = off_elems * fx.Index(BF16_BYTES)
            addr = fx.Int64(fx.ptrtoint(lds_array.ptr)) + fx.Int64(base_bytes)
            addr = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, addr.ir_value()))
            return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), addr)

        def _dma_to_lds(rsrc, lds_ptr, voff_elem):
            voff_b = (voff_elem * fx.Int32(BF16_BYTES)).ir_value()
            rocdl.raw_ptr_buffer_load_lds(
                rsrc,
                lds_ptr,
                arith.constant(DMA_BYTES, type=T.i32),
                voff_b,
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
                arith.constant(0, type=T.i32),
            )

        def _load_a(stage, k_base):
            kbase_i = fx.Index(k_base)
            cc_base = ckk_base = None
            if const_expr(SCALAR_K):
                # Loop-invariant and block-uniform, so folding ch_base in here costs no
                # per-load instructions on the SCALAR_K path.
                cc_base = kbase_i % CGP
                if const_expr(groups > 1):
                    cc_base = ch_base + cc_base
                ckk_base = kbase_i // CGP
            stage_tile = fx.Index(stage) * TILE_M * TILE_K
            for i in range_constexpr(LDG_A_COUNT):
                if const_expr(BIG_IN_NM):
                    addr_ret = _a_addr(i, kbase_i, cc_base, ckk_base)
                    g_off_i, valid, n_idx_i = addr_ret
                    sample_addr = x_base_addr + fx.Int64(n_idx_i) * fx.Int64(X_SAMPLE_BYTES)
                    x_rsrc_i = buffer_ops.create_buffer_resource_from_addr(sample_addr, num_records_bytes=BIG_IN_NR)
                    voff = fx.Int32(arith.select(valid, g_off_i, OOB_ELEM))
                    _dma_to_lds(x_rsrc_i, _lds_dma_ptr(a_lds, stage_tile, i), voff)
                else:
                    g_off_i, valid = _a_addr(i, kbase_i, cc_base, ckk_base)
                    voff = fx.Int32(arith.select(valid, g_off_i, OOB_ELEM))
                    _dma_to_lds(x_rsrc, _lds_dma_ptr(a_lds, stage_tile, i), voff)

        def _load_b(stage, k_base):
            stage_tile = fx.Index(stage) * TILE_N * TILE_K
            for i in range_constexpr(LDG_B_COUNT):
                g_off, col_valid = _b_addr(i, k_base)
                if const_expr(n_tail):
                    voff = fx.Int32(arith.select(col_valid, g_off, OOB_ELEM))
                else:
                    voff = g_off
                _dma_to_lds(w_rsrc, _lds_dma_ptr(b_lds, stage_tile, i), voff)

        # ---- single-vec ds_read (LDS -> register), indexed by per-wave MFMA row ----
        def read_a_vec(stage, mi):
            a_row = wave_m * WARP_M + mi * MFMA_M + lane_m
            return lds_load_vec8(a_lds, a_lds_off(stage, fx.Index(a_row), fx.Index(lane_k_a)))

        def read_b_vec(stage, ni):
            b_row = wave_n * WARP_N + ni * MFMA_N + lane_n
            return lds_load_vec8(b_lds, b_lds_off(stage, fx.Index(b_row), fx.Index(lane_k_b)))

        def mfma_one(a_frag, b_frag, c_frag):
            out = mfma_fn(
                T.vec(MFMA_C_VALUES, T.f32),
                [a_frag, b_frag, c_frag, 0, 0, 0],
            )
            rocdl.sched_mfma(1)
            return out

        def read_a_frags(stage):
            frags = [read_a_vec(stage, mi) for mi in range_constexpr(MI_M)]
            rocdl.sched_dsrd(MI_M)
            return frags

        def read_b_frags(stage):
            frags = [read_b_vec(stage, ni) for ni in range_constexpr(MI_N)]
            rocdl.sched_dsrd(MI_N)
            return frags

        def do_compute(acc_values, a_frag_values, b_frag_values):
            rocdl.s_setprio(1)
            for mi in range_constexpr(MI_M):
                for ni in range_constexpr(MI_N):
                    idx = mi * MI_N + ni
                    acc_values[idx] = mfma_one(a_frag_values[mi], b_frag_values[ni], acc_values[idx])
            rocdl.s_setprio(0)
            return acc_values

        # global->LDS software pipeline
        # ---- prologue: fill the pipeline with the first PREFETCH tiles' DMAs ----
        PREFETCH = PIPE_STAGES - 1
        for s in range_constexpr(PREFETCH):
            if const_expr(s < tiles_per_split):
                _load_a(s, k_off + s * TILE_K)
                _load_b(s, k_off + s * TILE_K)
        LDG_PER_TILE = LDG_A_COUNT + LDG_B_COUNT

        # ---- main loop: wait oldest tile, read frags, launch tile PREFETCH ahead, compute ----
        for kt_idx in range_constexpr(tiles_per_split):
            cur = kt_idx % PIPE_STAGES
            inflight_tiles = min(PREFETCH - 1, tiles_per_split - 1 - kt_idx)
            barrier(vmcnt=inflight_tiles * LDG_PER_TILE, lgkmcnt=0)
            a_frags = read_a_frags(cur)
            b_frags = read_b_frags(cur)
            nxt = kt_idx + PREFETCH
            if const_expr(nxt < tiles_per_split):
                _load_a(nxt % PIPE_STAGES, k_off + nxt * TILE_K)
                _load_b(nxt % PIPE_STAGES, k_off + nxt * TILE_K)
                rocdl.sched_vmem(LDG_A_COUNT + LDG_B_COUNT)
            acc = do_compute(acc, a_frags, b_frags)

        # grid.x x grid.z over-provisions the M axis whenever the tiles do not divide
        # evenly between them, so those blocks have to be masked out even at a tail-free npq.
        _row_chk = (npq % TILE_M != 0) or (grid_x * m_chunks > grid_m)
        _need_chk = _row_chk or n_tail
        # The vectorized store walks four consecutive M rows, which are contiguous only in
        # NCDHW. Under NDHWC those rows are K apart -- a lane holds one column of four rows,
        # so there is nothing to widen without a cross-lane shuffle first.
        _vec_store = (n == 1) and (not use_splitk) and (dhw % MFMA_C_VALUES == 0) and (not BIG_OUT) and (not out_ndhwc)

        if const_expr(BIG_OUT):
            y_elem_base = fx.Int64(buffer_ops.extract_base_index(y))

        def _big_store(off_nk_i64, value):
            addr = y_elem_base + off_nk_i64 * fx.Int64(BF16_BYTES)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            llvm.StoreOp(value.ir_value() if hasattr(value, "ir_value") else value, ptr, alignment=2)

        # col_loc is the column within its group; the tail check is per group because the N
        # grid is over-provisioned. At groups == 1 it is the same value as col.
        def _valid_raw(row, col_loc):
            if const_expr(_row_chk and n_tail):
                return arith.andi(row < fx.Index(npq), col_loc < fx.Index(KG))
            if const_expr(_row_chk):
                v = row < fx.Index(npq)
                return arith.andi(v, v)
            v = col_loc < fx.Index(KG)
            return arith.andi(v, v)

        def store_acc():
            for mi in range_constexpr(MI_M):
                row_base = m_offset + wave_m * WARP_M + mi * MFMA_M + c_m_vec
                for ni in range_constexpr(MI_N):
                    col_off = fx.Index(wave_n * WARP_N + ni * MFMA_N + c_n)
                    col = n_offset + col_off
                    col_loc = (n_local + col_off) if const_expr(groups > 1) else col
                    a = Vec(acc[mi * MI_N + ni])
                    if const_expr(has_bias and not use_splitk):
                        col_i = fx.Int32(col)  # bias is indexed by the global out-channel
                        if const_expr(n_tail):
                            col_i = arith.select(col_loc < fx.Index(KG), col_i, fx.Int32(0))
                        bias_val = fx.Float32(buffer_ops.buffer_load(bias_rsrc, col_i, vec_width=1, dtype=fx.Float32))

                    if const_expr(_vec_store):
                        row0 = fx.Index(row_base)
                        off_nk0 = col * dhw + row0

                        def _emit_vec():
                            vals = []
                            for i in range_constexpr(MFMA_C_VALUES):
                                cval = (a[i] + bias_val) if const_expr(has_bias) else a[i]
                                vals.append(cval.to(elem_ty))
                            v4 = fx.Vector.from_elements(vals, dtype=elem_ty)
                            buffer_ops.buffer_store(v4, y_rsrc, off_nk0)

                        if const_expr(_need_chk):
                            if _valid_raw(row0, col_loc):
                                _emit_vec()
                        else:
                            _emit_vec()
                        continue

                    for i in range_constexpr(MFMA_C_VALUES):
                        row = fx.Index(row_base + i)
                        off_sk = row * k + col

                        if const_expr(out_ndhwc):
                            # (n, do, ho, wo, k) is row-major over exactly the GEMM's own
                            # (npq, k) index space, so the scatter collapses to off_sk and
                            # the row decomposition disappears.
                            off_nk = off_sk
                        elif const_expr(n == 1):
                            off_nk = col * dhw + row
                        else:
                            n_idx = row // dhw
                            sp = row % dhw
                            off_nk = n_idx * (k * dhw) + col * dhw + sp

                        def _emit():
                            if const_expr(use_splitk):
                                off_b = fx.Int32(off_sk * 4)
                                z0 = fx.Int32(0)
                                buffer_atomic_add(a[i], y_rsrc, off_b, z0, z0)
                            else:
                                cval = (a[i] + bias_val).to(elem_ty) if const_expr(has_bias) else a[i].to(elem_ty)
                                if const_expr(BIG_OUT):
                                    _big_store(fx.Int64(off_nk), cval)
                                else:
                                    buffer_ops.buffer_store(cval, y_rsrc, off_nk)

                        if const_expr(_need_chk):
                            if _valid_raw(row, col_loc):
                                _emit()
                        else:
                            _emit()

        store_acc()

    @flyc.jit
    def launch(y: fx.Tensor, x: fx.Tensor, weight: fx.Tensor, bias: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        conv3d_implicit_kernel(y, x, weight, bias).launch(
            grid=(grid_x, grid_n, m_chunks * splitk), block=(BLOCK_THREADS, 1, 1), stream=stream
        )

    def _launch(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return launch(y, x, weight, bias, stream=_as_stream(stream))

    def _compile(y, x, weight, bias, stream=None):
        with CompilationContext.compile_hints(CONV_COMPILE_HINTS):
            return flyc.compile(launch, y, x, weight, bias, _as_stream(stream))

    _launch.compile = _compile
    return _launch


# Split-K accumulates into an (npq, k) f32 staging buffer through a buffer atomic, and
# that atomic's voffset is an unsigned 32-bit BYTE offset while the descriptor's
# num_records caps the window at 0xFFFFFFFF. Past 4 GB of staging `off_sk * 4` truncates
# and the high rows wrap onto the start of the buffer, accumulating into unrelated
# outputs; exactly at 4 GB the last element instead lands outside num_records and its
# atomic is dropped. Both are silent, so refuse split-K rather than teach the epilogue
# 64-bit addressing: split-K only pays when the tile grid is too small to fill the device,
# and an npq*k this large is already tens of thousands of tiles.
SPLITK_MAX_STAGING_BYTES = 0xFFFFFFFF


def _resolve_splitk(splitk, npq, crs, k, device, tile=DEFAULT_TILE, groups=1):
    k_tiles = (crs + TILE_K - 1) // TILE_K
    # Correctness bound, so it has to gate an explicit splitk too -- the auto branch's own
    # staging term below is a stricter memory-traffic heuristic, not this limit.
    if npq * k * 4 > SPLITK_MAX_STAGING_BYTES:
        return 1
    if splitk is None:
        tile_m, tile_n = tile[0], tile[1]
        kg = k // groups
        base = ((npq + tile_m - 1) // tile_m) * groups * ((kg + tile_n - 1) // tile_n)
        if (
            npq < 4096
            or k_tiles < 16
            or kg % tile_n != 0
            or npq % tile_m != 0
            or crs % TILE_K != 0
            or npq * k * 4 > 0x7FFFFFFF
        ):
            sk = 1
        else:
            try:
                num_cu = torch.cuda.get_device_properties(device).multi_processor_count
            except Exception:
                num_cu = 256
            if base >= (3 * num_cu) // 4:
                sk = 1
            else:
                sk = min(4, max(1, num_cu // base), k_tiles)
    else:
        sk = max(1, splitk)
    while sk > 1 and k_tiles % sk != 0:
        sk -= 1
    return sk


def _as_tuple(v, rank, name):
    """Normalize torch's int / length-1 / length-``rank`` sequence forms to a tuple.

    torch broadcasts a length-1 sequence across every spatial axis, so ``stride=(2,)``
    on a 3D conv means ``(2, 2, 2)``. Unpacking the sequence directly instead would
    raise ValueError on that form.
    """
    if isinstance(v, int):
        return (v,) * rank
    t = tuple(v)
    if len(t) == 1:
        return t * rank
    assert len(t) == rank, f"{name} must be an int or a sequence of 1 or {rank} ints, got {tuple(v)}"
    return t


def _resolve_padding(padding, kernel, stride, dilation):
    """Normalize torch's ``padding`` argument to a (low, high) pair of per-axis triples.

    An int or a triple is symmetric, so both sides come back the same. The two strings
    torch accepts are resolved here instead: "valid" is no padding, and "same" holds the
    output extent equal to the input's, which takes ``dilation * (kernel - 1)`` elements
    per axis. When that total is odd it cannot be split evenly and torch puts the extra
    element on the high side, so the two returned triples differ -- see ``_conv3d_impl``
    for how that case is lowered. "same" is only defined at stride 1, matching torch.
    """
    if not isinstance(padding, str):
        p = _as_tuple(padding, 3, "padding")
        assert min(p) >= 0, f"negative padding is not supported, got (pt, ph, pw) = {p}"
        return p, p
    if padding == "valid":
        return (0, 0, 0), (0, 0, 0)
    if padding != "same":
        raise ValueError(f"padding string must be 'same' or 'valid', got {padding!r}")
    assert all(
        s == 1 for s in stride
    ), f"padding='same' is not supported for strided convolutions, got stride {tuple(stride)}"
    total = [dl * (kn - 1) for kn, dl in zip(kernel, dilation)]
    return tuple(t // 2 for t in total), tuple(t - t // 2 for t in total)


def _conv3d_impl(
    x,
    weight,
    bias=None,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    splitk=None,
    stream=None,
    tile=None,
    autotune=None,
    input_layout="NCDHW",
    output_layout="NCDHW",
):
    _check_layouts(3, input_layout, output_layout)
    # in_ndhwc is not constant: the padding fallbacks below can hand back a channels-first
    # copy, after which the ordinary transpose path takes over again.
    in_ndhwc = input_layout == "NDHWC"
    out_ndhwc = output_layout == "NDHWC"
    n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)
    k, wc, kt, kh, kw = weight.shape
    # Device and dtype first: everything below this point either launches a kernel or
    # allocates on x.device, and a host tensor reaching that far faults the GPU instead
    # of raising.
    for name, t in (("x", x), ("weight", weight), ("bias", bias)):
        assert t is None or t.is_cuda, f"conv3d_implicit needs GPU tensors; {name} is on {t.device}"
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16, (
        f"conv3d_implicit is a bf16-only kernel; got x={x.dtype}, weight={weight.dtype}"
    )
    assert bias is None or (bias.dim() == 1 and bias.numel() == k), (
        f"bias must be a 1-D tensor of {k} elements, one per output channel; "
        f"got shape {tuple(bias.shape)}"
    )
    groups = int(groups)
    assert groups >= 1, f"groups must be >= 1, got {groups}"
    assert c % groups == 0, f"in-channels {c} not divisible by groups {groups}"
    assert k % groups == 0, f"out-channels {k} not divisible by groups {groups}"
    assert wc == c // groups, f"weight in-channels {wc} != C/groups = {c // groups}"
    st, sh, sw = _as_tuple(stride, 3, "stride")
    # Both of these are torch errors, and validating them here is what makes them say so:
    # stride 0 would otherwise divide by zero computing the output extent below, and a
    # negative stride or padding would collapse that extent and trip the unrelated
    # "dilated filter is larger than the padded input" assertion. The 1D/2D entries widen
    # their argument to three axes with 1s, so the triple reported here can be longer than
    # the one that was passed in.
    assert min(st, sh, sw) >= 1, f"non-positive stride is not supported, got (st, sh, sw) = {(st, sh, sw)}"
    dt, dh, dw = _as_tuple(dilation, 3, "dilation")
    assert min(dt, dh, dw) >= 1, f"dilation must be >= 1, got {(dt, dh, dw)}"
    pad_lo, pad_hi = _resolve_padding(padding, (kt, kh, kw), (st, sh, sw), (dt, dh, dw))
    pt, ph, pw = pad_lo
    assert padding_mode in PADDING_MODES, f"padding_mode must be one of {PADDING_MODES}, got {padding_mode!r}"

    # The bounds torch enforces inside its own pad. They also make the kernel's remap a
    # single step, so check them up front on both paths for one consistent message. An
    # uneven "same" pad is checked on its wider side, which is the one that can overrun.
    if padding_mode in ("reflect", "circular"):
        for ax, (p, ext) in enumerate(zip(map(max, pad_lo, pad_hi), (d, h, w))):
            if padding_mode == "reflect":
                assert p < ext, f"reflect padding {p} must be < input extent {ext} on spatial axis {ax}"
            else:
                assert p <= ext, f"circular padding {p} must be <= input extent {ext} on spatial axis {ax}"

    # An odd "same" total splits unevenly, and the kernel carries one pad per axis. Torch
    # has the same limitation and resolves it the same way -- by materializing a padded
    # copy of the input, which it warns about. Under "zeros" only the surplus on the high
    # side has to exist, because a zero tap past the high edge is already what the
    # gather's range mask produces; pad that side alone and keep convolving with pad_lo.
    # The other modes fill the whole border in one call, matching nn.ConvNd, because two
    # chained pads do not compose (reflecting by 1 then by 2 is not reflecting by 3).
    if pad_lo != pad_hi:
        if padding_mode == "zeros":
            x, in_ndhwc = _pad_spatial(x, in_ndhwc, (0, pad_hi[2] - pw, 0, pad_hi[1] - ph, 0, pad_hi[0] - pt))
        else:
            x, in_ndhwc = _pad_spatial(x, in_ndhwc, (pw, pad_hi[2], ph, pad_hi[1], pt, pad_hi[0]), mode=padding_mode)
            pt = ph = pw = 0
        n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)

    # Non-zero modes are resolved inside the im2col gather, which remaps an out-of-range
    # tap onto a real input coordinate instead of masking it to zero. No border is
    # materialized, so this costs no extra memory traffic.
    #
    # BIG_IN is the exception: that path rebases the input buffer per block to keep
    # offsets in 32 bits, and a reflected tap can resolve below the block's base. Those
    # inputs (> 2^31 elements) fall back to torch's pre-pad, which is always correct.
    inline_pad = padding_mode != "zeros" and bool(pt or ph or pw)
    if inline_pad and _big_in(n, c, groups, d, h, w, pt, ph, pw):
        x, in_ndhwc = _pad_spatial(x, in_ndhwc, (pw, pw, ph, ph, pt, pt), mode=padding_mode)
        n, c, d, h, w = _shape_ncdhw(x, in_ndhwc)
        pt = ph = pw = 0
        inline_pad = False
    pad_mode = padding_mode if inline_pad else "zeros"

    # 1x1x1 fast path: y[n,k,dhw] = sum_c weight[k,c] * x[n,c,dhw] — pure channel GEMM.
    # Grouped 1x1x1 is block-diagonal, so it goes through the kernel instead.
    if (
        groups == 1
        and kt == 1
        and kh == 1
        and kw == 1
        and st == 1
        and sh == 1
        and sw == 1
        and pt == 0
        and ph == 0
        and pw == 0
    ):
        wm = weight.reshape(k, c)
        if in_ndhwc:
            # Channels-last is already the GEMM's row-major (npq, C) operand, so this side
            # runs x @ W^T and lands directly in NDHWC.
            y = torch.matmul(x.reshape(n * d * h * w, c), wm.t()).reshape(n, d, h, w, k)
            if bias is not None:
                y = y + bias.to(y.dtype)
            return y if out_ndhwc else y.permute(0, 4, 1, 2, 3).contiguous()
        if n == 1:
            y = torch.matmul(wm, x.reshape(c, d * h * w)).reshape(n, k, d, h, w)
        else:
            y = torch.matmul(wm, x.reshape(n, c, d * h * w)).reshape(n, k, d, h, w)
        if bias is not None:
            y = y + bias.to(y.dtype).view(1, k, 1, 1, 1)
        return y.permute(0, 2, 3, 4, 1).contiguous() if out_ndhwc else y

    do = (d + 2 * pt - (dt * (kt - 1) + 1)) // st + 1
    ho = (h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    wo = (w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    assert min(do, ho, wo) >= 1, f"dilated filter is larger than the padded input: output ({do}, {ho}, {wo})"
    npq = n * do * ho * wo

    # An empty batch would launch the transpose with grid.z == 0 and the conv with grid.x == 0;
    # both return hipErrorInvalidValue and leave the HIP context unusable. Return the empty
    # output torch produces instead, before any launch.
    if n == 0:
        empty = (0, do, ho, wo, k) if out_ndhwc else (0, k, do, ho, wo)
        return torch.empty(empty, device=x.device, dtype=torch.bfloat16)

    # Zero-pad C to the gather's vector width; padded channels see zero weights. The pad is
    # PER GROUP, since the gather vectorizes along channels and must not cross into the next
    # group. _prep_weight pads the weight's own C the same way, so the two stay aligned.
    cg = c // groups
    cgp = _pad_channels(cg)
    if cgp != cg:
        if in_ndhwc:
            x = torch.nn.functional.pad(x.reshape(n, d, h, w, groups, cg), (0, cgp - cg))
            x = x.reshape(n, d, h, w, groups * cgp)
        else:
            x = torch.nn.functional.pad(x.reshape(n, groups, cg, d, h, w), (0, 0, 0, 0, 0, 0, 0, cgp - cg))
            x = x.reshape(n, groups * cgp, d, h, w)
    c = groups * cgp
    crs = cgp * kt * kh * kw

    launch_stream = torch.cuda.current_stream() if stream is None else stream
    has_bias = bias is not None
    bias_arg = bias.to(torch.float32).contiguous() if has_bias else torch.empty(1, device=x.device, dtype=torch.float32)

    # The gather is channels-last, so an NDHWC input is already in the kernel's layout and
    # only has to be made contiguous (a no-op for the ordinary case).
    x_ndhwc = x.contiguous() if in_ndhwc else _ncdhw_to_ndhwc(x, stream)
    w_packed = _prep_weight(weight, k, kt, kh, kw, wc)

    shape = (n, c, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, dt, dh, dw, pad_mode, has_bias, groups, out_ndhwc)

    def _run(the_tile, the_wgm=1):
        sk = _resolve_splitk(splitk, npq, crs, k, x.device, the_tile, groups)
        if sk > 1:
            y = torch.zeros((npq, k), device=x.device, dtype=torch.float32)
        else:
            out_shape = (n, do, ho, wo, k) if out_ndhwc else (n, k, do, ho, wo)
            y = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        exe = compile_conv3d_implicit(
            n,
            c,
            d,
            h,
            w,
            k,
            kt,
            kh,
            kw,
            st,
            sh,
            sw,
            pt,
            ph,
            pw,
            dt,
            dh,
            dw,
            pad_mode,
            has_bias,
            sk,
            the_tile,
            the_wgm,
            groups,
            out_ndhwc,
        )
        _dispatch(exe, y, x_ndhwc, w_packed, bias_arg, stream=launch_stream)
        return y, sk

    if tile is not None:
        chosen_tile = tuple(tile)
        chosen_wgm = 1
    elif autotune or (autotune is None and _autotune_enabled()):
        from kernels.conv.conv3d_autotune import BF16_CANDIDATES, WGM_VALUES, autotune_conv3d

        candidates = [(t, w) for t in BF16_CANDIDATES for w in WGM_VALUES]
        best = autotune_conv3d("bf16", shape, "bf16", candidates, x.device, lambda tw: _run(tw[0], tw[1])[0])
        chosen_tile, chosen_wgm = best
    else:
        # A tile never spans groups, so a 128-wide N tile sits mostly empty when K/groups is
        # small; drop to the narrowest legal candidate. Autotune picks properly when enabled.
        chosen_tile = (64, 64, 2, 2) if (groups > 1 and k // groups < DEFAULT_TILE[1]) else DEFAULT_TILE
        chosen_wgm = 1

    y, sk = _run(chosen_tile, chosen_wgm)
    if sk > 1:
        if has_bias:
            y = y + bias_arg.view(1, k)
        # The split-K accumulator is the raw GEMM shape, i.e. NDHWC, so the view below is a
        # metadata-only relabel. Requesting NDHWC therefore leaves only the f32 -> bf16 cast.
        if out_ndhwc:
            return y.view(n, do, ho, wo, k).to(torch.bfloat16)
        # Materialize contiguous NCDHW so the memory format does not depend on whether
        # split-K ran. copy_ folds the cast into the transpose, keeping this to the single
        # pass the cast cost anyway.
        out = torch.empty((n, k, do, ho, wo), device=x.device, dtype=torch.bfloat16)
        out.copy_(y.view(n, do, ho, wo, k).permute(0, 4, 1, 2, 3))
        return out
    return y


def _conv2d_impl(
    x, weight, bias=None, stride=1, padding=0, dilation=1, input_layout="NCHW", output_layout="NCHW", **kwargs
):
    assert x.dim() == 4 and weight.dim() == 4, "conv2d expects (N,C,H,W) / (K,C,R,S)"
    _check_layouts(2, input_layout, output_layout)
    sh, sw = _as_tuple(stride, 2, "stride")
    dh, dw = _as_tuple(dilation, 2, "dilation")
    # A padding string stays a string: the degenerate depth axis is a single 1-tap slice,
    # so "same" resolves to no padding on it anyway.
    if isinstance(padding, str):
        p3 = padding
    else:
        ph, pw = _as_tuple(padding, 2, "padding")
        p3 = (0, ph, pw)
    k, wc, r, s = weight.shape
    # The degenerate depth axis goes in front of the spatial ones in either layout, so both
    # reshapes stay views.
    if input_layout == "NHWC":
        n, h, w, c = x.shape
        x5, in5 = x.reshape(n, 1, h, w, c), "NDHWC"
    else:
        n, c, h, w = x.shape
        x5, in5 = x.reshape(n, c, 1, h, w), "NCDHW"
    out5 = "NDHWC" if output_layout == "NHWC" else "NCDHW"
    w5 = weight.reshape(k, wc, 1, r, s)
    y5 = _conv3d_impl(
        x5,
        w5,
        bias=bias,
        stride=(1, sh, sw),
        padding=p3,
        dilation=(1, dh, dw),
        input_layout=in5,
        output_layout=out5,
        **kwargs,
    )
    if output_layout == "NHWC":
        return y5.reshape(y5.shape[0], y5.shape[2], y5.shape[3], y5.shape[4])
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[3], y5.shape[4])


def _conv1d_impl(
    x, weight, bias=None, stride=1, padding=0, dilation=1, input_layout="NCW", output_layout="NCW", **kwargs
):
    assert x.dim() == 3 and weight.dim() == 3, "conv1d expects (N,C,W) / (K,C,S)"
    _check_layouts(1, input_layout, output_layout)
    (sw,) = _as_tuple(stride, 1, "stride")
    (dw,) = _as_tuple(dilation, 1, "dilation")
    if isinstance(padding, str):
        p3 = padding
    else:
        p3 = (0, 0, _as_tuple(padding, 1, "padding")[0])
    k, wc, s = weight.shape
    if input_layout == "NWC":
        n, w, c = x.shape
        x5, in5 = x.reshape(n, 1, 1, w, c), "NDHWC"
    else:
        n, c, w = x.shape
        x5, in5 = x.reshape(n, c, 1, 1, w), "NCDHW"
    out5 = "NDHWC" if output_layout == "NWC" else "NCDHW"
    w5 = weight.reshape(k, wc, 1, 1, s)
    y5 = _conv3d_impl(
        x5,
        w5,
        bias=bias,
        stride=(1, 1, sw),
        padding=p3,
        dilation=(1, 1, dw),
        input_layout=in5,
        output_layout=out5,
        **kwargs,
    )
    if output_layout == "NWC":
        return y5.reshape(y5.shape[0], y5.shape[3], y5.shape[4])
    return y5.reshape(y5.shape[0], y5.shape[1], y5.shape[4])


def conv3d_implicit(x, weight, bias=None, stride=1, padding=0, dilation=1, **kwargs):
    """Main implicit-GEMM conv entry; dispatches 1D/2D/3D by filter rank.

    Rank is taken from the filter (weight.dim() - 2): 3 -> 3D (N,C,D,H,W)/(K,C,T,R,S).

    ``input_layout`` and ``output_layout`` are independent and named per rank:
    "NCDHW"/"NDHWC", "NCHW"/"NHWC", "NCW"/"NWC". The weight stays KC*, and the batch axis
    leads in both, so an unbatched input works either way. Channels-last is the kernel's
    own layout on both sides: an NDHWC input skips the pre-transpose, and an NDHWC output
    is the (npq, K) index space the GEMM already writes, so it also skips the split-K
    epilogue's transpose. Channels-last output does give up the vectorized store on the
    ``n == 1`` fast path, since a lane's four accumulator values are four M rows and those
    are K apart once channels are innermost.

    ``padding`` takes an int, a per-axis tuple, or one of torch's two strings. "valid" is
    no padding. "same" pads so the output keeps the input's spatial extent, which needs
    ``dilation * (kernel - 1)`` elements per axis and, like torch, is only defined at
    stride 1. That total is normally even and costs nothing beyond an ordinary symmetric
    pad. An even-length filter under odd dilation makes it odd, and torch's rule of
    putting the extra element on the high side then asks for a pad the kernel cannot
    express with one value per axis; that case materializes a padded input first, exactly
    as torch does (it warns about the same copy). ``padding_mode`` applies to "same" too.

    ``dilation`` follows torch semantics: it spaces the filter taps by that factor
    over the input, shrinking the output to
    ``(D + 2*pad - dilation*(T-1) - 1)//stride + 1`` per axis. It costs nothing in the
    GEMM -- the K axis is still C/groups*T*R*S -- it only stretches the im2col gather,
    so a dilated filter reads a wider input footprint per output row and gets less
    reuse out of cache than the same filter undilated.

    ``groups`` follows torch semantics: C and K must both be divisible by it and the
    weight's channel dim is C/groups. Groups map onto the N grid axis, one tile never
    spanning two groups, so efficiency tracks how well K/groups fills TILE_N. Measured
    on gfx950 vs torch/MIOpen, moderate cardinality wins across the board (1.5-2.0x for
    K/groups in [8, 256]). True depthwise (groups == C, so C/groups == 1) is the one
    weak case at ~0.5x: C/groups=1 pads to the gather's 8-wide vector, wasting 7/8 of
    the K axis, while K/groups=1 leaves all but one column of the N tile masked.
    Narrower tiles recover little there -- depthwise wants its own kernel, not this
    single-GEMM mapping.
    """
    spatial_rank = weight.dim() - 2
    if spatial_rank not in (1, 2, 3):
        raise ValueError(f"conv3d_implicit supports 1D/2D/3D; got filter rank {weight.dim()}")
    # An unbatched (C, *spatial) input runs as a batch of one and loses the dim again on
    # the way out, matching torch.
    unbatched = x.dim() == weight.dim() - 1
    if unbatched:
        x = x.unsqueeze(0)
    assert x.dim() == weight.dim(), f"x rank {x.dim()} != weight rank {weight.dim()}"
    impl = {3: _conv3d_impl, 2: _conv2d_impl, 1: _conv1d_impl}[spatial_rank]
    y = impl(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, **kwargs)
    return y.squeeze(0) if unbatched else y
