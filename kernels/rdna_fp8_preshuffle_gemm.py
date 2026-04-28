"""Fast Float8 Preshuffle GEMM for RDNA4 (gfx120x, wave32).

Optimized for M=32, N=8192, K=6144 (decode-phase inference shape).

  C[M,N] = A[M,K] @ B[K,N]

Both A and B are fp8_e4m3fn with per-tensor scales.
Output is bf16.  Accumulation in f32.

A is loaded directly from raw [M,K] layout (no preshuffle needed).
Uses per-token (rowwise) scaling: scale_a[M] for activation, scale_b[N] for weight.
B must be preshuffled to [N0, K0, KLane=2, NLane=16, KPack=8] bytes.
  - No LDS needed — direct GMEM -> register -> WMMA pipeline
  - Software-pipelined K-loop with compile-time inner unrolling

Tile config (tuned for M=32):
  tile_m=32  (2 WMMA M-tiles)
  tile_n=128 (8 WMMA N-tiles)
  tile_k=32  (2 WMMA K-tiles)
  waves_m=1, waves_n=2 → 2 waves = 64 threads per block
  wave_reg_m=2, wave_reg_n=4 → 8 accumulators per wave
"""

import os
import functools
import time

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl, range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.expr.typing import T


WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


# =============================================================================
# Host-side preshuffle functions
# =============================================================================


def preshuffle_b_fp8(B_kn):
    """Preshuffle B[K,N] fp8 for WMMA B operand layout.

    Layout: [N0, K0, KLane=2, NLane=16, KPack=8] bytes.
    lane16 selects N column, klane selects K half.
    """
    import torch

    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    N0 = N // 16
    K0 = K // 16
    B_view = B_kn.view(torch.uint8)
    B_reshaped = B_view.reshape(K0, 2, 8, N0, 16)
    return B_reshaped.permute(3, 0, 1, 4, 2).contiguous()  # [N0, K0, 2, 16, 8]


def fp8_quantize_per_token(x_f32):
    """Quantize f32 tensor to fp8_e4m3fn with per-token (per-row) scale.

    Returns (x_fp8, scale_per_token) where:
      x_f32[m, :] ~ x_fp8[m, :].float() * scale_per_token[m]
      scale_per_token shape: [M]
    """
    import torch

    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0  # fp8_e4m3fn max = 448.0
    x_scaled = (x_f32 / scale).clamp(-448.0, 448.0)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale.squeeze(-1)  # [M]


def fp8_quantize_per_channel(x_f32):
    """Quantize f32 tensor to fp8_e4m3fn with per-channel (per-column) scale.

    Returns (x_fp8, scale_per_channel) where:
      x_f32[:, n] ~ x_fp8[:, n].float() * scale_per_channel[n]
      scale_per_channel shape: [N]
    """
    import torch

    amax = x_f32.abs().amax(dim=0).clamp(min=1e-12)
    scale = amax / 448.0
    x_scaled = (x_f32 / scale.unsqueeze(0)).clamp(-448.0, 448.0)
    x_fp8 = x_scaled.to(torch.float8_e4m3fn)
    return x_fp8, scale  # [N]


# =============================================================================
# Kernel compiler
# =============================================================================


@functools.lru_cache(maxsize=64)
def compile_fp8_gemm(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int = 32,
    tile_n: int = None,
    tile_k: int = 32,
    k_unroll: int = None,
    group_m: int = 8,
):
    """Compile fp8 GEMM for RDNA4.

    A is raw fp8 [M,K] (no preshuffle needed). B must be preshuffled.
    Optimized for small-M shapes (e.g., M=32, decode phase).

    Args:
        M, N, K: Matrix dimensions. Must be divisible by tile sizes.
        tile_m: Block tile M (default 32 for small-M).
        tile_n: Block tile N (default 128).
        tile_k: Block tile K (default 32 = 2 WMMA K-tiles).
        k_unroll: Inner K-loop unroll factor.
        group_m: L2 cache swizzle group size.

    Returns:
        launch(c, a_fp8_f32, b_shuf_f32, scale_a_per_token, scale_b_per_channel, stream)
    """
    # Auto-select tile_n and k_unroll based on shape
    if tile_n is None:
        tile_n = 256 if M >= 256 else 128
    if k_unroll is None:
        k_unroll = 1 if M >= 256 else 2

    gpu_arch = get_rocm_arch()

    WAVE_SIZE = 32
    assert tile_m % WMMA_M == 0, f"tile_m={tile_m} must be multiple of {WMMA_M}"
    assert tile_n % WMMA_N == 0, f"tile_n={tile_n} must be multiple of {WMMA_N}"
    assert tile_k % WMMA_K == 0, f"tile_k={tile_k} must be multiple of {WMMA_K}"
    assert M % tile_m == 0, f"M={M} must be multiple of tile_m={tile_m}"
    assert N % tile_n == 0, f"N={N} must be multiple of tile_n={tile_n}"
    assert K % tile_k == 0, f"K={K} must be multiple of tile_k={tile_k}"

    reg_m = tile_m // WMMA_M  # 32/16 = 2
    reg_n = tile_n // WMMA_N  # 128/16 = 8
    reg_k = tile_k // WMMA_K  # 32/16 = 2

    # Wave layout: for small M, put all waves along N
    if tile_m >= 128 and tile_n >= 128:
        waves_m, waves_n = 2, 2
    elif tile_m >= 64 and tile_n >= 128:
        waves_m, waves_n = 2, 2
    elif tile_n >= 256:
        waves_m, waves_n = 1, 2
    elif tile_m >= 64:
        waves_m, waves_n = 2, 1
    elif tile_n >= 128:
        waves_m, waves_n = 1, 2
    else:
        waves_m, waves_n = 1, 1

    NUM_WAVES = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE
    wave_reg_m = reg_m // waves_m
    wave_reg_n = reg_n // waves_n

    num_k_tiles = K // tile_k
    grid_m = M // tile_m
    grid_n = N // tile_n

    K0_total = K // 16  # total WMMA K-tiles across full K dimension

    # B preshuffle strides (byte-based for fp8)
    # B layout: [N0, K0, KLane=2, NLane=16, KPack=8] bytes
    B_KPACK = 8
    B_STRIDE_NLANE = B_KPACK  # 8
    B_STRIDE_KLANE = 16 * B_KPACK  # 128
    B_STRIDE_K0 = 2 * 16 * B_KPACK  # 256
    B_STRIDE_N0 = K0_total * B_STRIDE_K0

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
    ):
        # === Types ===
        f32 = T.f32
        bf16 = T.bf16
        i32 = T.i32
        v8f32_ty = T.vec(8, T.f32)
        v2i32_ty = T.i32x2

        # === Thread/block IDs ===
        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")

        c32 = fx.Index(32)
        c16 = fx.Index(16)
        c8 = fx.Index(8)
        wave_id = tid // c32
        lane = tid % c32
        lane16 = lane % c16
        klane = lane // c16

        # === L2 cache swizzle ===
        effective_group_m = min(group_m, grid_m)
        c_grid_n = fx.Index(grid_n)
        c_group_m = fx.Index(effective_group_m)
        num_pid_in_group = c_group_m * c_grid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * c_group_m
        group_size_m = c_group_m
        pid_in_group = pid % num_pid_in_group
        bid_m = first_pid_m + (pid_in_group % group_size_m)
        bid_n = pid_in_group // group_size_m

        # === Wave position within workgroup ===
        c_wn = fx.Index(waves_n)
        wave_m = wave_id // c_wn
        wave_n = wave_id % c_wn

        tile_m0 = bid_m * fx.Index(tile_m)
        tile_n0 = bid_n * fx.Index(tile_n)

        # === Buffer resources ===
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=True)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

        # === Scale resources (per-token scale_a[M], per-channel scale_b[N]) ===
        # Scales are loaded per-element in the epilogue

        # === Tile load functions ===

        def _load_a_tile(k_tile_idx):
            """Load A fp8 tile from raw A[M,K]. Returns [reg_k][wave_reg_m] of v2i32.

            Each lane loads 8 contiguous fp8 bytes from:
              row = tile_m0 + wave_m*wave_reg_m*16 + rm*16 + lane16
              col = k_tile_idx*tile_k + rk*16 + klane*8
            """
            a_vecs = []
            c_K = fx.Index(K)
            for rk in range_constexpr(reg_k):
                rk_vecs = []
                col_base = k_tile_idx * fx.Index(tile_k) + fx.Index(rk * 16) + klane * c8
                for rm in range_constexpr(wave_reg_m):
                    row = tile_m0 + wave_m * fx.Index(wave_reg_m * WMMA_M) + fx.Index(rm * WMMA_M) + lane16
                    byte_off = row * c_K + col_base
                    dword_off = byte_off // fx.Index(4)
                    a_raw = buffer_ops.buffer_load(a_rsrc, dword_off, vec_width=2, dtype=i32)
                    rk_vecs.append(a_raw)
                a_vecs.append(rk_vecs)
            return a_vecs

        def _load_b_tile(k_tile_idx):
            """Load B fp8 tile. Returns [reg_k][wave_reg_n] of v2i32."""
            b_vecs = []
            n0_base = tile_n0 // c16 + wave_n * fx.Index(wave_reg_n)
            for rk in range_constexpr(reg_k):
                rk_vecs = []
                k0 = k_tile_idx * fx.Index(reg_k) + fx.Index(rk)
                for rn in range_constexpr(wave_reg_n):
                    n0 = n0_base + fx.Index(rn)
                    byte_off = (
                        n0 * fx.Index(B_STRIDE_N0)
                        + k0 * fx.Index(B_STRIDE_K0)
                        + klane * fx.Index(B_STRIDE_KLANE)
                        + lane16 * fx.Index(B_STRIDE_NLANE)
                    )
                    dword_off = byte_off // fx.Index(4)
                    b_raw = buffer_ops.buffer_load(b_rsrc, dword_off, vec_width=2, dtype=i32)
                    rk_vecs.append(b_raw)
                b_vecs.append(rk_vecs)
            return b_vecs

        # === Compute function ===

        def _do_compute(accs_in, a_vecs, b_vecs):
            """Run WMMA fp8 multiply-accumulate for one tile."""
            new_accs = list(accs_in)
            for rk in range_constexpr(reg_k):
                # Load all B for this rk, then iterate A (minimize reg pressure)
                for rm in range_constexpr(wave_reg_m):
                    for rn in range_constexpr(wave_reg_n):
                        idx = rm * wave_reg_n + rn
                        new_accs[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                            v8f32_ty,
                            a_vecs[rk][rm],
                            b_vecs[rk][rn],
                            arith.unwrap(new_accs[idx]),
                        ).result
            return new_accs

        # === Initialize accumulators ===
        zero_acc = arith.constant_vector(0.0, v8f32_ty)
        accs = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n)]

        # === Software-pipelined K-loop ===
        # Prologue: load first tile
        a_cur = _load_a_tile(fx.Index(0))
        b_cur = _load_b_tile(fx.Index(0))

        full_outer_iters = (num_k_tiles - 1) // k_unroll
        remainder = (num_k_tiles - 1) % k_unroll

        # Flatten/unflatten helpers for loop-carried state
        def _flatten_tile(tile):
            flat = []
            for rk_vecs in tile:
                flat.extend(rk_vecs)
            return flat

        def _unflatten_a(flat):
            out = []
            idx = 0
            for rk in range_constexpr(reg_k):
                row = []
                for rm in range_constexpr(wave_reg_m):
                    row.append(flat[idx])
                    idx += 1
                out.append(row)
            return out

        def _unflatten_b(flat):
            out = []
            idx = 0
            for rk in range_constexpr(reg_k):
                row = []
                for rn in range_constexpr(wave_reg_n):
                    row.append(flat[idx])
                    idx += 1
                out.append(row)
            return out

        n_a = reg_k * wave_reg_m
        n_b = reg_k * wave_reg_n
        n_acc = wave_reg_m * wave_reg_n

        # Build initial state: [a_flat, accs, b_flat]
        init_state = _flatten_tile(a_cur) + list(accs) + _flatten_tile(b_cur)

        # Main K-loop: SCF outer with constexpr inner unroll
        if const_expr(full_outer_iters > 0):
            for iv, state in range(0, full_outer_iters * k_unroll, k_unroll, init=init_state):
                s_a = _unflatten_a(list(state[:n_a]))
                s_accs = list(state[n_a : n_a + n_acc])
                s_b = _unflatten_b(list(state[n_a + n_acc :]))

                # Inner unroll: pipeline load-before-compute
                for j in range_constexpr(k_unroll):
                    next_kt = iv + fx.Index(j + 1)
                    a_next = _load_a_tile(next_kt)
                    b_next = _load_b_tile(next_kt)
                    s_accs = _do_compute(s_accs, s_a, s_b)
                    s_a = _unflatten_a(_flatten_tile(a_next))
                    s_b = _unflatten_b(_flatten_tile(b_next))

                results = yield _flatten_tile(s_a) + list(s_accs) + _flatten_tile(s_b)

            a_cur = _unflatten_a(list(results[:n_a]))
            accs = list(results[n_a : n_a + n_acc])
            b_cur = _unflatten_b(list(results[n_a + n_acc :]))

        # Handle remainder tiles
        if const_expr(remainder > 0):
            for j in range_constexpr(remainder):
                next_kt = fx.Index(full_outer_iters * k_unroll + j + 1)
                a_next = _load_a_tile(next_kt)
                b_next = _load_b_tile(next_kt)
                accs = _do_compute(accs, a_cur, b_cur)
                a_cur = _unflatten_a(_flatten_tile(a_next))
                b_cur = _unflatten_b(_flatten_tile(b_next))

        # Epilogue: compute last loaded tile
        accs = _do_compute(accs, a_cur, b_cur)

        # === Store results with scaling ===
        c_n = fx.Index(N)
        base8 = klane * c8
        # Pre-load scale_b for each N column this lane writes to
        sb_cache = []
        for rn in range_constexpr(wave_reg_n):
            g_col = tile_n0 + wave_n * fx.Index(wave_reg_n * WMMA_N) + fx.Index(rn * WMMA_N) + lane16
            sb_cache.append(buffer_ops.buffer_load(scale_b_rsrc, g_col, vec_width=1, dtype=f32))

        for rm in range_constexpr(wave_reg_m):
            wmma_m_off = wave_m * fx.Index(wave_reg_m * WMMA_M) + fx.Index(rm * WMMA_M)
            # Pre-load scale_a for the 8 rows in this WMMA M tile
            sa_cache = []
            for si in range_constexpr(8):
                g_row_si = tile_m0 + wmma_m_off + base8 + fx.Index(si)
                sa_cache.append(buffer_ops.buffer_load(scale_a_rsrc, g_row_si, vec_width=1, dtype=f32))

            for rn in range_constexpr(wave_reg_n):
                idx = rm * wave_reg_n + rn
                wmma_n_off = wave_n * fx.Index(wave_reg_n * WMMA_N) + fx.Index(rn * WMMA_N)
                sb_val = sb_cache[rn]
                for si in range_constexpr(8):
                    g_row = tile_m0 + wmma_m_off + base8 + fx.Index(si)
                    g_col = tile_n0 + wmma_n_off + lane16
                    val = vector.extract(
                        accs[idx],
                        static_position=[si],
                        dynamic_position=[],
                    )
                    val = val * sa_cache[si] * sb_val
                    val_bf16 = arith.trunc_f(bf16, val)
                    elem_off = g_row * c_n + g_col
                    buffer_ops.buffer_store(val_bf16, c_rsrc, elem_off)

    # ── Host launcher ──────────────────────────────────────────────────────
    @flyc.jit
    def launch_fp8_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        stream: fx.Stream,
    ):
        c1 = 1
        total_blocks = grid_m * grid_n
        bk = THREADS_PER_BLOCK

        launcher = kernel_gemm(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b)
        launcher.launch(
            grid=(total_blocks, c1, c1),
            block=(bk, c1, c1),
            stream=stream,
        )

    return launch_fp8_gemm


__all__ = [
    "compile_fp8_gemm",
    "preshuffle_b_fp8",
    "fp8_quantize_per_token",
    "fp8_quantize_per_channel",
]
