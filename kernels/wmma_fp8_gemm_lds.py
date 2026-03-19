"""FP8 WMMA GEMM with LDS-buffered A for RDNA4 (gfx12xx).

Optimized for large-M compute-bound shapes (M >= 128, e.g., prefill / training).
For small-M decode shapes (M < 128), use wmma_fp8_gemm.py instead.

Architecture:
  A: raw [M,K] fp8 → coalesced GMEM load → LDS (ping-pong) → register → WMMA
  B: preshuffled [N0,K0,2,16,8] fp8 → GMEM direct → register → WMMA
  C: bf16 output with per-token/per-channel (rowwise) scaling

Design rationale:
  - A goes through LDS because activation tensors cannot be preshuffled at runtime.
    Coalesced GMEM→LDS load + LDS→register read is faster than strided GMEM→register.
  - B is preshuffled offline (weight matrix), loaded directly from GMEM into registers
    in WMMA-ready layout. No LDS needed for B.
  - LDS ping-pong double buffer hides A load latency: while computing from buffer 0,
    next A tile is loaded into buffer 1, then swap.
  - SCF loop carries only accumulators (16 x v8f32 = 128 VGPRs). A/B tiles are
    loaded fresh each iteration — no register rename overhead (zero v_dual_mov).

Tile parameters:
  BLOCK_M=128, BLOCK_N=128, BLOCK_K=32 (configurable via reg_m/reg_n/reg_k)
  4 waves (2x2), 128 threads per workgroup
  LDS: 12KB for A ping-pong (only A, not B)
  VGPR: ~170 (1 wave/SIMD occupancy)

Scaling:
  Per-token scale_a[M] for activation, per-channel scale_b[N] for weight.
  Compatible with torch._scaled_mm RowWise mode and vLLM/SGLang FP8 inference.

Performance (gfx1201, vs torch._scaled_mm):
  4096x4096x4096:  ~205 TFLOPS, ~90% of torch
  4096x4096x8192:  ~195 TFLOPS, ~105% of torch (exceeds torch)
  Best for M >= 2048 where compute dominates.
"""

from __future__ import annotations

import functools
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T
from flydsl.expr import arith, vector, gpu, buffer_ops, rocdl, range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
import flydsl.compiler as flyc
import flydsl.expr as fx

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16


def preshuffle_b_fp8(B_kn):
    """Preshuffle B[K,N] fp8 for WMMA B operand layout."""
    import torch

    K, N = B_kn.shape
    assert K % 16 == 0 and N % 16 == 0
    N0, K0 = N // 16, K // 16
    return B_kn.view(torch.uint8).reshape(K0, 2, 8, N0, 16).permute(3, 0, 1, 4, 2).contiguous()


def fp8_quantize_per_token(x_f32):
    """Quantize f32 tensor to fp8_e4m3fn with per-token (per-row) scale."""
    import torch

    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 448.0
    x_scaled = (x_f32 / scale).clamp(-448.0, 448.0)
    return x_scaled.to(torch.float8_e4m3fn), scale.squeeze(-1)


def fp8_quantize_per_channel(x_f32):
    """Quantize f32 tensor to fp8_e4m3fn with per-channel (per-column) scale."""
    import torch

    amax = x_f32.abs().amax(dim=0).clamp(min=1e-12)
    scale = amax / 448.0
    x_scaled = (x_f32 / scale.unsqueeze(0)).clamp(-448.0, 448.0)
    return x_scaled.to(torch.float8_e4m3fn), scale


@functools.lru_cache(maxsize=64)
def compile_fp8_gemm_lds(
    M: int,
    N: int,
    K: int,
    *,
    reg_m: int = 4,
    reg_n: int = 4,
    reg_k: int = 2,
    waves_m: int = 2,
    waves_n: int = 2,
    group_m: int = 8,
    a_k_pad: int = 16,
):
    """Compile FP8 GEMM with LDS-buffered A and preshuffled B.

    Args:
        M, N, K: matrix dimensions. Must be multiples of BLOCK_M/N/K.
        reg_m, reg_n: WMMA tile repeats per wave in M/N direction.
        reg_k: K-steps per tile (tile_k = reg_k * 16).
        waves_m, waves_n: wave grid within workgroup.
        group_m: L2 cache swizzle group size.
        a_k_pad: K-padding in LDS for A to avoid bank conflicts.

    Inputs:
        c: output bf16 [M, N]
        a: raw fp8 [M, K], viewed as float32 (A_fp8.view(torch.float32))
        b: preshuffled fp8, viewed as float32 (preshuffle_b_fp8(B).view(torch.float32))
        scale_a: per-token scale [M] (float32)
        scale_b: per-channel scale [N] (float32)
        stream: CUDA stream

    K-loop pipeline:
        Prologue: load A tile 0 to LDS buf0, barrier.
        Main loop (SCF, carry only accs):
            1. Issue async GMEM load for next A tile (coalesced, 128 threads)
            2. Load B tile from preshuffle GMEM (direct to register)
            3. Read current A from LDS, compute WMMAs with B registers
            4. Store next A to LDS write buffer
            5. Barrier, swap ping-pong buffers
        Epilogue: compute last tile, apply rowwise scales, store C.

    LDS layout:
        A[BLOCK_M, BLOCK_K + a_k_pad] x 2 buffers (ping-pong)
        fp8 = 1 byte/element, total ~12KB for default 128x48x2.
        K-padding (a_k_pad=16) avoids LDS bank conflicts on A reads.
    """
    gpu_arch = get_rocm_arch()

    BLOCK_M = WMMA_M * reg_m * waves_m
    BLOCK_N = WMMA_N * reg_n * waves_n
    BLOCK_K = WMMA_K * reg_k
    NUM_WAVES = waves_m * waves_n
    WAVE_SIZE = 32
    THREADS = NUM_WAVES * WAVE_SIZE

    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0
    num_k_tiles = K // BLOCK_K
    assert num_k_tiles >= 2
    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N

    wave_reg_m = reg_m
    wave_reg_n = reg_n

    # A LDS: fp8 = 1 byte/elem. Each thread loads 16 bytes (dwordx4).
    A_LDS_STRIDE = BLOCK_K + a_k_pad  # padded stride to avoid bank conflicts
    A_LDS_ONE = BLOCK_M * A_LDS_STRIDE  # one buffer in elements (fp8)
    A_LDS_TOTAL = 2 * A_LDS_ONE  # ping-pong double buffer

    A_LOAD_VEC = 16  # 16 fp8 elements = 4 dwords
    NUM_A_LOADS = (BLOCK_M * BLOCK_K) // (THREADS * A_LOAD_VEC)

    # B preshuffle strides
    K0_total = K // 16
    B_KPACK = 8
    B_STRIDE_NLANE = B_KPACK
    B_STRIDE_KLANE = 16 * B_KPACK
    B_STRIDE_K0 = 2 * 16 * B_KPACK
    B_STRIDE_N0 = K0_total * B_STRIDE_K0

    # LDS allocation
    elem_bytes = 1  # fp8
    allocator = SmemAllocator(None, arch=gpu_arch)
    lds_byte_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_byte_offset + A_LDS_TOTAL * elem_bytes

    @flyc.kernel
    def fp8_lds_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
    ):
        f32 = ir.F32Type.get()
        bf16 = ir.BF16Type.get()
        i32 = ir.IntegerType.get_signless(32)
        i8 = ir.IntegerType.get_signless(8)
        v8f32_ty = T.vec(8, T.f32)
        v2i32_ty = ir.VectorType.get([2], i32)
        v16i8_ty = ir.VectorType.get([16], i8)
        v4i32_ty = ir.VectorType.get([4], i32)

        lds_base = allocator.get_base()
        lds_view = SmemPtr(lds_base, lds_byte_offset, i8, shape=(A_LDS_TOTAL,)).get()

        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")

        c32 = arith.index(32)
        c16 = arith.index(16)
        c8 = arith.index(8)
        c4 = arith.index(4)
        c2 = arith.index(2)
        c_zero = arith.index(0)
        c_lds_buf_stride = arith.index(A_LDS_ONE)

        wave_id = tid // c32
        lane = tid % c32
        lane16 = lane % c16
        klane = lane // c16
        base8 = klane * c8

        # L2 swizzle
        effective_group_m = min(group_m, grid_m)
        c_grid_n = arith.index(grid_n)
        c_group_m = arith.index(effective_group_m)
        num_pid_in_group = c_group_m * c_grid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * c_group_m
        pid_in_group = pid % num_pid_in_group
        bid_m = first_pid_m + (pid_in_group % c_group_m)
        bid_n = pid_in_group // c_group_m

        c_wn = arith.index(waves_n)
        wave_m = wave_id // c_wn
        wave_n = wave_id % c_wn

        tile_m0 = bid_m * arith.index(BLOCK_M)
        tile_n0 = bid_n * arith.index(BLOCK_N)

        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=True)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

        # === Pre-compute A GMEM→LDS offsets ===
        a_lds_info = []
        for al in range_constexpr(NUM_A_LOADS):
            a_lin = tid * arith.index(A_LOAD_VEC) + arith.index(al * THREADS * A_LOAD_VEC)
            a_load_row = a_lin // arith.index(BLOCK_K)
            a_load_col = a_lin % arith.index(BLOCK_K)
            lds_rel = a_load_row * arith.index(A_LDS_STRIDE) + a_load_col
            g_row = tile_m0 + a_load_row
            a_lds_info.append((g_row, a_load_col, lds_rel))

        # === B preshuffle base ===
        n0_base = tile_n0 // c16 + wave_n * arith.index(wave_reg_n)

        # === GMEM load A tile → returns raw dwordx4 list ===
        def _gmem_load_a(k_base):
            raw = []
            for al in range_constexpr(NUM_A_LOADS):
                g_row, a_load_col, _ = a_lds_info[al]
                g_col = k_base + a_load_col
                # fp8: byte offset = row * K + col, dword offset = byte_off // 4
                byte_off = g_row * arith.index(K) + g_col
                dword_off = byte_off // c4
                a_raw = buffer_ops.buffer_load(a_rsrc, dword_off, vec_width=4, dtype=i32)
                raw.append(a_raw)
            return raw

        # === Store A to LDS ===
        def _lds_store_a(raw_data, buf_offset):
            for al in range_constexpr(NUM_A_LOADS):
                _, _, lds_rel = a_lds_info[al]
                a_vec = vector.bitcast(v16i8_ty, raw_data[al])
                lds_idx = buf_offset + lds_rel
                vector.store(a_vec, lds_view, [lds_idx])

        # === Load A from LDS for WMMA ===
        def _load_a_from_lds(rk, buf_offset):
            """Load A WMMA operands from LDS. Returns [reg_m] of v2i32."""
            vecs = []
            col_base = arith.index(rk * WMMA_K) + base8
            for rm in range_constexpr(wave_reg_m):
                row = wave_m * arith.index(wave_reg_m * WMMA_M) + arith.index(rm * WMMA_M) + lane16
                lds_idx = buf_offset + row * arith.index(A_LDS_STRIDE) + col_base
                # Load 8 fp8 bytes = 2 x i32
                a_raw = vector.load_op(ir.VectorType.get([8], i8), lds_view, [lds_idx])
                a_v2i32 = vector.bitcast(v2i32_ty, a_raw)
                vecs.append(a_v2i32)
            return vecs

        # === Load B from preshuffle GMEM ===
        def _load_b_tile(k_tile_idx):
            """Load B fp8 tile from preshuffled GMEM. Returns [reg_k][wave_reg_n] of v2i32."""
            b_vecs = []
            for rk in range_constexpr(reg_k):
                rk_vecs = []
                k0 = k_tile_idx * arith.index(reg_k) + arith.index(rk)
                for rn in range_constexpr(wave_reg_n):
                    n0 = n0_base + arith.index(rn)
                    byte_off = (
                        n0 * arith.index(B_STRIDE_N0)
                        + k0 * arith.index(B_STRIDE_K0)
                        + klane * arith.index(B_STRIDE_KLANE)
                        + lane16 * arith.index(B_STRIDE_NLANE)
                    )
                    dword_off = byte_off // c4
                    b_raw = buffer_ops.buffer_load(b_rsrc, dword_off, vec_width=2, dtype=i32)
                    rk_vecs.append(b_raw)
                b_vecs.append(rk_vecs)
            return b_vecs

        def _barrier():
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_wait_dscnt 0x0\ns_wait_storecnt 0x0\ns_barrier_signal -1\ns_barrier_wait -1",
                constraints="",
                has_side_effects=True,
            )

        # === Compute one K-step from LDS A + GMEM B ===
        def _do_compute_rk(accs_in, rk, buf_offset, b_vecs_rk):
            new_accs = list(accs_in)
            a_vecs = _load_a_from_lds(rk, buf_offset)
            for rm in range_constexpr(wave_reg_m):
                for rn in range_constexpr(wave_reg_n):
                    idx = rm * wave_reg_n + rn
                    new_accs[idx] = rocdl.wmma_f32_16x16x16_fp8_fp8(
                        v8f32_ty, a_vecs[rm], b_vecs_rk[rn], arith.unwrap(new_accs[idx])
                    ).result
            return new_accs

        # === Initialize accumulators ===
        zero_acc = arith.constant_vector(0.0, v8f32_ty)
        accs = [zero_acc for _ in range_constexpr(wave_reg_m * wave_reg_n)]

        # === Prologue: load first A tile to LDS buf0 ===
        prologue_a = _gmem_load_a(c_zero)
        _lds_store_a(prologue_a, c_zero)
        _barrier()

        # === Main K-loop: SCF, carry only accs ===
        n_acc = wave_reg_m * wave_reg_n

        for iv, state in range(0, num_k_tiles - 1, 1, init=list(accs)):
            s_accs = list(state[:n_acc])

            read_off = iv % c2 * c_lds_buf_stride
            write_off = (arith.index(1) - iv % c2) * c_lds_buf_stride

            # 1. Issue GMEM load for next A tile (async, non-blocking)
            next_k = (iv + arith.index(1)) * arith.index(BLOCK_K)
            next_a_data = _gmem_load_a(next_k)

            # 2. Load B tile from preshuffle GMEM (async)
            b_tile = _load_b_tile(iv)

            # 3. Compute from current LDS A + GMEM B
            for rk in range_constexpr(reg_k):
                s_accs = _do_compute_rk(s_accs, rk, read_off, b_tile[rk])

            # 4. Store next A to write buffer
            _lds_store_a(next_a_data, write_off)

            # 5. Barrier
            _barrier()

            results = yield list(s_accs)

        accs = list(results[:n_acc])

        # === Epilogue: last K-tile ===
        last_read_off = arith.index((num_k_tiles - 1) % 2) * c_lds_buf_stride
        b_tile_last = _load_b_tile(arith.index(num_k_tiles - 1))
        for rk in range_constexpr(reg_k):
            accs = _do_compute_rk(accs, rk, last_read_off, b_tile_last[rk])

        # === Store results with rowwise scaling ===
        c_n = arith.index(N)

        sb_cache = []
        for rn in range_constexpr(wave_reg_n):
            g_col = tile_n0 + wave_n * arith.index(wave_reg_n * WMMA_N) + arith.index(rn * WMMA_N) + lane16
            sb_cache.append(buffer_ops.buffer_load(scale_b_rsrc, g_col, vec_width=1, dtype=f32))

        for rm in range_constexpr(wave_reg_m):
            m_base = tile_m0 + wave_m * arith.index(wave_reg_m * WMMA_M) + arith.index(rm * WMMA_M) + base8
            sa_cache = []
            row_off_cache = []
            for si in range_constexpr(8):
                g_row = m_base + arith.index(si)
                sa_cache.append(buffer_ops.buffer_load(scale_a_rsrc, g_row, vec_width=1, dtype=f32))
                row_off_cache.append(g_row * c_n)

            for rn in range_constexpr(wave_reg_n):
                idx = rm * wave_reg_n + rn
                g_col = tile_n0 + wave_n * arith.index(wave_reg_n * WMMA_N) + arith.index(rn * WMMA_N) + lane16
                sb_val = sb_cache[rn]
                for si in range_constexpr(8):
                    val = vector.extract(accs[idx], static_position=[si], dynamic_position=[])
                    val = val * sa_cache[si] * sb_val
                    val_bf16 = arith.trunc_f(bf16, val)
                    buffer_ops.buffer_store(val_bf16, c_rsrc, row_off_cache[si] + g_col)

    @flyc.jit
    def launch_fp8_gemm_lds(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        stream: fx.Stream,
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        total_blocks = grid_m * grid_n
        launcher = fp8_lds_kernel(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b)
        launcher.launch(
            grid=(total_blocks, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch_fp8_gemm_lds


__all__ = [
    "compile_fp8_gemm_lds",
    "preshuffle_b_fp8",
    "fp8_quantize_per_token",
    "fp8_quantize_per_channel",
]
