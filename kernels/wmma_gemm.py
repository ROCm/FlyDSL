#!/usr/bin/env python3
"""Optimized WMMA GEMM kernel for RDNA4 (gfx12xx, wave32).

Computes C[M,N] = A[M,K] @ B[K,N] using v_wmma_f32_16x16x16_{f16,bf16}.
Supports f16/bf16 inputs with f32 accumulation, output in f32 or bf16.

Optimizations applied:
  - Configurable block tile with K-unroll (REG_K WMMA-K steps per tile)
  - Register tiling: each wave computes REG_M x REG_N WMMA tiles per K step
  - 4 waves (128 threads) per workgroup
  - L2 cache swizzle: grouped block scheduling for better L2 reuse
  - Inline asm batched global loads (s_clause 7 + 8x global_load_b128)
    to prevent LLVM VGPR reuse serialization
  - SW pipelining: next tile global loads overlap with current tile WMMA compute
  - Vectorized LDS loads for A operand (contiguous in K dim)
  - LDS B padding to reduce bank conflicts on scalar B reads
  - Dynamic K loop (scf.for via auto-lowered range()) for O(1) IR size

WMMA data layout for wave32 (verified on gfx1201):
  A operand: "row-of-cols" -- lane t loads A[t%16][(t/16)*8 + i], i=0..7
  B operand: "col-of-rows" -- lane t loads B[(t/16)*8 + i][t%16], i=0..7
  D result:  "col-of-rows" -- lane t holds D[(t/16)*8 + i][t%16], i=0..7

LDS layout:
  A: row-major [BLOCK_M][BLOCK_K] -- A[m][k], contiguous in K
  B: row-major [BLOCK_K][B_LDS_STRIDE] -- B[k][n+pad], padded stride
"""

from flydsl.dialects.ext import flir, arith, memref, vector, rocdl, gpu
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.lang.ir.types import T
from flydsl.utils import SmemAllocator
from _mlir import ir
from _mlir.dialects import llvm as _llvm
from _mlir.dialects import arith as _std_arith
from _mlir.dialects import memref as _std_memref
import _mlir.extras.types as Textra


# =============================================================================
# Kernel configuration
# =============================================================================

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16

# Register tiling: each wave handles REG_M x REG_N WMMA tiles
REG_M = 4  # 4 WMMA tiles vertically per wave
REG_N = 4  # 4 WMMA tiles horizontally per wave

# K-unroll: multiple WMMA-K steps per tile load
REG_K = 2  # 2 WMMA-K steps per K tile

# Waves per workgroup arranged as WAVES_M x WAVES_N
WAVES_M = 2
WAVES_N = 2
NUM_WAVES = WAVES_M * WAVES_N  # 4

# Derived block tile dimensions
BLOCK_M = WMMA_M * REG_M * WAVES_M  # 16*4*2 = 128
BLOCK_N = WMMA_N * REG_N * WAVES_N  # 16*4*2 = 128
BLOCK_K = WMMA_K * REG_K  # 16*2 = 32

THREADS_PER_BLOCK = NUM_WAVES * 32  # 128

# Elements per thread for cooperative global->LDS load
A_TILE_ELEMS = BLOCK_M * BLOCK_K
B_TILE_ELEMS = BLOCK_K * BLOCK_N
NUM_A_LOADS = A_TILE_ELEMS // (THREADS_PER_BLOCK * 8)
NUM_B_LOADS = B_TILE_ELEMS // (THREADS_PER_BLOCK * 8)

# LDS padding for B to reduce bank conflicts
B_PAD = 8
B_LDS_STRIDE = BLOCK_N + B_PAD

# LDS sizes (single buffer)
LDS_A_ELEMS = BLOCK_M * BLOCK_K
LDS_B_ELEMS = BLOCK_K * B_LDS_STRIDE

# L2 cache swizzle group size
GROUP_M = 8

# Number of inline-asm batched global loads (A loads + B loads)
TOTAL_GLOBAL_LOADS = NUM_A_LOADS + NUM_B_LOADS  # 4 + 4 = 8


def _unwrap(v):
    """Unwrap ArithValue to raw MLIR Value."""
    while hasattr(v, "_value"):
        v = v._value
    return v


def create_wmma_gemm_module(M: int, N: int, K: int, in_dtype="bf16", out_dtype="f32"):
    """Create an optimized WMMA GEMM module.

    Args:
        M, N, K: matrix dimensions (must be multiples of BLOCK_M/N/K)
        in_dtype: "bf16" or "f16"
        out_dtype: "f32" or "bf16"
    """
    gpu_arch = get_rocm_arch()
    S = ir.ShapedType.get_dynamic_size()

    assert M % BLOCK_M == 0, f"M={M} must be multiple of BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} must be multiple of BLOCK_N={BLOCK_N}"
    assert K % BLOCK_K == 0, f"K={K} must be multiple of BLOCK_K={BLOCK_K}"

    num_k_tiles = K // BLOCK_K
    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N
    is_bf16 = in_dtype == "bf16"

    def _in_elem_ty():
        return Textra.bf16() if is_bf16 else Textra.f16()

    def _out_elem_ty():
        return Textra.f32() if out_dtype == "f32" else Textra.bf16()

    def _wmma_op(result_type, a_vec, b_vec, acc, v8i16_ty):
        """Execute the correct WMMA op based on dtype, with bf16->i16 bitcast."""
        if is_bf16:
            a_i16 = vector.bitcast(v8i16_ty, a_vec)
            b_i16 = vector.bitcast(v8i16_ty, b_vec)
            return rocdl.wmma_f32_16x16x16_bf16(
                result_type,
                [a_i16, b_i16, arith.unwrap(acc)],
            )
        else:
            return rocdl.wmma_f32_16x16x16_f16(
                result_type,
                [arith.unwrap(a_vec), arith.unwrap(b_vec), arith.unwrap(acc)],
            )

    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    class _WmmaGemm(flir.MlirModule):
        GPU_MODULE_NAME = "wmma_gemm"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}">']

        def init_gpu_module(self):
            _state["s_a"] = allocator.allocate_array(_in_elem_ty(), LDS_A_ELEMS)
            _state["s_b"] = allocator.allocate_array(_in_elem_ty(), LDS_B_ELEMS)
            allocator.finalize()

        @flir.kernel
        def wmma_gemm_kernel(
            self: flir.T.i64,
            A: lambda: Textra.memref(S, S, _in_elem_ty()),
            B: lambda: Textra.memref(S, S, _in_elem_ty()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            # ---- Types ----
            in_ir_ty = ir.BF16Type.get() if is_bf16 else ir.F16Type.get()
            v8_in_ty = ir.VectorType.get([8], in_ir_ty)
            v8f32_ty = T.vec(8, T.f32)
            i16_ty = ir.IntegerType.get_signless(16)
            v8i16_ty = ir.VectorType.get([8], i16_ty)
            i64_ty = ir.IntegerType.get_signless(64)
            ptr_ty = ir.Type.parse("!llvm.ptr")
            v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
            struct_ty = _llvm.StructType.get_literal([v4i32_ty] * TOTAL_GLOBAL_LOADS)

            # ---- Thread / block IDs ----
            tid = flir.thread_idx("x")
            pid = flir.block_idx("x")  # linear program id

            c32 = arith.index(32)
            c16 = arith.index(16)
            c8 = arith.index(8)
            wave_id = tid // c32
            lane = tid % c32
            lane16 = lane % c16
            base8 = (lane // c16) * c8

            # ---- L2 cache swizzle (grouped block scheduling) ----
            effective_group_m = min(GROUP_M, grid_m)
            c_grid_n = arith.index(grid_n)
            c_group_m = arith.index(effective_group_m)
            num_pid_in_group = c_group_m * c_grid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * c_group_m
            group_size_m = c_group_m

            pid_in_group = pid % num_pid_in_group
            bid_m = first_pid_m + (pid_in_group % group_size_m)
            bid_n = pid_in_group // group_size_m

            # Wave position in the WAVES_M x WAVES_N grid
            c_wn = arith.index(WAVES_N)
            wave_m = wave_id // c_wn
            wave_n = wave_id % c_wn

            # Global tile origins
            tile_m0 = bid_m * arith.index(BLOCK_M)
            tile_n0 = bid_n * arith.index(BLOCK_N)

            # ---- LDS setup ----
            lds_base = allocator.get_base()
            As = _state["s_a"](lds_base)
            Bs = _state["s_b"](lds_base)
            lds_a_view = As.get()
            lds_b_view = Bs.get()

            # ---- Extract base pointers for inline asm loads ----
            elem_bytes = 2  # bf16 or f16 = 2 bytes
            a_base_i64 = _unwrap(
                _std_arith.IndexCastOp(
                    i64_ty,
                    _unwrap(
                        _std_memref.ExtractAlignedPointerAsIndexOp(_unwrap(A)).result
                    ),
                ).result
            )
            b_base_i64 = _unwrap(
                _std_arith.IndexCastOp(
                    i64_ty,
                    _unwrap(
                        _std_memref.ExtractAlignedPointerAsIndexOp(_unwrap(B)).result
                    ),
                ).result
            )

            # ---- Pre-compute thread-local LDS store addresses (invariant) ----
            a_lds_addrs = []
            b_lds_addrs = []
            for al in range_constexpr(NUM_A_LOADS):
                a_lin = tid * c8 + arith.index(al * THREADS_PER_BLOCK * 8)
                a_load_row = a_lin // arith.index(BLOCK_K)
                a_load_col = a_lin % arith.index(BLOCK_K)
                a_lds_addrs.append(a_load_row * arith.index(BLOCK_K) + a_load_col)

            for bl in range_constexpr(NUM_B_LOADS):
                b_lin = tid * c8 + arith.index(bl * THREADS_PER_BLOCK * 8)
                b_load_row = b_lin // arith.index(BLOCK_N)
                b_load_col = b_lin % arith.index(BLOCK_N)
                b_lds_addrs.append(b_load_row * arith.index(B_LDS_STRIDE) + b_load_col)

            # ---- Build inline asm strings (compile-time constants) ----
            # Load asm: s_clause + 8x global_load_b128, NO wait
            asm_load_lines = [f"s_clause {TOTAL_GLOBAL_LOADS - 1}"]
            for i in range_constexpr(TOTAL_GLOBAL_LOADS):
                asm_load_lines.append(
                    f"global_load_b128 ${i}, ${i + TOTAL_GLOBAL_LOADS}, off"
                )
            asm_load_str = "\n".join(asm_load_lines)

            out_constraints = ",".join(["=&v"] * TOTAL_GLOBAL_LOADS)
            in_constraints = ",".join(["v"] * TOTAL_GLOBAL_LOADS)
            asm_constraints = f"{out_constraints},{in_constraints}"

            # ---- Helper: compute flat addresses for a given k_base ----
            def _compute_load_addrs(k_base):
                """Compute 8 flat pointers for global loads at given k_base."""
                all_addrs = []
                for al in range_constexpr(NUM_A_LOADS):
                    a_lin = tid * c8 + arith.index(al * THREADS_PER_BLOCK * 8)
                    a_load_row = a_lin // arith.index(BLOCK_K)
                    a_load_col = a_lin % arith.index(BLOCK_K)
                    g_a_row = tile_m0 + a_load_row
                    g_a_col = k_base + a_load_col
                    byte_off = (g_a_row * arith.index(K) + g_a_col) * arith.index(
                        elem_bytes
                    )
                    byte_off_i64 = _unwrap(
                        _std_arith.IndexCastOp(
                            i64_ty, _unwrap(arith.unwrap(byte_off))
                        ).result
                    )
                    addr_i64 = _unwrap(
                        _std_arith.AddIOp(a_base_i64, byte_off_i64).result
                    )
                    addr_ptr = _unwrap(_llvm.IntToPtrOp(ptr_ty, addr_i64).result)
                    all_addrs.append(addr_ptr)

                for bl in range_constexpr(NUM_B_LOADS):
                    b_lin = tid * c8 + arith.index(bl * THREADS_PER_BLOCK * 8)
                    b_load_row = b_lin // arith.index(BLOCK_N)
                    b_load_col = b_lin % arith.index(BLOCK_N)
                    g_b_row = k_base + b_load_row
                    g_b_col = tile_n0 + b_load_col
                    byte_off = (g_b_row * arith.index(N) + g_b_col) * arith.index(
                        elem_bytes
                    )
                    byte_off_i64 = _unwrap(
                        _std_arith.IndexCastOp(
                            i64_ty, _unwrap(arith.unwrap(byte_off))
                        ).result
                    )
                    addr_i64 = _unwrap(
                        _std_arith.AddIOp(b_base_i64, byte_off_i64).result
                    )
                    addr_ptr = _unwrap(_llvm.IntToPtrOp(ptr_ty, addr_i64).result)
                    all_addrs.append(addr_ptr)
                return all_addrs

            # ---- Helper: issue batched loads (no wait) ----
            def _issue_loads(all_addrs):
                """Issue 8 batched global_load_b128 via inline asm. Returns struct."""
                return _llvm.inline_asm(
                    struct_ty,
                    all_addrs,
                    asm_load_str,
                    asm_constraints,
                    has_side_effects=True,
                )

            # ---- Helper: extract + bitcast load results ----
            def _extract_load_results(asm_result):
                """Extract 8 load results from asm struct, bitcast to bf16/f16."""
                all_vecs = []
                for i in range_constexpr(TOTAL_GLOBAL_LOADS):
                    pos_attr = ir.DenseI64ArrayAttr.get([i])
                    v4i32_val = _llvm.ExtractValueOp(
                        v4i32_ty, asm_result, pos_attr
                    ).result
                    bf16_vec = vector.bitcast(v8_in_ty, v4i32_val)
                    all_vecs.append(bf16_vec)
                return all_vecs[:NUM_A_LOADS], all_vecs[NUM_A_LOADS:]

            # ---- Helper: store load results to LDS ----
            def _store_to_lds(a_vecs, b_vecs):
                """Store A and B vectors to LDS."""
                for al in range_constexpr(NUM_A_LOADS):
                    vector.store(a_vecs[al], lds_a_view, [a_lds_addrs[al]])
                for bl in range_constexpr(NUM_B_LOADS):
                    vector.store(b_vecs[bl], lds_b_view, [b_lds_addrs[bl]])

            # ---- Inline asm infrastructure for batched B + A LDS reads ----
            # B operand: 4 packed i32 (8 bf16) per WMMA tile, 16 i32 per K-step
            # A operand: v4i32 (8 bf16) per WMMA tile, REG_M=4 tiles per K-step
            NUM_B_VGPRS_PER_K = REG_N * 4  # 16
            b_lds_stride_bytes = B_LDS_STRIDE * 2  # 272 bytes

            i32_ty = ir.IntegerType.get_signless(32)
            n = NUM_B_VGPRS_PER_K  # 16

            # B immediate offsets: even (u16) and odd (d16_hi) for each (rn, pi)
            even_offsets = []
            odd_offsets = []
            for rn in range_constexpr(REG_N):
                for pi in range_constexpr(4):
                    even_offsets.append((2 * pi * B_LDS_STRIDE + rn * WMMA_N) * 2)
                    odd_offsets.append(((2 * pi + 1) * B_LDS_STRIDE + rn * WMMA_N) * 2)

            # A immediate offsets for each rm
            a_rk_offsets = []
            for rm in range_constexpr(REG_M):
                a_rk_offsets.append(rm * WMMA_M * BLOCK_K * 2)

            # ---- Combined asm for rk=0: A + B loads with waits ----
            # Outputs: $0..$15 = 16 B i32, $16..$19 = 4 A v4i32
            # Inputs: $20 = B base (v), $21 = A base (v)
            NUM_COMBINED_OUTPUTS = n + REG_M  # 20
            combined_out_types = [i32_ty] * n + [v4i32_ty] * REG_M
            combined_struct_ty = _llvm.StructType.get_literal(combined_out_types)

            combined_lines = []
            b_base_idx = NUM_COMBINED_OUTPUTS  # $20
            a_base_idx = NUM_COMBINED_OUTPUTS + 1  # $21
            for i in range_constexpr(n):
                combined_lines.append(
                    f"ds_load_u16 ${i}, ${b_base_idx} offset:{even_offsets[i]}"
                )
            for i in range_constexpr(REG_M):
                combined_lines.append(
                    f"ds_load_b128 ${n + i}, ${a_base_idx} offset:{a_rk_offsets[i]}"
                )
            combined_lines.append("s_wait_dscnt 0x4")
            for i in range_constexpr(n):
                combined_lines.append(
                    f"ds_load_u16_d16_hi ${i}, ${b_base_idx} offset:{odd_offsets[i]}"
                )
            combined_lines.append("s_wait_dscnt 0x0")

            asm_combined_str = "\n".join(combined_lines)
            asm_combined_constraints = ",".join(["=&v"] * NUM_COMBINED_OUTPUTS) + ",v,v"

            # ---- Helpers: compute B/A base address and extract results ----
            def _compute_b_base(k_off):
                """Compute single VGPR base address for B LDS reads."""
                b_alloc_byte_off = _unwrap(
                    _std_arith.IndexCastOp(
                        i32_ty,
                        _unwrap(
                            _std_memref.ExtractAlignedPointerAsIndexOp(
                                _unwrap(lds_b_view)
                            ).result
                        ),
                    ).result
                )
                k_row = k_off + base8
                row_byte_off = k_row * arith.index(b_lds_stride_bytes)
                col_idx = wave_n * arith.index(REG_N * WMMA_N) + lane16
                col_byte_off = col_idx * arith.index(2)
                total_off = row_byte_off + col_byte_off
                total_off_i32 = _unwrap(
                    _std_arith.IndexCastOp(
                        i32_ty, _unwrap(arith.unwrap(total_off))
                    ).result
                )
                return _unwrap(
                    _std_arith.AddIOp(b_alloc_byte_off, total_off_i32).result
                )

            def _compute_a_base(k_off):
                """Compute single VGPR base address for A LDS reads."""
                a_alloc_byte_off = _unwrap(
                    _std_arith.IndexCastOp(
                        i32_ty,
                        _unwrap(
                            _std_memref.ExtractAlignedPointerAsIndexOp(
                                _unwrap(lds_a_view)
                            ).result
                        ),
                    ).result
                )
                a_row = wave_m * arith.index(REG_M * WMMA_M) + lane16
                a_off = (a_row * arith.index(BLOCK_K) + k_off + base8) * arith.index(2)
                a_off_i32 = _unwrap(
                    _std_arith.IndexCastOp(i32_ty, _unwrap(arith.unwrap(a_off))).result
                )
                return _unwrap(_std_arith.AddIOp(a_alloc_byte_off, a_off_i32).result)

            def _do_wmma_block(a_vecs, b_vecs, accs):
                """Execute REG_M * REG_N WMMAs for one K-step."""
                for rm in range_constexpr(REG_M):
                    for rn in range_constexpr(REG_N):
                        idx = rm * REG_N + rn
                        accs[idx] = _wmma_op(
                            v8f32_ty,
                            a_vecs[rm],
                            b_vecs[rn],
                            accs[idx],
                            v8i16_ty,
                        )

            # ---- Helper: WMMA compute phase (reads from LDS) ----
            def _do_compute(accs_in):
                """Execute WMMA compute for all K-steps.

                For each K-step, issues a single combined inline asm block:
                  16x ds_load_u16 (B low halves)
                  4x ds_load_b128 (A tiles)
                  s_wait_dscnt 0x4  (B_u16 done, A still in flight)
                  16x ds_load_u16_d16_hi (B high halves)
                  s_wait_dscnt 0x0  (everything done)
                Then executes REG_M * REG_N WMMAs.
                """
                new_accs = list(accs_in)

                for rk in range_constexpr(REG_K):
                    k_off = arith.index(rk * WMMA_K)
                    b_base = _compute_b_base(k_off)
                    a_base = _compute_a_base(k_off)

                    result = _llvm.inline_asm(
                        combined_struct_ty,
                        [b_base, a_base],
                        asm_combined_str,
                        asm_combined_constraints,
                        has_side_effects=True,
                    )

                    # Extract B: 16 i32 -> pack into 4 v4i32 -> bitcast to 4 v8bf16
                    b_vecs = []
                    for rn in range_constexpr(REG_N):
                        v4 = _unwrap(_llvm.UndefOp(v4i32_ty).result)
                        for pi in range_constexpr(4):
                            idx = rn * 4 + pi
                            pos_attr = ir.DenseI64ArrayAttr.get([idx])
                            val = _unwrap(
                                _llvm.ExtractValueOp(i32_ty, result, pos_attr).result
                            )
                            pi_val = _unwrap(
                                _std_arith.ConstantOp(
                                    i32_ty, ir.IntegerAttr.get(i32_ty, pi)
                                ).result
                            )
                            v4 = _unwrap(_llvm.InsertElementOp(v4, val, pi_val).result)
                        b_vecs.append(vector.bitcast(v8_in_ty, v4))

                    # Extract A: 4 v4i32 -> bitcast to 4 v8bf16
                    a_vecs = []
                    for rm in range_constexpr(REG_M):
                        pos_attr = ir.DenseI64ArrayAttr.get([n + rm])
                        v4 = _llvm.ExtractValueOp(v4i32_ty, result, pos_attr).result
                        a_vecs.append(vector.bitcast(v8_in_ty, v4))

                    _do_wmma_block(a_vecs, b_vecs, new_accs)

                return new_accs

            # ---- Initialize REG_M x REG_N accumulators (flat list) ----
            zero_acc = arith.constant_vector(0.0, v8f32_ty)
            accs = [zero_acc for _ in range_constexpr(REG_M * REG_N)]

            # ========================================================
            # SW-PIPELINED K LOOP
            # ========================================================

            # ---- Prologue: load tile 0, wait, store to LDS, barrier ----
            prologue_addrs = _compute_load_addrs(arith.index(0))
            prologue_result = _issue_loads(prologue_addrs)
            # Wait for prologue loads
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_wait_loadcnt 0x0",
                constraints="",
                has_side_effects=True,
            )
            a_vecs_p, b_vecs_p = _extract_load_results(prologue_result)
            _store_to_lds(a_vecs_p, b_vecs_p)
            gpu.barrier()

            if num_k_tiles == 1:
                # Single tile: just compute
                accs = _do_compute(accs)
            else:
                # ---- Main loop: tiles 0..num_k_tiles-2 ----
                for kt in range(num_k_tiles - 1):
                    # k_base for the NEXT tile's load
                    k_base_next = (kt + arith.index(1)) * arith.index(BLOCK_K)

                    # Issue loads for next tile (NO WAIT - overlap with compute)
                    next_addrs = _compute_load_addrs(k_base_next)
                    next_result = _issue_loads(next_addrs)

                    # Compute on current tile (from LDS)
                    accs = _do_compute(accs)

                    # Barrier: ensure all waves done reading LDS
                    gpu.barrier()

                    # Wait for next tile's loads to complete
                    _llvm.inline_asm(
                        res=None,
                        operands_=[],
                        asm_string="s_wait_loadcnt 0x0",
                        constraints="",
                        has_side_effects=True,
                    )

                    # Store next tile to LDS
                    a_vecs_n, b_vecs_n = _extract_load_results(next_result)
                    _store_to_lds(a_vecs_n, b_vecs_n)

                    # Barrier: ensure all waves done writing LDS
                    gpu.barrier()

                # ---- Epilogue: compute on last tile ----
                accs = _do_compute(accs)

            # ========== Store results (scalar stores) ==========
            for rm in range_constexpr(REG_M):
                for rn in range_constexpr(REG_N):
                    idx = rm * REG_N + rn
                    wmma_m_off = wave_m * arith.index(REG_M * WMMA_M) + arith.index(
                        rm * WMMA_M
                    )
                    wmma_n_off = wave_n * arith.index(REG_N * WMMA_N) + arith.index(
                        rn * WMMA_N
                    )

                    # D layout: col-of-rows => D[base8+i][lane16]
                    for si in range_constexpr(8):
                        g_row = tile_m0 + wmma_m_off + base8 + arith.index(si)
                        g_col = tile_n0 + wmma_n_off + lane16
                        val = vector.extract(
                            accs[idx],
                            static_position=[si],
                            dynamic_position=[],
                        )
                        if out_dtype == "bf16":
                            val = arith.trunc_f(ir.BF16Type.get(), val)
                        memref.store(val, C, [g_row, g_col])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: Textra.memref(S, S, _in_elem_ty()),
            B: lambda: Textra.memref(S, S, _in_elem_ty()),
            C: lambda: Textra.memref(S, S, _out_elem_ty()),
        ):
            c1 = arith.index(1)
            total_blocks = arith.index(grid_m * grid_n)
            bk = arith.index(THREADS_PER_BLOCK)
            flir.gpu_ext.LaunchFuncOp(
                ["wmma_gemm", "wmma_gemm_kernel"],
                grid_size=(total_blocks, c1, c1),
                block_size=(bk, c1, c1),
                kernel_operands=[A, B, C],
            )

    return _WmmaGemm()
