"""Softmax kernel builder used by tests.

This file intentionally keeps the kernel builder logic identical to the version
previously embedded in `tests/python/gpu/test_softmax.py` to preserve codegen and
performance. Only test-only helpers/imports are removed.
"""

import os

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import rocir
from . import reduce as reduce_utils
from rocdsl.runtime.hip_util import get_hip_arch
from rocdsl.utils import SmemAllocator
from _mlir import ir
import _mlir.extras.types as T


KERNEL_NAME = "softmax_kernel"


def unwrap(v):
    if hasattr(v, "value"):
        return v.value
    if hasattr(v, "_value"):
        return v._value
    if hasattr(v, "result"):
        return v.result
    return v


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


# Expose modules through Rocir interface (keep behavior/perf, avoid mlir.* imports).
gpu = rocir.gpu_ext          # extended wrapper (has set_container_module, etc.)
scf = rocir.scf_ext          # extended wrapper (yield_ helper, etc.)
arith = rocir.arith_ext      # extended wrapper (arith.constant(...), ArithValue, etc.)
memref = rocir.memref        # raw dialect module
vector = rocir.vector        # raw dialect module
mlir_math = rocir.math       # raw dialect module
llvm = rocir.llvm            # raw dialect module


def build_softmax_module(M, N, dtype_str="f32"):
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    gpu_arch = get_hip_arch()

    # Types
    if dtype_str == "f32":
        elem_type = T.f32()
    elif dtype_str == "f16":
        elem_type = T.f16()
    elif dtype_str == "bf16":
        elem_type = ir.BF16Type.get()
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    # For bf16, avoid repeated bf16<->f32 "unpack/align" sequences in ISA by doing all
    # math in f32 and only converting at load/store boundaries.
    compute_type = T.f32() if dtype_str == "bf16" else elem_type

    f32 = T.f32()  # still used for comparisons in tests
    idx = T.index()

    # Kernel Config
    # Adaptive Block Size
    BLOCK_SIZE = min(256, next_power_of_2(N))
    if BLOCK_SIZE < 32:
        BLOCK_SIZE = 32  # Min block size for warp ops

    # Vector Width
    # Use 8 for aligned, small vectors for unaligned? To keep simple in Python gen, we use 8 and fallback to scalar for tail.
    VEC_WIDTH = 8

    # Nontemporal (cache-bypass) hints for global vector loads/stores.
    # MLIR vector.load/store support `nontemporal` as a BoolAttr. We apply it only on the aligned vector path.
    USE_NONTEMPORAL = True
    # Conservative alignment hint (bytes) for vector load/store.
    # For f16/bf16 and VEC_WIDTH=8 -> 16B; for f32 -> 32B. Use 16B universally.
    VEC_ALIGN = 16

    # Arch-dependent BF16 pack:
    # - gfx950 (and gfx95x): hardware supports v_cvt_pk_bf16_f32, so no manual pack needed.
    # - gfx942: does not support v_cvt_pk_bf16_f32; keep manual pack.
    USE_HW_CVT_PK_BF16_F32 = (gpu_arch == "gfx950") or gpu_arch.startswith("gfx95")

    # NOTE: Remaining bf16 "unpack/align" ops (e.g. 0xffff0000) mainly come from
    # unpacking packed bf16 values in 16B global vector loads. In this pipeline, scalarizing the loads tends to be re-vectorized back to the same pattern.
    BF16_SCALARIZE_VEC_LOAD = False

    # Allocator for Shared Memory (Warp Reductions)
    allocator = SmemAllocator(ctx, arch=gpu_arch)
    # Reduction scratch: one slot per wave (lane0 writes partials) + reuse slot 0 for broadcast.
    WARP_SIZE = 64
    RED_SLOTS = max(1, (BLOCK_SIZE + WARP_SIZE - 1) // WARP_SIZE)
    smem_red = allocator.allocate_array(compute_type, RED_SLOTS)

    @gpu.module(f"softmax_{dtype_str}", [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">'])
    def gpu_mod():
        allocator.finalize()

        @gpu.func(emit=True)
        def softmax_kernel(A: T.memref(M, N, elem_type), C: T.memref(M, N, elem_type)):
            row = rocir.block_idx("x")
            tid = rocir.thread_idx("x")

            base_ptr = allocator.get_base()
            s_red = smem_red(base_ptr).get()
            # Rocir-style: tensor views + tiled copies (like elementwise_add_kernel).
            c0_idx = rocir.const_index(0)
            tile_cols = BLOCK_SIZE * VEC_WIDTH  # python int
            tensor_A = rocir.make_tensor(A, shape=(M, N), strides=(N, 1))
            tensor_C = rocir.make_tensor(C, shape=(M, N), strides=(N, 1))
            s_red_tv = rocir.make_tensor(s_red, shape=(RED_SLOTS,), strides=(1,))
            gA = rocir.zipped_divide(tensor_A, (1, tile_cols))
            gC = rocir.zipped_divide(tensor_C, (1, tile_cols))

            thr_layout = rocir.make_ordered_layout((1, BLOCK_SIZE), order=(1, 0))
            val_layout = rocir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0))
            copy_atom_load = rocir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            copy_atom_store = rocir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tiled_copy_A = rocir.make_tiled_copy_tv(
                copy_atom_load,
                thr_layout,
                val_layout,
                thr_shape=(1, BLOCK_SIZE),
                val_shape=(1, VEC_WIDTH),
            )
            tiled_copy_C = rocir.make_tiled_copy_tv(
                copy_atom_store,
                thr_layout,
                val_layout,
                thr_shape=(1, BLOCK_SIZE),
                val_shape=(1, VEC_WIDTH),
            )
            thr_copy_A = tiled_copy_A.get_slice(unwrap(tid))
            thr_copy_C = tiled_copy_C.get_slice(unwrap(tid))

            # Element-type constants
            c_zero = arith.constant(0.0, type=compute_type).value
            c_neg_inf = arith.constant(float("-inf"), type=compute_type).value
            c_zero_idx = rocir.const_index(0)
            # exp(x) -> exp2(x * log2(e))
            c_log2e = arith.constant(1.4426950408889634, type=compute_type).value  # log2(e)
            fm_fast = rocir.arith.FastMathFlags.fast

            # Helper: Block Reduction (wave64 shuffle + wave0 finalize)
            block_reduce = reduce_utils.make_block_reduce(
                tid=tid,
                BLOCK_SIZE=BLOCK_SIZE,
                compute_type=compute_type,
                arith=arith,
                gpu=gpu,
                rocir=rocir,
                s_red_tv=s_red_tv,
                T=T,
                ir=ir,
                c_zero=c_zero,
                c_neg_inf=c_neg_inf,
                c_zero_idx=c_zero_idx,
                fm_fast=fm_fast,
            )

            # 1. Load Data into Registers (Buffering)
            # List of buffered values (vector or scalar with validity)
            row_buffer = []

            # Stride = BLOCK_SIZE * VEC_WIDTH
            step = BLOCK_SIZE * VEC_WIDTH

            # Base offset for this thread
            thread_offset_base = rocir.arith.MulIOp(unwrap(tid), rocir.const_index(VEC_WIDTH)).result

            # Loop range(0, N, step)
            for base_idx_int in range(0, N, step):
                # Current global index base for this thread
                # global_idx = base_idx_int + thread_offset_base
                c_base = rocir.const_index(base_idx_int)
                curr_idx = rocir.arith.AddIOp(c_base, unwrap(thread_offset_base)).result

                # Check bounds
                # If fully within N, vector load We can check statically for the loop unroll? Since N is compile time constant, we check specific offsets. However, thread_id is dynamic. We rely on logic: If (base_idx_int + BLOCK_SIZE*VEC_WIDTH) <= N, then ALL threads are safe? No. tid=255 accesses last chunk. Safe logic: if (base_idx_int + (BLOCK_SIZE-1)*WIDTH + WIDTH) <= N.

                is_safe_vector = (base_idx_int + (BLOCK_SIZE - 1) * VEC_WIDTH + VEC_WIDTH) <= N

                if is_safe_vector:
                    # Rocir tiled copy: global -> rmem fragment, then load vector from fragment.
                    tile_i = base_idx_int // tile_cols  # python int
                    blkA = gA[(unwrap(row), tile_i)]
                    thrA = thr_copy_A.partition_S(blkA)
                    frgA = rocir.make_fragment_like(thrA, elem_type)
                    rocir.copy(
                        tiled_copy_A,
                        thrA,
                        frgA,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                    )
                    vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                    vec_val_e = vector.load(vec_type_e, frgA.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                    if dtype_str == "bf16":
                        vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                        vec_val = rocir.arith.extf(vec_type_c, unwrap(vec_val_e))
                    else:
                        vec_val = vec_val_e
                    row_buffer.append(vec_val)

                else:
                    # Scalar tail handling with validity mask
                    for k in range(VEC_WIDTH):
                        c_k = rocir.const_index(k)
                        idx_k = rocir.arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result

                        c_N = rocir.const_index(N)
                        is_valid = rocir.arith.CmpIOp(rocir.arith.CmpIPredicate.ult, unwrap(idx_k), unwrap(c_N)).result

                        if_load = rocir.scf_ext.IfOp(unwrap(is_valid), [compute_type], hasElse=True)
                        with ir.InsertionPoint(if_load.then_block):
                            val_e = tensor_A[(unwrap(row), unwrap(idx_k))]
                            if dtype_str == "bf16":
                                val_c = rocir.arith.extf(compute_type, unwrap(val_e))
                                rocir.scf_ext.yield_([unwrap(val_c)])
                            else:
                                rocir.scf_ext.yield_([unwrap(val_e)])
                        with ir.InsertionPoint(if_load.else_block):
                            rocir.scf_ext.yield_([unwrap(c_neg_inf)])

                        row_buffer.append((if_load.results[0], is_valid))

            # 2. Local Max
            thread_max = unwrap(c_neg_inf)

            reduce_vec_max = lambda vec_val: reduce_utils.reduce_vec_max(
                vec_val, VEC_WIDTH=VEC_WIDTH, compute_type=compute_type, vector=vector
            )
            reduce_vec_sum = lambda vec_val: reduce_utils.reduce_vec_sum(
                vec_val, VEC_WIDTH=VEC_WIDTH, compute_type=compute_type, vector=vector, fm_fast=fm_fast
            )

            for item in row_buffer:
                if isinstance(item, tuple):  # Scalar with validity mask
                    val, valid = item
                    # Select: if valid, val, else -inf
                    safe_val = rocir.arith.SelectOp(unwrap(valid), unwrap(val), unwrap(c_neg_inf)).result
                    thread_max = rocir.arith.MaximumFOp(unwrap(thread_max), unwrap(safe_val)).result
                else:  # Vector
                    vec_val = item
                    red = reduce_vec_max(vec_val)
                    thread_max = rocir.arith.MaximumFOp(unwrap(thread_max), unwrap(red)).result

            # 3. Global Max
            global_max = block_reduce(thread_max, "max")

            # 4. Local Sum & Exp
            thread_sum = unwrap(c_zero)

            # Update buffer in place with Exp values
            new_buffer = []

            g_max_splat_vec = None  # Cache splat
            log2e_splat = None  # Cache splat (vector)

            for i, item in enumerate(row_buffer):
                if isinstance(item, tuple):
                    val, valid = item
                    sub = rocir.arith.SubFOp(unwrap(val), unwrap(global_max), fastmath=fm_fast).result
                    scaled = rocir.arith.MulFOp(unwrap(sub), unwrap(c_log2e), fastmath=fm_fast).result
                    exp_val = mlir_math.exp2(unwrap(scaled), fastmath=fm_fast)

                    # Accumulate sum only if valid
                    safe_exp = rocir.arith.SelectOp(unwrap(valid), unwrap(exp_val), unwrap(c_zero)).result
                    thread_sum = rocir.arith.AddFOp(unwrap(thread_sum), unwrap(safe_exp), fastmath=fm_fast).result

                    new_buffer.append((exp_val, valid))  # Store exp
                else:
                    vec_val = item
                    if g_max_splat_vec is None:
                        vec_type = ir.VectorType.get([VEC_WIDTH], compute_type)
                        g_max_splat_vec = vector.splat(vec_type, unwrap(global_max))
                        log2e_splat = vector.splat(vec_type, unwrap(c_log2e))

                    sub = rocir.arith.SubFOp(unwrap(vec_val), unwrap(g_max_splat_vec), fastmath=fm_fast).result
                    scaled = rocir.arith.MulFOp(unwrap(sub), unwrap(log2e_splat), fastmath=fm_fast).result
                    exp_vec = mlir_math.exp2(unwrap(scaled), fastmath=fm_fast)

                    red = reduce_vec_sum(exp_vec)
                    thread_sum = rocir.arith.AddFOp(unwrap(thread_sum), unwrap(red), fastmath=fm_fast).result

                    new_buffer.append(exp_vec)

            row_buffer = new_buffer

            # 5. Global Sum
            global_sum = block_reduce(thread_sum, "sum")

            # 6. Normalize & Store
            c_one = arith.constant(1.0, type=compute_type).value
            inv_sum = rocir.arith.DivFOp(unwrap(c_one), unwrap(global_sum), fastmath=fm_fast).result

            inv_sum_splat_vec = None

            # Reconstruct indices for store
            buf_idx = 0
            thread_offset_base = rocir.arith.MulIOp(unwrap(tid), rocir.const_index(VEC_WIDTH)).result

            for base_idx_int in range(0, N, step):
                c_base = rocir.const_index(base_idx_int)
                curr_idx = rocir.arith.AddIOp(unwrap(c_base), unwrap(thread_offset_base)).result

                is_safe_vector = (base_idx_int + (BLOCK_SIZE - 1) * VEC_WIDTH + VEC_WIDTH) <= N

                if is_safe_vector:
                    vec_exp = row_buffer[buf_idx]
                    buf_idx += 1

                    if inv_sum_splat_vec is None:
                        vec_type = ir.VectorType.get([VEC_WIDTH], compute_type)
                        inv_sum_splat_vec = vector.splat(vec_type, unwrap(inv_sum))

                    # Prefer fast-math for normalization multiply
                    norm_vec = rocir.arith.MulFOp(vec_exp, inv_sum_splat_vec, fastmath=fm_fast).result

                    if dtype_str == "bf16":
                        if USE_HW_CVT_PK_BF16_F32:
                            # gfx95x: rely on f32->bf16 vector truncation lowering, which should
                            # map to hardware v_cvt_pk_bf16_f32 on these targets.
                            vec_type_bf16 = ir.VectorType.get([VEC_WIDTH], elem_type)
                            out_bf16 = rocir.arith.truncf(vec_type_bf16, unwrap(norm_vec))
                        else:
                            # === BF16 fast-pack store path (manual pack, toolchain-safe) ===
                            vec_i32_ty = ir.VectorType.get([VEC_WIDTH], T.i32())
                            vec4_i32_ty = ir.VectorType.get([VEC_WIDTH // 2], T.i32())
                            vec_bf16_ty = ir.VectorType.get([VEC_WIDTH], elem_type)

                            c16_i32 = arith.constant(16, type=T.i32()).value
                            c7fff_i32 = arith.constant(0x7FFF, type=T.i32()).value
                            c1_i32 = arith.constant(1, type=T.i32()).value

                            c16_i32_v = vector.splat(vec_i32_ty, unwrap(c16_i32))
                            c7fff_i32_v = vector.splat(vec_i32_ty, unwrap(c7fff_i32))
                            c1_i32_v = vector.splat(vec_i32_ty, unwrap(c1_i32))

                            u = rocir.arith.bitcast(vec_i32_ty, unwrap(norm_vec))
                            hi = rocir.arith.ShRUIOp(unwrap(u), unwrap(c16_i32_v)).result
                            lsb = rocir.arith.AndIOp(unwrap(hi), unwrap(c1_i32_v)).result
                            bias = rocir.arith.AddIOp(unwrap(c7fff_i32_v), unwrap(lsb)).result
                            u_round = rocir.arith.AddIOp(unwrap(u), unwrap(bias)).result
                            bf16_bits = rocir.arith.ShRUIOp(unwrap(u_round), unwrap(c16_i32_v)).result

                            even = vector.shuffle(bf16_bits, bf16_bits, mask=[0, 2, 4, 6])
                            odd = vector.shuffle(bf16_bits, bf16_bits, mask=[1, 3, 5, 7])
                            odd_sh = rocir.arith.ShLIOp(
                                unwrap(odd),
                                unwrap(vector.splat(vec4_i32_ty, unwrap(c16_i32))),
                            ).result
                            packed = rocir.arith.OrIOp(unwrap(even), unwrap(odd_sh)).result
                            out_bf16 = vector.bitcast(vec_bf16_ty, unwrap(packed))

                        tile_i = base_idx_int // tile_cols  # python int
                        blkC = gC[(unwrap(row), tile_i)]
                        thrC = thr_copy_C.partition_S(blkC)
                        frgC = rocir.make_fragment_like(thrC, elem_type)
                        vector.store(out_bf16, frgC.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                        rocir.copy(
                            tiled_copy_C,
                            frgC,
                            thrC,
                            nontemporal=USE_NONTEMPORAL,
                            alignment=VEC_ALIGN,
                        )
                    else:
                        # Store directly in element type (no upcast)
                        tile_i = base_idx_int // tile_cols  # python int
                        blkC = gC[(unwrap(row), tile_i)]
                        thrC = thr_copy_C.partition_S(blkC)
                        frgC = rocir.make_fragment_like(thrC, elem_type)
                        vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                        norm_e = norm_vec if dtype_str != "bf16" else rocir.arith.truncf(vec_type_e, unwrap(norm_vec))
                        vector.store(unwrap(norm_e), frgC.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                        rocir.copy(
                            tiled_copy_C,
                            frgC,
                            thrC,
                            nontemporal=USE_NONTEMPORAL,
                            alignment=VEC_ALIGN,
                        )

                else:
                    for k in range(VEC_WIDTH):
                        item = row_buffer[buf_idx]
                        buf_idx += 1
                        val_exp, valid = item

                        # If valid, store
                        if_store = rocir.scf_ext.IfOp(unwrap(valid))
                        with ir.InsertionPoint(if_store.then_block):
                            norm_val = rocir.arith.MulFOp(unwrap(val_exp), unwrap(inv_sum), fastmath=fm_fast).result
                            if dtype_str == "bf16":
                                norm_val = rocir.arith.truncf(elem_type, unwrap(norm_val))

                            c_k = rocir.const_index(k)
                            idx_k = rocir.arith.AddIOp(unwrap(curr_idx), unwrap(c_k)).result
                            tensor_C[(unwrap(row), unwrap(idx_k))] = unwrap(norm_val)
                            rocir.scf_ext.yield_([])

    return ctx


