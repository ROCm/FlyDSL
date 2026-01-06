"""Softmax kernel builder used by tests.

This file intentionally keeps the kernel builder logic identical to the version
previously embedded in `tests/kernels/test_softmax.py` to preserve codegen and
performance. Only test-only helpers/imports are removed.
"""

import os

from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from . import reduce as reduce_utils
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
from _mlir import ir
import _mlir.extras.types as T


KERNEL_NAME = "softmax_kernel"


def val(v):
    if hasattr(v, "value"):
        return v.value
    if hasattr(v, "_value"):
        return v._value
    if hasattr(v, "result"):
        return v.result
    return v


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


# Expose modules through Flir interface (keep behavior/perf, avoid mlir.* imports).
gpu = flir.gpu_ext          # extended wrapper (has set_container_module, etc.)
scf = flir.scf_ext          # extended wrapper (yield_ helper, etc.)
arith = flir.arith_ext      # extended wrapper (arith.constant(...), ArithValue, etc.)
memref = flir.memref        # raw dialect module
vector = flir.vector        # raw dialect module
mlir_math = flir.math       # raw dialect module
llvm = flir.llvm            # raw dialect module


def build_softmax_module(M, N, dtype_str="f32"):
    gpu_arch = get_hip_arch()

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
    allocator = SmemAllocator(None, arch=gpu_arch)
    # Reduction scratch: one slot per wave (lane0 writes partials) + reuse slot 0 for broadcast.
    WARP_SIZE = 64
    RED_SLOTS = max(1, (BLOCK_SIZE + WARP_SIZE - 1) // WARP_SIZE)
    _state = {}

    class _Softmax(flir.MlirModule):
        GPU_MODULE_NAME = f"softmax_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            # Types (must be created under an active MLIR Context).
            if dtype_str == "f32":
                elem_type = T.f32()
            elif dtype_str == "f16":
                elem_type = T.f16()
            elif dtype_str == "bf16":
                elem_type = ir.BF16Type.get()
            else:
                raise ValueError(f"Unsupported dtype: {dtype_str}")

            # For bf16, do math in f32 and only convert at load/store boundaries.
            compute_type = T.f32() if dtype_str == "bf16" else elem_type
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type

            _state["smem_red"] = allocator.allocate_array(compute_type, RED_SLOTS)
            allocator.finalize()

        @flir.kernel
        def softmax_kernel(
            self: flir.T.i64,
            A: lambda: T.memref(M, N, _state["elem_type"]),
            C: lambda: T.memref(M, N, _state["elem_type"]),
        ):
            row = flir.block_idx("x")
            tid = flir.thread_idx("x")

            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            base_ptr = allocator.get_base()
            s_red = _state["smem_red"](base_ptr).get()
            # Rocir-style: tensor views + tiled copies (like elementwise_add_kernel).
            c0_idx = flir.const_index(0)
            tile_cols = BLOCK_SIZE * VEC_WIDTH  # python int
            tensor_A = flir.make_tensor(A, shape=(M, N), strides=(N, 1))
            tensor_C = flir.make_tensor(C, shape=(M, N), strides=(N, 1))
            s_red_tv = flir.make_tensor(s_red, shape=(RED_SLOTS,), strides=(1,))
            gA = flir.zipped_divide(tensor_A, (1, tile_cols))
            gC = flir.zipped_divide(tensor_C, (1, tile_cols))

            thr_layout = flir.make_ordered_layout((1, BLOCK_SIZE), order=(1, 0))
            val_layout = flir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0))
            copy_atom_load = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            copy_atom_store = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tiled_copy_A = flir.make_tiled_copy_tv(
                copy_atom_load,
                thr_layout,
                val_layout,
                thr_shape=(1, BLOCK_SIZE),
                val_shape=(1, VEC_WIDTH),
            )
            tiled_copy_C = flir.make_tiled_copy_tv(
                copy_atom_store,
                thr_layout,
                val_layout,
                thr_shape=(1, BLOCK_SIZE),
                val_shape=(1, VEC_WIDTH),
            )
            thr_copy_A = tiled_copy_A.get_slice(val(tid))
            thr_copy_C = tiled_copy_C.get_slice(val(tid))

            # Element-type constants
            c_zero = arith.constant(0.0, type=compute_type).value
            c_neg_inf = arith.constant(float("-inf"), type=compute_type).value
            c_zero_idx = flir.const_index(0)
            # exp(x) -> exp2(x * log2(e))
            c_log2e = arith.constant(1.4426950408889634, type=compute_type).value  # log2(e)
            fm_fast = flir.arith.FastMathFlags.fast

            # Helper: Block Reduction (wave64 shuffle + wave0 finalize)
            block_reduce = reduce_utils.make_block_reduce(
                tid=tid,
                BLOCK_SIZE=BLOCK_SIZE,
                compute_type=compute_type,
                arith=arith,
                gpu=gpu,
                flir=flir,
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
            thread_offset_base = flir.arith.MulIOp(val(tid), flir.const_index(VEC_WIDTH)).result

            # Loop range(0, N, step)
            for base_idx_int in range_constexpr(0, N, step):
                # Current global index base for this thread
                # global_idx = base_idx_int + thread_offset_base
                c_base = flir.const_index(base_idx_int)
                curr_idx = flir.arith.AddIOp(c_base, val(thread_offset_base)).result

                # Check bounds
                # If fully within N, vector load We can check statically for the loop unroll? Since N is compile time constant, we check specific offsets. However, thread_id is dynamic. We rely on logic: If (base_idx_int + BLOCK_SIZE*VEC_WIDTH) <= N, then ALL threads are safe? No. tid=255 accesses last chunk. Safe logic: if (base_idx_int + (BLOCK_SIZE-1)*WIDTH + WIDTH) <= N.

                is_safe_vector = (base_idx_int + (BLOCK_SIZE - 1) * VEC_WIDTH + VEC_WIDTH) <= N

                if is_safe_vector:
                    # Flir tiled copy: global -> rmem fragment, then load vector from fragment.
                    tile_i = base_idx_int // tile_cols  # python int
                    blkA = gA[(val(row), tile_i)]
                    thrA = thr_copy_A.partition_S(blkA)
                    frgA = flir.make_fragment_like(thrA, elem_type)
                    flir.copy(
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
                        vec_val = flir.arith.extf(vec_type_c, val(vec_val_e))
                    else:
                        vec_val = vec_val_e
                    row_buffer.append(vec_val)

                else:
                    # Scalar tail handling with validity mask
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = flir.const_index(k)
                        idx_k = flir.arith.AddIOp(val(curr_idx), val(c_k)).result

                        c_N = flir.const_index(N)
                        is_valid = flir.arith.CmpIOp(flir.arith.CmpIPredicate.ult, val(idx_k), val(c_N)).result

                        # IMPORTANT: `is_valid` is an MLIR i1 Value, not a Python bool.
                        # Use predicated load to avoid OOB memory access.
                        idx_safe = flir.arith.SelectOp(val(is_valid), val(idx_k), val(c0_idx)).result
                        val_e = tensor_A[(val(row), val(idx_safe))]
                        val_c = flir.arith.extf(compute_type, val(val_e)) if dtype_str == "bf16" else val_e
                        val = flir.arith.SelectOp(val(is_valid), val(val_c), val(c_neg_inf)).result

                        row_buffer.append((val, is_valid))

            # 2. Local Max
            thread_max = val(c_neg_inf)

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
                    safe_val = flir.arith.SelectOp(val(valid), val(val), val(c_neg_inf)).result
                    thread_max = flir.arith.MaximumFOp(val(thread_max), val(safe_val)).result
                else:  # Vector
                    vec_val = item
                    red = reduce_vec_max(vec_val)
                    thread_max = flir.arith.MaximumFOp(val(thread_max), val(red)).result

            # 3. Global Max
            global_max = block_reduce(thread_max, "max")

            # 4. Local Sum & Exp
            thread_sum = val(c_zero)

            # Update buffer in place with Exp values
            new_buffer = []

            g_max_splat_vec = None  # Cache splat
            log2e_splat = None  # Cache splat (vector)

            for i, item in enumerate(row_buffer):
                if isinstance(item, tuple):
                    val, valid = item
                    sub = flir.arith.SubFOp(val(val), val(global_max), fastmath=fm_fast).result
                    scaled = flir.arith.MulFOp(val(sub), val(c_log2e), fastmath=fm_fast).result
                    exp_val = mlir_math.exp2(val(scaled), fastmath=fm_fast)

                    # Accumulate sum only if valid
                    safe_exp = flir.arith.SelectOp(val(valid), val(exp_val), val(c_zero)).result
                    thread_sum = flir.arith.AddFOp(val(thread_sum), val(safe_exp), fastmath=fm_fast).result

                    new_buffer.append((exp_val, valid))  # Store exp
                else:
                    vec_val = item
                    if g_max_splat_vec is None:
                        vec_type = ir.VectorType.get([VEC_WIDTH], compute_type)
                        g_max_splat_vec = vector.splat(vec_type, val(global_max))
                        log2e_splat = vector.splat(vec_type, val(c_log2e))

                    sub = flir.arith.SubFOp(val(vec_val), val(g_max_splat_vec), fastmath=fm_fast).result
                    scaled = flir.arith.MulFOp(val(sub), val(log2e_splat), fastmath=fm_fast).result
                    exp_vec = mlir_math.exp2(val(scaled), fastmath=fm_fast)

                    red = reduce_vec_sum(exp_vec)
                    thread_sum = flir.arith.AddFOp(val(thread_sum), val(red), fastmath=fm_fast).result

                    new_buffer.append(exp_vec)

            row_buffer = new_buffer

            # 5. Global Sum
            global_sum = block_reduce(thread_sum, "sum")

            # 6. Normalize & Store
            c_one = arith.constant(1.0, type=compute_type).value
            inv_sum = flir.arith.DivFOp(val(c_one), val(global_sum), fastmath=fm_fast).result

            inv_sum_splat_vec = None

            # Reconstruct indices for store
            buf_idx = 0
            thread_offset_base = flir.arith.MulIOp(val(tid), flir.const_index(VEC_WIDTH)).result

            for base_idx_int in range_constexpr(0, N, step):
                c_base = flir.const_index(base_idx_int)
                curr_idx = flir.arith.AddIOp(val(c_base), val(thread_offset_base)).result

                is_safe_vector = (base_idx_int + (BLOCK_SIZE - 1) * VEC_WIDTH + VEC_WIDTH) <= N

                if is_safe_vector:
                    vec_exp = row_buffer[buf_idx]
                    buf_idx += 1

                    if inv_sum_splat_vec is None:
                        vec_type = ir.VectorType.get([VEC_WIDTH], compute_type)
                        inv_sum_splat_vec = vector.splat(vec_type, val(inv_sum))

                    # Prefer fast-math for normalization multiply
                    norm_vec = flir.arith.MulFOp(vec_exp, inv_sum_splat_vec, fastmath=fm_fast).result

                    if dtype_str == "bf16":
                        if USE_HW_CVT_PK_BF16_F32:
                            # gfx95x: rely on f32->bf16 vector truncation lowering, which should
                            # map to hardware v_cvt_pk_bf16_f32 on these targets.
                            vec_type_bf16 = ir.VectorType.get([VEC_WIDTH], elem_type)
                            out_bf16 = flir.arith.truncf(vec_type_bf16, val(norm_vec))
                        else:
                            # === BF16 fast-pack store path (manual pack, toolchain-safe) ===
                            vec_i32_ty = ir.VectorType.get([VEC_WIDTH], T.i32())
                            vec4_i32_ty = ir.VectorType.get([VEC_WIDTH // 2], T.i32())
                            vec_bf16_ty = ir.VectorType.get([VEC_WIDTH], elem_type)

                            c16_i32 = arith.constant(16, type=T.i32()).value
                            c7fff_i32 = arith.constant(0x7FFF, type=T.i32()).value
                            c1_i32 = arith.constant(1, type=T.i32()).value

                            c16_i32_v = vector.splat(vec_i32_ty, val(c16_i32))
                            c7fff_i32_v = vector.splat(vec_i32_ty, val(c7fff_i32))
                            c1_i32_v = vector.splat(vec_i32_ty, val(c1_i32))

                            u = flir.arith.bitcast(vec_i32_ty, val(norm_vec))
                            hi = flir.arith.ShRUIOp(val(u), val(c16_i32_v)).result
                            lsb = flir.arith.AndIOp(val(hi), val(c1_i32_v)).result
                            bias = flir.arith.AddIOp(val(c7fff_i32_v), val(lsb)).result
                            u_round = flir.arith.AddIOp(val(u), val(bias)).result
                            bf16_bits = flir.arith.ShRUIOp(val(u_round), val(c16_i32_v)).result

                            even = vector.shuffle(bf16_bits, bf16_bits, mask=[0, 2, 4, 6])
                            odd = vector.shuffle(bf16_bits, bf16_bits, mask=[1, 3, 5, 7])
                            odd_sh = flir.arith.ShLIOp(
                                val(odd),
                                val(vector.splat(vec4_i32_ty, val(c16_i32))),
                            ).result
                            packed = flir.arith.OrIOp(val(even), val(odd_sh)).result
                            out_bf16 = vector.bitcast(vec_bf16_ty, val(packed))

                        tile_i = base_idx_int // tile_cols  # python int
                        blkC = gC[(val(row), tile_i)]
                        thrC = thr_copy_C.partition_S(blkC)
                        frgC = flir.make_fragment_like(thrC, elem_type)
                        vector.store(out_bf16, frgC.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                        flir.copy(
                            tiled_copy_C,
                            frgC,
                            thrC,
                            nontemporal=USE_NONTEMPORAL,
                            alignment=VEC_ALIGN,
                        )
                    else:
                        # Store directly in element type (no upcast)
                        tile_i = base_idx_int // tile_cols  # python int
                        blkC = gC[(val(row), tile_i)]
                        thrC = thr_copy_C.partition_S(blkC)
                        frgC = flir.make_fragment_like(thrC, elem_type)
                        vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                        norm_e = norm_vec if dtype_str != "bf16" else flir.arith.truncf(vec_type_e, val(norm_vec))
                        vector.store(val(norm_e), frgC.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                        flir.copy(
                            tiled_copy_C,
                            frgC,
                            thrC,
                            nontemporal=USE_NONTEMPORAL,
                            alignment=VEC_ALIGN,
                        )

                else:
                    for k in range_constexpr(VEC_WIDTH):
                        item = row_buffer[buf_idx]
                        buf_idx += 1
                        val_exp, valid = item

                        # If valid, store
                        with scf.if_(valid) as then_blk:
                            with ir.InsertionPoint(then_blk):
                                norm_val = flir.arith.MulFOp(val(val_exp), val(inv_sum), fastmath=fm_fast).result
                                if dtype_str == "bf16":
                                    norm_val = flir.arith.truncf(elem_type, val(norm_val))

                                c_k = flir.const_index(k)
                                idx_k = flir.arith.AddIOp(val(curr_idx), val(c_k)).result
                                tensor_C[(val(row), val(idx_k))] = val(norm_val)
                                scf.yield_([])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            A: lambda: T.memref(M, N, _state["elem_type"]),
            C: lambda: T.memref(M, N, _state["elem_type"]),
        ):
            c1 = val(flir.arith_ext.index(1))
            gx = val(flir.arith_ext.index(M))
            bx = val(flir.arith_ext.index(BLOCK_SIZE))
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "softmax_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[A, C],
            )

    return _Softmax()


