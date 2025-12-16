#!/usr/bin/env python3
"""
Softmax Operator Test with Manual Vectorization and Register Buffering
Implementation based on high-performance C++ kernel logic:
- Vectorized Loads/Stores (WIDTH=8/4)
- Register Buffering (Row kept in registers)
- Warp Reductions (Shuffle)
- Shared Memory Block Reductions
"""

import sys
import os
import math

# Add paths (prefer embedded MLIR to avoid mixing multiple runtimes)
repo_root = os.path.join(os.path.dirname(__file__), "../../..")
embedded_pkgs = os.path.join(repo_root, "build", "python_packages", "rocdsl")
if os.path.isdir(os.path.join(embedded_pkgs, "_mlir")):
    sys.path.insert(0, embedded_pkgs)
else:
    sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH', ''), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))
sys.path.insert(0, repo_root)

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import rocir
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from rocdsl.utils import SmemAllocator
from tests.utils import compile_to_hsaco
from _mlir import ir
import _mlir.extras.types as T
try:
    from hip import hip
except ImportError:
    print("HIP module not found. Skipping GPU tests.")
    sys.exit(0)

import numpy as np
import ctypes

# Expose modules through Rocir interface (keep behavior/perf, avoid mlir.* imports).
gpu = rocir.gpu_ext          # extended wrapper (has set_container_module, etc.)
scf = rocir.scf_ext          # extended wrapper (yield_ helper, etc.)
arith = rocir.arith_ext      # extended wrapper (arith.constant(...), ArithValue, etc.)
memref = rocir.memref        # raw dialect module
vector = rocir.vector        # raw dialect module
mlir_math = rocir.math       # raw dialect module
llvm = rocir.llvm            # raw dialect module

def unwrap(v):
    if hasattr(v, "value"): return v.value
    if hasattr(v, "_value"): return v._value
    if hasattr(v, "result"): return v.result
    return v

def bf16_to_fp32_cpu(arr_bf16_uint16):
    arr_u32 = arr_bf16_uint16.astype(np.uint32) << 16
    return np.frombuffer(arr_u32.tobytes(), dtype=np.float32).reshape(arr_bf16_uint16.shape)

def fp32_to_bf16_cpu(arr_fp32):
    arr_u32 = np.frombuffer(arr_fp32.astype(np.float32).tobytes(), dtype=np.uint32)
    return (arr_u32 >> 16).astype(np.uint16).reshape(arr_fp32.shape)

def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

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
    if BLOCK_SIZE < 32: BLOCK_SIZE = 32 # Min block size for warp ops
    
    # Vector Width
    # Use 8 for aligned, small vectors for unaligned? To keep simple in Python gen, we use 8 and fallback to scalar for tail.
    VEC_WIDTH = 8

    # Nontemporal (cache-bypass) hints for global vector loads/stores.
    # MLIR vector.load/store support `nontemporal` as a BoolAttr. We apply it only on the aligned vector path.
    USE_NONTEMPORAL = True
    # Conservative alignment hint (bytes) for vector load/store.
    # For f16/bf16 and VEC_WIDTH=8 -> 16B; for f32 -> 32B. Use 16B universally.
    VEC_ALIGN = 16

    # Optional: use LLVM AMDGPU bf16 pack intrinsic (requires LLVM support).
    # - 0 (default): use manual bf16 pack (works on current toolchain) 
    # - 1: use llvm.amdgcn.cvt.pk.bf16.f32 (after you add the intrinsic to your LLVM build)
    # - NOTE: gfx942 does not support `v_cvt_pk_bf16_f32` (llvm-mc will report instruction not supported).
    USE_BF16_PACK_INTR = (os.environ.get("ROCDSL_BF16_PACK_INTR", "0") == "1") and (gpu_arch != "gfx942")

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
                copy_atom_load, thr_layout, val_layout,
                thr_shape=(1, BLOCK_SIZE), val_shape=(1, VEC_WIDTH)
            )
            tiled_copy_C = rocir.make_tiled_copy_tv(
                copy_atom_store, thr_layout, val_layout,
                thr_shape=(1, BLOCK_SIZE), val_shape=(1, VEC_WIDTH)
            )
            thr_copy_A = tiled_copy_A.get_slice(unwrap(tid))
            thr_copy_C = tiled_copy_C.get_slice(unwrap(tid))
            
            # Element-type constants
            c_zero = arith.constant(0.0, type=compute_type).value
            c_neg_inf = arith.constant(float('-inf'), type=compute_type).value
            c_zero_idx = rocir.const_index(0)
            # exp(x) -> exp2(x * log2(e))
            c_log2e = arith.constant(1.4426950408889634, type=compute_type).value  # log2(e)
            fm_fast = rocir.arith.FastMathFlags.fast

            # Helper: Block Reduction (wave64 shuffle + wave0 finalize)
            def block_reduce(val, reduce_op_name):
                # AMD wavefront size is 64 on gfx9+/gfx10+/gfx11.
                WARP_SIZE = 64
                NUM_WAVES = (BLOCK_SIZE + WARP_SIZE - 1) // WARP_SIZE  # python int
                # Use Rocir layout algebra to compute LDS indices for the reduction scratch.
                c_num_waves = rocir.const_index(NUM_WAVES)
                c1 = rocir.const_index(1)
                shape_red = rocir.make_shape(unwrap(c_num_waves))
                stride_red = rocir.make_stride(unwrap(c1))
                layout_red = rocir.make_layout(shape_red, stride_red)

                tid_i32 = rocir.arith.IndexCastOp(T.i32(), unwrap(tid)).result
                c_warp_i32 = arith.constant(WARP_SIZE, type=T.i32()).value
                lane_i32 = rocir.arith.RemUIOp(unwrap(tid_i32), unwrap(c_warp_i32)).result
                wave_i32 = rocir.arith.DivUIOp(unwrap(tid_i32), unwrap(c_warp_i32)).result

                width_i32 = arith.constant(WARP_SIZE, type=T.i32()).value
                w = unwrap(val)

                # Intra-wave reduction via xor shuffle
                for sh in [32, 16, 8, 4, 2, 1]:
                    off = arith.constant(sh, type=T.i32()).value
                    peer = gpu.ShuffleOp(unwrap(w), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
                    if reduce_op_name == "max":
                        w = rocir.arith.MaximumFOp(unwrap(w), unwrap(peer)).result
                    else:
                        w = rocir.arith.AddFOp(unwrap(w), unwrap(peer), fastmath=fm_fast).result

                # lane0 writes per-wave partial into LDS s_red[wave_id]
                is_lane0 = rocir.arith.CmpIOp(
                    rocir.arith.CmpIPredicate.eq,
                    unwrap(lane_i32),
                    unwrap(arith.constant(0, type=T.i32()).value),
                ).result
                if_lane0 = rocir.scf_ext.IfOp(unwrap(is_lane0))
                with ir.InsertionPoint(if_lane0.then_block):
                    wave_idx = rocir.arith.IndexCastOp(T.index(), unwrap(wave_i32)).result
                    red_idx = rocir.crd2idx(rocir.make_coord(unwrap(wave_idx)), layout_red)
                    s_red_tv[unwrap(red_idx)] = unwrap(w)
                    rocir.scf_ext.yield_([])
                gpu.barrier()

                # wave0 reduces NUM_WAVES partials (still using shuffle)
                is_wave0 = rocir.arith.CmpIOp(
                    rocir.arith.CmpIPredicate.eq,
                    unwrap(wave_i32),
                    unwrap(arith.constant(0, type=T.i32()).value),
                ).result
                if_wave0 = rocir.scf_ext.IfOp(unwrap(is_wave0), [compute_type], hasElse=True)
                with ir.InsertionPoint(if_wave0.then_block):
                    in_range = rocir.arith.CmpIOp(
                        rocir.arith.CmpIPredicate.ult,
                        unwrap(lane_i32),
                        unwrap(arith.constant(NUM_WAVES, type=T.i32()).value),
                    ).result
                    if_in = rocir.scf_ext.IfOp(unwrap(in_range), [compute_type], hasElse=True)
                    with ir.InsertionPoint(if_in.then_block):
                        lane_idx = rocir.arith.IndexCastOp(T.index(), unwrap(lane_i32)).result
                        red_idx = rocir.crd2idx(rocir.make_coord(unwrap(lane_idx)), layout_red)
                        v = s_red_tv[unwrap(red_idx)]
                        rocir.scf_ext.yield_([unwrap(v)])
                    with ir.InsertionPoint(if_in.else_block):
                        neutral = c_neg_inf if reduce_op_name == "max" else c_zero
                        rocir.scf_ext.yield_([unwrap(neutral)])

                    ww = if_in.results[0]
                    for sh in [32, 16, 8, 4, 2, 1]:
                        off = arith.constant(sh, type=T.i32()).value
                        peer = gpu.ShuffleOp(unwrap(ww), unwrap(off), unwrap(width_i32), mode="xor").shuffleResult
                        if reduce_op_name == "max":
                            ww = rocir.arith.MaximumFOp(unwrap(ww), unwrap(peer)).result
                        else:
                            ww = rocir.arith.AddFOp(unwrap(ww), unwrap(peer), fastmath=fm_fast).result

                    # lane0 writes final to s_red[0]
                    is_lane0_2 = rocir.arith.CmpIOp(
                        rocir.arith.CmpIPredicate.eq,
                        unwrap(lane_i32),
                        unwrap(arith.constant(0, type=T.i32()).value),
                    ).result
                    if_lane0_2 = rocir.scf_ext.IfOp(unwrap(is_lane0_2))
                    with ir.InsertionPoint(if_lane0_2.then_block):
                        red_idx0 = rocir.crd2idx(rocir.make_coord(unwrap(c_zero_idx)), layout_red)
                        s_red_tv[unwrap(red_idx0)] = unwrap(ww)
                        rocir.scf_ext.yield_([])

                    rocir.scf_ext.yield_([unwrap(ww)])
                with ir.InsertionPoint(if_wave0.else_block):
                    red_idx0 = rocir.crd2idx(rocir.make_coord(unwrap(c_zero_idx)), layout_red)
                    keep = s_red_tv[unwrap(red_idx0)]
                    rocir.scf_ext.yield_([unwrap(keep)])
                gpu.barrier()

                red_idx0 = rocir.crd2idx(rocir.make_coord(unwrap(c_zero_idx)), layout_red)
                return s_red_tv[unwrap(red_idx0)]

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
            
            def reduce_vec_max(vec_val):
                if VEC_WIDTH == 1:
                    return vector.extract(vec_val, static_position=[0], dynamic_position=[])
                # Avoid fastmath on bf16 max reduction; some backends can fail to select.
                return vector.reduction(compute_type, "maxnumf", unwrap(vec_val))
            
            def reduce_vec_sum(vec_val):
                if VEC_WIDTH == 1:
                    return vector.extract(vec_val, static_position=[0], dynamic_position=[])
                return vector.reduction(compute_type, "add", unwrap(vec_val), fastmath=fm_fast)
            
            for item in row_buffer:
                if isinstance(item, tuple): # Scalar with validity mask
                    val, valid = item
                    # Select: if valid, val, else -inf
                    safe_val = rocir.arith.SelectOp(unwrap(valid), unwrap(val), unwrap(c_neg_inf)).result
                    thread_max = rocir.arith.MaximumFOp(unwrap(thread_max), unwrap(safe_val)).result
                else: # Vector
                    vec_val = item
                    red = reduce_vec_max(vec_val)
                    thread_max = rocir.arith.MaximumFOp(unwrap(thread_max), unwrap(red)).result

            # 3. Global Max
            global_max = block_reduce(thread_max, "max")
            
            # 4. Local Sum & Exp
            thread_sum = unwrap(c_zero)
            
            # Update buffer in place with Exp values
            new_buffer = []
            
            g_max_splat_vec = None # Cache splat
            log2e_splat = None     # Cache splat (vector)
            
            for i, item in enumerate(row_buffer):
                if isinstance(item, tuple):
                    val, valid = item
                    sub = rocir.arith.SubFOp(unwrap(val), unwrap(global_max), fastmath=fm_fast).result
                    scaled = rocir.arith.MulFOp(unwrap(sub), unwrap(c_log2e), fastmath=fm_fast).result
                    exp_val = mlir_math.exp2(unwrap(scaled), fastmath=fm_fast)
                    
                    # Accumulate sum only if valid
                    safe_exp = rocir.arith.SelectOp(unwrap(valid), unwrap(exp_val), unwrap(c_zero)).result
                    thread_sum = rocir.arith.AddFOp(unwrap(thread_sum), unwrap(safe_exp), fastmath=fm_fast).result
                    
                    new_buffer.append((exp_val, valid)) # Store exp
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
                        if USE_BF16_PACK_INTR:
                            # === BF16 fast-pack store path (LLVM AMDGPU intrinsic) ===
                            # After patching/rebuilding LLVM, this should exist and lower to native pack on gfx9+: llvm.amdgcn.cvt.pk.bf16.f32(float lo, float hi) -> <2 x bf16>
                            vec_type_bf16 = ir.VectorType.get([VEC_WIDTH], elem_type)
                            bf16_zero = arith.constant(0.0, type=elem_type).value
                            out_bf16 = vector.splat(vec_type_bf16, unwrap(bf16_zero))

                            pair_ty = ir.VectorType.get([2], elem_type)
                            intr = "llvm.amdgcn.cvt.pk.bf16.f32"

                            for pj in range(VEC_WIDTH // 2):
                                f0 = vector.extract(norm_vec, static_position=[2 * pj], dynamic_position=[])
                                f1 = vector.extract(norm_vec, static_position=[2 * pj + 1], dynamic_position=[])
                                pair = llvm.call_intrinsic(
                                    pair_ty,
                                    intr,
                                    [unwrap(f0), unwrap(f1)],
                                    [],
                                    [],
                                )
                                b0 = vector.extract(pair, static_position=[0], dynamic_position=[])
                                b1 = vector.extract(pair, static_position=[1], dynamic_position=[])
                                out_bf16 = vector.insert(unwrap(b0), unwrap(out_bf16), dynamic_position=[], static_position=[2 * pj])
                                out_bf16 = vector.insert(unwrap(b1), unwrap(out_bf16), dynamic_position=[], static_position=[2 * pj + 1])
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
                            odd_sh = rocir.arith.ShLIOp(unwrap(odd), unwrap(vector.splat(vec4_i32_ty, unwrap(c16_i32)))).result
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

def run_test(M, N, dtype_str):
    print(f"\nTesting Softmax (Vectorized): M={M}, N={N}, dtype={dtype_str}")
    
    try:
        ctx = build_softmax_module(M, N, dtype_str)
        hsaco = compile_to_hsaco(ctx.module, kernel_name="softmax_kernel")
    except Exception as e:
        print(f"❌ Compilation Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    np.random.seed(42)
    a_f32 = (np.random.rand(M, N).astype(np.float32) * 4.0) - 2.0
    
    max_vals = np.max(a_f32, axis=1, keepdims=True)
    exp_vals = np.exp(a_f32 - max_vals)
    sum_vals = np.sum(exp_vals, axis=1, keepdims=True)
    expected_f32 = exp_vals / sum_vals
    
    if dtype_str == "f32":
        a_host = a_f32
        c_host = np.zeros_like(a_f32)
        nbytes = M * N * 4
    elif dtype_str == "f16":
        a_host = a_f32.astype(np.float16)
        c_host = np.zeros_like(a_host)
        nbytes = M * N * 2
    elif dtype_str == "bf16":
        a_host = fp32_to_bf16_cpu(a_f32)
        c_host = np.zeros_like(a_host)
        nbytes = M * N * 2
    
    d_a = hip_check(hip.hipMalloc(nbytes))
    d_c = hip_check(hip.hipMalloc(nbytes))
    
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, nbytes, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"softmax_kernel"))
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_c))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Dynamic block size based on N
    blk = min(256, next_power_of_2(N))
    if blk < 32: blk = 32
    
    hip_check(hip.hipModuleLaunchKernel(
        kernel_func, 
        M, 1, 1, 
        blk, 1, 1, 
        0, 0, args, None
    ))
    hip_check(hip.hipDeviceSynchronize())
    
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, nbytes, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    if dtype_str == "f32":
        res_f32 = c_host
        atol = 1e-5
    elif dtype_str == "f16":
        res_f32 = c_host.astype(np.float32)
        atol = 1e-2 
    elif dtype_str == "bf16":
        res_f32 = bf16_to_fp32_cpu(c_host)
        atol = 2e-2 
        
    diff = np.abs(res_f32 - expected_f32)
    max_err = np.max(diff)
    print(f"  Max Absolute Error: {max_err:.2e} (atol={atol})")
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    
    if max_err < atol:
        print("  ✅ Passed")
        return True
    else:
        print("  ❌ Failed")
        return False

def test_all():
    print("="*80)
    print("Running Softmax Vectorized Tests")
    print("="*80)
    
    configs = [
        # (64, 256, "f32"),    # Aligned
        # (128, 1024, "f32"),  # Aligned
        # (32, 128, "f16"),    # Aligned
        # (64, 2000, "f32"),   # Unaligned (tail handling)
        # (16, 512, "bf16"),   # BF16
        # (1024, 8192, "bf16"),# BF16
        (32768, 8192, "bf16"),  # BF16
    ]
    
    failures = 0
    for M, N, dtype in configs:
        if not run_test(M, N, dtype):
            failures += 1
            
    print("\n" + "="*80)
    if failures == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"{failures} TESTS FAILED")
    print("="*80)

if __name__ == "__main__":
    test_all()
