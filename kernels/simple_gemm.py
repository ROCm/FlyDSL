"""Simple GEMM kernel implementation using FlyDSL (MFMA 16x16x16).

This module provides a simple GEMM kernel (C = A × B^T) for AMD GPUs using MFMA instructions.

Configuration:
- Block: 256 threads = 4 waves × 64 lanes
- Tile: M=16 × N=64 × K=128 (configurable)
- Currently supports bf16/fp16 input, f32 accumulator, bf16/fp16 output
- Supports non-aligned M, N, K dimensions:
  - M and N: kernel boundary checks on output stores
  - K: caller pads to multiple of 8 (vector load granularity)

A matrix loading:
- GM → GPR → LDS: 256 threads cooperatively load the A tile with XOR16 swizzle
- LDS → GPR: Each wave reads 16×32 sub-block for MFMA

B matrix loading:
- Direct load: Each wave handles 16 columns of N
- Wave 0: N[0:16], Wave 1: N[16:32], Wave 2: N[32:48], Wave 3: N[48:64]

Output C matrix (16×64):
- Wave 0 → C[0:16, 0:16]
- Wave 1 → C[0:16, 16:32]
- Wave 2 → C[0:16, 32:48]
- Wave 3 → C[0:16, 48:64]
- Boundary check: skip stores for out-of-bounds elements
"""

import functools

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator

from _mlir import ir
from _mlir.dialects import scf as _scf

from flydsl.dialects.ext import arith, gpu, buffer_ops, vector, rocdl, scf
from flydsl.lang.ir.types import T, memref


@functools.lru_cache(maxsize=1024)
def compile_simple_gemm(
    *,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 128,
    in_dtype: str = "bf16",
):
    """Compile a simple GEMM kernel and return the compiled executable.

    This kernel supports non-aligned M and N dimensions via output boundary checks.
    The caller is responsible for:
    - Passing M_pad, N_pad, K_pad as the actual tensor dimensions (padded to tile sizes)
    - Passing M_orig, N_orig as the original dimensions for boundary checking on output

    Args:
        tile_m, tile_n, tile_k: Block tile sizes.
        in_dtype: Input data type ("bf16" or "fp16").
    """
    if in_dtype not in ("bf16", "fp16"):
        raise ValueError(f"in_dtype must be 'bf16' or 'fp16', got {in_dtype!r}")

    is_bf16 = in_dtype == "bf16"
    elem_bytes = 2  # bf16 and fp16 are both 2 bytes

    # Validate tile configuration
    tile_k_bytes = tile_k * elem_bytes
    if tile_k_bytes % 64 != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes}"
        )

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    DYN = ir.ShapedType.get_dynamic_size()
    total_threads = 256

    # LDS configuration: tile_m × tile_k elements
    lds_stride = tile_k  # No padding for simplicity

    # Type helpers
    def _elem_type():
        return T.bf16 if is_bf16 else T.f16

    def _vec8_type():
        """16B vector (8 bf16/fp16 elements)."""
        return T.bf16x8 if is_bf16 else T.f16x8

    def _out_type():
        """Output element type."""
        return T.bf16 if is_bf16 else T.f16

    module_name = f"simple_gemm_{in_dtype}_t{tile_m}x{tile_n}x{tile_k}".replace("-", "_")

    class _GEMM(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # Allocate LDS for A tile: tile_m × tile_k elements
            lds_a_elems = tile_m * lds_stride
            _state["lds_a_decl"] = allocator.allocate_array(_elem_type(), lds_a_elems)
            allocator.finalize()

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_type()),
            arg_a: lambda: memref(DYN, _elem_type()),
            arg_b: lambda: memref(DYN, _elem_type()),
            c_m_pad: lambda: T.index,  # Padded M dimension (tensor size)
            c_n_pad: lambda: T.index,  # Padded N dimension (tensor size)
            c_k_pad: lambda: T.index,  # Padded K dimension (tensor size)
            c_m_orig: lambda: T.index,  # Original M dimension (for boundary check)
            c_n_orig: lambda: T.index,  # Original N dimension (for boundary check)
        ):
            # ================= Types =================
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.f32x4
            vec4_i16 = T.i16x4
            vec4_f16 = T.f16x4
            vec8_elem = _vec8_type()
            vec1_i64 = T.vec(1, i64)
            vec2_i64 = T.vec(2, i64)

            # Accumulator initialization
            acc_init = arith.constant_vector(0.0, vec4_f32)

            # ================= Layouts =================
            # A layout: [M_pad, K_pad] row-major
            layout_a = flir.make_layout((c_m_pad, c_k_pad), stride=(c_k_pad, 1))

            # B layout: [N_pad, K_pad] row-major (B^T in standard GEMM)
            layout_b = flir.make_layout((c_n_pad, c_k_pad), stride=(c_k_pad, 1))

            # C layout: [M_pad, N_pad] row-major
            layout_c = flir.make_layout((c_m_pad, c_n_pad), stride=(c_n_pad, 1))

            # LDS layout: [tile_m, tile_k]
            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            # XOR16 swizzle parameter (in 16-byte blocks)
            k_blocks16 = arith.constant(tile_k_bytes // 16, index=True)

            # ================= Thread/Block IDs =================
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")  # M dimension
            by = gpu.block_id("y")  # N dimension

            # Base addresses for this block
            bx_m = bx * arith.constant(tile_m, index=True)
            by_n = by * arith.constant(tile_n, index=True)

            # ================= Thread Decomposition =================
            # tx -> (wave_id, lane_id)
            layout_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            coord_wave_lane = flir.idx2crd(tx, layout_wave_lane)
            wave_id = flir.get(coord_wave_lane, 0)
            lane_id = flir.get(coord_wave_lane, 1)

            # lane_id -> (lane_div_16, lane_mod_16)
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            coord_lane16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_lane16, 0)
            lane_mod_16 = flir.get(coord_lane16, 1)

            # ================= LDS Setup =================
            base_ptr = allocator.get_base()
            lds_a_ptr = _state["lds_a_decl"](base_ptr)
            lds_a = lds_a_ptr.get()

            # ================= Buffer Resources =================
            a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=False)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=False)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False)

            # ================= Wave/Lane Mappings =================
            # For MFMA 16x16x16:
            # - A row index: lane_mod_16 (0..15)
            # - K pack offset: lane_div_16 * 4 (each lane group handles 4 elements)
            row_a_lds = lane_mod_16

            # K element offset for LDS reads (16 elements per pack, 4 packs per K64)
            kpack_elems = 8  # 8 bf16 = 16 bytes
            col_offset_base = lane_div_16 * arith.constant(kpack_elems, index=True)
            # Convert to bytes for swizzle
            col_offset_base_bytes = col_offset_base * arith.constant(elem_bytes, index=True)

            # ================= Tile Configuration =================
            m_repeat = tile_m // 16  # Number of M-dimension repeats
            k_unroll = tile_k_bytes // 64  # K64-byte micro-steps
            num_waves = 4
            n_per_wave = tile_n // num_waves  # Columns per wave
            num_acc_n = n_per_wave // 16  # Accumulators per wave along N

            # Wave's N tile base
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            n_tile_base = wave_id * c_n_per_wave

            # ================= A Tile Loading (GM -> LDS) =================
            # 256 threads load tile_m × tile_k elements (16B per thread)
            bytes_a_per_tile = tile_m * tile_k * elem_bytes
            bytes_per_thread_a = bytes_a_per_tile // total_threads
            num_a_loads = bytes_per_thread_a // 16  # 16B loads

            # A tile layout in dwords for addressing
            tile_k_dwords = (tile_k * elem_bytes) // 4
            layout_a_tile_div4 = flir.make_layout(
                (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
            )

            # Convert K to dwords for A addressing
            c_k_div4bytes = (c_k_pad * arith.constant(elem_bytes, index=True)) / arith.constant(4, index=True)
            layout_a_div4 = flir.make_layout((c_m_pad, c_k_div4bytes), stride=(c_k_div4bytes, 1))

            c4 = arith.constant(4, index=True)
            tx_i32_base = tx * c4

            atom_a_g2r = flir.make_copy_atom(_elem_type(), vector_size=8)
            atom_a_lds = flir.make_copy_atom(_elem_type(), vector_size=8)

            def a_tile_chunk_coord(i: int):
                """Map (thread, chunk_id) -> (row_local, col_local_i32) for A loads."""
                chunk_off = arith.constant(i * total_threads * 4, index=True)
                tile_idx = tx_i32_base + chunk_off
                coord_local = flir.idx2crd(tile_idx, layout_a_tile_div4)
                row_local = flir.get(coord_local, 0)
                col_local_i32 = flir.get(coord_local, 1)
                return row_local, col_local_i32

            def load_a_tile(base_k):
                """Load A tile from global memory (tile_m × tile_k)."""
                base_k_bytes = base_k * arith.constant(elem_bytes, index=True)
                base_k_div4 = base_k_bytes / arith.constant(4, index=True)
                parts = []
                for i in range_constexpr(num_a_loads):
                    row_local, col_local_i32 = a_tile_chunk_coord(i)
                    row_global = bx_m + row_local
                    coord_a_g = flir.make_coord(row_global, base_k_div4 + col_local_i32)
                    idx_i32 = flir.crd2idx(coord_a_g, layout_a_div4)
                    # Convert dword index to element index
                    idx_elem = idx_i32 * arith.constant(2, index=True)  # 2 bf16 per dword

                    a_view = flir.TensorView(
                        arg_a,
                        (8,),  # 8 bf16 elements = 16 bytes
                        strides=(1,),
                        base_indices=(idx_elem,),
                        element_type=_elem_type(),
                    )
                    a_vec = flir.copy(
                        atom_a_g2r,
                        a_view,
                        None,
                        alignment=16,
                        return_vector=True,
                        src_buffer_resource=None,  # Use memref directly
                    )
                    parts.append(vector.bitcast(T.i32x4, a_vec))
                return parts

            def store_a_tile_to_lds(a_parts):
                """Store A tile to LDS with XOR16 swizzle."""
                for i in range_constexpr(num_a_loads):
                    row_local, col_local_i32 = a_tile_chunk_coord(i)
                    # Apply XOR16 swizzle
                    col_local_bytes = col_local_i32 * c4
                    col_swz_bytes = flir.swizzle_xor16(row_local, col_local_bytes, k_blocks16)
                    col_swz = col_swz_bytes / arith.constant(elem_bytes, index=True)
                    coord_store = flir.make_coord(row_local, col_swz)
                    idx0 = flir.crd2idx(coord_store, layout_lds)
                    v8 = vector.bitcast(vec8_elem, a_parts[i])
                    s_view = flir.TensorView(
                        lds_a,
                        (8,),
                        strides=(1,),
                        base_indices=(idx0,),
                        element_type=_elem_type(),
                    )
                    flir.copy(atom_a_lds, v8, s_view, alignment=16)

            # ================= B Tile Loading (Direct to GPR) =================
            def load_b_packs_k64(base_k, ku: int, ni: int):
                """Load B pack for MFMA (16B -> 2 × i64 for K64-byte step)."""
                # Global N index for this wave/lane
                n_offset = arith.constant(ni * 16, index=True)
                n_global = by_n + n_tile_base + n_offset + lane_mod_16

                # K index within the K64 block
                ki64 = arith.constant(ku * 32, index=True)  # 64 bytes = 32 bf16
                k_base = base_k + ki64

                # lane_div_16 determines which 8 elements to load (0-3 -> 0, 8, 16, 24 offset)
                k_lane_offset = lane_div_16 * arith.constant(8, index=True)
                k_global = k_base + k_lane_offset

                # Calculate linear index
                coord_b = flir.make_coord(n_global, k_global)
                idx_b = flir.crd2idx(coord_b, layout_b)

                # Load 8 elements (16 bytes)
                b_view = flir.TensorView(
                    arg_b,
                    (8,),
                    strides=(1,),
                    base_indices=(idx_b,),
                    element_type=_elem_type(),
                )
                b_vec = flir.copy(
                    atom_a_g2r,  # Same atom type
                    b_view,
                    None,
                    alignment=16,
                    return_vector=True,
                    src_buffer_resource=None,
                )

                # Split into two i64 halves for two MFMA K16 steps
                b_i64x2 = vector.bitcast(vec2_i64, b_vec)
                b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])

                # Convert to MFMA operand type
                if is_bf16:
                    # bf16 uses i16 bit patterns
                    b0_v1 = vector.from_elements(vec1_i64, [b0_i64])
                    b1_v1 = vector.from_elements(vec1_i64, [b1_i64])
                    return vector.bitcast(vec4_i16, b0_v1), vector.bitcast(vec4_i16, b1_v1)
                else:
                    # fp16 uses f16 directly
                    b0_v1 = vector.from_elements(vec1_i64, [b0_i64])
                    b1_v1 = vector.from_elements(vec1_i64, [b1_i64])
                    return vector.bitcast(vec4_f16, b0_v1), vector.bitcast(vec4_f16, b1_v1)

            def load_b_tile(base_k):
                """Load entire B tile for K loop."""
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        b0, b1 = load_b_packs_k64(base_k, ku, ni)
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile

            # ================= A LDS Load =================
            def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_base):
                """Load A pack from LDS for MFMA (16B -> 2 × i64)."""
                # Apply XOR16 swizzle
                col_swz_bytes = flir.swizzle_xor16(curr_row_a_lds, col_base_bytes, k_blocks16)
                col_swz = col_swz_bytes / arith.constant(elem_bytes, index=True)
                coord_a = flir.make_coord(curr_row_a_lds, col_swz)
                idx_a = flir.crd2idx(coord_a, layout_lds)
                idx_a = idx_a + lds_base

                # Load 8 elements
                loaded_a = vector.load_op(vec8_elem, lds_a, [idx_a])
                a_i64x2 = vector.bitcast(vec2_i64, loaded_a)
                a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])

                # Convert to MFMA operand type
                if is_bf16:
                    a0_v1 = vector.from_elements(vec1_i64, [a0_i64])
                    a1_v1 = vector.from_elements(vec1_i64, [a1_i64])
                    return vector.bitcast(vec4_i16, a0_v1), vector.bitcast(vec4_i16, a1_v1)
                else:
                    a0_v1 = vector.from_elements(vec1_i64, [a0_i64])
                    a1_v1 = vector.from_elements(vec1_i64, [a1_i64])
                    return vector.bitcast(vec4_f16, a0_v1), vector.bitcast(vec4_f16, a1_v1)

            # ================= MFMA Computation =================
            mfma_res_ty = vec4_f32
            if is_bf16:
                mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
            else:
                mfma_fn = rocdl.mfma_f32_16x16x16f16

            def mfma_step(acc_in, a, b):
                return mfma_fn(mfma_res_ty, [a, b, acc_in, 0, 0, 0])

            def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
                """K64-byte wrapper: two MFMA K16 ops."""
                acc_mid = mfma_step(acc_in, a0, b0)
                return mfma_step(acc_mid, a1, b1)

            def compute_tile(accs_in, b_tile_in, lds_base):
                """Compute one tile of MFMA operations."""
                current_accs = list(accs_in)

                for ku in range_constexpr(k_unroll):
                    b_packs0, b_packs1 = b_tile_in[ku]
                    # K byte offset for this ku
                    ki64 = ku * 64  # 64 bytes per ku
                    col_base = col_offset_base_bytes + arith.constant(ki64, index=True)

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val

                        # Load A pack from LDS
                        a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_base)

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            current_accs[acc_idx] = mfma_k64_bytes(
                                current_accs[acc_idx],
                                a0, a1,
                                b_packs0[ni], b_packs1[ni],
                            )

                return current_accs

            # ================= Epilogue (Store C) with boundary checks =================
            def store_output(final_accs):
                """Store accumulated results to C with boundary checks."""
                lane_div_16_mul4 = lane_div_16 * arith.constant(4, index=True)

                for mi in range_constexpr(m_repeat):
                    mi_base = arith.constant(mi * 16, index=True)
                    for ii in range_constexpr(4):  # 4 rows per lane group
                        ii_idx = arith.constant(ii, index=True)
                        row_off = lane_div_16_mul4 + ii_idx
                        row_in_tile = mi_base + row_off
                        row = bx_m + row_in_tile

                        col_base = by_n + n_tile_base + lane_mod_16

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            acc = final_accs[acc_idx]
                            val = vector.extract(acc, static_position=[ii], dynamic_position=[])

                            # Convert f32 to output type
                            val_out = arith.trunc_f(_out_type(), val)

                            col = col_base + arith.constant(ni * 16, index=True)

                            # Boundary check: row < M_orig and col < N_orig
                            row_valid = arith.cmpu(row, c_m_orig, "ult")
                            col_valid = arith.cmpu(col, c_n_orig, "ult")
                            in_bounds = arith.andi(row_valid, col_valid)

                            # Conditional store using IfOp
                            if_op = scf.IfOp(in_bounds)
                            with if_op.then():
                                coord_c = flir.make_coord(row, col)
                                idx_c = flir.crd2idx(coord_c, layout_c)
                                buffer_ops.buffer_store(val_out, c_rsrc, idx_c)

            # ================= Main Pipeline =================
            # Single LDS buffer, simple pipeline
            lds_base = arith.constant(0, index=True)

            # Initialize accumulators
            accs = [acc_init] * (num_acc_n * m_repeat)

            # K loop
            c_tile_k = arith.constant(tile_k, index=True)
            for k_base in range(arith.constant(0, index=True), c_k_pad, c_tile_k):
                # Load A tile to LDS
                a_parts = load_a_tile(k_base)
                store_a_tile_to_lds(a_parts)
                gpu.barrier()

                # Load B tile directly to GPR
                b_tile = load_b_tile(k_base)

                # Compute MFMA
                accs = compute_tile(accs, b_tile, lds_base)

                # Barrier before next iteration (if any)
                gpu.barrier()

            # Store output (with boundary checks)
            store_output(accs)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_type()),
            arg_a: lambda: memref(DYN, _elem_type()),
            arg_b: lambda: memref(DYN, _elem_type()),
            c_m_pad: lambda: T.index,
            c_n_pad: lambda: T.index,
            c_k_pad: lambda: T.index,
            c_m_orig: lambda: T.index,
            c_n_orig: lambda: T.index,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(256, index=True)
            tm = arith.constant(tile_m, index=True)
            tn = arith.constant(tile_n, index=True)
            one = arith.constant(1, index=True)
            # Use padded dimensions for grid size
            gx = (c_m_pad + tm - one) / tm
            gy = (c_n_pad + tn - one) / tn
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "kernel_gemm"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_c,
                    arg_a,
                    arg_b,
                    c_m_pad,
                    c_n_pad,
                    c_k_pad,
                    c_m_orig,
                    c_n_orig,
                ],
            )

    m = _GEMM()
    return flydsl.compile(m)


def _align_up(val: int, align: int) -> int:
    """Round up val to the next multiple of align."""
    return ((val + align - 1) // align) * align


def run_simple_gemm(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int = 16,
    tile_n: int = 64,
    tile_k: int = 128,
    in_dtype: str = "bf16",
    A=None,
    B=None,
):
    """Run simple GEMM: C = A @ B^T.

    This function supports non-aligned M, N, K dimensions.
    - M and N: handled by kernel boundary checks on output stores
    - K: padded to multiple of tile_k to ensure correct computation

    Args:
        M, N, K: Matrix dimensions (A[M,K], B[N,K], C[M,N]).
        tile_m, tile_n, tile_k: Tile sizes.
        in_dtype: Input data type ("bf16" or "fp16").
        A: Optional input tensor A[M,K]. If None, creates random tensor.
        B: Optional input tensor B[N,K]. If None, creates random tensor.

    Returns:
        C: Output tensor C[M,N].
    """
    import torch

    # Determine torch dtype
    if in_dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    device = "cuda"

    # Create input tensors if not provided
    if A is None:
        A = torch.randn(M, K, dtype=torch_dtype, device=device)
    if B is None:
        B = torch.randn(N, K, dtype=torch_dtype, device=device)

    # Ensure inputs are contiguous and on correct device
    A = A.contiguous().to(device=device, dtype=torch_dtype)
    B = B.contiguous().to(device=device, dtype=torch_dtype)

    # Pad all dimensions to tile sizes for kernel execution
    M_pad = _align_up(M, tile_m)
    N_pad = _align_up(N, tile_n)
    K_pad = _align_up(K, tile_k)

    # Create padded tensors
    A_pad = torch.zeros(M_pad, K_pad, dtype=torch_dtype, device=device)
    B_pad = torch.zeros(N_pad, K_pad, dtype=torch_dtype, device=device)
    C_pad = torch.zeros(M_pad, N_pad, dtype=torch_dtype, device=device)
    
    # Copy original data to padded tensors
    A_pad[:M, :K] = A
    B_pad[:N, :K] = B

    # Compile and run kernel
    exe = compile_simple_gemm(
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        in_dtype=in_dtype,
    )

    # Flatten tensors for kernel interface
    A_flat = A_pad.view(-1)
    B_flat = B_pad.view(-1)
    C_flat = C_pad.view(-1)

    # Pass both padded and original dimensions
    exe(C_flat, A_flat, B_flat, M_pad, N_pad, K_pad, M, N)
    torch.cuda.synchronize()

    # Extract the actual output (only the M×N portion)
    C = C_pad[:M, :N].contiguous()

    return C
