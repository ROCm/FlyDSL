"""Preshuffle GEMM kernel implementations (FLIR MFMA FP8/INT8).

This module intentionally contains the **kernel builder code** for the preshuffle GEMM,
extracted from `tests/kernels/test_preshuffle_gemm.py` in the same style as
`kernels/moe_gemm_2stage.py`:
- `kernels/` holds the implementation (compile functions)
- `tests/` holds correctness/perf harnesses

Pipelines:
- `pingpong`: tuned 2-stage pipeline with ping-pong LDS for A (2 LDS buffers)
- `ck_v1_single_lds`: CK-like Intrawave + bpreshuffle v1 spirit (single LDS buffer for A)
"""

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator, SmemPtr

from _mlir import ir

from flydsl.dialects.ext import arith, gpu, buffer_ops, vector, rocdl
from flydsl.lang.ir.types import T, memref
from kernels.kernels_common import stream_ptr_to_async_token
from flydsl.compiler.compiler import _apply_waves_per_eu_hint

from kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
    MfmaPipeline,
    EpilogPipeline,
    mfma_pipeline_dicts,
    PreshufflePipelineManager,
    block_mfma_block_scale_f8f6f4,
    block_mfma_PTPC_f8f6f4,
    block_mfma_16x16,
)
from kernels.mfma_epilogues import mfma_epilog


def compile_preshuffle_gemm(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp8",
    b_dtype: str = "fp8",
    out_dtype: str = "fp16",
    lds_stage: int = 2,
    # Epilogue options
    use_cshuffle_epilog: bool = False,
    # Occupancy control
    waves_per_eu: int = None,
    # Memory transfer options
    use_async_copy: bool = False,
):
    """Compile the preshuffle GEMM kernel and return the compiled executable.

    Args:
        M, N, K: GEMM sizes (A[M,K], B[N,K], C[M,N]).
        tile_m, tile_n, tile_k: block tile sizes.
        a_dtype:
          - "fp8": A/B are fp8 (1B/elem)
          - "int8": A/B are int8 (1B/elem)
          - "a8w4": W4A8 path: A is int8 (1B/elem).
        b_dtype:
          - "fp8": A/B are fp8 (1B/elem)
          - "int8": A/B are int8 (1B/elem)
          - "int4": W4A8 path: B is packed int4 (2 values per byte) and unpacked to int8 in-kernel.
        out_dtype:
          - "fp16": Output is fp16 (2B/elem)
          - "bf16": Output is bf16 (2B/elem)
          - "f32": Output is f32 (4B/elem)
        lds_stage: 
          - 2: ping-pong LDS for A (2 LDS buffers), tuned schedule (original).
          - 1: single LDS buffer for A .
        use_cshuffle_epilog: Enable cross-shuffle epilogue for better cache behavior.
        waves_per_eu: Number of wavefronts per execution unit (occupancy hint).
          - None: No hint, use default occupancy (default)
          - 1-4: Limit occupancy to improve register/LDS availability per wave
          Higher values increase latency hiding but reduce per-wave resources.
          Common values: 1 (max resources), 2 (balanced), 4 (max occupancy).
        use_async_copy: Use async DMA for A tile global-to-LDS transfer.
    """

    total_threads = 256

    pipeline_manager = PreshufflePipelineManager(a_dtype, b_dtype, out_dtype)
    pipeline_manager.check_type_valid()

    epilog_pipeline = pipeline_manager.get_epilog_pipeline()
    mfma_pipeline = pipeline_manager.get_mfma_pipeline()
    mfma_fn = pipeline_manager.get_mfma_fn()

    a_elem_bytes = pipeline_manager.get_a_elem_bytes()
    b_elem_bytes = pipeline_manager.get_b_elem_bytes()

    a_elem_pack = pipeline_manager.a_elem_pack
    b_elem_pack = pipeline_manager.b_elem_pack

    # Pipeline is byte-addressed along K (16B loads, XOR16 swizzle in bytes).
    # For fp16/bf16 (2B/elem), user passes tile_k halved so b_tile_k_bytes stays constant.
    b_tile_k_bytes = int(tile_k) * int(b_elem_bytes)

    # K64-byte micro-step wrapper uses 2x "half-pack" MFMA.
    # - fp8/i8: mfma K32, wrapper covers 64B (=64 elems)
    # - fp16/bf16: mfma K16, wrapper covers 64B (=32 elems)  -> effective K halves
    if (b_tile_k_bytes % 64) != 0:
        raise ValueError(
            f"b_tile_k_bytes must be divisible by 64, got b_tile_k_bytes={b_tile_k_bytes} "
            f"(tile_k={tile_k}, b_elem_bytes={b_elem_bytes})"
        )

    # For FP4: A is packed (2 elems/byte), so actual bytes = tile_k * a_elem_bytes / a_elem_pack
    a_tile_k_bytes = int(tile_k) * int(a_elem_bytes) // a_elem_pack

    gpu_arch = get_hip_arch()
    if use_async_copy and gpu_arch != "gfx950":
        raise NotImplementedError(f"Async_copy is only supported on gfx950, but got '{gpu_arch}'")

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name = "smem_pong")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name = "smem_ping")
    _state = {}

    DYN = ir.ShapedType.get_dynamic_size()

    # a is copied by the whole block, so we need to calculate the bytes per thread.
    a_bytes_per_thread = pipeline_manager.get_a_bytes_per_thread(tile_m, tile_k)

    # CK-style LDS128: stride is in BYTES along K (for XOR16 swizzle).
    # a_tile_k_bytes already accounts for packing, so no further division needed
    lds_stride_bytes = a_tile_k_bytes

    def _get_mfma_dict_value(key, pipeline):
        value = mfma_pipeline_dicts[key][pipeline]
        return value() if callable(value) else value

    def _a_elem_type():
        return _get_mfma_dict_value("a_elem_type", mfma_pipeline)
    def _b_elem_type():
        return _get_mfma_dict_value("b_elem_type", mfma_pipeline)
    def _scale_elem_type():
        return _get_mfma_dict_value("scale_elem_type", mfma_pipeline)
    def _out_elem_type():
        return _get_mfma_dict_value("out_elem_type", epilog_pipeline)
    def _a_vec16_type():
        return _get_mfma_dict_value("a_vec16_type", mfma_pipeline)
    def _b_vec16_type():
        return _get_mfma_dict_value("b_vec16_type", mfma_pipeline)
    def _mfma_input_pack_ty():
        return _get_mfma_dict_value("mfma_input_pack_ty", mfma_pipeline)
    def _mfma_output_pack_ty():
        return _get_mfma_dict_value("mfma_output_pack_ty", mfma_pipeline)

    is_f16_or_bf16 = mfma_pipeline in [MfmaPipeline.F16F16_16x16_PIPELINE,
                                       MfmaPipeline.BF16BF16_16x16_PIPELINE]
    is_int4 = mfma_pipeline in [MfmaPipeline.I8I4_16x16_PIPELINE]
    is_int8 = mfma_pipeline in [MfmaPipeline.I8I8_16x16_PIPELINE]
    is_fp4 = mfma_pipeline in [MfmaPipeline.F4F4_MXFP4_PIPELINE,
                               MfmaPipeline.F8F4_MXFP4_PIPELINE]

    no_epilogue_dequant = is_fp4 or is_f16_or_bf16

    # 350 16x16x128 adtype(cbsz) & bdtype(blgp)
    cbsz = 4 if mfma_pipeline == MfmaPipeline.F4F4_MXFP4_PIPELINE else 0
    blgp = 4 if mfma_pipeline in [MfmaPipeline.F4F4_MXFP4_PIPELINE, MfmaPipeline.F8F4_MXFP4_PIPELINE] else 0

    pack_M = 2
    pack_N = 2
    pack_K = 2

    use_mfma_scale_128 = (
        str(gpu_arch).startswith("gfx95")
        and (not is_int8)
        and (not is_int4)
        and (not is_f16_or_bf16)
    )

    # GEMM epilogue toggle: optional LDS CShuffle + vectorized stores.
    # Default: off (current measured cases show no benefit).
    module_name = f"mfma_preshuffle_{lds_stage}stages_{mfma_pipeline}_{epilog_pipeline}".replace("-", "_")

    class _GEMM(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # LDS scratch:
            # - A tiles (fp8/int8): tile_m * lds_stride bytes per buffer
            # - optional CShuffle output tile (fp16): tile_m * tile_n * 2 bytes
            #
            # For 2-stage pipeline: create separate ping/pong allocations for no-alias optimization
            # For 1-stage pipeline: single buffer allocation
            #
            # IMPORTANT: When CShuffle is enabled, lds_out REUSES one of the A buffers (they are
            # dead after mainloop). This saves LDS memory.

            lds_tile_bytes = int(tile_m) * int(lds_stride_bytes)
            lds_out_bytes = 2 * int(tile_m) * int(tile_n) if use_cshuffle_epilog else 0
<<<<<<< HEAD
            lds_total_bytes = max(lds_a_bytes, lds_out_bytes)
            # Keep LDS allocation sized in bytes: element_size * num_elems.
            # Allocate element type == _elem_type() and scale element count accordingly.
            a_lds_total_elems = lds_total_bytes // a_elem_bytes
            _state["lds_a_decl"] = allocator.allocate_array(_a_elem_type(), a_lds_total_elems)
            allocator.finalize()
=======

            if int(lds_stage) == 2:
                # Separate ping/pong buffers for no-alias guarantee
                # Size each buffer to fit max(A_tile, CShuffle_output) to enable reuse
                assert lds_out_bytes % 2 == 0, "lds_out_bytes should be mutiple of 2"
                buffer_size_bytes = max(lds_tile_bytes, lds_out_bytes // lds_stage)
                # Convert bytes to element count based on element size
                buffer_size_elems = buffer_size_bytes if elem_bytes == 1 else (buffer_size_bytes // 2)
                _state["lds_a_pong"] = allocator_pong.allocate_array(_elem_type(), buffer_size_elems)
                _state["lds_a_ping"] = allocator_ping.allocate_array(_elem_type(), buffer_size_elems)
            else:
                # Single buffer for 1-stage pipeline
                lds_total_bytes = max(lds_tile_bytes, lds_out_bytes)
                lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)
                _state["lds_a_decl"] = allocator_pong.allocate_array(_elem_type(), lds_total_elems)

            allocator_pong.finalize()
            allocator_ping.finalize()
>>>>>>> main

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_elem_type()),
            arg_a: lambda: memref(DYN, _a_elem_type()),
            arg_b: lambda: memref(DYN, _b_elem_type()),
            arg_scale_a: lambda: memref(DYN, _scale_elem_type()),
            arg_scale_b: lambda: memref(DYN, _scale_elem_type()),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            # ---- Types ----
            # NOTE: Some environments have multiple `flydsl` builds on PYTHONPATH.
            # Use explicit MLIR Values (not Python ints / wrapper objects) for ROCDL ops.
            acc_init = arith.unwrap(arith.constant_vector(0, _mfma_output_pack_ty()))

            # Layouts
            layout_c = flir.make_layout((c_m, c_n), stride=(c_n, 1))

            # A uses dword indexing (buffer-load dwordx4). Convert element index -> dword index:
            #   dword_index = (elem_index * elem_bytes) / 4
<<<<<<< HEAD
            c_k_div4bytes = c_k * a_elem_bytes / 4 / a_elem_pack
=======
            # Also create byte-indexed layout for DMA path.
            if (int(elem_bytes) == 2):
                c_k_div4bytes = (c_k * 2) / 4
                c_k_bytes = c_k * 2
            else:
                c_k_div4bytes = c_k / 4
                c_k_bytes = c_k
            layout_a = flir.make_layout((c_m, c_k_bytes), stride=(c_k_bytes, 1))
>>>>>>> main
            layout_a_div4 = flir.make_layout((c_m, c_k_div4bytes), stride=(c_k_div4bytes, 1))

            # B preshuffle layout (shared with MoE kernels).
            # For FP4: B is packed (2 elems/byte), so adjust c_k accordingly
            b_kpack_bytes = 16
            c_k_b = c_k // b_elem_pack
            layout_b = make_preshuffle_b_layout(
                flir, arith, c_n=c_n, c_k=c_k_b, kpack_bytes=b_kpack_bytes, elem_bytes=b_elem_bytes
            ).layout_b

            # Scale layouts for FP4/MXFP4 (block-scale MFMA).
            layout_a_scale = make_preshuffle_scale_layout(flir, arith, c_mn=c_m, c_k=c_k) if is_fp4 else None
            layout_b_scale = make_preshuffle_scale_layout(flir, arith, c_mn=c_n, c_k=c_k) if is_fp4 else None

            # LDS layout is element-indexed, but XOR16 swizzle is byte-based.
            # Represent LDS as (tile_m, tile_k) in elements and scale swizzle math by elem_bytes.
            # For FP4: A is packed (2 elems/byte), so LDS K dimension is tile_k / a_elem_pack
            lds_tile_k = tile_k // a_elem_pack if is_fp4 else tile_k
            shape_lds = flir.make_shape(tile_m, lds_tile_k)
            stride_lds = flir.make_stride(lds_tile_k, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            # CK-style XOR16 swizzle parameter (const).
            a_k_blocks16 = arith.index(a_tile_k_bytes // 16)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")

<<<<<<< HEAD
            base_ptr = allocator.get_base()
            lds_a_ptr = _state["lds_a_decl"](base_ptr)
            lds_a = lds_a_ptr.get()
            lds_out = (
                SmemPtr(base_ptr, lds_a_ptr.byte_offset, _out_elem_type(), shape=(tile_m * tile_n,)).get()
                if use_cshuffle_epilog
                else None
            )
=======
            base_ptr, base_ptr1 = allocator_pong.get_base(), allocator_ping.get_base()

            # Get LDS pointers based on pipeline stage
            if lds_stage == 2:
                # Separate ping/pong buffers - compiler knows they don't alias!
                lds_a_pong_ptr = _state["lds_a_pong"](base_ptr)
                lds_a_ping_ptr = _state["lds_a_ping"](base_ptr1)

                # View as fp8/int8 arrays for A tiles
                # The buffer size is max(tile_bytes, out_bytes) but we just need tile_m*tile_k elements
                lds_a_pong = SmemPtr(
                    base_ptr, lds_a_pong_ptr.byte_offset, _elem_type(), shape=(tile_m * tile_k,)
                ).get()
                lds_a_ping = SmemPtr(
                    base_ptr1, lds_a_ping_ptr.byte_offset, _elem_type(), shape=(tile_m * tile_k,)
                ).get()

                # CShuffle output buffer: REUSE pong buffer (A tiles are dead after mainloop)
                if use_cshuffle_epilog:
                    lds_out = SmemPtr(
                        base_ptr, lds_a_pong_ptr.byte_offset, T.f16, shape=(tile_m * tile_n,)
                    ).get()
                else:
                    lds_out = None
            else:
                # Single buffer for 1-stage pipeline
                lds_a_ptr = _state["lds_a_decl"](base_ptr)
                lds_a_pong = lds_a_ptr.get()
                lds_a_ping = lds_a_pong  # Reuse same buffer
                lds_out = (
                    SmemPtr(base_ptr, lds_a_ptr.byte_offset, T.f16, shape=(tile_m * tile_n,)).get()
                    if use_cshuffle_epilog
                    else None
                )
>>>>>>> main

            # Note: We assume N is aligned (no N-tail support in this kernel).
            a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=False)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False)
            scale_a_rsrc = None if is_f16_or_bf16 else buffer_ops.create_buffer_resource(arg_scale_a, max_size=False)

            b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
            scale_b_rsrc = None if is_f16_or_bf16 else buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

            bx_m = bx * tile_m
            by_n = by * tile_n

            # (thread_id.x) -> (wave_id, lane_id) via FLIR.
            layout_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            coord_wave_lane = flir.idx2crd(tx, layout_wave_lane)
            wave_id = flir.get(coord_wave_lane, 0)
            lane_id = flir.get(coord_wave_lane, 1)

            # lane_id -> (lane_div_16, lane_mod_16) via FLIR.
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            coord_lane16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_lane16, 0)
            lane_mod_16 = flir.get(coord_lane16, 1)

            row_a_lds = lane_mod_16
            # Per-`k1` (KLane) base offset along K inside a 64B K0 block.
            #
            # CK preshuffle uses KPackBytes=16 across dtypes, but KPackElems differs:
            # - fp8/int8: 16 elems (1B)
            # - fp16/bf16: 8 elems (2B)
            #
            # We express `col_offset_base` in *elements*.
            kpack_elems = 16
            a_kpack_elems = kpack_elems / a_elem_bytes
            col_offset_base = lane_div_16 * arith.constant(int(a_kpack_elems), index=True)
            # `col_offset_base` is in element units (multiples of 16). We do LDS swizzle/math
            # in bytes, so scale by element size for fp16/bf16.
            col_offset_base_bytes = (
                col_offset_base * arith.constant(int(a_elem_bytes), index=True)
            )

            m_repeat = tile_m // 16
            # K stepping is byte-addressed:
            # - For FP4/MXFP4 (mfma 16x16x128): one micro-step is 128 bytes
            # - For FP8/INT8 (mfma 16x16x32): one micro-step is 64 bytes (K32 x 2)
            if is_fp4:
                k_unroll = b_tile_k_bytes // 64 // b_elem_pack
            else:
                k_unroll = b_tile_k_bytes // 64

            # --- Dynamic tiling along N (4 waves) ---
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16

            c_n_per_wave = arith.constant(n_per_wave, index=True)
            n_tile_base = wave_id * c_n_per_wave

            # Decompose global_n -> (n_blk, n_intra) once per ni.
            c_n0 = c_n / 16
            layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
            n_intra_list = []
            n_blk_list = []
            for i in range_constexpr(num_acc_n):
                offset = i * 16
                c_offset = arith.constant(offset, index=True)
                global_n = by_n + n_tile_base + c_offset + lane_mod_16
                coord_n = flir.idx2crd(global_n, layout_n_blk_intra)
                n_blk_list.append(flir.get(coord_n, 0))
                n_intra_list.append(flir.get(coord_n, 1))

            # FP4/MXFP4 pack parameters for block-scale MFMA.
            k_unroll_packed = k_unroll // pack_K if is_fp4 else 0
            m_repeat_packed = m_repeat // pack_M if is_fp4 else 0
            num_acc_n_packed = num_acc_n // pack_N if is_fp4 else 0

            # --- Scale load logic for FP4/MXFP4 ---
            def load_scale(arg_scale, rsrc, layout, ku, mni):
                """Load a single scale value for FP4/MXFP4 block-scale MFMA."""
                k_lane = lane_div_16
                n_lane = lane_mod_16
                coord_pack = flir.make_coord(mni, ku, k_lane, n_lane)
                idx_pack = flir.crd2idx(coord_pack, layout)
                scale_view = flir.TensorView(
                    arg_scale,
                    (1,),
                    strides=(1,),
                    base_indices=(idx_pack,),
                    element_type=_scale_elem_type(),
                )
                scale = flir.copy(
                    flir.make_copy_atom(_scale_elem_type(), vector_size=1),
                    scale_view,
                    None,
                    alignment=8,
                    return_vector=True,
                    src_buffer_resource=rsrc,
                    src_buffer_offset_in_bytes=False,
                )
                return scale

            def load_b_scale_tile(base_k):
                """Load B scale tile for FP4/MXFP4."""
                b_scale_tile = []
                for ku in range_constexpr(k_unroll_packed):
                    for ni in range_constexpr(num_acc_n_packed):
                        scale = load_scale(
                            arg_scale_b,
                            scale_b_rsrc,
                            layout_b_scale,
                            ku + base_k,
                            ni + (by_n + n_tile_base) // pack_N // 16,
                        )
                        b_scale_tile.append(scale)
                return b_scale_tile

            def load_a_scale_tile(base_k):
                """Load A scale tile for FP4/MXFP4."""
                a_scale_tile = []
                for ku in range_constexpr(k_unroll_packed):
                    for mi in range_constexpr(m_repeat_packed):
                        scale = load_scale(
                            arg_scale_a,
                            scale_a_rsrc,
                            layout_a_scale,
                            ku + base_k,
                            mi + bx_m // pack_M // 16,
                        )
                        a_scale_tile.append(scale)
                return a_scale_tile

            def prefetch_ab_scale_tile(base_k):
                """Prefetch A and B scale tiles for FP4/MXFP4."""
                if not is_fp4:
                    return None, None
                # return load_a_scale_tile(base_k), load_b_scale_tile(base_k)
                return load_a_scale_tile(0), load_b_scale_tile(0)

            # --- B load logic ---
            # Shared loader supports:
            # - FP8/INT8: explicit 16B load (one full KPack) + extract 8B for this micro-step
            # - INT4 (W4A8): 4B load + 7-op unpack to 8B (no v_perm)
            def load_b_pack(base_k, ki_step, ni):
                return load_b_pack_k32(
                    buffer_ops,
                    flir,
                    arith,
                    vector,
                    arg_b=arg_b,
                    b_rsrc=b_rsrc,
                    layout_b=layout_b,
                    base_k=base_k,
                    ki_step=ki_step,
                    n_blk=n_blk_list[ni],
                    n_intra=n_intra_list[ni],
                    lane_div_16=lane_div_16,
                    elem_type=_b_elem_type(),
                    kpack_bytes=b_kpack_bytes,
                    elem_bytes=b_elem_bytes,
                    unpack_int4=is_int4,
                )

            # For FP8/INT8 we can load one 16B pack and extract both 8B halves (K64 bytes).
            # For INT4 (packed), reuse the existing K32 loader twice (2x4B loads + unpack).
            atom_b_g2r16 = flir.make_copy_atom(_b_elem_type(), vector_size=16)
            c64_b = 64
            c0_idx = 0

            def load_b_packs_k64(base_k, ku: int, ni: int):
                if is_int4:
                    ki0 = (ku * 2) + 0
                    ki1 = (ku * 2) + 1
                    return load_b_pack(base_k, ki0, ni), load_b_pack(base_k, ki1, ni)

                # FP8/INT8/FP16/BF16: load 16 bytes (one full KPack).
                base_k_bytes = base_k * arith.constant(int(b_elem_bytes), index=True)
                k0_base = base_k_bytes / c64_b
                k0 = k0_base + ku
                k1 = lane_div_16
                coord_pack = flir.make_coord(n_blk_list[ni], k0, k1, n_intra_list[ni], c0_idx)
                idx_pack = flir.crd2idx(coord_pack, layout_b)
                vec_elems = kpack_elems / b_elem_bytes

                b_view = flir.TensorView(
                    arg_b,
                    (vec_elems,),
                    strides=(1,),
                    base_indices=(idx_pack,),
                    element_type=_b_elem_type(),
                )
                b16 = flir.copy(
                    flir.make_copy_atom(_b_elem_type(), vector_size=vec_elems),
                    b_view,
                    None,
                    alignment=8,
                    return_vector=True,
                    src_buffer_resource=(b_rsrc if b_elem_bytes == 1 else None),
                    src_buffer_offset_in_bytes=(b_elem_bytes == 1),
                )

                # Split 16B pack into two 8B halves.
                b_i64x2 = vector.bitcast(T.i64x2, b16)
                b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
                b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])

                if not is_f16_or_bf16:
                    return b0_i64, b1_i64

                # For fp16/bf16 MFMA 16x16x16, operands are 8B packs:
                # - fp16: v4f16
                # - bf16: v4i16 (bit pattern) for *_bf16_1k
                vec1_i64_ty = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                b0_v1 = vector.from_elements(vec1_i64_ty, [b0_i64])
                b1_v1 = vector.from_elements(vec1_i64_ty, [b1_i64])
                return vector.bitcast(_mfma_input_pack_ty(), b0_v1), vector.bitcast(_mfma_input_pack_ty(), b1_v1)

            def load_b_tile(base_k):
                # b_tile[ku] = (packs_half0[ni], packs_half1[ni])
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

            def lds_load_16b(curr_row_a_lds, col_base, lds_buffer):
                # Swizzle in bytes, then convert to element offset for memref indexing.
                col_base_swz_bytes = flir.swizzle_xor16(curr_row_a_lds, col_base, a_k_blocks16)
                col_base_swz = col_base_swz_bytes if a_elem_bytes == 1 else (col_base_swz_bytes / 2)
                coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swz)
                idx_a16 = flir.crd2idx(coord_a16, layout_lds)
<<<<<<< HEAD
                idx_a16 = idx_a16 + lds_base
                return vector.load_op(_a_vec16_type(), lds_a, [idx_a16])
=======
                return vector.load_op(_vec16_type(), lds_buffer, [idx_a16])
>>>>>>> main

            # --- A LDS load helper for K64-bytes (load 16B once, extract 2x i64 halves) ---
            def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
                loaded_a16 = lds_load_16b(curr_row_a_lds, col_base, lds_buffer)
                a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
                a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])

                if not is_f16_or_bf16:
                    return a0_i64, a1_i64

                vec1_i64_ty = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                a0_v1 = vector.from_elements(vec1_i64_ty, [a0_i64])
                a1_v1 = vector.from_elements(vec1_i64_ty, [a1_i64])
                return vector.bitcast(_mfma_input_pack_ty(), a0_v1), vector.bitcast(_mfma_input_pack_ty(), a1_v1)

            # --- A load/store (16B chunks), XOR16 swizzle ---
<<<<<<< HEAD
            a_load_bytes = 16
            num_a_loads = a_bytes_per_thread // a_load_bytes
=======
            # Original register-based approach (commented out, kept for reference)
            num_a_loads = bytes_per_thread_a // a_load_bytes
>>>>>>> main
            # A tile mapping in dwords along K:
            #   tile_k_dwords = (tile_k * elem_bytes) / 4
            # For FP4: A is packed (2 elems/byte), so divide by a_elem_pack
            if is_fp4:
                a_tile_k_dwords = tile_k * a_elem_bytes // 4 // a_elem_pack
            else:
                a_tile_k_dwords = tile_k * a_elem_bytes // 4
            layout_a_tile_div4 = flir.make_layout((tile_m, a_tile_k_dwords), stride=(a_tile_k_dwords, 1))
            c4 = arith.constant(4, index=True)
            tx_i32_base = tx * c4
            atom_a_g2r16 = flir.make_copy_atom(_a_elem_type(), vector_size=a_kpack_elems)

            def load_a_16(idx_elem):
                return buffer_copy_gmem16_dwordx4(
                    flir,
                    arg=arg_a,
                    elem_type=_a_elem_type(),
                    idx_i32=idx_elem,
                    atom_g2r16=atom_a_g2r16,
                    rsrc=a_rsrc,
                    vec_elems=a_kpack_elems,
                )

            def a_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(
                    flir,
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_a_tile_div4,
                )

            # Original register-based load/store (kept for reference)
            def load_a_tile(base_k_div4):
                parts = []
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    row_a_global = bx_m + row_a_local
                    coord_a_g = flir.make_coord(row_a_global, base_k_div4 + col_a_local_i32)
                    idx_i32 = flir.crd2idx(coord_a_g, layout_a_div4)
                    # `idx_i32` is a dword offset. For 2B element types (fp16/bf16),
                    # convert to element offset so the generic `vector.load` path reads
                    # the right address (FLIR only specializes buffer_load_dwordx4 for 1B types).
                    idx_elem = idx_i32 * arith.constant(a_elem_bytes, index=True)

                    a_16B = load_a_16(idx_elem)
                    parts.append(vector.bitcast(T.i32x4, a_16B))
                return parts

            def store_a_tile_to_lds(vec_a_parts, lds_buffer):
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    lds_store_16b_xor16(
                        flir,
                        arith,
                        vector,
<<<<<<< HEAD
                        lds_memref=lds_a,
                        vec16_ty=_a_vec16_type(),
                        elem_type=_a_elem_type(),
=======
                        lds_memref=lds_buffer,
                        vec16_ty=_vec16_type(),
                        elem_type=_elem_type(),
>>>>>>> main
                        atom_s16=atom_a_g2r16,
                        layout_lds=layout_lds,
                        row_local=row_a_local,
                        col_local_i32=col_a_local_i32,
                        tx_c4=c4,
<<<<<<< HEAD
                        k_blocks16=a_k_blocks16,
                        lds_base=lds_base,
=======
                        k_blocks16=k_blocks16,
                        lds_base=arith.constant(0, index=True),
>>>>>>> main
                        vec_part_i32x4=vec_a_parts[i],
                        elem_bytes=a_elem_bytes,
                    )

            def prefetch_ab_tile(base_k):
<<<<<<< HEAD
                # Convert element index to byte index, then to dword index.
                base_k_bytes = base_k * arith.constant(int(a_elem_bytes), index=True)
                base_k_div4 = base_k_bytes / 4
                if is_fp4:
                    # For FP4/MXFP4: A and B are packed (2 elems/byte), need to adjust offsets
                    a_regs = load_a_tile(base_k_div4 // a_elem_pack)
                    b_regs = load_b_tile(base_k // b_elem_pack)
                else:
                    a_regs = load_a_tile(base_k_div4)
                    b_regs = load_b_tile(base_k)
                return a_regs, b_regs

            def compute_tile(accs_in, b_tile_in, lds_base, *, is_last_tile=False, a0_prefetch=None, a_scale=None, b_scale=None):
=======
                a_regs = prefetch_a_tile(base_k)
                b_regs = prefetch_b_tile(base_k)
                return a_regs, b_regs

            def prefetch_a_tile(base_k):
                base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
                base_k_div4 = base_k_bytes / 4
                return load_a_tile(base_k_div4)

            def prefetch_b_tile(base_k):
                return load_b_tile(base_k)

            # DMA async version: direct global-to-LDS transfer
            def dma_a_tile_to_lds(base_k_div4, lds_buffer):
                from _mlir.dialects import llvm, memref as memref_dialect

                # GFX942 doesn't support 16B buffer_load_lds, use 4B per operation
                # GFX950+ (MI355) supports 16B
                use_16b_dma = str(gpu_arch).startswith("gfx95")
                dma_bytes = 16 if use_16b_dma else 4

                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    col_a_local_sw = flir.swizzle_xor16(row_a_local, col_a_local_i32 * c4, k_blocks16)
                    row_a_global = bx_m + row_a_local
                    coord_a_g = flir.make_coord(row_a_global, base_k_div4 * c4 + col_a_local_sw)
                    global_offset = arith.index_cast(T.i32, flir.crd2idx(coord_a_g, layout_a))

                    if i == 0:
                        col_local_bytes = (col_a_local_i32 * c4)
                        coord_lds = flir.make_coord(row_a_local, col_local_bytes)
                        idx_lds = flir.crd2idx(coord_lds, layout_lds)

                        # Get LDS pointer from the specific buffer (no offset needed - separate allocations)
                        lds_addr = memref_dialect.extract_aligned_pointer_as_index(lds_buffer) + idx_lds * elem_bytes
                        lds_ptr_i64_lane0 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                    else:
                        lds_ptr_i64_lane0 += 1024 * 4
                    lds_ptr = buffer_ops.create_llvm_ptr(lds_ptr_i64_lane0, address_space=3)

                    # DMA from global to LDS using buffer_load_lds
                    size_i32 = arith.constant(dma_bytes, type=T.i32)
                    soffset = arith.constant(0, type=T.i32)
                    offset_imm = arith.constant(0, type=T.i32)
                    aux = arith.constant(1, type=T.i32)

                    rocdl.raw_ptr_buffer_load_lds(
                        a_rsrc,
                        lds_ptr,
                        arith.unwrap(size_i32),
                        arith.unwrap(global_offset),
                        arith.unwrap(soffset),
                        arith.unwrap(offset_imm),
                        arith.unwrap(aux),
                    )

            def prefetch_a_to_lds(base_k, lds_buffer):
                """Load A tile from global memory to LDS via DMA.

                Args:
                    base_k: Starting K coordinate
                    lds_buffer: The specific LDS buffer (lds_a_ping or lds_a_pong) to write to
                """
                base_k_div4 = base_k / 4
                dma_a_tile_to_lds(base_k_div4, lds_buffer)

            def compute_tile(accs_in, b_tile_in, lds_buffer, *, is_last_tile=False, a0_prefetch=None):
>>>>>>> main
                scales_pf = {}

                mfma_res_ty = _mfma_output_pack_ty()
                if is_last_tile and (not no_epilogue_dequant):
                    # Prefetch scales for non-FP4 scaled paths (fp8/int8/int4 with per-tensor scale).
                    s_b_vals = []
                    for ni in range_constexpr(num_acc_n):
                        offset = ni * 16
                        c_offset = arith.constant(offset, index=True)
                        col_g = by_n + n_tile_base + c_offset + lane_mod_16
                        s_b_vals.append(
                            buffer_ops.buffer_load(scale_b_rsrc, col_g, vec_width=1, dtype=T.f32)
                        )
                    scales_pf["s_b_vals"] = s_b_vals
                    scales_pf["s_a_vecs"] = []
                    row_off_base = lane_div_16 * 4
                    for mi in range_constexpr(m_repeat):
                        row_base_m = bx_m + (mi * 16)
                        row_g_base = row_base_m + row_off_base
                        s_a_vec = buffer_ops.buffer_load(
                            scale_a_rsrc, row_g_base, vec_width=4, dtype=T.f32
                        )
                        scales_pf["s_a_vecs"].append(vector.bitcast(T.f32x4, s_a_vec))

                current_accs_list = list(accs_in)
<<<<<<< HEAD
                if is_fp4:
                    # FP4/MXFP4 path: use block-scale MFMA with per-block scales.
                    current_accs_list = block_mfma_block_scale_f8f6f4(
                        current_accs_list,
                        b_tile_in,
                        a_scale,
                        b_scale,
                        lds_base,
                        lds_load_packs_k64=lds_load_packs_k64,
                        col_offset_base_bytes=col_offset_base_bytes,
                        row_a_lds=row_a_lds,
                        mfma_fn=rocdl.mfma_scale_f32_16x16x128_f8f6f4,
                        mfma_res_ty=mfma_res_ty,
                        cbsz=cbsz,
                        blgp=blgp,
                        a_elem_vec_pack=a_elem_pack,
                        k_unroll=k_unroll,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        pack_K=pack_K,
                        pack_M=pack_M,
                        pack_N=pack_N,
                        a0_prefetch=a0_prefetch,
                    )
=======

                # ---------------- gfx95 fast path (K128 MFMA scale) ----------------
                # This is the key optimization from `zhimding/develop_0107` for FP8:
                # use mfma.scale 16x16x128 to reduce instruction count in the hot loop.
                #
                # Notes:
                # - Only valid for fp8 path (not int8/int4) and gfx95+
                # - Requires tile_k divisible by 128
                # - mfma.scale takes 9 operands: 3 vectors + 6 i32 flags/scales.
                use_mfma_scale_128 = (
                    str(gpu_arch).startswith("gfx95")
                    and (not is_int8)
                    and (not is_int4)
                    and (not is_f16_or_bf16)
                )
                if use_mfma_scale_128:
                    if (int(tile_k) % 128) != 0:
                        raise ValueError(
                            f"tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_k}"
                        )

                    mfma_res_ty = T.f32x4
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)
                    # flir.print(f"k_unroll = {k_unroll}\n")
                    for ku128 in range_constexpr(k_unroll // 2):
                        ku0 = ku128 * 2
                        ku1 = ku0 + 1

                        b0_packs0, b0_packs1 = b_tile_in[ku0]
                        b1_packs0, b1_packs1 = b_tile_in[ku1]

                        col_base0 = col_offset_base_bytes + (ku0 * 64)
                        col_base1 = col_offset_base_bytes + (ku1 * 64)

                        for mi in range_constexpr(m_repeat):
                            mi_val = arith.constant(mi * 16, index=True)
                            curr_row_a_lds = row_a_lds + mi_val

                            # if (mi == 0) and bx == 0 and by == 0:
                            #     flir.printf("mfma_scale_128: ku0=%d, ku1=%d, col_base0=%d, col_base1=%d\n", ku0, ku1, col_base0, col_base1)

                            if (a0_prefetch is not None) and (ku0 == 0) and (mi == 0):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_buffer)
                            a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, lds_buffer)
                            a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                            for ni in range_constexpr(num_acc_n):
                                b0 = b0_packs0[ni]
                                b1 = b0_packs1[ni]
                                b2 = b1_packs0[ni]
                                b3 = b1_packs1[ni]
                                b128 = pack_i64x4_to_i32x8(b0, b1, b2, b3)

                                acc_idx = mi * num_acc_n + ni
                                current_accs_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    mfma_res_ty,
                                    [
                                        a128,
                                        b128,
                                        current_accs_list[acc_idx],
                                        # cbsz, abid, blgp: 0
                                        0,
                                        0,
                                        0,
                                        # op_sel_a + scale_a (1.0f as i32 bits)
                                        0x3F800000,
                                        # op_sel_b + scale_b (1.0f as i32 bits)
                                        0,
                                        0x3F800000,
                                    ],
                                )
>>>>>>> main
                    return current_accs_list, scales_pf

                if use_mfma_scale_128:
                    current_accs_list = block_mfma_PTPC_f8f6f4(
                        current_accs_list,
                        b_tile_in,
                        lds_base,
                        col_offset_base_bytes=col_offset_base_bytes,
                        row_a_lds=row_a_lds,
                        lds_load_packs_k64=lds_load_packs_k64,
                        mfma_fn=rocdl.mfma_scale_f32_16x16x128_f8f6f4,
                        mfma_res_ty=mfma_res_ty,
                        k_unroll=k_unroll,
                        num_acc_n=num_acc_n,
                        m_repeat=m_repeat,
                        a0_prefetch=a0_prefetch,
                    )
                    return current_accs_list, scales_pf

<<<<<<< HEAD
                current_accs_list = block_mfma_16x16(
                    current_accs_list,
                    b_tile_in,
                    lds_base,
                    col_offset_base_bytes=col_offset_base_bytes,
                    row_a_lds=row_a_lds,
                    lds_load_packs_k64=lds_load_packs_k64,
                    mfma_fn=mfma_fn,
                    mfma_res_ty=mfma_res_ty,
                    k_unroll=k_unroll,
                    num_acc_n=num_acc_n,
                    m_repeat=m_repeat,
                    a0_prefetch=a0_prefetch,
                )
=======
                if is_int8:
                    mfma_fn = mfma_i32_k32
                elif is_f16:
                    # gfx942 fp16 MFMA: 16x16x16 f16 (operands are v4f16, 8B packs)
                    mfma_fn = rocdl.mfma_f32_16x16x16f16
                elif is_bf16:
                    # bf16 MFMA K16 variant uses i16 bit-pattern packs (v4i16)
                    mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
                else:
                    mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

                def mfma_step(acc_in, a, b):
                    return mfma_fn(mfma_res_ty, [a, b, acc_in, 0, 0, 0])

                # "K64-byte wrapper": two back-to-back MFMA/WMMA ops using the two 8B halves.
                def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
                    acc_mid = mfma_step(acc_in, a0, b0)
                    return mfma_step(acc_mid, a1, b1)

                for ku in range_constexpr(k_unroll):
                    b_packs0, b_packs1 = b_tile_in[ku]
                    # Byte-addressed K stepping (64B per ku).
                    ki64 = ku * 64
                    col_base = col_offset_base_bytes + ki64
                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val
                        if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                            a0, a1 = a0_prefetch
                        else:
                            a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer)
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            current_accs_list[acc_idx] = mfma_k64_bytes(
                                current_accs_list[acc_idx],
                                a0,
                                a1,
                                b_packs0[ni],
                                b_packs1[ni],
                            )
>>>>>>> main
                return current_accs_list, scales_pf

            vec1_f16 = ir.VectorType.get([1], ir.F16Type.get())

            def store_output(final_accs, scales):
                # fp16/bf16: no scale fetch, no scale multiply in epilogue.
                if no_epilogue_dequant:
                    s_b_vals = None
                    s_a_vecs = None
                else:
                    s_b_vals = scales["s_b_vals"]
                    s_a_vecs = scales["s_a_vecs"]

                if use_cshuffle_epilog:
                    if lds_out is None:
                        raise RuntimeError(
                            "use_cshuffle_epilog=True but lds_out is not allocated/aliased."
                        )
                    # We reuse the A LDS allocation as `lds_out` for the cshuffle epilogue.
                    # Add a block-wide barrier before starting to write into LDS to avoid
                    # racing with the tail of the mainloop's LDS reads (different waves can
                    # reach the epilogue at slightly different times).
                    gpu.barrier()

                    def write_row_to_lds(
                        *,
                        mi: int,
                        ii: int,
                        row_in_tile,
                        row,
                        row_base_lds,
                        col_base_local,
                        num_acc_n: int,
                        lds_out,
                    ):
                        # CShuffle write (non-pack):
                        # - Each lane computes one f16 element for its `col_local`
                        # - Write directly to LDS in row-major [tile_m, tile_n] order
                        # - The shuffle/remap happens in the LDS read phase
                        if not is_f16_or_bf16:
                            s_a_vec4 = s_a_vecs[mi]
                            s_a = vector.extract(
                                s_a_vec4, static_position=[ii], dynamic_position=[]
                            )
                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            acc = final_accs[acc_idx]
                            val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                            # if is_int8:
                            if is_int8 or is_int4:
                                val = arith.sitofp(T.f32, val)
                            if no_epilogue_dequant:
                                val_s = val
                            else:
                                val_s = (val * s_a) * s_b_vals[ni]
                            v16 = arith.trunc_f(_out_elem_type(), val_s)

                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(vec1_f16, [v16])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        # Store vector<EVecxf16> to C at (row, col_g0).
                        #
                        # IMPORTANT:
                        # RawPtrBufferStoreOp offsets are in BYTES. `buffer_ops.buffer_store()`
                        # will scale by element bytes based on the *data type*. For f16 vectors,
                        # some backends/paths can be fragile. We explicitly bitcast to i32
                        # and pass a byte offset to keep the store well-defined.
                        idx_out = flir.crd2idx(flir.make_coord(row, col_g0), layout_c)  # f16 element offset
                        byte_off = idx_out * arith.constant(2, index=True)  # bytes

                        if e_vec == 4:
                            frag_i32x2 = vector.bitcast(T.vec(2, T.i32), frag)
                            buffer_ops.buffer_store(
                                frag_i32x2, c_rsrc, byte_off, offset_is_bytes=True
                            )
                        else:
                            # e_vec == 2: pack 2xf16 -> 1xi32
                            frag_i32x1 = vector.bitcast(T.vec(1, T.i32), frag)
                            frag_i32 = vector.extract(
                                frag_i32x1, static_position=[0], dynamic_position=[]
                            )
                            buffer_ops.buffer_store(
                                frag_i32, c_rsrc, byte_off, offset_is_bytes=True
                            )

                    # Prefer 16B stores when possible:
                    # - EVec=4 => 4xf16 (8B) per store (and matches tile_n multiples of 128)
                    # - EVec=2 => 2xf16 (4B) per store (tile_n multiples of 64)
                    e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                    mfma_epilog(
                        use_cshuffle=True,
                        arith=arith,
                        vector=vector,
                        gpu=gpu,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        e_vec=e_vec,
                        m_repeat=m_repeat,
                        num_acc_n=num_acc_n,
                        tx=tx,
                        lane_div_16=lane_div_16,
                        lane_mod_16=lane_mod_16,
                        bx_m=bx_m,
                        by_n=by_n,
                        n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        write_row_to_lds=write_row_to_lds,
                        store_pair=store_pair,
                    )
                    return

                def body_row(*, mi: int, ii: int, row_in_tile, row):
                    if not no_epilogue_dequant:
                        s_a_vec4 = s_a_vecs[mi]
                        s_a = vector.extract(s_a_vec4, static_position=[ii], dynamic_position=[])
                    col_base = by_n + n_tile_base + lane_mod_16
                    idx_base = flir.crd2idx(flir.make_coord(row, col_base), layout_c)
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                        if is_int8 or is_int4:
                            # INT8/INT4 paths use i32 accumulators; convert to f32 for scaled epilogue.
                            val = arith.sitofp(T.f32, val)
                        if no_epilogue_dequant:
                            val_s = val
                        else:
                            val_s = (val * s_a) * s_b_vals[ni]
                        val_out = arith.trunc_f(_out_elem_type(), val_s)
                        idx_out = idx_base + arith.constant(ni * 16, index=True)
                        buffer_ops.buffer_store(val_out, c_rsrc, idx_out)

                mfma_epilog(
                    use_cshuffle=False,
                    arith=arith,
                    range_constexpr=range_constexpr,
                    m_repeat=m_repeat,
                    lane_div_16=lane_div_16,
                    bx_m=bx_m,
                    body_row=body_row,
                )

            # ---------------- Scheduling hints (match CK-style) ----------------
            # These sched_group_barrier hints help the backend interleave VMEM/DS/MFMA
            # similarly to CK's tuned pipelines.
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                # - MFMA group size per "slot": num_acc_n
                # - Total MFMA per tile: (2*K32 per K64) * k_unroll * m_repeat * num_acc_n
                # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                mfma_group = num_acc_n
                mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

                # DS-read preload (CK default is 2).
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                if tile_m == 16:
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma(1)
                if tile_m == 16:
                    rocdl.sched_vmem(1)
                if num_acc_n < 4:
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if tile_m == 16:
                        rocdl.sched_vmem(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if tile_m == 16:
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)

                # DS-write hints near the end: match total A LDS-store micro-ops per thread.
                dswr_tail = num_a_loads
                if dswr_tail > sche_iters:
                    dswr_tail = sche_iters
                dswr_start = sche_iters - dswr_tail

                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group)
                    if (not use_async_copy) and (sche_i >= dswr_start - 1):
                        rocdl.sched_dswr(1)
                rocdl.sched_barrier(0)

            # ---------------- Pipeline ----------------
<<<<<<< HEAD
            # LDS base offsets are in *elements* of `_elem_type()`.
            # We keep LDS laid out as (tile_m, tile_k) in element units.
            # For FP4: A is packed (2 elems/byte), so divide by a_elem_pack
            lds_tile_elems_val = tile_m * tile_k // a_elem_pack if is_fp4 else tile_m * tile_k
            lds_tile_elems = arith.constant(lds_tile_elems_val, index=True)
            lds_base0 = arith.constant(0, index=True)
            lds_base1 = lds_tile_elems

=======
            # LDS tile size in elements (not bytes) for offset calculation
            lds_tile_elems = arith.constant(tile_m * tile_k, index=True)
            # Note: For 2-stage pipeline, we now use separate LDS buffers (lds_a_ping/pong)
            # instead of a single buffer with different base offsets. This gives stronger
            # no-alias guarantees to the compiler.
            b_tile_ping = None
            b_tile_pong = None
>>>>>>> main
            if lds_stage == 2:
                # ---------------- Ping-pong pipeline (2 separate LDS buffers) ----------------
                # Cross-tile A0 LDS prefetch (default-on):
                # issue the first A-pack DS read for the next tile *between* barriers,
                # so it can overlap with the VMEM prefetch of the following tile.

                def prefetch_a0_pack(lds_buffer):
                    # (mi=0, ku=0): prefetch both K32 halves (K64) for the first A-pack.
                    return lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_buffer)

                # Prologue: tile-0
                k0 = arith.constant(0, index=True)
<<<<<<< HEAD
                a_regs0, b_tile0 = prefetch_ab_tile(k0)
                # Prefetch scales for FP4/MXFP4 at tile-0.
                a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(k0 // 256) if is_fp4 else (k0, k0)

                store_a_tile_to_lds(a_regs0, lds_base0)
=======
                b_tile_pong = prefetch_b_tile(k0)
                if use_async_copy:
                    prefetch_a_to_lds(k0, lds_a_pong)  # Load into pong buffer
                else:
                    store_a_tile_to_lds(prefetch_a_tile(k0), lds_a_pong)
>>>>>>> main
                gpu.barrier()
                accs = [acc_init] * (num_acc_n * m_repeat)

                c_k_main = c_k - tile_k

                # Prefetch A0 for the first compute tile (overlap with the next VMEM prefetch).
                a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)

                num_tiles = K // tile_k
                if (num_tiles % 2) == 1:
                    for k_iv in range(0, c_k_main, tile_k * 2):
                        next_k1 = k_iv + tile_k
<<<<<<< HEAD
                        a_regs_ping, b_tile_ping = prefetch_ab_tile(next_k1)
                        a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(next_k1 // 256) if is_fp4 else (k0, k0)

                        accs, _ = compute_tile(
                            accs, b_tile_pong, lds_base_pong,
                            a0_prefetch=a0_prefetch_pong,
                            a_scale=a_scale_pong, b_scale=b_scale_pong,
=======
                        b_tile_ping = prefetch_b_tile(next_k1)
                        if use_async_copy:
                            prefetch_a_to_lds(next_k1, lds_a_ping)
                        else:
                            store_a_tile_to_lds(prefetch_a_tile(next_k1), lds_a_ping)

                        accs, _ = compute_tile(
                            accs, b_tile_pong, lds_a_pong, a0_prefetch=a0_prefetch_pong
>>>>>>> main
                        )
                        a0_prefetch_pong = None

                        hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the ping tile we are about to compute.
                        a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

                        next_k2 = k_iv + tile_k * 2
<<<<<<< HEAD
                        a_regs_pong, b_tile_pong = prefetch_ab_tile(next_k2)
                        a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(next_k2 // 256) if is_fp4 else (k0, k0)

                        accs, _ = compute_tile(
                            accs, b_tile_ping, lds_base_ping,
                            a0_prefetch=a0_prefetch_ping,
                            a_scale=a_scale_ping, b_scale=b_scale_ping,
=======
                        b_tile_pong = prefetch_b_tile(next_k2)
                        if use_async_copy:
                            prefetch_a_to_lds(next_k2, lds_a_pong)
                        else:
                            store_a_tile_to_lds(prefetch_a_tile(next_k2), lds_a_pong)

                        accs, _ = compute_tile(
                            accs, b_tile_ping, lds_a_ping, a0_prefetch=a0_prefetch_ping
>>>>>>> main
                        )
                        a0_prefetch_ping = None

                        hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the next pong tile.
                        a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)

                    final_accs, scales = compute_tile(
                        accs,
                        b_tile_pong,
                        lds_a_pong,
                        is_last_tile=True,
                        a0_prefetch=a0_prefetch_pong,
                        a_scale=a_scale_pong, b_scale=b_scale_pong,
                    )
                else:
                    c_k_stop = c_k - (tile_k * 3)
                    for k_iv in range(0, c_k_stop, tile_k * 2):
                        next_k1 = k_iv + tile_k
<<<<<<< HEAD
                        a_regs_ping, b_tile_ping = prefetch_ab_tile(next_k1)
                        a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(next_k1 // 256) if is_fp4 else (k0, k0)

                        accs, _ = compute_tile(
                            accs, b_tile_pong, lds_base_pong,
                            a0_prefetch=a0_prefetch_pong,
                            a_scale=a_scale_pong, b_scale=b_scale_pong,
=======
                        b_tile_ping = prefetch_b_tile(next_k1)
                        if use_async_copy:
                            prefetch_a_to_lds(next_k1, lds_a_ping)
                        else:
                            store_a_tile_to_lds(prefetch_a_tile(next_k1), lds_a_ping)
                        accs, _ = compute_tile(
                            accs, b_tile_pong, lds_a_pong, a0_prefetch=a0_prefetch_pong
>>>>>>> main
                        )
                        a0_prefetch_pong = None
                        hot_loop_scheduler()
                        gpu.barrier()

                        a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

                        next_k2 = k_iv + tile_k * 2
<<<<<<< HEAD
                        a_regs_pong, b_tile_pong = prefetch_ab_tile(next_k2)
                        a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(next_k2 // 256) if is_fp4 else (k0, k0)

                        accs, _ = compute_tile(
                            accs, b_tile_ping, lds_base_ping,
                            a0_prefetch=a0_prefetch_ping,
                            a_scale=a_scale_ping, b_scale=b_scale_ping,
=======
                        b_tile_pong = prefetch_b_tile(next_k2)
                        if use_async_copy:
                            prefetch_a_to_lds(next_k2, lds_a_pong)
                        else:
                            store_a_tile_to_lds(prefetch_a_tile(next_k2), lds_a_pong)
                        accs, _ = compute_tile(
                            accs, b_tile_ping, lds_a_ping, a0_prefetch=a0_prefetch_ping
>>>>>>> main
                        )
                        a0_prefetch_ping = None

                        hot_loop_scheduler()
                        gpu.barrier()

                        a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)
                    last_k = c_k - tile_k
<<<<<<< HEAD
                    a_regs_ping, b_tile_ping = prefetch_ab_tile(last_k)
                    a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(last_k // 256) if is_fp4 else (k0, k0)

                    accs, _ = compute_tile(
                        accs, b_tile_pong, lds_base_pong,
                        a0_prefetch=a0_prefetch_pong,
                        a_scale=a_scale_pong, b_scale=b_scale_pong,
=======
                    b_tile_ping = prefetch_b_tile(last_k)
                    if use_async_copy:
                        prefetch_a_to_lds(last_k, lds_a_ping)
                    else:
                        store_a_tile_to_lds(prefetch_a_tile(last_k), lds_a_ping)

                    accs, _ = compute_tile(
                        accs, b_tile_pong, lds_a_pong, a0_prefetch=a0_prefetch_pong
>>>>>>> main
                    )
                    a0_prefetch_pong = None

                    hot_loop_scheduler()
                    gpu.barrier()

                    a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

                    final_accs, scales = compute_tile(
                        accs,
                        b_tile_ping,
                        lds_a_ping,
                        is_last_tile=True,
                        a0_prefetch=a0_prefetch_ping,
                        a_scale=a_scale_ping, b_scale=b_scale_ping,
                    )

                store_output(final_accs, scales)
<<<<<<< HEAD
            # else:
            #     # CK-like bpreshuffle v1 spirit:
            #     # - Intrawave schedule
            #     # - Global prefetch 2 (regs double-buffer)
            #     # - Local shared memory buffer 1 (single LDS tile for A)
            #     # Prologue: tile-0
            #     k0 = arith.constant(0, index=True)
            #     a_regs0, b_tile0 = prefetch_ab_tile(k0)
            #     store_a_tile_to_lds(a_regs0, lds_base0)
            #     gpu.barrier()
            #     accs = [acc_init] * (num_acc_n * m_repeat)

            #     lds_base = lds_base0
            #     b_tile_cur = b_tile0

            #     # For each tile except last: prefetch next tile, compute current, then overwrite LDS.
            #     for k_base in range(0, c_k - tile_k, tile_k):
            #         next_k = k_base + tile_k
            #         a_next, b_next = prefetch_ab_tile(next_k)
            #         accs, _ = compute_tile(accs, b_tile_cur, lds_base)
            #         # Single LDS buffer: ensure *all* waves are done reading A from LDS
            #         # before any wave overwrites it with the next tile.
            #         gpu.barrier()
            #         store_a_tile_to_lds(a_next, lds_base)
            #         hot_loop_scheduler()
            #         gpu.barrier()
            #         b_tile_cur = b_next

            #     final_accs, scales = compute_tile(
            #         accs, b_tile_cur, lds_base, is_last_tile=True
            #     )
            #     store_output(final_accs, scales)
=======
            else:
                # CK-like bpreshuffle v1 spirit:
                # - Intrawave schedule
                # - Global prefetch 2 (regs double-buffer)
                # - Local shared memory buffer 1 (single LDS tile for A)
                # Prologue: tile-0
                k0 = arith.constant(0, index=True)
                prefetch_a_to_lds(k0, lds_a_pong)  # Single buffer (reuses same)
                b_tile0 = load_b_tile(k0)
                gpu.barrier()
                accs = [acc_init] * (num_acc_n * m_repeat)

                b_tile_cur = b_tile0

                # For each tile except last: prefetch next tile, compute current, then overwrite LDS.
                for k_base in range(0, c_k - tile_k, tile_k):
                    next_k = k_base + tile_k
                    prefetch_a_to_lds(next_k, lds_a_pong)  # Reuse same buffer
                    b_next = load_b_tile(next_k)
                    accs, _ = compute_tile(accs, b_tile_cur, lds_a_pong)
                    # Single LDS buffer: ensure *all* waves are done reading A from LDS
                    # before any wave overwrites it with the next tile.
                    gpu.barrier()
                    hot_loop_scheduler()
                    gpu.barrier()
                    b_tile_cur = b_next

                final_accs, scales = compute_tile(
                    accs, b_tile_cur, lds_a_pong, is_last_tile=True
                )
                store_output(final_accs, scales)
>>>>>>> main

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: memref(DYN, _out_elem_type()),
            arg_a: lambda: memref(DYN, _a_elem_type()),
            arg_b: lambda: memref(DYN, _b_elem_type()),
            arg_scale_a: lambda: memref(DYN, _scale_elem_type()),
            arg_scale_b: lambda: memref(DYN, _scale_elem_type()),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
            stream_ptr: lambda: T.i64,  # PyTorch stream pointer passed at runtime
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(256, index=True)
            tm = arith.constant(tile_m, index=True)
            tn = arith.constant(tile_n, index=True)
            one = arith.constant(1, index=True)
            gx = (c_m + tm - one) / tm
            gy = c_n / tn
            # Convert runtime stream_ptr (i64) to async_token for gpu.launch_func
            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "kernel_gemm"],
                grid_size=(gx, gy, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_c,
                    arg_a,
                    arg_b,
                    arg_scale_a,
                    arg_scale_b,
                    c_m,
                    c_n,
                    c_k,
                ],
                async_dependencies=[stream_token],
            )

    m = _GEMM()

    # Apply waves_per_eu hint if specified (before final compilation)
    if waves_per_eu is not None:
        _apply_waves_per_eu_hint(m.module, waves_per_eu)

    return flydsl.compile(
        m,
        use_bare_ptr_memref_call_conv=False,
        use_bare_pointers_for_host=False,
        use_bare_pointers_for_kernels=False,
    )


__all__ = ["compile_preshuffle_gemm"]
