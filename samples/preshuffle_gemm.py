"""Preshuffle GEMM kernel implementations (FLIR MFMA FP8/INT8).

This module intentionally contains the **kernel builder code** for the preshuffle GEMM,
extracted from `tests/python/gpu/test_preshuffle_gemm.py` in the same style as
`samples/moe_gemm_2stage.py`:
- `samples/` holds the implementation (compile functions)
- `tests/` holds correctness/perf harnesses

Pipelines:
- `pingpong`: tuned 2-stage pipeline with ping-pong LDS for A (2 LDS buffers)
- `ck_v1_single_lds`: CK-like Intrawave + bpreshuffle v1 spirit (single LDS buffer for A)
"""

import pyflir
from pyflir.dialects.ext import flir
from pyflir.dialects.ext.python_control_flow import range_constexpr
from pyflir.runtime.device import get_rocm_arch as get_hip_arch
from pyflir.utils import SmemAllocator

from pyflir.dialects.ext import arith, gpu, buffer_ops, vector, rocdl
from pyflir.lang.ir.types import T, memref

from samples.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_load_pack_k32,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
)


def compile_preshuffle_gemm_a8(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    lds_stage: int = 2,
):
    """Compile the preshuffle GEMM kernel and return the compiled executable.

    Args:
        M, N, K: GEMM sizes (A[M,K], B[N,K], C[M,N]).
        tile_m, tile_n, tile_k: block tile sizes.
        in_dtype:
          - "fp8": A/B are fp8 (1B/elem)
          - "int8": A/B are int8 (1B/elem)
          - "int4": W4A8 path: A is int8, B is packed int4 (2 values per byte) and unpacked to int8 in-kernel.
        lds_stage: 
          - 2: ping-pong LDS for A (2 LDS buffers), tuned schedule (original).
          - 1: single LDS buffer for A .
    """
    if in_dtype not in ("fp8", "int8", "int4"):
        raise ValueError(f"in_dtype must be 'fp8', 'int8', or 'int4', got {in_dtype!r}")
    is_int4 = in_dtype == "int4"
    is_int8 = (in_dtype == "int8") or is_int4

    # INT8 must use a K32 MFMA so the micro-step matches the FP8 path (strict alignment).
    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    size_c = int(M) * int(N)
    size_a = int(M) * int(K)
    # B is packed int4 for W4A8: 2 values per byte.
    size_b = (int(N) * int(K)) // 2 if is_int4 else (int(N) * int(K))

    # Vector width calc (assume full tiles / no tail guards).
    total_threads = 256
    elems_a_per_tile = tile_m * tile_k
    if elems_a_per_tile % total_threads != 0:
        raise ValueError(
            f"tile_m*tile_k must be divisible by {total_threads}: tile_m={tile_m}, tile_k={tile_k}"
        )
    elems_per_thread_a = elems_a_per_tile // total_threads
    bytes_per_thread_a = elems_per_thread_a  # 1B elems

    # Assume A loads are always 16B-aligned and use fixed dwordx4 (16B) buffer loads.
    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load_bytes}"
        )

    # CK-style LDS128: stride == tile_k, XOR16 swizzle.
    lds_stride = tile_k

    def _elem_type():
        return T.i8 if is_int8 else T.f8

    def _vec16_type():
        return T.i8x16 if is_int8 else T.f8x16

    module_name = f"mfma_preshuffle_{lds_stage}stages_{in_dtype}".replace("-", "_")

    class _GEMM(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # LDS for A: either ping-pong (2 tiles) or single tile (CK v1 spirit).
            _state["lds_a_decl"] = allocator.allocate_array(
                _elem_type(), lds_stage * tile_m * lds_stride
            )
            allocator.finalize()

        @flir.kernel
        def kernel_gemm(
            self: flir.T.i64,
            arg_c: lambda: memref(size_c, T.f16),
            arg_a: lambda: memref(size_a, _elem_type()),
            arg_b: lambda: memref(size_b, _elem_type()),
            arg_scale_a: lambda: memref(M, T.f32),
            arg_scale_b: lambda: memref(N, T.f32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            # ---- Types ----
            # NOTE: Some environments have multiple `pyflir` builds on PYTHONPATH.
            # Use explicit MLIR Values (not Python ints / wrapper objects) for ROCDL ops.
            acc_init = arith.unwrap(
                arith.constant_vector(0, T.i32x4)
                if is_int8
                else arith.constant_vector(0.0, T.f32x4)
            )
            c0_i32 = arith.unwrap(0, type=T.i32)

            # Layouts
            layout_c = flir.make_layout((c_m, c_n), stride=(c_n, 1))

            # A is 1B/elem but we use dwordx4 (16B) buffer loads, so build a /4 layout.
            c_k_div4 = c_k / 4
            layout_a_div4 = flir.make_layout((c_m, c_k_div4), stride=(c_k_div4, 1))

            # B preshuffle layout (shared with MoE kernels).
            kpack_bytes = 8 if is_int4 else 16
            layout_b = make_preshuffle_b_layout(
                flir, arith, c_n=c_n, c_k=c_k, kpack_bytes=kpack_bytes
            ).layout_b

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            # CK-style XOR16 swizzle parameter (const).
            k_blocks16 = arith.constant(tile_k // 16, index=True)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")

            base_ptr = allocator.get_base()
            lds_a = _state["lds_a_decl"](base_ptr).get()

            a_rsrc = buffer_ops.create_buffer_resource(arg_a)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c)
            scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a)
            scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b)

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
            # lane_div_16 * 16 via FLIR crd2idx((lane_div_16,0), layout=(4,16):stride=(16,1))
            col_offset_base = flir.crd2idx(flir.make_coord(lane_div_16, 0), layout_lane16)

            m_repeat = tile_m // 16
            k_unroll = tile_k // 32  # K32 micro-step

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

            # --- B load logic (K32 micro-step), return i64 packs ---
            # Shared loader supports:
            # - FP8/INT8: 8B direct load from preshuffled layout
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
                    elem_type=_elem_type(),
                    kpack_bytes=kpack_bytes,
                    unpack_int4=is_int4,
                )

            def load_b_tile(base_k):
                b_tile = []
                for ki_step in range_constexpr(k_unroll):
                    packs = []
                    for ni in range_constexpr(num_acc_n):
                        packs.append(load_b_pack(base_k, ki_step, ni))
                    b_tile.append(packs)
                return b_tile

            # --- A load/store (16B chunks), XOR16 swizzle ---
            num_a_loads = bytes_per_thread_a // a_load_bytes
            layout_a_tile_div4 = flir.make_layout(
                (tile_m, tile_k // 4), stride=(tile_k // 4, 1)
            )
            c4 = arith.constant(4, index=True)
            tx_i32_base = tx * c4
            atom_a_g2r16 = flir.make_copy_atom(_elem_type(), vector_size=16)

            def load_a_16(idx_i32):
                return buffer_copy_gmem16_dwordx4(
                    flir,
                    arg=arg_a,
                    elem_type=_elem_type(),
                    idx_i32=idx_i32,
                    atom_g2r16=atom_a_g2r16,
                    rsrc=a_rsrc,
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

            def load_a_tile(base_k_div4):
                parts = []
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    row_a_global = bx_m + row_a_local
                    coord_a_g = flir.make_coord(row_a_global, base_k_div4 + col_a_local_i32)
                    idx_i32 = flir.crd2idx(coord_a_g, layout_a_div4)
                    a_16B = load_a_16(idx_i32)
                    parts.append(vector.bitcast(T.i32x4, a_16B))
                return parts

            def store_a_tile_to_lds(vec_a_parts, lds_base):
                for i in range_constexpr(num_a_loads):
                    row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                    lds_store_16b_xor16(
                        flir,
                        arith,
                        vector,
                        lds_memref=lds_a,
                        vec16_ty=_vec16_type(),
                        elem_type=_elem_type(),
                        atom_s16=atom_a_g2r16,
                        layout_lds=layout_lds,
                        row_local=row_a_local,
                        col_local_i32=col_a_local_i32,
                        tx_c4=c4,
                        k_blocks16=k_blocks16,
                        lds_base=lds_base,
                        vec_part_i32x4=vec_a_parts[i],
                    )

            def prefetch_ab_tile(base_k):
                base_k_div4 = base_k / 4
                a_regs = load_a_tile(base_k_div4)
                b_regs = load_b_tile(base_k)
                return a_regs, b_regs

            def compute_tile(accs_in, b_tile_in, lds_base, *, is_last_tile=False):
                scales_pf = {}
                if is_last_tile:
                    # Prefetch scales (same as original kernel).
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
                mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
                mfma_fn = mfma_i32_k32 if is_int8 else rocdl.mfma_f32_16x16x32_fp8_fp8

                for ki_step in range_constexpr(k_unroll):
                    b_packs = b_tile_in[ki_step]
                    ki64 = (ki_step // 2) * 64
                    half = ki_step % 2
                    col_base = col_offset_base + ki64
                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val
                        col_base_swz = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                        a_pack = lds_load_pack_k32(
                            flir,
                            arith,
                            vector,
                            lds_memref=lds_a,
                            layout_lds=layout_lds,
                            k_blocks16=k_blocks16,
                            curr_row_a_lds=curr_row_a_lds,
                            col_base=col_base,
                            half=half,
                            lds_base=lds_base,
                            ck_lds128=True,
                            vec16_ty=_vec16_type(),
                            vec8_ty=_vec16_type(),  # unused in ck_lds128=True path
                            vec2_i64_ty=T.i64x2,
                            vec1_i64_ty=T.vec(1, T.i64),
                        )
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            current_accs_list[acc_idx] = mfma_fn(
                                mfma_res_ty,
                                [
                                    a_pack,
                                    b_packs[ni],
                                    current_accs_list[acc_idx],
                                    c0_i32,
                                    c0_i32,
                                    c0_i32,
                                ],
                            )
                return current_accs_list, scales_pf

            def store_output(final_accs, scales):
                s_b_vals = scales["s_b_vals"]
                s_a_vecs = scales["s_a_vecs"]
                for mi in range_constexpr(m_repeat):
                    row_base_m = bx_m + (mi * 16)
                    s_a_vec4 = s_a_vecs[mi]
                    for i in range_constexpr(4):
                        row_off = (lane_div_16 * 4) + i
                        row_g = row_base_m + row_off
                        s_a = vector.extract(s_a_vec4, static_position=[i], dynamic_position=[])
                        col_base = by_n + n_tile_base + lane_mod_16
                        idx_base = flir.crd2idx(flir.make_coord(row_g, col_base), layout_c)
                        byte_offset_base = idx_base * 2
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            acc = final_accs[acc_idx]
                            val = vector.extract(acc, static_position=[i], dynamic_position=[])
                            if is_int8:
                                val = arith.sitofp(T.f32, val)
                            val_s = (val * s_a) * s_b_vals[ni]
                            val_f16 = arith.trunc_f(T.f16, val_s)
                            byte_off = byte_offset_base + 32 * ni
                            buffer_ops.buffer_store(val_f16, c_rsrc, byte_off, offset_is_bytes=True)

            # ---------------- Scheduling hints (match CK-style) ----------------
            # These sched_group_barrier hints help the backend interleave VMEM/DS/MFMA
            # similarly to CK's tuned pipelines.
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                if tile_m == 16:
                    return
                # - MFMA group size per "slot": num_acc_n
                # - Total MFMA per tile: k_unroll * m_repeat * num_acc_n
                # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                mfma_group = num_acc_n
                mfma_total = k_unroll * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

                # DS-read preload (CK default is 2).
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                rocdl.sched_mfma(1)
                if num_acc_n < 4:
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(1)
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
                    if sche_i >= dswr_start - 1:
                        rocdl.sched_dswr(1)

                rocdl.sched_barrier(0)

            # ---------------- Pipeline ----------------
            lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
            lds_base0 = arith.constant(0, index=True)
            lds_base1 = lds_tile_elems

            if lds_stage == 2:
                # Prologue: tile-0
                k0 = arith.constant(0, index=True)
                a_regs0, b_tile0 = prefetch_ab_tile(k0)
                store_a_tile_to_lds(a_regs0, lds_base0)
                gpu.barrier()
                accs = [acc_init] * (num_acc_n * m_repeat)

                # Ping-pong LDS (2 buffers), same as the original tuned kernel.
                lds_base_pong = lds_base0
                lds_base_ping = lds_base1
                b_tile_pong = b_tile0
                c_k_main = c_k - tile_k

                # Compile-time parity selection (kernel specialized to Python-level K).
                num_tiles = K // tile_k
                if (num_tiles % 2) == 1:
                    for k_iv in range(0, c_k_main, tile_k * 2):
                        next_k1 = k_iv + tile_k
                        a_regs_ping, b_tile_ping = prefetch_ab_tile(next_k1)
                        accs, _ = compute_tile(accs, b_tile_pong, lds_base_pong)
                        store_a_tile_to_lds(a_regs_ping, lds_base_ping)
                        hot_loop_scheduler()
                        gpu.barrier()
                        next_k2 = k_iv + tile_k * 2
                        a_regs_pong, b_tile_pong = prefetch_ab_tile(next_k2)
                        accs, _ = compute_tile(accs, b_tile_ping, lds_base_ping)
                        store_a_tile_to_lds(a_regs_pong, lds_base_pong)
                        hot_loop_scheduler()
                        gpu.barrier()
                    final_accs, scales = compute_tile(
                        accs, b_tile_pong, lds_base_pong, is_last_tile=True
                    )
                else:
                    c_k_stop = c_k - (tile_k * 3)
                    for k_iv in range(0, c_k_stop, tile_k * 2):
                        next_k1 = k_iv + tile_k
                        a_regs_ping, b_tile_ping = prefetch_ab_tile(next_k1)
                        accs, _ = compute_tile(accs, b_tile_pong, lds_base_pong)
                        store_a_tile_to_lds(a_regs_ping, lds_base_ping)
                        hot_loop_scheduler()
                        gpu.barrier()
                        next_k2 = k_iv + tile_k * 2
                        a_regs_pong, b_tile_pong = prefetch_ab_tile(next_k2)
                        accs, _ = compute_tile(accs, b_tile_ping, lds_base_ping)
                        store_a_tile_to_lds(a_regs_pong, lds_base_pong)
                        hot_loop_scheduler()
                        gpu.barrier()
                    last_k = c_k - tile_k
                    a_regs_ping, b_tile_ping = prefetch_ab_tile(last_k)
                    accs, _ = compute_tile(accs, b_tile_pong, lds_base_pong)
                    store_a_tile_to_lds(a_regs_ping, lds_base_ping)
                    hot_loop_scheduler()
                    gpu.barrier()
                    final_accs, scales = compute_tile(
                        accs, b_tile_ping, lds_base_ping, is_last_tile=True
                    )

                store_output(final_accs, scales)
            else:
                # CK-like bpreshuffle v1 spirit:
                # - Intrawave schedule
                # - Global prefetch 2 (regs double-buffer)
                # - Local shared memory buffer 1 (single LDS tile for A)
                # Prologue: tile-0
                k0 = arith.constant(0, index=True)
                a_regs0, b_tile0 = prefetch_ab_tile(k0)
                store_a_tile_to_lds(a_regs0, lds_base0)
                gpu.barrier()
                accs = [acc_init] * (num_acc_n * m_repeat)

                lds_base = lds_base0
                b_tile_cur = b_tile0

                # For each tile except last: prefetch next tile, compute current, then overwrite LDS.
                for k_base in range(0, c_k - tile_k, tile_k):
                    next_k = k_base + tile_k
                    a_next, b_next = prefetch_ab_tile(next_k)
                    accs, _ = compute_tile(accs, b_tile_cur, lds_base)
                    # Single LDS buffer: ensure *all* waves are done reading A from LDS
                    # before any wave overwrites it with the next tile.
                    gpu.barrier()
                    store_a_tile_to_lds(a_next, lds_base)
                    hot_loop_scheduler()
                    gpu.barrier()
                    b_tile_cur = b_next

                final_accs, scales = compute_tile(
                    accs, b_tile_cur, lds_base, is_last_tile=True
                )
                store_output(final_accs, scales)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_c: lambda: memref(size_c, T.f16),
            arg_a: lambda: memref(size_a, _elem_type()),
            arg_b: lambda: memref(size_b, _elem_type()),
            arg_scale_a: lambda: memref(M, T.f32),
            arg_scale_b: lambda: memref(N, T.f32),
            c_m: lambda: T.index,
            c_n: lambda: T.index,
            c_k: lambda: T.index,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(256, index=True)
            gx = arith.constant(M // tile_m, index=True)
            gy = arith.constant(N // tile_n, index=True)
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
            )

    m = _GEMM()
    return pyflir.compile(m)


__all__ = ["compile_preshuffle_gemm_a8"]

