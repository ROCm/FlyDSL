"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8).

This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)

It is extracted from `tests/python/gpu/test_moe_gemm.py` so that:
- `samples/` holds the implementation
- `tests/` holds correctness/perf harnesses
"""

import os

import pyflir
from pyflir.dialects.ext import flir
from pyflir.dialects.ext.python_control_flow import range_constexpr
from pyflir.runtime.device import get_rocm_arch as get_hip_arch
from pyflir.utils import SmemAllocator

from _mlir import ir
import _mlir.extras.types as T
from pyflir.lang.ir.types import T as I

from pyflir.dialects.ext import arith, gpu, buffer_ops, llvm, vector, rocdl

from samples.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_load_pack_k32,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
)


def compile_moe_gemm1(
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    sorted_size: int,
    size_expert_ids: int,
    doweight_stage1: bool,
    in_dtype: str = "fp8",
):
    """Compile stage1 kernel (`moe_gemm1`) and return the compiled executable.

    in_dtype:
      - "fp8": X/W are fp8
      - "int8": X/W are int8
      - "int4": W4A8 path: X is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    if in_dtype not in ("fp8", "int8", "int4"):
        raise ValueError(f"in_dtype must be 'fp8', 'int8', or 'int4', got {in_dtype!r}")
    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: X is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype == "int8") or is_int4

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

    size_out = tokens * topk * inter_dim
    size_x = tokens * model_dim
    # W is packed int4 for W4A8: 2 values per byte.
    size_w = (experts * (2 * inter_dim) * model_dim) // 2 if is_int4 else (experts * (2 * inter_dim) * model_dim)
    size_sorted = int(sorted_size)
    size_expert_ids = int(size_expert_ids)

    total_threads = 256
    elems_x_per_tile = tile_m * tile_k
    elems_per_thread_x = elems_x_per_tile // total_threads
    bytes_per_thread_x = elems_per_thread_x  # 1B elems (fp8 or int8)
    # Keep MoE stage1 X gmem->LDS pipeline consistent with the optimized GEMM kernel:
    # split into <=16B pieces and use `flir.copy(load-only)` for buffer_load_dwordx4.
    # (Compute the split lens inside the kernel so the code matches GEMM structure.)

    # CK-style LDS128 mode (same idea as test_preshuffle_gemm.py):
    # - LDS stride == tile_k (no extra padding) + XOR16 swizzle
    # - Use ds_{read,write}_b128 (16B) and extract 8B halves for MFMA steps
    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in ("1", "true", "True", "YES", "yes")
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k

    module_name = f"mfma_moe1_{in_dtype}".replace("-", "_")

    class _MOE1(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # Ping-pong LDS for X (2-stage pipeline), matching the tuned GEMM kernel structure.
            _state["lds_x_decl"] = allocator.allocate_array(I.i8 if is_int8 else I.f8, 2 * tile_m * lds_stride)
            allocator.finalize()

        @flir.kernel
        def moe_gemm1(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, T.f16()),
            arg_x: lambda: T.memref(size_x, I.i8 if is_int8 else I.f8),
            arg_w: lambda: T.memref(size_w, I.i8 if is_int8 else I.f8),
            arg_scale_x: lambda: T.memref(tokens, T.f32()),
            arg_scale_w: lambda: T.memref(experts * (2 * inter_dim), T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            x_elem = I.i8 if is_int8 else I.f8
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = I.i8 if is_int8 else I.f8
            f32 = I.f32
            i32 = I.i32
            i64 = I.i64
            vec4_f32 = I.vec(4, f32)
            vec4_i32 = I.vec(4, i32)
            vec8_x = I.vec(8, x_elem)
            vec16_x = I.vec(16, x_elem)
            vec1_i64 = I.vec(1, i64)
            vec2_i64 = I.vec(2, i64)

            c0 = arith.constant(0, index=True)
            c4 = arith.constant(4, index=True)
            c1 = arith.constant(1, index=True)
            c16 = arith.constant(16, index=True)
            c64 = arith.constant(64, index=True)
            c256 = arith.constant(256, index=True)
            c1024 = arith.constant(1024, index=True)
            c_tile_k = arith.constant(tile_k, index=True)
            c0_i32 = arith.unwrap(0, type=i32)

            c0f = arith.constant(0.0, type=f32)
            c1f = arith.constant(1.0, type=f32)
            # CK-style silu uses exp2(log2e * x) + rcp, which maps to v_exp_f32 + v_rcp_f32
            # and avoids the full-precision div fixup sequence (and its cndmask-heavy guards).
            c_log2e = arith.constant(1.4426950408889634, type=f32)  # log2(e)
            c_log2e_neg = arith.constant(-1.4426950408889634, type=f32)
            c3f = arith.constant(3.0, type=f32)
            c1_div_6 = arith.constant(0.1666666716337204, type=f32)  # 1/6 as f32

            def silu(x):
                # Align with CK's device fast path:
                #   emu = exp(-x)  ~= exp2(log2e * (-x))  -> v_exp_f32
                #   sig = rcp(1 + emu)                   -> v_rcp_f32
                #   y = x * sig
                #
                # Using llvm.amdgcn intrinsics prevents lowering to the div_scale/div_fixup
                # sequences that introduce extra compares/cndmasks.
                t = x * c_log2e_neg
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                den = c1f + emu
                sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                return x * sig

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # Layouts
            layout_x = flir.make_layout((tokens_in, k_in), stride=(k_in, 1))

            # B preshuffle layout: match GEMM test helper exactly.
            c_n_total = arith.constant(experts * (2 * inter_dim), index=True)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                flir, arith, c_n=c_n_total, c_k=k_in, kpack_bytes=kpack_bytes
            )
            layout_b = b_layout.layout_b
            c_k0 = k_in / c64

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")  # tile along sorted M
            by = gpu.block_id("y")  # tile along inter_dim

            # Common constants/atoms (hoisted): keep IR small like GEMM.
            # CK-style XOR16 swizzle parameter (constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k // 16, index=True)
            atom_x_s16 = flir.make_copy_atom(x_elem, vector_size=16)
            atom_x_s8 = flir.make_copy_atom(x_elem, vector_size=8)
            atom_x_g2r16 = flir.make_copy_atom(x_elem, vector_size=16)
            atom_x_g2r8 = flir.make_copy_atom(x_elem, vector_size=8)
            layout_tx_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))

            base_ptr = allocator.get_base()
            lds_x = _state["lds_x_decl"](base_ptr).get()

            # Use logical buffer sizes (descriptor num_records) so hardware OOB checking can be
            # used directly (CK-style). This allows us to avoid `select`-based masking for
            # invalid lanes and rely on the buffer instruction's built-in bounds behavior.
            x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False)
            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False)
            sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x, max_size=False)
            sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)
            sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False)
            expert_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False)
            sorted_w_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=False)

            # Expert id for this M tile (keep address math in `index`)
            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=i32)
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            inter2_idx = arith.constant(2 * inter_dim, index=True)
            expert_off_idx = expert_idx * inter2_idx  # index

            bx_m = bx * arith.constant(tile_m, index=True)

            # ---- X gmem->reg prefetch (match test_preshuffle_gemm.py + CK gather style) ----
            # Use the same 16B chunk mapping as preshuffle_gemm:
            #   chunk_linear = tx + i*total_threads
            #   chunk_i32_base = chunk_linear * 4  (16B == 4 dwords)
            x_load_bytes = 16
            if bytes_per_thread_x % x_load_bytes != 0:
                raise ValueError(
                    f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by {x_load_bytes}"
                )
            num_x_loads = bytes_per_thread_x // x_load_bytes

            c_k_div4 = k_in / c4
            layout_x_div4 = flir.make_layout((tokens_in, c_k_div4), stride=(c_k_div4, 1))
            layout_x_tile_div4 = flir.make_layout((tile_m, tile_k // 4), stride=(tile_k // 4, 1))
            tx_i32_base = tx * c4
            mask24 = arith.i32(0xFFFFFF)
            # Keep i32 constants available for epilogue index math.
            tokens_i32 = arith.i32(tokens)
            topk_i32 = arith.i32(topk)

            def x_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(
                    flir,
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_x_tile_div4,
                )

            # CK-aligned: decode token once (per thread's M-slice) and build a base row offset.
            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)

                sorted_row_i = bx_m + row_local
                fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32)
                t_i32 = arith.andi(fused_i, mask24)
                t_idx = arith.index_cast(ir.IndexType.get(), t_i32)
                x_row_base_div4.append(arith.ArithValue(t_idx) * c_k_div4)

            vec2_i32 = I.vec(2, i32)
            vec4_i32 = I.vec(4, i32)

            def load_x_16(idx_i32):
                """Load 16 bytes from X (gmem) into a vector via buffer_load backend.

                `idx_i32` is an i32-element (dword) offset (not bytes), matching GEMM.
                """
                return buffer_copy_gmem16_dwordx4(
                    flir,
                    arg=arg_x,
                    elem_type=x_elem,
                    idx_i32=idx_i32,
                    atom_g2r16=atom_x_g2r16,
                    rsrc=x_rsrc,
                )

            def load_x_tile(base_k):
                """Prefetch the per-thread X tile portion (gmem -> regs) for a given K base (in elements)."""
                base_k_div4 = base_k / c4
                parts = []
                for i in range_constexpr(num_x_loads):
                    idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                    x_f8 = load_x_16(idx_i32)
                    parts.append(vector.bitcast(vec4_i32, x_f8))
                return parts

            # tx -> wave/lane (GEMM-style decomposition).
            coord_wl = flir.idx2crd(tx, layout_tx_wave_lane)
            wave_id = flir.get(coord_wl, 0)
            lane_id = flir.get(coord_wl, 1)
            coord_l16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_l16, 0)
            lane_mod_16 = flir.get(coord_l16, 1)

            # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col base is lane_div_16*16.
            row_a_lds = lane_mod_16
            col_offset_base = flir.crd2idx(flir.make_coord(lane_div_16, 0), layout_lane16)

            # Dynamic N tiling within block (same as existing kernels)
            by_n = by * arith.constant(tile_n, index=True)
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_mod_4 = wave_id % c4
            n_tile_base = wave_mod_4 * c_n_per_wave

            # Precompute n_blk/n_intra for gate and up rows (GEMM-style: idx2crd/get)
            n_intra_gate = []
            n_blk_gate = []
            n_intra_up = []
            n_blk_up = []
            col_g_list = []
            valid_col_list = []
            inter_idx = arith.constant(inter_dim, index=True)
            # layout for (row -> (blk,intra)) where intra is 0..15
            c_n0 = c_n_total / c16
            layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
            for ni in range_constexpr(num_acc_n):
                offset = arith.constant(ni * 16, index=True)
                col_g = by_n + n_tile_base
                col_g = col_g + offset
                col_g = col_g + lane_mod_16
                col_g_list.append(col_g)

                row_gate = expert_off_idx + col_g
                row_up = row_gate + inter_idx

                coord_gate = flir.idx2crd(row_gate, layout_n_blk_intra)
                n_blk_gate.append(flir.get(coord_gate, 0))
                n_intra_gate.append(flir.get(coord_gate, 1))

                coord_up = flir.idx2crd(row_up, layout_n_blk_intra)
                n_blk_up.append(flir.get(coord_up, 0))
                n_intra_up.append(flir.get(coord_up, 1))

                valid_col_list.append(arith.ult(col_g, inter_idx))

            m_repeat = tile_m // 16
            k_unroll = tile_k // 32

            # --- B Load Logic (K32) - shared with preshuffle GEMM ---
            def load_b_pack(base_k, ki_step, ni, blk_list, intra_list):
                return load_b_pack_k32(
                    buffer_ops,
                    flir,
                    arith,
                    vector,
                    arg_b=arg_w,
                    b_rsrc=w_rsrc,
                    layout_b=layout_b,
                    base_k=base_k,
                    ki_step=ki_step,
                    n_blk=blk_list[ni],
                    n_intra=intra_list[ni],
                    lane_div_16=lane_div_16,  # 0..3
                    elem_type=w_elem,
                    kpack_bytes=kpack_bytes,
                    unpack_int4=is_int4,
                )

            def load_b_tile(base_k, blk_list, intra_list):
                """Prefetch the entire per-thread B tile (gmem -> regs) for a given K base."""
                b_tile = []
                for ki_step in range_constexpr(k_unroll):
                    packs = []
                    for ni in range_constexpr(num_acc_n):
                        packs.append(load_b_pack(base_k, ki_step, ni, blk_list, intra_list))
                    b_tile.append(packs)
                return b_tile

            acc_gate = [acc_init] * (num_acc_n * m_repeat)
            acc_up = [acc_init] * (num_acc_n * m_repeat)

            # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
            def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                for i in range_constexpr(num_x_loads):
                    # Match test_preshuffle_gemm.py exactly: per-thread 16B chunk mapping.
                    row_local = x_row_local[i]
                    col_local_i32 = x_col_local_i32[i]
                    lds_store_16b_xor16(
                        flir,
                        arith,
                        vector,
                        lds_memref=lds_x,
                        vec16_ty=vec16_x,
                        elem_type=x_elem,
                        atom_s16=atom_x_s16,
                        layout_lds=layout_lds,
                        row_local=row_local,
                        col_local_i32=col_local_i32,
                        tx_c4=c4,
                        k_blocks16=k_blocks16,
                        lds_base=lds_base,
                        vec_part_i32x4=vec_x_in_parts[i],
                    )

            def compute_tile(
                acc_gate_in,
                acc_up_in,
                b_gate_tile_in,
                b_up_tile_in,
                lds_base,
                *,
                prefetch_epilogue: bool = False,
            ):
                gate_list = list(acc_gate_in)
                up_list = list(acc_up_in)
                mfma_res_ty = vec4_i32 if is_int8 else vec4_f32
                mfma_fn = mfma_i32_k32 if is_int8 else rocdl.mfma_f32_16x16x32_fp8_fp8

                # Optional: prefetch epilogue scales while we are about to run the last MFMA tile,
                # matching the preshuffle GEMM pattern of overlapping scale loads with MFMA.
                epilogue_pf = None
                if prefetch_epilogue:
                    expert_off_pf = arith.ArithValue(expert_off_idx)
                    sw_gate_pf = []
                    sw_up_pf = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        valid_col = valid_col_list[ni]
                        row_gate_idx = expert_off_pf + col_g
                        row_up_idx = row_gate_idx + inter_idx
                        sw_gate_pf.append(
                            buffer_ops.buffer_load(sw_rsrc, row_gate_idx, vec_width=1, dtype=f32)
                        )
                        sw_up_pf.append(
                            buffer_ops.buffer_load(sw_rsrc, row_up_idx, vec_width=1, dtype=f32)
                        )
                    epilogue_pf = (sw_gate_pf, sw_up_pf)

                for ki_step in range_constexpr(k_unroll):
                    b_gate_packs = b_gate_tile_in[ki_step]
                    b_up_packs = b_up_tile_in[ki_step]

                    half = ki_step % 2
                    ki64 = (ki_step // 2) * 64
                    col_base = col_offset_base + ki64

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val

                        a_pack = lds_load_pack_k32(
                            flir,
                            arith,
                            vector,
                            lds_memref=lds_x,
                            layout_lds=layout_lds,
                            k_blocks16=k_blocks16,
                            curr_row_a_lds=curr_row_a_lds,
                            col_base=col_base,
                            half=half,
                            lds_base=lds_base,
                            ck_lds128=bool(_ck_lds128),
                            vec16_ty=vec16_x,
                            vec8_ty=vec8_x,
                            vec2_i64_ty=vec2_i64,
                            vec1_i64_ty=vec1_i64,
                        )

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            gate_list[acc_idx] = mfma_fn(
                                mfma_res_ty,
                                [
                                    a_pack,
                                    b_gate_packs[ni],
                                    gate_list[acc_idx],
                                    c0_i32,
                                    c0_i32,
                                    c0_i32,
                                ],
                            )
                            up_list[acc_idx] = mfma_fn(
                                mfma_res_ty,
                                [
                                    a_pack,
                                    b_up_packs[ni],
                                    up_list[acc_idx],
                                    c0_i32,
                                    c0_i32,
                                    c0_i32,
                                ],
                            )
                return gate_list, up_list, epilogue_pf

            # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
            lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
            lds_base_cur = arith.constant(0, index=True)
            lds_base_nxt = lds_tile_elems

            # Optional scheduler hints (copied from tuned GEMM); can be disabled via env.
            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                mfma_group = num_acc_n * 2
                mfma_total = k_unroll * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

                # DS-read preload (CK default is 2); clamp to non-negative.
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(2)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(1)

                # DS-write hints near the end: match total X LDS-store micro-ops per thread.
                dswr_tail = num_x_loads
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

            # Prologue: prefetch tile0, store to LDS(cur), sync.
            k0 = arith.constant(0, index=True)
            x_regs0 = load_x_tile(k0)
            b_gate_cur = load_b_tile(k0, n_blk_gate, n_intra_gate)
            b_up_cur = load_b_tile(k0, n_blk_up, n_intra_up)
            store_x_tile_to_lds(x_regs0, lds_base_cur)
            gpu.barrier()

            # Loop-carried ping/pong state.
            lds_base_pong = lds_base_cur  # current/compute
            lds_base_ping = lds_base_nxt  # next/load+store

            # Unrolled ping-pong main loop (2 tiles per iteration), leaving 2 tail tiles.
            c2_tile_k = arith.constant(tile_k * 2, index=True)
            c_k_main2 = k_in - c2_tile_k

            for k_iv in range(c0, c_k_main2, c2_tile_k):
                # ---- stage 0: prefetch+store ping, compute pong ----
                next_k1 = k_iv + c_tile_k
                x_regs_ping = load_x_tile(next_k1)
                b_gate_ping = load_b_tile(next_k1, n_blk_gate, n_intra_gate)
                b_up_ping = load_b_tile(next_k1, n_blk_up, n_intra_up)

                acc_gate, acc_up, _ = compute_tile(acc_gate, acc_up, b_gate_cur, b_up_cur, lds_base_pong)
                store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                hot_loop_scheduler()
                gpu.barrier()

                # ---- stage 1: prefetch+store pong, compute ping ----
                next_k2 = k_iv + c2_tile_k
                x_regs_pong = load_x_tile(next_k2)
                b_gate_next = load_b_tile(next_k2, n_blk_gate, n_intra_gate)
                b_up_next = load_b_tile(next_k2, n_blk_up, n_intra_up)

                acc_gate, acc_up, _ = compute_tile(acc_gate, acc_up, b_gate_ping, b_up_ping, lds_base_ping)
                store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                hot_loop_scheduler()
                gpu.barrier()

                # Advance pong state to next_k2 for next iteration.
                b_gate_cur = b_gate_next
                b_up_cur = b_up_next

            # Tail: 2 remaining tiles at (k_in - 2*tile_k) and (k_in - tile_k).
            k_tail1 = k_in - c_tile_k
            x_regs_ping = load_x_tile(k_tail1)
            b_gate_ping = load_b_tile(k_tail1, n_blk_gate, n_intra_gate)
            b_up_ping = load_b_tile(k_tail1, n_blk_up, n_intra_up)

            acc_gate, acc_up, _ = compute_tile(acc_gate, acc_up, b_gate_cur, b_up_cur, lds_base_pong)
            store_x_tile_to_lds(x_regs_ping, lds_base_ping)
            hot_loop_scheduler()
            gpu.barrier()

            # Epilogue: compute last tile with epilogue scale prefetch to overlap loads with MFMA.
            acc_gate, acc_up, epilogue_pf = compute_tile(
                acc_gate,
                acc_up,
                b_gate_ping,
                b_up_ping,
                lds_base_ping,
                prefetch_epilogue=True,
            )

            # Store epilogue to out[t, slot, inter]
            expert_off = arith.ArithValue(expert_off_idx)
            bx_m0 = arith.ArithValue(bx_m)
            tokens_i32_v = arith.ArithValue(tokens_i32)
            topk_i32_v = arith.ArithValue(topk_i32)
            inter_i32_v = arith.ArithValue(arith.i32(inter_dim))
            mask24_i32 = arith.i32(0xFFFFFF)

            if epilogue_pf is not None:
                sw_gate_vals, sw_up_vals = epilogue_pf
            else:
                sw_gate_vals = []
                sw_up_vals = []
                for ni in range_constexpr(num_acc_n):
                    col_g = col_g_list[ni]
                    row_gate_idx = expert_off + col_g
                    row_up_idx = row_gate_idx + inter_idx
                    sw_gate_vals.append(
                        buffer_ops.buffer_load(sw_rsrc, row_gate_idx, vec_width=1, dtype=f32)
                    )
                    sw_up_vals.append(
                        buffer_ops.buffer_load(sw_rsrc, row_up_idx, vec_width=1, dtype=f32)
                    )

            # Epilogue hoists to keep IR + Python build time small:
            col_i32_list = []
            for ni in range_constexpr(num_acc_n):
                col_i32_list.append(arith.ArithValue(arith.index_cast(i32, col_g_list[ni])))

            lane_div_16_mul4 = arith.ArithValue(lane_div_16) * 4
            ii_idx_list = [arith.constant(ii, index=True) for ii in range(4)]
            inter_i32_local = inter_i32_v

            for mi in range_constexpr(m_repeat):
                mi_base = arith.constant(mi * 16, index=True)
                for ii in range_constexpr(4):
                    row_off = lane_div_16_mul4 + ii_idx_list[ii]
                    row_in_tile = mi_base + row_off
                    sorted_row2 = bx_m0 + row_in_tile

                    fused2 = buffer_ops.buffer_load(sorted_rsrc, sorted_row2, vec_width=1, dtype=i32)
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24
                    # No explicit mask: rely on buffer descriptor OOB to zero-fill when t2 is the
                    # sentinel (t2 == tokens) or otherwise out-of-range.
                    sx = buffer_ops.buffer_load(sx_rsrc, t2, vec_width=1, dtype=f32)

                    # out linear index base = ((t*topk + s)*inter_dim) (invariant across ni)
                    idx0 = (t2 * topk_i32_v + s2) * inter_i32_local

                    # Sorted weight aligned with `sorted_row2` (matches aiter moe_sorting output).
                    if doweight_stage1:
                        tw = buffer_ops.buffer_load(
                            sorted_w_rsrc, sorted_row2, vec_width=1, dtype=f32
                        )

                    for ni in range_constexpr(num_acc_n):
                        col_i32 = col_i32_list[ni]
                        sw_gate = sw_gate_vals[ni]
                        sw_up = sw_up_vals[ni]

                        acc_idx = mi * num_acc_n + ni
                        vg = vector.extract(acc_gate[acc_idx], static_position=[ii], dynamic_position=[])
                        vu = vector.extract(acc_up[acc_idx], static_position=[ii], dynamic_position=[])

                        if is_int8:
                            vg = arith.sitofp(f32, vg)
                            vu = arith.sitofp(f32, vu)
                        vg = vg * sx * sw_gate
                        vu = vu * sx * sw_up

                        y = silu(vg) * vu
                        if doweight_stage1:
                            y = y * tw
                        y = arith.trunc_f(T.f16(), y)
                        idx_out = idx0 + col_i32
                        buffer_ops.buffer_store(y, out_rsrc, idx_out)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, T.f16()),
            arg_x: lambda: T.memref(size_x, I.i8 if is_int8 else I.f8),
            arg_w: lambda: T.memref(size_w, I.i8 if is_int8 else I.f8),
            arg_scale_x: lambda: T.memref(tokens, T.f32()),
            arg_scale_w: lambda: T.memref(experts * (2 * inter_dim), T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            bdx = 256
            gx = size_expert_ids
            gy = inter_dim // tile_n
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "moe_gemm1"],
                grid_size=(gx, gy, 1),
                block_size=(bdx, 1, 1),
                kernel_operands=[
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    tokens_in,
                    inter_in,
                    k_in,
                ],
            )

    m = _MOE1()
    exe = pyflir.compile(m)
    return exe


def compile_moe_gemm2(
    *,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    sorted_size: int,
    size_expert_ids: int,
    doweight_stage2: bool,
    in_dtype: str = "fp8",
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    in_dtype:
      - "fp8": A2/W are fp8
      - "int8": A2/W are int8
      - "int4": W4A8 path: A2 is int8, W is packed int4 unpacked to int8 in-kernel
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    if in_dtype not in ("fp8", "int8", "int4"):
        raise ValueError(f"in_dtype must be 'fp8', 'int8', or 'int4', got {in_dtype!r}")
    is_int4 = in_dtype == "int4"
    # INT4 here means W4A8: A2 is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = (in_dtype == "int8") or is_int4

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

    size_out = tokens * model_dim
    size_x = tokens * topk * inter_dim
    # W is packed int4 for W4A8: 2 values per byte.
    size_w = (experts * model_dim * inter_dim) // 2 if is_int4 else (experts * model_dim * inter_dim)
    size_sorted = int(sorted_size)
    size_expert_ids = int(size_expert_ids)

    total_threads = 256
    elems_x_per_tile = tile_m * tile_k
    elems_per_thread_x = elems_x_per_tile // total_threads
    bytes_per_thread_x = elems_per_thread_x  # 1B elems (fp8 or int8)

    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in ("1", "true", "True", "YES", "yes")
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k

    module_name = f"mfma_moe2_{in_dtype}".replace("-", "_")

    class _MOE2(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            # Ping-pong LDS for A2 (2-stage pipeline).
            _state["lds_x_decl"] = allocator.allocate_array(I.i8 if is_int8 else I.f8, 2 * tile_m * lds_stride)
            allocator.finalize()

        @flir.kernel
        def moe_gemm2(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, T.f32()),
            arg_x: lambda: T.memref(size_x, I.i8 if is_int8 else I.f8),
            arg_w: lambda: T.memref(size_w, I.i8 if is_int8 else I.f8),
            arg_scale_x: lambda: T.memref(tokens * topk, T.f32()),
            arg_scale_w: lambda: T.memref(experts * model_dim, T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            x_elem = I.i8 if is_int8 else I.f8
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = I.i8 if is_int8 else I.f8
            f32 = I.f32
            i32 = I.i32
            i64 = I.i64
            vec4_f32 = I.vec(4, f32)
            vec4_i32 = I.vec(4, i32)
            vec8_x = I.vec(8, x_elem)
            vec16_x = I.vec(16, x_elem)
            vec1_i64 = I.vec(1, i64)
            vec2_i64 = I.vec(2, i64)

            c0 = arith.constant(0, index=True)
            c4 = arith.constant(4, index=True)
            c1 = arith.constant(1, index=True)
            c16 = arith.constant(16, index=True)
            c64 = arith.constant(64, index=True)
            c_tile_k = arith.constant(tile_k, index=True)
            c0_i32 = arith.unwrap(0, type=i32)

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # A2 layout (flatten token-slot -> M).
            topk_idx = arith.constant(topk, index=True)
            m_in = tokens_in * topk_idx
            layout_x = flir.make_layout((m_in, k_in), stride=(k_in, 1))

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.constant(experts * model_dim, index=True)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                flir, arith, c_n=c_n_total, c_k=k_in, kpack_bytes=kpack_bytes
            )
            layout_b = b_layout.layout_b
            c_k0 = k_in / c64

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")  # tile along sorted M
            by = gpu.block_id("y")  # tile along model_dim

            # CK-style XOR16 swizzle parameter (constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k // 16, index=True)
            atom_x_s16 = flir.make_copy_atom(x_elem, vector_size=16)
            atom_x_g2r16 = flir.make_copy_atom(x_elem, vector_size=16)
            layout_tx_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            layout_lin_rowcol = flir.make_layout((tile_m, tile_k), stride=(tile_k, 1))

            base_ptr = allocator.get_base()
            lds_x = _state["lds_x_decl"](base_ptr).get()

            # Buffer resources (logical sizes: allow hardware OOB checks).
            x_rsrc = buffer_ops.create_buffer_resource(arg_x, max_size=False)
            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=False)
            sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x, max_size=False)
            sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)
            sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids, max_size=False)
            expert_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids, max_size=False)
            sorted_w_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights, max_size=False)

            # Expert id for this M tile.
            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=i32)
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            n_idx = arith.constant(model_dim, index=True)
            expert_off_idx = expert_idx * n_idx  # index

            bx_m = bx * arith.constant(tile_m, index=True)

            # ---- X gmem->reg prefetch (match test_preshuffle_gemm.py exactly) ----
            # Reindex A2 at dword granularity so lanes load contiguous 16B chunks:
            x_load_bytes = 16
            if bytes_per_thread_x % x_load_bytes != 0:
                raise ValueError(
                    f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by {x_load_bytes}"
                )
            num_x_loads = bytes_per_thread_x // x_load_bytes  # 16B chunks per thread
            vec4_i32 = I.vec(4, i32)

            c_k_div4 = k_in / c4
            layout_x_div4 = flir.make_layout((m_in, c_k_div4), stride=(c_k_div4, 1))
            layout_x_tile_div4 = flir.make_layout((tile_m, tile_k // 4), stride=(tile_k // 4, 1))
            tx_i32_base = tx * c4

            topk_i32 = arith.i32(topk)
            mask24 = arith.i32(0xFFFFFF)
            tokens_i32 = arith.i32(tokens)

            def x_tile_chunk_coord_i32(i: int):
                return tile_chunk_coord_i32(
                    flir,
                    arith,
                    tx_i32_base=tx_i32_base,
                    i=i,
                    total_threads=total_threads,
                    layout_tile_div4=layout_x_tile_div4,
                )

            def load_x_16(idx_i32):
                return buffer_copy_gmem16_dwordx4(
                    flir,
                    arg=arg_x,
                    elem_type=x_elem,
                    idx_i32=idx_i32,
                    atom_g2r16=atom_x_g2r16,
                    rsrc=x_rsrc,
                )

            # CK-aligned: decode routed token once (per thread's M-slice) and build a base offset.
            x_row_base_div4 = []
            x_col_local_i32 = []
            x_row_local = []
            for i in range_constexpr(num_x_loads):
                row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                x_row_local.append(row_local)
                x_col_local_i32.append(col_local_i32)

                sorted_row_i = bx_m + row_local
                fused_i = buffer_ops.buffer_load(sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32)
                t_i32 = arith.andi(fused_i, mask24)
                s_i32 = arith.shrui(fused_i, arith.i32(24))
                # Guard sentinel padded token id (t_i32 == tokens) / any OOB:
                # clamp row index so global loads stay in-bounds; masked stores/atomics
                # will zero out contributions for invalid rows.
                t_valid = flir.arith.CmpIOp(
                    flir.arith.CmpIPredicate.ult,
                    arith.ArithValue(t_i32).value,
                    arith.ArithValue(tokens_i32).value,
                ).result
                t_i32_safe = flir.arith.SelectOp(
                    arith.ArithValue(t_valid).value,
                    arith.ArithValue(t_i32).value,
                    arith.constant(0, type=T.i32()).value,
                ).result
                # A2 row index = t*topk + s (still i32 here).
                row_ts_i32 = arith.ArithValue(t_i32_safe) * topk_i32 + s_i32
                row_ts_idx = arith.index_cast(ir.IndexType.get(), row_ts_i32)
                # Base row offset in dword units: row_ts_idx * (k_in/4)
                x_row_base_div4.append(arith.ArithValue(row_ts_idx) * c_k_div4)

            def load_x_tile(base_k):
                base_k_div4 = base_k / c4
                parts = []
                for i in range_constexpr(num_x_loads):
                    idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                    x_f8 = load_x_16(idx_i32)
                    parts.append(vector.bitcast(vec4_i32, x_f8))
                return parts

            # tx -> wave/lane (GEMM-style decomposition).
            coord_wl = flir.idx2crd(tx, layout_tx_wave_lane)
            wave_id = flir.get(coord_wl, 0)
            lane_id = flir.get(coord_wl, 1)
            coord_l16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_l16, 0)
            lane_mod_16 = flir.get(coord_l16, 1)

            row_a_lds = lane_mod_16
            col_offset_base = flir.crd2idx(flir.make_coord(lane_div_16, 0), layout_lane16)

            # Dynamic N tiling within block.
            by_n = by * arith.constant(tile_n, index=True)
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_mod_4 = wave_id % arith.constant(4, index=True)
            n_tile_base = wave_mod_4 * c_n_per_wave

            # Precompute (n_blk, n_intra) for B, and col indices for output.
            n_intra_list = []
            n_blk_list = []
            col_g_list = []
            c_n0 = c_n_total / c16
            layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
            for ni in range_constexpr(num_acc_n):
                offset = arith.constant(ni * 16, index=True)
                col_g = by_n + n_tile_base + offset + lane_mod_16
                col_g_list.append(col_g)

                row_w = expert_off_idx + col_g
                coord_w = flir.idx2crd(row_w, layout_n_blk_intra)
                n_blk_list.append(flir.get(coord_w, 0))
                n_intra_list.append(flir.get(coord_w, 1))

            m_repeat = tile_m // 16
            k_unroll = tile_k // 32

            # --- B Load Logic (K32) ---
            def load_b_pack(base_k, ki_step, ni):
                return load_b_pack_k32(
                    buffer_ops,
                    flir,
                    arith,
                    vector,
                    arg_b=arg_w,
                    b_rsrc=w_rsrc,
                    layout_b=layout_b,
                    base_k=base_k,
                    ki_step=ki_step,
                    n_blk=n_blk_list[ni],
                    n_intra=n_intra_list[ni],
                    lane_div_16=lane_div_16,  # 0..3
                    elem_type=w_elem,
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

            # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
            def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                for i in range_constexpr(num_x_loads):
                    row_local = x_row_local[i]
                    col_local_i32 = x_col_local_i32[i]
                    lds_store_16b_xor16(
                        flir,
                        arith,
                        vector,
                        lds_memref=lds_x,
                        vec16_ty=vec16_x,
                        elem_type=x_elem,
                        atom_s16=atom_x_s16,
                        layout_lds=layout_lds,
                        row_local=row_local,
                        col_local_i32=col_local_i32,
                        tx_c4=c4,
                        k_blocks16=k_blocks16,
                        lds_base=lds_base,
                        vec_part_i32x4=vec_x_in_parts[i],
                    )

            def compute_tile(acc_in, b_tile_in, lds_base, *, prefetch_epilogue: bool = False):
                acc_list = list(acc_in)
                mfma_res_ty = vec4_i32 if is_int8 else vec4_f32
                mfma_fn = mfma_i32_k32 if is_int8 else rocdl.mfma_f32_16x16x32_fp8_fp8

                epilogue_pf = None
                if prefetch_epilogue:
                    expert_off_pf = arith.ArithValue(expert_off_idx)
                    sw_pf = []
                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        row_w_idx = expert_off_pf + col_g
                        sw_pf.append(
                            buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=1, dtype=f32)
                        )
                    epilogue_pf = sw_pf

                for ki_step in range_constexpr(k_unroll):
                    b_packs = b_tile_in[ki_step]
                    half = ki_step % 2
                    ki64 = (ki_step // 2) * 64
                    col_base = col_offset_base + ki64

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val

                        a_pack = lds_load_pack_k32(
                            flir,
                            arith,
                            vector,
                            lds_memref=lds_x,
                            layout_lds=layout_lds,
                            k_blocks16=k_blocks16,
                            curr_row_a_lds=curr_row_a_lds,
                            col_base=col_base,
                            half=half,
                            lds_base=lds_base,
                            ck_lds128=bool(_ck_lds128),
                            vec16_ty=vec16_x,
                            vec8_ty=vec8_x,
                            vec2_i64_ty=vec2_i64,
                            vec1_i64_ty=vec1_i64,
                        )

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            acc_list[acc_idx] = mfma_fn(
                                mfma_res_ty,
                                [
                                    a_pack,
                                    b_packs[ni],
                                    acc_list[acc_idx],
                                    c0_i32,
                                    c0_i32,
                                    c0_i32,
                                ],
                            )
                return acc_list, epilogue_pf

            # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
            lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
            lds_base_cur = arith.constant(0, index=True)
            lds_base_nxt = lds_tile_elems

            rocdl.sched_barrier(0)

            def hot_loop_scheduler():
                mfma_group = num_acc_n
                mfma_total = k_unroll * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                rocdl.sched_mfma(1)
                if num_acc_n < 4:
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(2)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(2)
                    rocdl.sched_vmem(1)

                dswr_tail = num_x_loads
                if dswr_tail > sche_iters:
                    dswr_tail = sche_iters
                dswr_start = sche_iters - dswr_tail
                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_mfma(mfma_group // 2)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group // 2)
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    if sche_i >= dswr_start - 1:
                        rocdl.sched_dswr(1)
                rocdl.sched_barrier(0)

            # Prologue.
            k0 = arith.constant(0, index=True)
            x_regs0 = load_x_tile(k0)
            b_cur = load_b_tile(k0)
            store_x_tile_to_lds(x_regs0, lds_base_cur)
            gpu.barrier()

            acc = [acc_init] * (num_acc_n * m_repeat)
            lds_base_pong = lds_base_cur
            lds_base_ping = lds_base_nxt

            # Main loop leaves 2 tail tiles.
            c2_tile_k = arith.constant(tile_k * 2, index=True)
            c_k_main2 = k_in - c2_tile_k
            for k_iv in range(c0, c_k_main2, c2_tile_k):
                next_k1 = k_iv + c_tile_k
                x_regs_ping = load_x_tile(next_k1)
                b_ping = load_b_tile(next_k1)

                acc, _ = compute_tile(acc, b_cur, lds_base_pong)
                store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                hot_loop_scheduler()
                gpu.barrier()

                next_k2 = k_iv + c2_tile_k
                x_regs_pong = load_x_tile(next_k2)
                b_next = load_b_tile(next_k2)

                acc, _ = compute_tile(acc, b_ping, lds_base_ping)
                store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                hot_loop_scheduler()
                gpu.barrier()

                b_cur = b_next

            # Tail: 2 remaining tiles.
            k_tail1 = k_in - c_tile_k
            x_regs_ping = load_x_tile(k_tail1)
            b_ping = load_b_tile(k_tail1)

            acc, _ = compute_tile(acc, b_cur, lds_base_pong)
            store_x_tile_to_lds(x_regs_ping, lds_base_ping)
            hot_loop_scheduler()
            gpu.barrier()

            # Epilogue tile with sw prefetch.
            acc, sw_pf = compute_tile(acc, b_ping, lds_base_ping, prefetch_epilogue=True)

            # Store epilogue: atomic-add into out[t, n].
            expert_off = arith.ArithValue(expert_off_idx)
            bx_m0 = arith.ArithValue(bx_m)
            mask24_i32 = arith.i32(0xFFFFFF)
            model_i32 = arith.ArithValue(arith.i32(model_dim))
            topk_i32_v = arith.ArithValue(topk_i32)
            c4_i32 = arith.ArithValue(arith.i32(4))

            zero_i32 = arith.i32(0)

            def atomic_add_f32(val_f32, byte_off_i32):
                rocdl.raw_ptr_buffer_atomic_fadd(
                    val_f32,
                    out_rsrc,
                    byte_off_i32,
                    zero_i32,
                    zero_i32,
                )

            # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
            if sw_pf is not None:
                sw_vals = sw_pf
            else:
                sw_vals = []
                for ni in range_constexpr(num_acc_n):
                    col_g = col_g_list[ni]
                    row_w_idx = expert_off + col_g
                    sw_vals.append(buffer_ops.buffer_load(sw_rsrc, row_w_idx, vec_width=1, dtype=f32))

            col_i32_list = []
            for ni in range_constexpr(num_acc_n):
                col_i32_list.append(arith.ArithValue(arith.index_cast(i32, col_g_list[ni])))

            lane_div_16_mul4 = arith.ArithValue(lane_div_16) * 4
            ii_idx_list = [arith.constant(ii, index=True) for ii in range(4)]

            for mi in range_constexpr(m_repeat):
                mi_base = arith.constant(mi * 16, index=True)
                for ii in range_constexpr(4):
                    row_off = lane_div_16_mul4 + ii_idx_list[ii]
                    row_in_tile = mi_base + row_off
                    sorted_row2 = bx_m0 + row_in_tile

                    fused2 = buffer_ops.buffer_load(sorted_rsrc, sorted_row2, vec_width=1, dtype=i32)
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24

                    # a2_scale index = t*topk + s (i32). Hardware OOB handles sentinel padding.
                    ts2 = arith.ArithValue(t2) * topk_i32_v + s2
                    sx = buffer_ops.buffer_load(sx_rsrc, ts2, vec_width=1, dtype=f32)

                    # out index base (in elements): t*model_dim
                    idx0 = arith.ArithValue(t2) * model_i32

                    if doweight_stage2:
                        tw = buffer_ops.buffer_load(sorted_w_rsrc, sorted_row2, vec_width=1, dtype=f32)

                    for ni in range_constexpr(num_acc_n):
                        col_i32 = col_i32_list[ni]
                        sw = sw_vals[ni]
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(acc[acc_idx], static_position=[ii], dynamic_position=[])
                        if is_int8:
                            v = arith.sitofp(f32, v)
                        v = v * sx * sw
                        if doweight_stage2:
                            v = v * tw

                        idx_elem = idx0 + col_i32
                        byte_off = idx_elem * c4_i32
                        atomic_add_f32(v, byte_off)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, T.f32()),
            arg_x: lambda: T.memref(size_x, I.i8 if is_int8 else I.f8),
            arg_w: lambda: T.memref(size_w, I.i8 if is_int8 else I.f8),
            arg_scale_x: lambda: T.memref(tokens * topk, T.f32()),
            arg_scale_w: lambda: T.memref(experts * model_dim, T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            n_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            bdx = 256
            gx = size_expert_ids
            gy = model_dim // tile_n
            flir.gpu_ext.LaunchFuncOp(
                [module_name, "moe_gemm2"],
                grid_size=(gx, gy, 1),
                block_size=(bdx, 1, 1),
                kernel_operands=[
                    arg_out,
                    arg_x,
                    arg_w,
                    arg_scale_x,
                    arg_scale_w,
                    arg_sorted_token_ids,
                    arg_expert_ids,
                    arg_sorted_weights,
                    tokens_in,
                    n_in,
                    k_in,
                ],
            )

    m = _MOE2()
    exe = pyflir.compile(m)
    return exe

