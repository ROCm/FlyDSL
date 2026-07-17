# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""4-wave MXFP8 GEMM for AMD CDNA4/gfx950.

Algorithm derived from the HipKittens MXFP8_4wave kernel
(https://github.com/HazyResearch/HipKittens/blob/a288366e4245528f74540b3fe446637cf8345745/kernels/cdna4/gemm/mxfp8/MXFP8_4wave/4_wave.cu#L1).

The kernel targets ``v_mfma_scale_f32_16x16x128_f8f6f4`` using MXFP8 E4M3
inputs and E8M0 per-K32 scaling factors. Scale operands are prepacked on the
host into the format consumed by the scaled MFMA instruction and are loaded
through a dedicated scale pipeline separate from the A/B data path.

A and B operands are staged through XOR-swizzled LDS buffers using a
HipKittens-style 4-wave schedule with overlapped global loads, LDS reads,
and MFMA execution. The implementation uses direct ROCDL scheduling controls
to preserve the intended ordering of memory and compute operations.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.fp8_gemm_utils import divmod, pack_i32x4_i32x8, swizzle_128


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle(num_pid_m, num_pid_n):
    """Map the linear workgroup ID to a GEMM tile coordinate.

    Uses row-major ordering for small or uneven grids. Otherwise, partitions
    workgroups evenly across XCDs and applies grouped-M ordering to improve
    operand reuse and cache locality.
    """
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x
    num_wg = num_pid_m * num_pid_n

    # Simple row-major path.
    simple_m, simple_n = divmod(wgid, num_pid_n)

    # XCD-remapped grouped-M path.
    intra_xcd, xcd = divmod(wgid, NUM_XCDS)
    wgid_remap = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = divmod(wgid_remap, num_wgid_in_group)
    first_pid_m = group_id * WGM
    group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m

    use_simple = (num_wg < SWIZZLE_THRESHOLD) | (num_wg % NUM_XCDS != 0)
    return (
        arith.select(use_simple, simple_m, pid_m),
        arith.select(use_simple, simple_n, pid_n),
    )


LDS_VECTOR_BYTES = 16


def _encode_waitcnt(vmcnt=63, lgkmcnt=15):
    """Encode the CDNA4/gfx950 ``S_WAITCNT`` SIMM16 operand.

    ``rocdl.s_waitcnt`` accepts the raw 16-bit immediate operand of the
    32-bit ``S_WAITCNT`` ISA instruction. On CDNA4, that SIMM16 field is:

        SIMM16[3:0]   = vmcnt[3:0]
        SIMM16[6:4]   = expcnt[2:0]
        SIMM16[11:8]  = lgkmcnt[3:0]
        SIMM16[15:14] = vmcnt[5:4]

    ``vmcnt`` is therefore one six-bit counter split across two noncontiguous
    fields; bits [5:4] are placed in SIMM16[15:14], while bits [3:0] remain
    in SIMM16[3:0].

    A wait-counter field set to its maximum representable value is effectively
    unconstrained: the instruction does not wait on that counter. This helper
    always encodes ``expcnt=7`` and defaults to ``vmcnt=63`` and ``lgkmcnt=15``,
    so callers specify only the counters on which they intend to wait.

    For example, ``_encode_waitcnt(lgkmcnt=0)`` returns ``0xC07F``, which the
    assembler renders as ``s_waitcnt lgkmcnt(0)``.
    See: https://llvm.org/docs/AMDGPU/gfx9_waitcnt.html
    """
    if not 0 <= vmcnt <= 63:
        raise ValueError(f"vmcnt must be in [0, 63], got {vmcnt}")
    if not 0 <= lgkmcnt <= 15:
        raise ValueError(f"lgkmcnt must be in [0, 15], got {lgkmcnt}")

    return (
        (7 << 4)  # expcnt=7 -> SIMM16[6:4] (unconstrained)
        | (vmcnt & 0x0F)  # vmcnt[3:0] -> SIMM16[3:0]
        | ((lgkmcnt & 0x0F) << 8)  # lgkmcnt[3:0] -> SIMM16[11:8]
        | ((vmcnt & 0x30) << 10)  # vmcnt[5:4] -> SIMM16[15:14]
    )


def _barrier(vmcnt=63, lgkmcnt=15):
    if vmcnt != 63 or lgkmcnt != 15:
        rocdl.s_waitcnt(_encode_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt))
    rocdl.s_barrier()


def compile_mxfp8_gemm_4w(*, K: int, BLOCK_M: int = 256, BLOCK_N: int = 256, use_xcd_remap: bool = True):
    """Build the specialized 4-wave kernel for compile-time ``K``.

    ``K`` must contain at least four K128 tiles. Runtime M/N are expected to
    be exact multiples of ``BLOCK_M``/``BLOCK_N``; the kernel has no edge masks.
    """
    BLOCK_K = 128
    NUM_THREADS = 256
    WARP_SIZE = 64
    NUM_WAVES = NUM_THREADS // WARP_SIZE

    SUBTILE_M = 64
    SUBTILE_N = 64

    MFMA_M = 16
    MFMA_N = 16

    SUBTILES_PER_WAVE = 4
    MFMA_M_PER_SUBTILE = SUBTILE_M // MFMA_M
    MFMA_N_PER_SUBTILE = SUBTILE_N // MFMA_N
    ACCS_PER_WAVE = SUBTILES_PER_WAVE * MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE

    ELEM_BYTES = 1
    VEC_BYTES = LDS_VECTOR_BYTES

    LDS_ELEMS_A = BLOCK_M * BLOCK_K
    LDS_ELEMS_B = BLOCK_N * BLOCK_K
    LDS_BYTES_A = LDS_ELEMS_A * ELEM_BYTES
    LDS_BYTES_B = LDS_ELEMS_B * ELEM_BYTES

    LOAD_PASSES_A = LDS_BYTES_A // (NUM_THREADS * VEC_BYTES)
    LOAD_PASSES_B = LDS_BYTES_B // (NUM_THREADS * VEC_BYTES)
    LOAD_PASSES_A_SUBTILE = LOAD_PASSES_A // 2
    LOAD_PASSES_B_SUBTILE = LOAD_PASSES_B // 2
    LOAD_PASSES_SCALES = 16

    assert K % BLOCK_K == 0, f"K must be a multiple of {BLOCK_K}, got {K}"
    NUM_K_TILES = K // BLOCK_K
    assert NUM_K_TILES >= 4, f"K={K} gives {NUM_K_TILES} K128 tiles; the two-page pipeline needs at least 4"

    LDS_SYM_A0 = "mxfp8_pp_smem_a0"
    LDS_SYM_A1 = "mxfp8_pp_smem_a1"
    LDS_SYM_B0 = "mxfp8_pp_smem_b0"
    LDS_SYM_B1 = "mxfp8_pp_smem_b1"
    # Model each ping-pong LDS page as a distinct LLVM alias scope. Loads from
    # the page being consumed can then be scheduled independently of direct
    # global-to-LDS refills targeting a different page.
    LDS_ALIAS_DOMAIN = '#llvm.alias_scope_domain<id = "mxfp8_pp_lds">'
    SCOPE_IDS = ("a0", "a1", "b0", "b1")

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def kernel_gemm(
        A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, As: fx.Tensor, Bs: fx.Tensor, c_m: fx.Int32, c_n: fx.Int32
    ):
        _LDS_PTR_TY = ir.Type.parse("!llvm.ptr<3>")
        _I8_TY = T.i8
        _GEP_DYN = -(2**31)

        def _gep_lds(base_ptr, byte_offset_i32):
            return llvm.getelementptr(_LDS_PTR_TY, base_ptr, [byte_offset_i32], [_GEP_DYN], _I8_TY, None)

        def _scope_attr(ids):
            """Build an LLVM alias-scope array for the named LDS pages."""
            inner = ", ".join(f'#llvm.alias_scope<id = "{sid}", domain = {LDS_ALIAS_DOMAIN}>' for sid in ids)
            return ir.Attribute.parse(f"[{inner}]")

        # Tag each access with alias.scope(page) and noalias(other pages).
        # These are compile-time promises, justified because A0/A1/B0/B1 are
        # separate LDS globals; incorrect metadata here could permit invalid
        # memory reordering.
        _SCOPE = {sid: _scope_attr((sid,)) for sid in SCOPE_IDS}
        _NOALIAS = {sid: _scope_attr(tuple(other for other in SCOPE_IDS if other != sid)) for sid in SCOPE_IDS}

        lds_a0_base_ptr = llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYM_A0)
        lds_a1_base_ptr = llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYM_A1)
        lds_b0_base_ptr = llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYM_B0)
        lds_b1_base_ptr = llvm.mlir_addressof(_LDS_PTR_TY, LDS_SYM_B1)

        def _make_lds_view(base_ptr, my_scope, other_scopes):
            """Create a byte-addressed LDS view carrying LLVM alias metadata."""
            vec_t = Vec.make_type(4, fx.Int32)

            def vec_load_i32x4_byte_offset(byte_off_idx):
                # Attach the page's alias.scope and the other pages' noalias
                # scopes to every LDS read. LLVM can then keep reads from the
                # current page independent of refills targeting another page.
                byte_off_i32 = arith.index_cast(T.i32, byte_off_idx)
                gep = _gep_lds(base_ptr, byte_off_i32)
                return llvm.LoadOp(
                    vec_t,
                    gep,
                    alignment=4,
                    alias_scopes=my_scope,
                    noalias_scopes=other_scopes,
                ).result

            view = type("AliasScopedF8LDSView", (), {})()
            view.vec_load_i32x4_byte_offset = vec_load_i32x4_byte_offset
            return view

        lds_a0 = _make_lds_view(lds_a0_base_ptr, _SCOPE["a0"], _NOALIAS["a0"])
        lds_a1 = _make_lds_view(lds_a1_base_ptr, _SCOPE["a1"], _NOALIAS["a1"])
        lds_b0 = _make_lds_view(lds_b0_base_ptr, _SCOPE["b0"], _NOALIAS["b0"])
        lds_b1 = _make_lds_view(lds_b1_base_ptr, _SCOPE["b1"], _NOALIAS["b1"])
        a_rsrc = buffer_ops.create_buffer_resource(A, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(B, max_size=True)
        as_rsrc = buffer_ops.create_buffer_resource(As, max_size=True)
        bs_rsrc = buffer_ops.create_buffer_resource(Bs, max_size=True)
        tx = gpu.thread_id("x")

        num_blocks_m = c_m // BLOCK_M
        num_blocks_n = c_n // BLOCK_N

        if const_expr(use_xcd_remap):
            pid_m, pid_n = _xcd_swizzle(num_blocks_m, num_blocks_n)
        else:
            pid_m, pid_n = divmod(fx.block_idx.x, num_blocks_n)

        bx_m = pid_m * BLOCK_M
        by_n = pid_n * BLOCK_N

        # The flattened/XCD-swizzled block coordinates are i32, while global
        # address arithmetic below is expressed in MLIR index type.  Convert
        # once here and use these index-typed tile bases for every address.
        bx_m_idx = fx.Index(bx_m)
        by_n_idx = fx.Index(by_n)

        layout_wave_lane = fx.make_layout((NUM_WAVES, WARP_SIZE), (WARP_SIZE, 1))
        coord_wave_lane = fx.idx2crd(fx.Int32(tx), layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(fx.Int32(lane), layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # C can exceed the signed-i32 element/byte offset range for large M*N.
        # Bias the buffer descriptor base once per CTA using an index/i64 GEP,
        # then store with only tile-local i32 offsets.  This keeps the hot store
        # instruction form unchanged while avoiding i32 wrap in buffer_store().
        c_n_idx_for_base = fx.Index(c_n)
        c_tile_base_elems = bx_m_idx * c_n_idx_for_base + by_n_idx
        c_tile_base_bytes = c_tile_base_elems * fx.Index(2)  # C is f16.
        c_rsrc = buffer_ops.create_buffer_resource(
            C,
            max_size=True,
            base_byte_offset=c_tile_base_bytes,
        )

        PIN_ACC_BASE = 0

        def _reg_list(prefix, start, end):
            return ",".join(f"~{{{prefix}{r}}}" for r in range(start, end + 1))

        def reserve_pinned_accumulators():
            # Reserve a fixed physical AGPR bank for all accumulators. In the
            # SSA-lowered path, the compiler generated heavy AGPR <-> VGPR traffic,
            # including v_accvgpr_mov/read sequences, s_nop stalls, and accumulator
            # spills. Pinning each f32x4 accumulator to a stable AGPR range keeps the
            # scaled MFMA accumulation in place and avoids those transfers and spills.
            #
            # ACCS_PER_WAVE = 64 accumulator objects and each object is f32x4,
            # so the physical bank is exactly 64 * 4 = 256 AGPRs: a[0:255].
            clobbers = _reg_list("a", PIN_ACC_BASE, PIN_ACC_BASE + ACCS_PER_WAVE * 4 - 1)
            llvm.InlineAsmOp(
                None,
                [],
                "",
                clobbers,
                has_side_effects=True,
            )

        def zero_pinned_accumulators():
            for ai in range_constexpr(ACCS_PER_WAVE * 4):
                llvm.InlineAsmOp(
                    None,
                    [],
                    f"v_accvgpr_write_b32 a[{PIN_ACC_BASE + ai}], 0",
                    f"~{{a{PIN_ACC_BASE + ai}}}",
                    has_side_effects=True,
                )

        def _inline_asm_i32(asm_string, constraints, operands=None):
            op = llvm.InlineAsmOp(
                T.i32,
                operands or [],
                asm_string,
                constraints,
                has_side_effects=True,
            )
            return _one_i32_result(op)

        def _unwrap_mlir_value(x):
            """Return the bare MLIR value for the known MFMA operand types.

            A/B fragments are ``flydsl.expr.typing.Vector`` wrappers and expose
            their underlying MLIR value through ``ir_value()``. Scale operands
            are ``flydsl.expr.utils.arith.ArithValue`` instances, which are
            already accepted directly by ``llvm.InlineAsmOp``.
            """
            if isinstance(x, Vec):
                return x.ir_value()
            return x

        def _one_i32_result(op):
            # Accept the result attribute names exposed by the supported MLIR Python bindings.
            return getattr(op, "result", getattr(op, "res", op.results[0]))

        def read_pinned_accumulator(acc_idx):
            acc_pin = PIN_ACC_BASE + acc_idx * 4
            r0 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 0}]", "=v")
            r1 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 1}]", "=v")
            r2 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 2}]", "=v")
            r3 = _inline_asm_i32(f"v_accvgpr_read_b32 $0, a[{acc_pin + 3}]", "=v")
            return Vec.from_elements([r0, r1, r2, r3], fx.Int32).bitcast(fx.Float32)

        # As/Bs are MFMA-ready packed scale words: [K128, row] uint32.
        # Each loaded dword already contains the four 16-row/16-col MFMA scale
        # bytes for this lane's 64-row A/B half.  The MFMA instruction selects
        # the byte via op_sel/op_sel_hi, so there is intentionally no hot-loop
        # byte extraction and no 0x01010101 broadcast here.
        c_m_idx = fx.Index(c_m)
        c_n_idx = fx.Index(c_n)

        def hot_loop_scheduler_q_refill_2n():
            # 8 chunks:
            #   1 VMEM/LDS-refill pass
            #   2 MFMA
            for _ in range_constexpr(8):
                rocdl.sched_vmem(1)
                rocdl.sched_mfma(2)

            rocdl.sched_barrier(0)

        def load_a_scale_row(k128, row):
            packed = buffer_ops.buffer_load(
                as_rsrc,
                k128 * c_m_idx + bx_m_idx + row,
                vec_width=1,
                dtype=T.i32,
            )
            return packed

        def load_b_scale_row(k128, row):
            packed = buffer_ops.buffer_load(
                bs_rsrc,
                k128 * c_n_idx + by_n_idx + row,
                vec_width=1,
                dtype=T.i32,
            )
            return packed

        def load_a_scale_subtile(k128, sm):
            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)
            a_row = subtile_m_idx * fx.Index(SUBTILE_M) + fx.Index(lane)
            a_scale = load_a_scale_row(k128, a_row)
            return (a_scale, a_scale, a_scale, a_scale)

        def load_b_scale_subtile(k128, sn):
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            b_row = subtile_n_idx * fx.Index(SUBTILE_N) + fx.Index(lane)
            b_scale = load_b_scale_row(k128, b_row)
            return (b_scale, b_scale, b_scale, b_scale)

        def load_scale_tile(k128):
            # Load all scale VGPRs needed by this wave for this K128 tile once.
            # Return order: A-top, A-bottom, B-left, B-right.
            return (
                load_a_scale_subtile(k128, 0),
                load_a_scale_subtile(k128, 1),
                load_b_scale_subtile(k128, 0),
                load_b_scale_subtile(k128, 1),
            )

        def stage_a_pass(k_base, load_i, lds_a_base_ptr, lds_a_scope, lds_a_noalias):
            linear = fx.Index(tx) + fx.Index(load_i * NUM_THREADS)
            byte_base = linear * fx.Index(VEC_BYTES)

            row = byte_base // fx.Index(BLOCK_K)
            col = byte_base % fx.Index(BLOCK_K)

            # Keep the LDS destination physically contiguous for
            # raw_ptr_buffer_load_lds, but load the logical K column that maps
            # to this physical slot under the XOR layout.
            a_phys_col_bytes = col
            _, a_log_col_bytes = swizzle_128(row, a_phys_col_bytes)
            a_global_byte = (bx_m_idx + row) * fx.Index(K) + (k_base + a_log_col_bytes)
            a_lds_byte_i32 = arith.index_cast(T.i32, byte_base)
            a_lds_byte_uniform = rocdl.readfirstlane(T.i32, a_lds_byte_i32)
            a_lds_ptr = _gep_lds(lds_a_base_ptr, a_lds_byte_uniform)

            rocdl.raw_ptr_buffer_load_lds(
                a_rsrc,
                a_lds_ptr,
                fx.Int32(VEC_BYTES),
                fx.Int32(a_global_byte),
                fx.Int32(0),
                fx.Int32(0),
                fx.Int32(1),
                alias_scopes=lds_a_scope,
                noalias_scopes=lds_a_noalias,
            )

        def stage_b_pass(k_base, load_i, lds_b_base_ptr, lds_b_scope, lds_b_noalias):
            linear = fx.Index(tx) + fx.Index(load_i * NUM_THREADS)
            byte_base = linear * fx.Index(VEC_BYTES)

            row = byte_base // fx.Index(BLOCK_K)
            col = byte_base % fx.Index(BLOCK_K)

            # Same physical-contiguous LDS write / logical-swizzled global
            # column scheme as A.
            b_phys_col_bytes = col
            _, b_log_col_bytes = swizzle_128(row, b_phys_col_bytes)
            b_global_byte = (by_n_idx + row) * fx.Index(K) + (k_base + b_log_col_bytes)
            b_lds_byte_i32 = arith.index_cast(T.i32, byte_base)
            b_lds_byte_uniform = rocdl.readfirstlane(T.i32, b_lds_byte_i32)
            b_lds_ptr = _gep_lds(lds_b_base_ptr, b_lds_byte_uniform)

            rocdl.raw_ptr_buffer_load_lds(
                b_rsrc,
                b_lds_ptr,
                fx.Int32(VEC_BYTES),
                fx.Int32(b_global_byte),
                fx.Int32(0),
                fx.Int32(0),
                fx.Int32(1),
                alias_scopes=lds_b_scope,
                noalias_scopes=lds_b_noalias,
            )

        def stage_a_subtile_pass(k_base, subtile, pass_in_subtile, lds_a_base_ptr, lds_a_scope, lds_a_noalias):
            load_i = subtile * LOAD_PASSES_A_SUBTILE + pass_in_subtile
            stage_a_pass(k_base, load_i, lds_a_base_ptr, lds_a_scope, lds_a_noalias)

        def stage_b_subtile_pass(k_base, subtile, pass_in_subtile, lds_b_base_ptr, lds_b_scope, lds_b_noalias):
            load_i = subtile * LOAD_PASSES_B_SUBTILE + pass_in_subtile
            stage_b_pass(k_base, load_i, lds_b_base_ptr, lds_b_scope, lds_b_noalias)

        def stage_a_subtile(k_base, subtile, lds_a_base_ptr, lds_a_scope, lds_a_noalias):
            start_pass = subtile * LOAD_PASSES_A_SUBTILE
            stop_pass = start_pass + LOAD_PASSES_A_SUBTILE
            for load_i in range_constexpr(start_pass, stop_pass):
                stage_a_pass(k_base, load_i, lds_a_base_ptr, lds_a_scope, lds_a_noalias)

        def stage_b_subtile(k_base, subtile, lds_b_base_ptr, lds_b_scope, lds_b_noalias):
            start_pass = subtile * LOAD_PASSES_B_SUBTILE
            stop_pass = start_pass + LOAD_PASSES_B_SUBTILE
            for load_i in range_constexpr(start_pass, stop_pass):
                stage_b_pass(k_base, load_i, lds_b_base_ptr, lds_b_scope, lds_b_noalias)

        def load_frag_at_byte_base(lds_view, row_byte_base):
            # The two K fragments use lane-invariant physical LDS columns.
            x0 = lds_view.vec_load_i32x4_byte_offset(row_byte_base + reg_lds_k_col0)
            x1 = lds_view.vec_load_i32x4_byte_offset(row_byte_base + reg_lds_k_col1)
            return pack_i32x4_i32x8(x0, x1)

        def load_a_frag(lds_a, local_row):
            return load_frag_at_byte_base(lds_a, local_row * fx.Index(BLOCK_K))

        def load_b_frag(lds_b, local_row):
            # B is stored as [N, K], but after staging it uses the same
            # XOR-swizzled LDS fragment addressing as A.
            return load_frag_at_byte_base(lds_b, local_row * fx.Index(BLOCK_K))

        def _acc_idx(subtile_id, mi, ni):
            return subtile_id * MFMA_M_PER_SUBTILE * MFMA_N_PER_SUBTILE + mi * MFMA_N_PER_SUBTILE + ni

        def pinned_mfma(acc_idx, a_frag, b_frag, a_scale, b_scale, mi, ni):
            # Fixed physical accumulator bank, visible SSA A/B/scale operands.
            # acc_idx maps directly to a[PIN_ACC_BASE + 4*acc_idx : +3].
            # The scale operands are MFMA-ready packed dwords.  mi/ni choose
            # which of the four bytes inside the A/B scale dword the MFMA uses.
            acc_pin = PIN_ACC_BASE + acc_idx * 4
            llvm.InlineAsmOp(
                None,
                [
                    _unwrap_mlir_value(a_frag),
                    _unwrap_mlir_value(b_frag),
                    _unwrap_mlir_value(a_scale),
                    _unwrap_mlir_value(b_scale),
                ],
                (
                    f"v_mfma_scale_f32_16x16x128_f8f6f4 "
                    f"a[{acc_pin}:{acc_pin + 3}], "
                    f"$0, $1, "
                    f"a[{acc_pin}:{acc_pin + 3}], "
                    f"$2, $3 "
                    f"op_sel:[{mi & 1},{ni & 1},0] "
                    f"op_sel_hi:[{mi >> 1},{ni >> 1},0]"
                ),
                (f"v,v,v,v,~{{a{acc_pin}}},~{{a{acc_pin + 1}}},~{{a{acc_pin + 2}}},~{{a{acc_pin + 3}}}"),
                has_side_effects=True,
            )

        def mfma_4n(acc_base, a_frag, a_scale, b0, b1, b2, b3, bs0, bs1, bs2, bs3):
            """Emit four N-direction scaled MFMAs into fixed physical AGPR accumulators."""
            mi = (acc_base // MFMA_N_PER_SUBTILE) % MFMA_M_PER_SUBTILE
            pinned_mfma(acc_base + 0, a_frag, b0, a_scale, bs0, mi, 0)
            pinned_mfma(acc_base + 1, a_frag, b1, a_scale, bs1, mi, 1)
            pinned_mfma(acc_base + 2, a_frag, b2, a_scale, bs2, mi, 2)
            pinned_mfma(acc_base + 3, a_frag, b3, a_scale, bs3, mi, 3)

        def mfma_2n(acc_base, a_frag, a_scale, b0, b1, bs0, bs1, ni_base):
            mi = (acc_base // MFMA_N_PER_SUBTILE) % MFMA_M_PER_SUBTILE
            pinned_mfma(acc_base + 0, a_frag, b0, a_scale, bs0, mi, ni_base + 0)
            pinned_mfma(acc_base + 1, a_frag, b1, a_scale, bs1, mi, ni_base + 1)

        def store_output():
            c_n_idx = fx.Index(c_n)
            for sm in range_constexpr(2):
                # HK store mapping: each wave owns the same 64x64 warp-position
                # inside all four 128x128 quadrants, not one contiguous 128x128 tile.
                subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)

                for sn in range_constexpr(2):
                    subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
                    subtile_id = sm * 2 + sn

                    subtile_m_base = subtile_m_idx * SUBTILE_M
                    subtile_n_base = subtile_n_idx * SUBTILE_N

                    for mi in range_constexpr(MFMA_M_PER_SUBTILE):
                        row_base = subtile_m_base + fx.Index(mi * MFMA_M) + lane_div_16 * 4

                        for ni in range_constexpr(MFMA_N_PER_SUBTILE):
                            col = subtile_n_base + fx.Index(ni * MFMA_N) + lane_mod_16
                            acc_idx = _acc_idx(subtile_id, mi, ni)
                            acc = read_pinned_accumulator(acc_idx)

                            for ii in range_constexpr(4):
                                row = row_base + fx.Index(ii)
                                c_idx = row * c_n_idx + col
                                val = Vec(acc)[ii].to(fx.Float16)
                                buffer_ops.buffer_store(val, c_rsrc, c_idx)

        # Explicit register coordinates for HK-style four-quadrant mapping.
        # BLOCK_M/BLOCK_N are 256x256.  Four waves map to warp positions
        # inside each 128x128 quadrant:
        #   cA: (warp_m,     warp_n)
        #   cB: (warp_m,     warp_n + 2)
        #   cC: (warp_m + 2, warp_n)
        #   cD: (warp_m + 2, warp_n + 2)
        reg_k_col0 = lane_div_16 * 16
        reg_k_col1 = 64 + lane_div_16 * 16

        # Every fragment row differs only by multiples of 16, so row % 16 is
        # always lane_mod_16. Hoist the logical->physical XOR mapping once.
        _, reg_lds_k_col0 = swizzle_128(lane_mod_16, reg_k_col0)
        _, reg_lds_k_col1 = swizzle_128(lane_mod_16, reg_k_col1)

        reg_subtile_m_idx0 = wave_id // 2
        reg_subtile_n_idx0 = wave_id % 2

        reserve_pinned_accumulators()
        zero_pinned_accumulators()

        def load_b_subtile_ni_regs(lds_b, scale_tile, sn, ni):
            # Fine-grained B register load for one 16-row N-direction MFMA slice.
            # Return one packed B fragment and its matching scale operand.
            subtile_n_idx = reg_subtile_n_idx0 + fx.Index(sn * 2)
            b_scales = scale_tile[2] if sn == 0 else scale_tile[3]

            b_row_addr = subtile_n_idx * fx.Index(SUBTILE_N) + fx.Index(ni * MFMA_N) + lane_mod_16
            b_ni = load_b_frag(lds_b, b_row_addr)
            b_scale_ni = b_scales[ni]
            return b_ni, b_scale_ni

        def load_b_subtile_regs(lds_b, scale_tile, sn):
            b0, bs0 = load_b_subtile_ni_regs(lds_b, scale_tile, sn, 0)
            b1, bs1 = load_b_subtile_ni_regs(lds_b, scale_tile, sn, 1)
            b2, bs2 = load_b_subtile_ni_regs(lds_b, scale_tile, sn, 2)
            b3, bs3 = load_b_subtile_ni_regs(lds_b, scale_tile, sn, 3)
            return b0, b1, b2, b3, bs0, bs1, bs2, bs3

        def load_a_subtile_mi_regs(lds_a, scale_tile, sm, mi):
            # Fine-grained A register load for one 16-row M-direction MFMA slice.
            subtile_m_idx = reg_subtile_m_idx0 + fx.Index(sm * 2)
            a_scales = scale_tile[0] if sm == 0 else scale_tile[1]
            a_row_addr = subtile_m_idx * fx.Index(SUBTILE_M) + fx.Index(mi * MFMA_M) + lane_mod_16
            a_mi = load_a_frag(lds_a, a_row_addr)
            a_scale_mi = a_scales[mi]
            return a_mi, a_scale_mi

        def hk_one_k_with_refill(
            k128,
            cur_a,
            cur_b,
            next_a,
            next_b,
            refill_a_base,
            refill_b_base,
            refill_a_scope,
            refill_a_noalias,
            refill_b_scope,
            refill_b_noalias,
            a0_regs,
            b0_regs,
            cur_scales,
            prev_refill_scales,
        ):
            # Scale invariant:
            #   cur_scales is HK MFMA-ready for K.
            #   prev_refill_scales is HK MFMA-ready for K+1.
            #   This iteration issues K+2 scale loads and returns them for the
            #   next steady iteration or final tail.

            # Wait only far enough for the current page; the next-page refill may remain in flight.
            _barrier(vmcnt=2 * LOAD_PASSES_A_SUBTILE + 2 * LOAD_PASSES_B_SUBTILE, lgkmcnt=0)
            rocdl.sched_barrier(0)

            # Immediately issue MFMA-ready K+2 scale loads.
            # They are returned for the next iteration without any in-kernel
            # byte extraction or broadcast.
            refill_scales = load_scale_tile(fx.Index(k128 + 2))
            next_scales_ready = prev_refill_scales
            # B-left is carried as a complete register tile, so its LDS page can be refilled
            # immediately. Only A-top mi=0 is carried to limit the A-fragment live range.
            # Split the carried B tile into individual MFMA slices for refill interleaving.
            b00, b01, b02, b03, bs00, bs01, bs02, bs03 = b0_regs
            a00, as00 = a0_regs
            # Read the remaining A-top slices before refilling this LDS page; afterwards the
            # old contents may be overwritten. Keeping the reads together also preserves contiguous MFMA groups.
            a01, as01 = load_a_subtile_mi_regs(cur_a, cur_scales, 0, 1)
            a02, as02 = load_a_subtile_mi_regs(cur_a, cur_scales, 0, 2)
            a03, as03 = load_a_subtile_mi_regs(cur_a, cur_scales, 0, 3)

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            b10, bs10 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 0)
            b11, bs11 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 1)
            b12, bs12 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 2)
            b13, bs13 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 3)

            # Refill the current ping-pong page with K+2, alternating A and B passes.
            k_refill = fx.Index((k128 + 2) * BLOCK_K)

            rocdl.sched_barrier(0)
            stage_a_subtile_pass(k_refill, 0, 0, refill_a_base, refill_a_scope, refill_a_noalias)
            mfma_2n(_acc_idx(0, 0, 0), a00, as00, b00, b01, bs00, bs01, 0)

            stage_b_subtile_pass(k_refill, 0, 0, refill_b_base, refill_b_scope, refill_b_noalias)
            mfma_2n(_acc_idx(0, 0, 2), a00, as00, b02, b03, bs02, bs03, 2)

            stage_a_subtile_pass(k_refill, 0, 1, refill_a_base, refill_a_scope, refill_a_noalias)
            mfma_2n(_acc_idx(0, 1, 0), a01, as01, b00, b01, bs00, bs01, 0)

            stage_b_subtile_pass(k_refill, 0, 1, refill_b_base, refill_b_scope, refill_b_noalias)
            mfma_2n(_acc_idx(0, 1, 2), a01, as01, b02, b03, bs02, bs03, 2)

            stage_a_subtile_pass(k_refill, 0, 2, refill_a_base, refill_a_scope, refill_a_noalias)
            mfma_2n(_acc_idx(0, 2, 0), a02, as02, b00, b01, bs00, bs01, 0)

            stage_b_subtile_pass(k_refill, 0, 2, refill_b_base, refill_b_scope, refill_b_noalias)
            mfma_2n(_acc_idx(0, 2, 2), a02, as02, b02, b03, bs02, bs03, 2)

            stage_a_subtile_pass(k_refill, 0, 3, refill_a_base, refill_a_scope, refill_a_noalias)
            mfma_2n(_acc_idx(0, 3, 0), a03, as03, b00, b01, bs00, bs01, 0)

            stage_b_subtile_pass(k_refill, 0, 3, refill_b_base, refill_b_scope, refill_b_noalias)
            mfma_2n(_acc_idx(0, 3, 2), a03, as03, b02, b03, bs02, bs03, 2)

            hot_loop_scheduler_q_refill_2n()

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a10, as10 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 0)
            a11, as11 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 1)
            a12, as12 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 2)
            a13, as13 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 3)

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            stage_b_subtile_pass(k_refill, 1, 0, refill_b_base, refill_b_scope, refill_b_noalias)
            mfma_2n(_acc_idx(1, 0, 0), a00, as00, b10, b11, bs10, bs11, 0)

            stage_a_subtile_pass(k_refill, 1, 0, refill_a_base, refill_a_scope, refill_a_noalias)
            mfma_2n(_acc_idx(1, 0, 2), a00, as00, b12, b13, bs12, bs13, 2)

            stage_b_subtile_pass(k_refill, 1, 1, refill_b_base, refill_b_scope, refill_b_noalias)
            mfma_2n(_acc_idx(1, 1, 0), a01, as01, b10, b11, bs10, bs11, 0)

            stage_a_subtile_pass(k_refill, 1, 1, refill_a_base, refill_a_scope, refill_a_noalias)
            mfma_2n(_acc_idx(1, 1, 2), a01, as01, b12, b13, bs12, bs13, 2)

            stage_b_subtile_pass(k_refill, 1, 2, refill_b_base, refill_b_scope, refill_b_noalias)
            mfma_2n(_acc_idx(1, 2, 0), a02, as02, b10, b11, bs10, bs11, 0)

            stage_a_subtile_pass(k_refill, 1, 2, refill_a_base, refill_a_scope, refill_a_noalias)
            mfma_2n(_acc_idx(1, 2, 2), a02, as02, b12, b13, bs12, bs13, 2)

            stage_b_subtile_pass(k_refill, 1, 3, refill_b_base, refill_b_scope, refill_b_noalias)
            mfma_2n(_acc_idx(1, 3, 0), a03, as03, b10, b11, bs10, bs11, 0)

            stage_a_subtile_pass(k_refill, 1, 3, refill_a_base, refill_a_scope, refill_a_noalias)
            mfma_2n(_acc_idx(1, 3, 2), a03, as03, b12, b13, bs12, bs13, 2)
            hot_loop_scheduler_q_refill_2n()

            # Leave exactly the K+2 refill and scale loads outstanding. The following
            # LDS reads consume the already-ready next page, not the page being refilled.
            rocdl.sched_barrier(0)
            _barrier(vmcnt=2 * LOAD_PASSES_A_SUBTILE + 2 * LOAD_PASSES_B_SUBTILE + LOAD_PASSES_SCALES, lgkmcnt=0)
            rocdl.sched_barrier(0)

            next_a0_regs = load_a_subtile_mi_regs(next_a, next_scales_ready, 0, 0)

            mfma_4n(_acc_idx(2, 0, 0), a10, as10, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(2, 1, 0), a11, as11, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(2, 2, 0), a12, as12, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(2, 3, 0), a13, as13, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            next_b0_regs = load_b_subtile_regs(next_b, next_scales_ready, 0)
            mfma_4n(_acc_idx(3, 0, 0), a10, as10, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(3, 1, 0), a11, as11, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(3, 2, 0), a12, as12, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(3, 3, 0), a13, as13, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            return next_a0_regs, next_b0_regs, next_scales_ready, refill_scales

        def hk_one_k_tail_with_next(cur_a, cur_b, next_a, next_b, a0_regs, b0_regs, cur_scales, next_scales):
            _barrier(vmcnt=2 * LOAD_PASSES_A_SUBTILE + 2 * LOAD_PASSES_B_SUBTILE, lgkmcnt=0)

            b00, b01, b02, b03, bs00, bs01, bs02, bs03 = b0_regs
            a00, as00 = a0_regs
            a01, as01 = load_a_subtile_mi_regs(cur_a, cur_scales, 0, 1)
            a02, as02 = load_a_subtile_mi_regs(cur_a, cur_scales, 0, 2)
            a03, as03 = load_a_subtile_mi_regs(cur_a, cur_scales, 0, 3)

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            b10, bs10 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 0)
            b11, bs11 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 1)
            b12, bs12 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 2)
            b13, bs13 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 3)

            mfma_4n(_acc_idx(0, 0, 0), a00, as00, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(0, 1, 0), a01, as01, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(0, 2, 0), a02, as02, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(0, 3, 0), a03, as03, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a10, as10 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 0)
            a11, as11 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 1)
            a12, as12 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 2)
            a13, as13 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 3)

            mfma_4n(_acc_idx(1, 0, 0), a00, as00, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(1, 1, 0), a01, as01, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(1, 2, 0), a02, as02, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(1, 3, 0), a03, as03, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            rocdl.sched_barrier(0)
            _barrier(LOAD_PASSES_A_SUBTILE + LOAD_PASSES_B_SUBTILE, lgkmcnt=0)
            rocdl.sched_barrier(0)

            next_a0_regs = load_a_subtile_mi_regs(next_a, next_scales, 0, 0)

            mfma_4n(_acc_idx(2, 0, 0), a10, as10, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(2, 1, 0), a11, as11, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(2, 2, 0), a12, as12, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(2, 3, 0), a13, as13, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            next_b0_regs = load_b_subtile_regs(next_b, next_scales, 0)
            mfma_4n(_acc_idx(3, 0, 0), a10, as10, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(3, 1, 0), a11, as11, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(3, 2, 0), a12, as12, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(3, 3, 0), a13, as13, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            return next_a0_regs, next_b0_regs

        def hk_one_k_final(cur_a, cur_b, a0_regs, b0_regs, cur_scales):
            _barrier(vmcnt=0, lgkmcnt=0)

            b00, b01, b02, b03, bs00, bs01, bs02, bs03 = b0_regs
            a00, as00 = a0_regs
            a01, as01 = load_a_subtile_mi_regs(cur_a, cur_scales, 0, 1)
            a02, as02 = load_a_subtile_mi_regs(cur_a, cur_scales, 0, 2)
            a03, as03 = load_a_subtile_mi_regs(cur_a, cur_scales, 0, 3)
            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            b10, bs10 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 0)
            b11, bs11 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 1)
            b12, bs12 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 2)
            b13, bs13 = load_b_subtile_ni_regs(cur_b, cur_scales, 1, 3)

            mfma_4n(_acc_idx(0, 0, 0), a00, as00, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(0, 1, 0), a01, as01, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(0, 2, 0), a02, as02, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(0, 3, 0), a03, as03, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            a10, as10 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 0)
            a11, as11 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 1)
            a12, as12 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 2)
            a13, as13 = load_a_subtile_mi_regs(cur_a, cur_scales, 1, 3)

            mfma_4n(_acc_idx(1, 0, 0), a00, as00, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(1, 1, 0), a01, as01, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(1, 2, 0), a02, as02, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(1, 3, 0), a03, as03, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

            rocdl.sched_barrier(0)
            _barrier(lgkmcnt=0)
            rocdl.sched_barrier(0)

            mfma_4n(_acc_idx(2, 0, 0), a10, as10, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(2, 1, 0), a11, as11, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(2, 2, 0), a12, as12, b00, b01, b02, b03, bs00, bs01, bs02, bs03)
            mfma_4n(_acc_idx(2, 3, 0), a13, as13, b00, b01, b02, b03, bs00, bs01, bs02, bs03)

            mfma_4n(_acc_idx(3, 0, 0), a10, as10, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(3, 1, 0), a11, as11, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(3, 2, 0), a12, as12, b10, b11, b12, b13, bs10, bs11, bs12, bs13)
            mfma_4n(_acc_idx(3, 3, 0), a13, as13, b10, b11, b12, b13, bs10, bs11, bs12, bs13)

        # Prologue: stage K0/K1 data into ping-pong LDS pages. Scales are not staged in
        # LDS: As/Bs are already MFMA-ready preshuffled packed uint32 [K128, row],
        # and load_scale_tile returns the current wave's scale operands in VGPRs.

        # Load scales first, so that they become the oldest VMEM ops.
        scales0 = load_scale_tile(fx.Index(0))
        scales1 = load_scale_tile(fx.Index(1))

        stage_a_subtile(fx.Index(0), 0, lds_a0_base_ptr, _SCOPE["a0"], _NOALIAS["a0"])
        stage_b_subtile(fx.Index(0), 0, lds_b0_base_ptr, _SCOPE["b0"], _NOALIAS["b0"])
        stage_b_subtile(fx.Index(0), 1, lds_b0_base_ptr, _SCOPE["b0"], _NOALIAS["b0"])
        stage_a_subtile(fx.Index(0), 1, lds_a0_base_ptr, _SCOPE["a0"], _NOALIAS["a0"])

        stage_a_subtile(fx.Index(BLOCK_K), 0, lds_a1_base_ptr, _SCOPE["a1"], _NOALIAS["a1"])
        stage_b_subtile(fx.Index(BLOCK_K), 0, lds_b1_base_ptr, _SCOPE["b1"], _NOALIAS["b1"])
        stage_b_subtile(fx.Index(BLOCK_K), 1, lds_b1_base_ptr, _SCOPE["b1"], _NOALIAS["b1"])
        stage_a_subtile(fx.Index(BLOCK_K), 1, lds_a1_base_ptr, _SCOPE["a1"], _NOALIAS["a1"])

        rocdl.sched_barrier(0)
        _barrier(vmcnt=3 * LOAD_PASSES_A_SUBTILE + 4 * LOAD_PASSES_B_SUBTILE)
        rocdl.sched_barrier(0)

        # scales0 is already MFMA-ready; no byte extraction or broadcast is needed.
        # Keep the hot loop consistent for k=0 and k>0:
        # K0 is consumed directly.  K1 MFMA-ready scales are carried as
        # prev_refill_scales and become next_scales_ready at loop entry.

        # Carry only A-top mi=0 across iterations to limit VGPR pressure;
        # b0_regs carries the complete B-left 64-row register tile.
        a0_regs = load_a_subtile_mi_regs(lds_a0, scales0, 0, 0)

        rocdl.sched_barrier(0)
        _barrier(vmcnt=3 * LOAD_PASSES_A_SUBTILE + 3 * LOAD_PASSES_B_SUBTILE)
        rocdl.sched_barrier(0)

        # Preload B-left for K0 to registers.
        b0_regs = load_b_subtile_regs(lds_b0, scales0, 0)

        # Main HK loop: exactly one logical K128 per iteration.
        # Even k consumes and refills LDS0; odd k does the same for LDS1.
        # Scale tiles follow the same K128 progression but remain in VGPRs.
        refill_scales = scales1  # K1 scales become the next ready scale tile at loop entry
        for k128 in range_constexpr(NUM_K_TILES - 2):
            if (k128 % 2) == 0:
                a0_regs, b0_regs, scales1, refill_scales = hk_one_k_with_refill(
                    k128,
                    lds_a0,
                    lds_b0,
                    lds_a1,
                    lds_b1,
                    lds_a0_base_ptr,
                    lds_b0_base_ptr,
                    _SCOPE["a0"],
                    _NOALIAS["a0"],
                    _SCOPE["b0"],
                    _NOALIAS["b0"],
                    a0_regs,
                    b0_regs,
                    scales0,
                    refill_scales,
                )
            else:
                a0_regs, b0_regs, scales0, refill_scales = hk_one_k_with_refill(
                    k128,
                    lds_a1,
                    lds_b1,
                    lds_a0,
                    lds_b0,
                    lds_a1_base_ptr,
                    lds_b1_base_ptr,
                    _SCOPE["a1"],
                    _NOALIAS["a1"],
                    _SCOPE["b1"],
                    _NOALIAS["b1"],
                    a0_regs,
                    b0_regs,
                    scales1,
                    refill_scales,
                )

        # Common two-page tail.  After the steady loop, a0_regs/b0_regs are
        # for the next page after the last consumed K128 tile.  The final scale
        # tile returned by the last refill belongs to the LDS page that was just
        # refilled, so the tail page order depends on parity:
        #   even NUM_K_TILES: consume LDS0 then final LDS1
        #   odd  NUM_K_TILES: consume LDS1 then final LDS0
        if (NUM_K_TILES % 2) == 0:
            scales1 = refill_scales
            a0_regs, b0_regs = hk_one_k_tail_with_next(
                lds_a0,
                lds_b0,
                lds_a1,
                lds_b1,
                a0_regs,
                b0_regs,
                scales0,
                scales1,
            )
            hk_one_k_final(lds_a1, lds_b1, a0_regs, b0_regs, scales1)
        else:
            scales0 = refill_scales
            a0_regs, b0_regs = hk_one_k_tail_with_next(
                lds_a1,
                lds_b1,
                lds_a0,
                lds_b0,
                a0_regs,
                b0_regs,
                scales1,
                scales0,
            )
            hk_one_k_final(lds_a0, lds_b0, a0_regs, b0_regs, scales0)

        rocdl.sched_barrier(0)
        _barrier()
        rocdl.sched_barrier(0)

        store_output()

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            linkage = ir.Attribute.parse("#llvm.linkage<external>")
            for sym, size in (
                (LDS_SYM_A0, LDS_BYTES_A),
                (LDS_SYM_A1, LDS_BYTES_A),
                (LDS_SYM_B0, LDS_BYTES_B),
                (LDS_SYM_B1, LDS_BYTES_B),
            ):
                llvm.GlobalOp(
                    global_type=ir.Type.parse(f"!llvm.array<{size} x i8>"),
                    sym_name=sym,
                    linkage=linkage,
                    addr_space=3,
                    alignment=1024,
                )

        # The integration only dispatches aligned shapes; no partial-tile masking exists.
        grid_x = (c_m // BLOCK_M) * (c_n // BLOCK_N)
        kernel_gemm(
            A,
            B,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(NUM_THREADS, 1, 1), smem=0, stream=stream)

    return launch_gemm
