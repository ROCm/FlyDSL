"""MLA (Multi-Latent Attention) decode kernel for FlyDSL.

Optimized for short query sequences (seqlen_q <= 4) with GQA.
Implements absorbed MLA where the KV cache stores [c_kv || k_rope] per token.

Key design:
- MFMA 16x16x16f16 for both GEMM stages.
- 4 waves per workgroup, each wave handles one seqlen_q position.
  Each MFMA tile computes 16 Q heads simultaneously (lane_mod_16 = head).
- No if-else boundary checks for KV loads (assumes seqlen_kv % BLOCK_N == 0).
- Online softmax in registers; P kept in registers for direct GEMM2 feed.
- Three LDS buffers: two for K (double-buffered prefetch), one for V^T.
  V is NOT reloaded from global memory — it is read directly from the
  current K buffer in LDS (since c_kv serves as both K_nope and V),
  transposed in registers, and written to the V^T buffer.
- BLOCK_N=16 to fit all three buffers within 64 KB LDS.
- Paged KV cache via block_table: logical KV positions are mapped to
  physical blocks through a per-batch block table (i32 indices).

GEMM1: S = K @ Q^T   using HEAD_DIM_QK = kv_lora_rank + qk_rope_head_dim
GEMM2: O^T = V^T @ P  using HEAD_DIM_V  = kv_lora_rank

Layout (1D flattened):
  Q       : [batch, seqlen_q,  num_q_heads,  HEAD_DIM_QK]
  KV      : [num_physical_blocks, page_block_size, num_kv_heads, HEAD_DIM_QK]
  Mid_O   : [batch*seqlen_q, num_kv_splits, num_q_heads, HEAD_DIM_V]  (fp32)
  Mid_lse : [batch*seqlen_q, num_kv_splits, num_q_heads]              (fp32)
  block_table : [batch, max_num_blocks]   (i32 physical block ids)

KV split mode:
  The KV sequence is partitioned into num_kv_splits contiguous chunks.
  Each split produces normalised partial output (fp32) and log-sum-exp
  (lse = m + log(l)), compatible with the stage-2 combine kernel that
  merges splits via online log-sum-exp.

Grid:  (batch, num_head_groups, num_kv_splits)  -- num_head_groups = nhead/16
Block: (256,) -- 4 waves of 64, each wave handles one seqlen_q row

LDS layout (f16 element offsets):
  K_buf_0 : [0,                     K_BUF_ELEMS)
  K_buf_1 : [K_BUF_ELEMS,           2*K_BUF_ELEMS)
  V^T_buf : [2*K_BUF_ELEMS,         2*K_BUF_ELEMS + VT_BUF_ELEMS)

Requires: kv_lora_rank % 16 == 0, qk_rope_head_dim % 16 == 0,
          GQA group size >= 4, seqlen_kv % BLOCK_N == 0,
          page_block_size % BLOCK_N == 0, page_block_size <= 64.
"""

import math

from flydsl.dialects.ext import flir, arith, gpu, scf, rocdl, buffer_ops, llvm
from flydsl.dialects.ext import vector as vec_ext
from flydsl.dialects.ext import memref as memref_ext
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.dialects.ext.scf import yield_ as scf_yield
from _mlir.dialects import memref as _memref
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
from _mlir import ir
import _mlir.extras.types as T
from _mlir.dialects import memref as memref_dialect


KERNEL_NAME = "mla_decode_kernel"


def _set_mfma_vgpr_form():
    """Force MFMA to use ACC_CD=0 (D/C in ArchVGPR) via LLVM cl::opt.

    This allows MFMA src_b to reside in AccVGPR (e.g. Q loaded via
    ds_read with ACC=1) while keeping the accumulator in ArchVGPR,
    avoiding the register-class cascade that inflates agpr_count.
    """
    import ctypes
    import os
    lib_dir = os.path.dirname(
        __import__('_mlir._mlir_libs', fromlist=['_mlir_libs']).__file__
    )
    lib = ctypes.CDLL(os.path.join(lib_dir, 'libFlirPythonCAPI.so'))
    parse_fn = lib.LLVMParseCommandLineOptions
    parse_fn.restype = None
    parse_fn.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_char_p,
    ]
    argv = [b'mlir', b'-amdgpu-mfma-vgpr-form']
    argv_arr = (ctypes.c_char_p * len(argv))(*argv)
    parse_fn(len(argv), argv_arr, None)


_set_mfma_vgpr_form()


def build_mla_decode_module(
    num_q_heads=16,
    num_kv_heads=1,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    page_block_size=64,
    causal=True,
    dtype_str="f16",
    sm_scale=None,
):
    gpu_arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()

    BLOCK_N = 16
    NUM_WAVES = 4
    WARP_SIZE = 64
    BLOCK_SIZE = NUM_WAVES * WARP_SIZE
    HEADS_PER_WAVE = 16

    NUM_Q_HEADS = num_q_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM_QK = kv_lora_rank + qk_rope_head_dim
    HEAD_DIM_V = kv_lora_rank
    GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS
    NUM_HEAD_GROUPS = NUM_Q_HEADS // HEADS_PER_WAVE
    CAUSAL = causal
    PAGE_BLOCK_SIZE = page_block_size

    assert NUM_Q_HEADS % HEADS_PER_WAVE == 0
    assert GQA_GROUP >= HEADS_PER_WAVE
    assert HEAD_DIM_V % 16 == 0
    assert qk_rope_head_dim % 16 == 0
    assert dtype_str == "f16"
    assert PAGE_BLOCK_SIZE % BLOCK_N == 0, (
        f"page_block_size ({PAGE_BLOCK_SIZE}) must be a multiple of BLOCK_N ({BLOCK_N})"
    )
    assert PAGE_BLOCK_SIZE <= 64, (
        f"page_block_size ({PAGE_BLOCK_SIZE}) must be <= 64"
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    K_STEPS = HEAD_DIM_QK // 32
    N_KV_SUBTILES = BLOCK_N // 16  # 1
    D_CHUNKS = HEAD_DIM_V // 16
    PV_K_STEPS = N_KV_SUBTILES     # 1

    STRIDE_TOKEN_Q = NUM_Q_HEADS * HEAD_DIM_QK
    STRIDE_TOKEN_KV = NUM_KV_HEADS * HEAD_DIM_QK
    STRIDE_PAGE = PAGE_BLOCK_SIZE * STRIDE_TOKEN_KV

    K_STRIDE = (
        (HEAD_DIM_QK + 2 * WARP_SIZE - 1)
        // (2 * WARP_SIZE) * (2 * WARP_SIZE)
    )  # 640: exactly 5 buffer_load_dword calls per wave fill one row
    VT_STRIDE = (BLOCK_N // 4) * 8  # 32  (per dim-pair: [seqlen/4, 2, 4], ds_read_b128)

    # ---- LDS buffer sizes (f16 elements) ----
    K_BUF_ELEMS = BLOCK_N * K_STRIDE              # 16 * 640 = 10240
    VT_BUF_ELEMS = (HEAD_DIM_V // 2) * VT_STRIDE  # 256 * 32 = 8192
    VT_BUF_OFFSET = 2 * K_BUF_ELEMS               # 20480
    KV_LDS_SIZE = 2 * K_BUF_ELEMS + VT_BUF_ELEMS  # 28672

    # ---- Per-wave K cooperative load ----
    K_CALLS_PER_ROW = K_STRIDE // (2 * WARP_SIZE)  # 5
    K_ROWS_PER_WAVE = BLOCK_N // NUM_WAVES          # 4

    # ---- Q preload buffer (64×64 tiled, overlaps LDS before K/V phase) ----
    # All 4 seqlens × 16 heads = 64 rows; loop over dim in 64-wide tiles.
    MAX_SEQLEN_Q = 4
    Q_TILE_DIM = 64
    Q_TILE_ROWS = MAX_SEQLEN_Q * HEADS_PER_WAVE            # 64
    Q_DIM_TILES = HEAD_DIM_QK // Q_TILE_DIM                 # 9  (576/64)
    Q_TILE_ELEMS = Q_TILE_ROWS * Q_TILE_DIM                 # 4096 f16
    assert HEAD_DIM_QK % Q_TILE_DIM == 0, (
        f"HEAD_DIM_QK ({HEAD_DIM_QK}) must be a multiple of Q_TILE_DIM ({Q_TILE_DIM})"
    )
    Q_CALLS_PER_WAVE = HEADS_PER_WAVE // 2                   # 8  (2 rows per call)
    Q_MAX_BATCH = KV_LDS_SIZE // Q_TILE_ELEMS                # 7  tiles/batch

    LDS_SIZE = max(KV_LDS_SIZE, Q_TILE_ELEMS)               # 28672

    # ---- V^T transpose: [dim/2, seqlen/4, 2, 4] layout ----
    assert HEAD_DIM_V // 2 == BLOCK_SIZE  # 128 threads × 2 dim-pairs each

    # Pre-build K load inline asm string (pure Python, no MLIR)
    _kload_lines = []
    for _r in range(K_ROWS_PER_WAVE):
        _m0off = _r * K_STRIDE * 2
        if _r == 0:
            _kload_lines.append("s_mov_b32 m0, $2")
        else:
            _kload_lines.append("s_add_i32 m0, $2, " + hex(_m0off))
        _kload_lines.append("s_nop 0")
        _soff = "0" if _r == 0 else ("$" + str(_r + 2))
        for _j in range(K_CALLS_PER_ROW):
            _off = "" if _j == 0 else (
                " offset:" + str(_j * 2 * WARP_SIZE * 2))
            _kload_lines.append(
                "buffer_load_dword $1, $0, "
                + _soff + " offen" + _off + " sc0 lds")
    _KLOAD_ASM = "\n".join(_kload_lines)
    _KLOAD_CONSTRAINTS = ",".join(
        ["s", "v", "s"]
        + ["s"] * (K_ROWS_PER_WAVE - 1)
        + ["~{m0}"])

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name = "smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name = "smem1")
    allocator_vt = SmemAllocator(None, arch=gpu_arch, global_sym_name = "smemvt")
    allocator_red = SmemAllocator(None, arch=gpu_arch, global_sym_name = "smem_red")
    _state = {}

    class _MLADecode(flir.MlirModule):
        GPU_MODULE_NAME = f"mla_decode_{dtype_str}"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type = T.f16()
            _state["elem_type"] = elem_type
            _state["lds_k_cur"] = allocator_pong.allocate_array(elem_type, K_BUF_ELEMS)
            _state["lds_k_next"] = allocator_ping.allocate_array(elem_type, K_BUF_ELEMS)
            _state["lds_vt"] = allocator_vt.allocate_array(elem_type, VT_BUF_ELEMS)
            _state["lds_red"] = allocator_red.allocate_array(T.f32(), 256)
            allocator_pong.finalize()
            allocator_ping.finalize()
            allocator_vt.finalize()
            allocator_red.finalize()

        @flir.kernel
        def mla_decode_kernel(
            self: flir.T.i64,
            Q: lambda: T.memref(DYN, _state["elem_type"]),
            KV: lambda: T.memref(DYN, _state["elem_type"]),
            Mid_O: lambda: T.memref(DYN, T.f32()),
            Mid_lse: lambda: T.memref(DYN, T.f32()),
            block_table: lambda: T.memref(DYN, T.i32()),
            batch_size: lambda: T.index(),
            seqlen_q: lambda: T.index(),
            kv_indptr: lambda: T.memref(DYN, T.i32()),
            max_num_blocks: lambda: T.index(),
            num_kv_splits: lambda: T.index(),
        ):
            compute_type = T.f32()
            elem_type = _state["elem_type"]
            fm_fast = flir.arith.FastMathFlags.fast

            v4f16_type = ir.VectorType.get([4], elem_type)
            v4f32_type = ir.VectorType.get([4], compute_type)
            v8f16_type = ir.VectorType.get([8], elem_type)
            i32_type = ir.IntegerType.get_signless(32)
            v2f16_type = ir.VectorType.get([2], elem_type)
            v1i32_type = ir.VectorType.get([1], i32_type)

            seqlen_q_v = arith.as_value(seqlen_q)
            max_num_blocks_v = arith.as_value(max_num_blocks)
            batch_size_v = arith.as_value(batch_size)

            base_ptr = allocator_pong.get_base()
            base_ptr1 = allocator_ping.get_base()
            base_ptr_vt = allocator_vt.get_base()
            base_ptr_red = allocator_red.get_base()

            lds_k_cur_buf = _state["lds_k_cur"](base_ptr).get()
            lds_k_next_buf = _state["lds_k_next"](base_ptr1).get()
            lds_vt = _state["lds_vt"](base_ptr_vt).get()
            lds_red = _state["lds_red"](base_ptr_red).get()

            # ---- Thread / block indices ----
            batch_idx = flir.const_index(flir.block_idx("x"))
            # s_load_dwordx2: scalar load kv_indptr[batch_idx] and kv_indptr[batch_idx+1]
            ptr_idx = arith.as_value(
                _memref.ExtractAlignedPointerAsIndexOp(kv_indptr).result
            )
            i64_type = ir.IntegerType.get_signless(64)
            base_ptr_i64 = arith.as_value(
                flir.arith.IndexCastOp(i64_type, ptr_idx).result
            )
            batch_byte_off = buffer_ops.index_cast_to_i32(
                (arith.ArithValue(batch_idx) * 4).value
            )
            v2i32_kv = ir.VectorType.get([2], i32_type)
            asm_op = llvm.InlineAsmOp(
                v2i32_kv,
                [base_ptr_i64, batch_byte_off],
                "s_load_dwordx2 $0, $1, $2\ns_waitcnt lgkmcnt(0)",
                "=s,s,s",
                has_side_effects=False,
                is_align_stack=False,
            )
            kv_pair = arith.as_value(asm_op.res)
            kv_start_i32 = arith.as_value(
                vec_ext.extract(kv_pair, static_position=[0], dynamic_position=[])
            )
            kv_end_i32 = arith.as_value(
                vec_ext.extract(kv_pair, static_position=[1], dynamic_position=[])
            )
            kv_len_i32 = arith.as_value(
                flir.arith.SubIOp(kv_end_i32, kv_start_i32).result
            )
            seqlen_kv_v = arith.as_value(
                flir.arith.IndexCastOp(T.index(), kv_len_i32).result
            )
            head_group_idx = flir.const_index(flir.block_idx("y"))
            split_id = flir.const_index(flir.block_idx("z"))
            tid = flir.const_index(flir.thread_idx("x"))

            c_ws = flir.const_index(WARP_SIZE)
            wave_id = arith.as_value(flir.arith.DivUIOp(tid, c_ws).result)
            lane = arith.as_value(flir.arith.RemUIOp(tid, c_ws).result)

            c16 = flir.const_index(16)
            lane_mod_16 = arith.as_value(flir.arith.RemUIOp(lane, c16).result)
            lane_div_16 = arith.as_value(flir.arith.DivUIOp(lane, c16).result)

            _c2 = flir.const_index(2)
            lane_mod_16_x2 = arith.as_value(
                flir.arith.MulIOp(lane_mod_16, _c2).result
            )

            # Per-lane head index: lane_mod_16 selects one of 16 heads
            q_head_idx = (
                arith.ArithValue(head_group_idx) * HEADS_PER_WAVE
                + arith.ArithValue(lane_mod_16)
            ).value
            q_head_base = (
                arith.ArithValue(head_group_idx) * HEADS_PER_WAVE
            ).value
            c_gqa = flir.const_index(GQA_GROUP)
            kv_head_idx = arith.as_value(
                flir.arith.DivUIOp(q_head_base, c_gqa).result
            )

            kv_head_offset = (arith.ArithValue(kv_head_idx) * HEAD_DIM_QK).value

            # ---- KV split range ----
            num_kv_splits_v = arith.as_value(num_kv_splits)
            c_bn_split = flir.const_index(BLOCK_N)
            c1_split = flir.const_index(1)
            total_kv_tiles = arith.as_value(
                flir.arith.DivUIOp(seqlen_kv_v, c_bn_split).result
            )
            tiles_nkv_m1 = arith.as_value(
                flir.arith.SubIOp(num_kv_splits_v, c1_split).result
            )
            tiles_plus = arith.as_value(
                flir.arith.AddIOp(total_kv_tiles, tiles_nkv_m1).result
            )
            tiles_per_split = arith.as_value(
                flir.arith.DivUIOp(tiles_plus, num_kv_splits_v).result
            )
            kv_per_split = (
                arith.ArithValue(tiles_per_split) * BLOCK_N
            ).value
            kv_split_start = (
                arith.ArithValue(split_id)
                * arith.ArithValue(kv_per_split)
            ).value
            kv_split_end = arith.as_value(
                flir.arith.MinUIOp(
                    (arith.ArithValue(kv_split_start)
                     + arith.ArithValue(kv_per_split)).value,
                    seqlen_kv_v,
                ).result
            )

            # ---- Mid_O / Mid_lse strides ----
            # Mid_O:   [total_q, num_kv_splits, NUM_Q_HEADS, HEAD_DIM_V]
            # Mid_lse: [total_q, num_kv_splits, NUM_Q_HEADS]
            stride_mid_o_token = (
                arith.ArithValue(num_kv_splits_v)
                * (NUM_Q_HEADS * HEAD_DIM_V)
            ).value
            stride_mid_lse_token = (
                arith.ArithValue(num_kv_splits_v) * NUM_Q_HEADS
            ).value

            # ---- V^T transpose decomposition: each thread handles two dim pairs ----
            # 128 threads per row × 4 f16/thread = 512 = HEAD_DIM_V.
            # Two halves (tid<128 vs tid>=128) cover interleaved groups.
            c_128 = flir.const_index(128)
            tid_mod_128 = arith.as_value(flir.arith.RemUIOp(tid, c_128).result)
            tid_div_128 = arith.as_value(flir.arith.DivUIOp(tid, c_128).result)
            vt_dim_quad = (arith.ArithValue(tid_mod_128) * 4).value
            c_perm_lo = arith.constant(0x05040100, type=i32_type)
            c_perm_hi = arith.constant(0x07060302, type=i32_type)

            # ---- Global index helpers (Q / O only — KV uses paged addressing) ----
            def q_global_idx(q_row, d_col):
                token = (
                    arith.ArithValue(batch_idx) * arith.ArithValue(seqlen_q_v)
                    + arith.ArithValue(q_row)
                )
                return (
                    token * STRIDE_TOKEN_Q
                    + arith.ArithValue(q_head_idx) * HEAD_DIM_QK
                    + arith.ArithValue(d_col)
                ).value

            def mid_o_global_idx(q_row, d_col):
                total_q = (
                    arith.ArithValue(batch_idx) * arith.ArithValue(seqlen_q_v)
                    + arith.ArithValue(q_row)
                )
                return (
                    total_q * arith.ArithValue(stride_mid_o_token)
                    + arith.ArithValue(split_id) * (NUM_Q_HEADS * HEAD_DIM_V)
                    + arith.ArithValue(q_head_idx) * HEAD_DIM_V
                    + arith.ArithValue(d_col)
                ).value

            def mid_lse_global_idx(q_row):
                total_q = (
                    arith.ArithValue(batch_idx) * arith.ArithValue(seqlen_q_v)
                    + arith.ArithValue(q_row)
                )
                return (
                    total_q * arith.ArithValue(stride_mid_lse_token)
                    + arith.ArithValue(split_id) * NUM_Q_HEADS
                    + arith.ArithValue(q_head_idx)
                ).value

            # ---- Block table lookup ----
            c_pbs = flir.const_index(PAGE_BLOCK_SIZE)

            # max valid log_block = max_num_blocks - 1 (for clamping speculative access)
            max_log_block = arith.as_value(
                flir.arith.SubIOp(max_num_blocks_v, flir.const_index(1)).result
            )

            def lookup_page(kv_abs_pos):
                """Map absolute KV position → (page_base, page_offset).

                page_base  = physical_block * STRIDE_PAGE  (raw index SSA)
                page_offset = kv_abs_pos % page_block_size (raw index SSA)

                log_block is clamped to [0, max_num_blocks-1] so that
                speculative execution on GPU does not cause OOB block table access.
                """
                log_block = arith.as_value(
                    flir.arith.DivUIOp(kv_abs_pos, c_pbs).result
                )
                log_block = arith.as_value(
                    flir.arith.MinUIOp(log_block, max_log_block).result
                )
                p_off = arith.as_value(
                    flir.arith.RemUIOp(kv_abs_pos, c_pbs).result
                )
                bt_idx = (
                    arith.ArithValue(batch_idx)
                    * arith.ArithValue(max_num_blocks_v)
                    + log_block
                ).value
                phys_i32 = memref_ext.load(block_table, [bt_idx])
                phys = arith.index_cast(T.index(), phys_i32)
                page_base = (phys * STRIDE_PAGE).value
                return page_base, p_off

            bt_rsrc = buffer_ops.create_buffer_resource(block_table)
            _bt_aux = arith.constant(0, type=T.i32())
            _bt_base_i32 = buffer_ops.index_cast_to_i32(
                (arith.ArithValue(batch_idx)
                 * arith.ArithValue(max_num_blocks_v)).value
            )
            _c6_i32 = arith.constant(6, type=T.i32())
            _c2_i32 = arith.constant(2, type=T.i32())
            _bt_soff = arith.as_value(
                flir.arith.ShLIOp(
                    arith.unwrap(_bt_base_i32),
                    arith.unwrap(_c2_i32),
                ).result
            )
            _c_pbs_mask_i32 = arith.constant(
                PAGE_BLOCK_SIZE - 1, type=T.i32()
            )

            def lookup_page_issue(kv_abs_pos, clamp=True,
                                   pos_i32=None):
                """Issue page-table buffer load (VMEM, vmcnt-tracked).

                clamp: clamp log_block to max_log_block (for speculative
                       loads near boundaries).  Pass False when the
                       position is guaranteed valid to skip v_cmp+s_cselect
                       and keep the entire computation in 32-bit SALU.
                pos_i32: pre-computed i32 position (avoids 64→32 trunc
                         that LLVM folds into v_alignbit_b32).
                """
                if clamp:
                    p_off = arith.as_value(
                        flir.arith.RemUIOp(kv_abs_pos, c_pbs).result
                    )
                    log_block = arith.as_value(
                        flir.arith.DivUIOp(kv_abs_pos, c_pbs).result
                    )
                    log_block = arith.as_value(
                        flir.arith.MinUIOp(
                            log_block, max_log_block
                        ).result
                    )
                    bt_byte_off = buffer_ops.index_cast_to_i32(
                        (arith.ArithValue(log_block) * 4).value
                    )
                else:
                    if pos_i32 is None:
                        pos_i32 = buffer_ops.index_cast_to_i32(
                            kv_abs_pos)
                    _and_res = flir.arith.AndIOp(
                        arith.unwrap(pos_i32),
                        arith.unwrap(_c_pbs_mask_i32),
                    ).result
                    p_off = arith.as_value(
                        flir.arith.IndexCastOp(
                            T.index(), arith.unwrap(_and_res)
                        ).result
                    )
                    log_block_i32 = arith.as_value(
                        flir.arith.ShRUIOp(
                            arith.unwrap(pos_i32),
                            arith.unwrap(_c6_i32),
                        ).result
                    )
                    bt_byte_off = arith.as_value(
                        flir.arith.ShLIOp(
                            arith.unwrap(log_block_i32),
                            arith.unwrap(_c2_i32),
                        ).result
                    )
                phys_i32 = rocdl.raw_ptr_buffer_load(
                    i32_type, bt_rsrc, bt_byte_off,
                    arith.unwrap(_bt_soff), arith.unwrap(_bt_aux),
                )
                return phys_i32, p_off

            def lookup_page_resolve(phys_i32):
                """Convert a loaded phys_i32 → page_base index."""
                phys = arith.index_cast(T.index(), phys_i32)
                return (phys * STRIDE_PAGE).value

            # Per-row soffset constants for per-wave K load
            K_ROW_SOFFSETS = [
                arith.constant(r * STRIDE_TOKEN_KV * 2, type=T.i32())
                for r in range(K_ROWS_PER_WAVE)
            ]
            K_J_OFFSETS = [
                arith.constant(j * 2 * WARP_SIZE * 2, type=T.i32())
                for j in range(K_CALLS_PER_ROW)
            ]

            # ---- Cooperative K load: paged global → K_buf (k_buf memref) ----
            # Uses llvm.GEPOp for LDS address generation to preserve pointer
            # provenance, preventing the compiler from inserting false-dep
            # vmcnt(0) between buffer_load_lds and subsequent ds_reads.
            def coop_load_k(kv_page_base, page_off, k_buf):
                g_f16_wave_row0 = (
                    arith.ArithValue(kv_page_base)
                    + (
                        arith.ArithValue(page_off)
                        + arith.ArithValue(wave_id) * K_ROWS_PER_WAVE
                    )
                    * STRIDE_TOKEN_KV
                    + arith.ArithValue(kv_head_offset)
                ).value
                voff_base = (
                    arith.ArithValue(g_f16_wave_row0) * 2
                    + arith.ArithValue(lane) * 4
                ).value
                voff_base_i32 = buffer_ops.index_cast_to_i32(voff_base)

                lds_offset = (
                    arith.ArithValue(wave_id)
                    * (K_ROWS_PER_WAVE * K_STRIDE * 2)
                ).value
                lds_base = memref_dialect.extract_aligned_pointer_as_index(k_buf)
                lds_base_lane0 = rocdl.readfirstlane(T.i64(), arith.index_cast(T.i64(), lds_base))
                lds_ptr_base = buffer_ops.create_llvm_ptr(lds_base_lane0, address_space=3)
                lds_ptr_row0 = buffer_ops.get_element_ptr(
                    lds_ptr_base, lds_offset,
                )

                for r_off in range_constexpr(K_ROWS_PER_WAVE):
                    lds_ptr = buffer_ops.get_element_ptr(
                        lds_ptr_row0,
                        static_byte_offset=r_off * (K_STRIDE * 2),
                    )
                    for j in range_constexpr(K_CALLS_PER_ROW):
                        rocdl.raw_ptr_buffer_load_lds(
                            kv_rsrc,
                            lds_ptr,
                            arith.unwrap(c_dword_sz),
                            voff_base_i32,
                            arith.unwrap(K_ROW_SOFFSETS[r_off]),
                            arith.unwrap(K_J_OFFSETS[j]),
                            arith.unwrap(aux),
                        )

            # ---- Cooperative K load (inline asm): hides buffer_load_dword_lds
            #      from the compiler's SIInsertWaitcnts pass, preventing
            #      false-dependency vmcnt(0) before subsequent ds_reads that
            #      target a different LDS region (e.g., VT_buf).
            def coop_load_k_asm(kv_page_base, page_off, k_buf):
                g_f16_wave_row0 = (
                    arith.ArithValue(kv_page_base)
                    + (
                        arith.ArithValue(page_off)
                        + arith.ArithValue(wave_id) * K_ROWS_PER_WAVE
                    )
                    * STRIDE_TOKEN_KV
                    + arith.ArithValue(kv_head_offset)
                ).value
                voff_base = (
                    arith.ArithValue(g_f16_wave_row0) * 2
                    + arith.ArithValue(lane) * 4
                ).value
                voff_base_i32 = buffer_ops.index_cast_to_i32(voff_base)

                buf_byte_idx = arith.as_value(
                    _memref.ExtractAlignedPointerAsIndexOp(k_buf).result
                )
                lds_row0_byte = (
                    arith.ArithValue(buf_byte_idx)
                    + arith.ArithValue(wave_id)
                    * (K_ROWS_PER_WAVE * K_STRIDE * 2)
                ).value

                m0_base_i32 = buffer_ops.index_cast_to_i32(lds_row0_byte)
                m0_base_sgpr = arith.as_value(
                    rocdl.readfirstlane(T.i32(), m0_base_i32)
                )

                operands = [
                    arith.unwrap(kv_rsrc),
                    voff_base_i32,
                    arith.unwrap(m0_base_sgpr),
                ]
                for r in [1, 2, 3]:
                    operands.append(
                        arith.unwrap(K_ROW_SOFFSETS[r])
                    )

                llvm.InlineAsmOp(
                    res=None,
                    operands_=operands,
                    asm_string=_KLOAD_ASM,
                    constraints=_KLOAD_CONSTRAINTS,
                    has_side_effects=True,
                    is_align_stack=False,
                )

            # ---- V transpose: K_buf(LDS) → V^T_buf(LDS) ----
            # Split into two phases:
            #   1. coop_load_v: ds_read from K buffer → raw dwords in VGPRs
            #   2. coop_perm_store_v: v_perm rearrangement + ds_write to VT buffer
            v2i32_type = ir.VectorType.get([2], i32_type)

            def coop_load_v(k_buf):
                """Phase 1: ds_read V data from K buffer into VGPRs.

                Returns list of (dwords_dp0, dwords_dp1) per g_pair.
                Each dwords_dpX is a list of 4 i32 values.
                """
                vt_read_base = (
                    arith.ArithValue(tid_div_128) * (4 * K_STRIDE)
                    + arith.ArithValue(vt_dim_quad)
                )
                v_raw = []
                for g_pair in range_constexpr(BLOCK_N // 8):
                    dwords_dp0 = []
                    dwords_dp1 = []
                    for s in range_constexpr(4):
                        lds_idx = (
                            vt_read_base
                            + (g_pair * 8 + s) * K_STRIDE
                        ).value
                        quad = arith.as_value(
                            vec_ext.load_op(v4f16_type, k_buf, [lds_idx])
                        )
                        dw_pair = arith.as_value(
                            vec_ext.bitcast(v2i32_type, quad)
                        )
                        dwords_dp0.append(arith.as_value(
                            vec_ext.extract(
                                dw_pair,
                                static_position=[0],
                                dynamic_position=[],
                            )
                        ))
                        dwords_dp1.append(arith.as_value(
                            vec_ext.extract(
                                dw_pair,
                                static_position=[1],
                                dynamic_position=[],
                            )
                        ))
                    v_raw.append((dwords_dp0, dwords_dp1))
                return v_raw

            def coop_perm_store_v(v_raw):
                """Phase 2: v_perm rearrangement + ds_write to VT buffer.

                v_raw: list of (dwords_dp0, dwords_dp1) per g_pair,
                       as returned by coop_load_v.
                """
                vt_write_base = (
                    arith.ArithValue(tid_mod_128) * (2 * VT_STRIDE)
                    + arith.ArithValue(tid_div_128) * 8
                )
                for g_pair in range_constexpr(BLOCK_N // 8):
                    dwords_dp0, dwords_dp1 = v_raw[g_pair]
                    for dp_idx, dwords in enumerate([dwords_dp0, dwords_dp1]):
                        out = []
                        for src_pair, sel in [
                            ((1, 0), c_perm_lo), ((3, 2), c_perm_lo),
                            ((1, 0), c_perm_hi), ((3, 2), c_perm_hi),
                        ]:
                            dw = arith.as_value(llvm.call_intrinsic(
                                i32_type, "llvm.amdgcn.perm",
                                [dwords[src_pair[0]], dwords[src_pair[1]], sel],
                                [], [],
                            ))
                            v1 = arith.as_value(
                                vec_ext.from_elements(v1i32_type, [dw])
                            )
                            out.append(arith.as_value(
                                vec_ext.bitcast(v2f16_type, v1)
                            ))
                        v4_lo = arith.as_value(
                            vec_ext.shuffle(out[0], out[1], [0, 1, 2, 3])
                        )
                        v4_hi = arith.as_value(
                            vec_ext.shuffle(out[2], out[3], [0, 1, 2, 3])
                        )
                        vec = arith.as_value(vec_ext.shuffle(
                            v4_lo, v4_hi, [0, 1, 2, 3, 4, 5, 6, 7],
                        ))
                        vt_idx = (
                            vt_write_base
                            + dp_idx * VT_STRIDE
                            + g_pair * 16
                        ).value
                        vec_ext.store(vec, lds_vt, [vt_idx])

            def coop_transpose_v_from_lds(k_buf):
                """Combined load + perm + store (backward-compatible wrapper)."""
                v_raw = coop_load_v(k_buf)
                coop_perm_store_v(v_raw)

            # ---- Preload Q via 64×64 tiled loop ----
            # 64 rows = 4 seqlens × 16 heads; loop over dim in Q_TILE_DIM=64 chunks.
            # Each iteration: 256 threads cooperatively load a [64, 64] tile from
            # global Q to LDS, then each wave reads its 16 rows into registers.
            c1_idx = flir.const_index(1)
            q_row = wave_id

            q_rsrc = buffer_ops.create_buffer_resource(Q)

            # K loading reads K_STRIDE (padded to 2*WARP_SIZE alignment) f16
            # per row, which exceeds the actual STRIDE_TOKEN_KV per row.
            # For the last physical block's last row, the overread would
            # access memory beyond the KV allocation.  Setting NUM_RECORDS
            # to the actual buffer size enables hardware OOB protection
            # (returns 0 instead of faulting).
            kv_total_blocks = (
                arith.ArithValue(batch_size_v)
                * arith.ArithValue(max_num_blocks_v)
            ).value
            kv_total_elems = (
                arith.ArithValue(kv_total_blocks) * STRIDE_PAGE
            ).value
            kv_size_bytes = (arith.ArithValue(kv_total_elems) * 2).value
            c_max_records = flir.const_index(0x7FFFFFFE)
            kv_size_capped = arith.as_value(
                flir.arith.MinUIOp(kv_size_bytes, c_max_records).result
            )
            kv_rsrc = buffer_ops.create_buffer_resource(
                KV, num_records_bytes=kv_size_capped
            )
            lds_base_byte_idx = arith.as_value(
                _memref.ExtractAlignedPointerAsIndexOp(lds_k_cur_buf).result
            )
            c_dword_sz = arith.constant(4, type=T.i32())
            c_zero_i32 = arith.constant(0, type=T.i32())
            aux = arith.constant(1, type=T.i32())

            q_b_packs_lo = [None] * K_STEPS
            q_b_packs_hi = [None] * K_STEPS

            # Per-lane decomposition for buffer_load_dword_lds:
            # 64 lanes → 32 lanes/row × 2 rows/call, 8 calls/wave = 16 rows/wave.
            c32 = flir.const_index(32)
            lane_div_32 = arith.as_value(
                flir.arith.DivUIOp(lane, c32).result
            )
            lane_mod_32 = arith.as_value(
                flir.arith.RemUIOp(lane, c32).result
            )

            q_voff_base_f16 = (
                arith.ArithValue(batch_idx)
                * arith.ArithValue(seqlen_q_v)
                * STRIDE_TOKEN_Q
                + arith.ArithValue(wave_id) * STRIDE_TOKEN_Q
                + arith.ArithValue(head_group_idx)
                * (HEADS_PER_WAVE * HEAD_DIM_QK)
                + arith.ArithValue(lane_div_32) * HEAD_DIM_QK
                + arith.ArithValue(lane_mod_32) * 2
            ).value

            Q_LDS_PAIR_BYTES = 2 * Q_TILE_DIM * 2              # 256
            Q_GLOBAL_PAIR_BYTES = 2 * HEAD_DIM_QK * 2          # 2304
            Q_CALL_SOFFSETS = [
                arith.constant(
                    c * (Q_GLOBAL_PAIR_BYTES - Q_LDS_PAIR_BYTES),
                    type=i32_type,
                )
                for c in range(Q_CALLS_PER_WAVE)
            ]
            Q_CALL_INST_OFFSETS = [
                arith.constant(c * Q_LDS_PAIR_BYTES, type=i32_type)
                for c in range(Q_CALLS_PER_WAVE)
            ]

            Q_TILE_BYTES = Q_TILE_ELEMS * 2               # 8192 bytes/tile

            # Issue page-table lookups for the first TWO K tiles BEFORE
            # the Q preload.  The global_load_dword latency (~300 cycles) is
            # fully hidden by the Q preload (~500+ cycles).  Q preload's
            # vmcnt(0) waits for both Q loads AND these page-table loads.
            pf0_phys_i32, pf0_p_off = lookup_page_issue(kv_split_start)
            tile1_start_early = (
                arith.ArithValue(kv_split_start) + BLOCK_N
            ).value
            pf1_phys_early, pf1_p_off_early = lookup_page_issue(
                tile1_start_early
            )

            d_tile_start = 0
            while d_tile_start < Q_DIM_TILES:
                batch_size = min(Q_DIM_TILES - d_tile_start, Q_MAX_BATCH)

                # ---- Issue all buffer_load_dword…lds for this batch ----
                for t in range_constexpr(batch_size):
                    d_tile = d_tile_start + t

                    tile_lds_byte = (
                        arith.ArithValue(lds_base_byte_idx)
                        + t * Q_TILE_BYTES
                        + arith.ArithValue(wave_id)
                        * (HEADS_PER_WAVE * Q_TILE_DIM * 2)
                    ).value
                    tile_lds_i64 = arith.as_value(
                        flir.arith.IndexCastOp(
                            T.i64(), tile_lds_byte
                        ).result
                    )
                    tile_lds_scalar = rocdl.readfirstlane(
                        T.i64(), tile_lds_i64
                    )
                    tile_lds_ptr = buffer_ops.create_llvm_ptr(
                        tile_lds_scalar, address_space=3
                    )

                    q_voff_byte = (
                        (arith.ArithValue(q_voff_base_f16)
                         + d_tile * Q_TILE_DIM) * 2
                    ).value
                    q_voff_i32 = buffer_ops.index_cast_to_i32(
                        q_voff_byte
                    )

                    for c in range_constexpr(Q_CALLS_PER_WAVE):
                        rocdl.raw_ptr_buffer_load_lds(
                            q_rsrc, tile_lds_ptr,
                            arith.unwrap(c_dword_sz),
                            q_voff_i32,
                            arith.unwrap(Q_CALL_SOFFSETS[c]),
                            arith.unwrap(Q_CALL_INST_OFFSETS[c]),
                            arith.unwrap(aux),
                        )

                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()

                # ---- Read all tiles from LDS into AccVGPR (ds_read ACC=1) ----
                # Two ds_read_b64 per step produce q_lo/q_hi directly in
                # AccVGPR without bitcast/shuffle (which would move values
                # back to ArchVGPR).
                for t in range_constexpr(batch_size):
                    d_tile = d_tile_start + t
                    q_tile_byte_base = (
                        arith.ArithValue(lds_base_byte_idx)
                        + (flir.const_index(t * Q_TILE_ELEMS)
                           + (arith.ArithValue(wave_id)
                              * HEADS_PER_WAVE
                              + arith.ArithValue(lane_mod_16))
                           * Q_TILE_DIM
                           + arith.ArithValue(lane_div_16) * 8) * 2
                    ).value
                    q_tile_addr_i32 = buffer_ops.index_cast_to_i32(
                        q_tile_byte_base
                    )
                    for local_ks in range_constexpr(2):
                        global_ks = d_tile * 2 + local_ks
                        _base_off = local_ks * 64
                        _ds_128 = llvm.InlineAsmOp(
                            res=v8f16_type,
                            operands_=[q_tile_addr_i32],
                            asm_string=(
                                f"ds_read_b128 $0, $1"
                                f" offset:{_base_off}"
                            ),
                            constraints="=a,v",
                            has_side_effects=True,
                            is_align_stack=False,
                        )
                        q_wide = arith.as_value(_ds_128.result)
                        q_b_packs_lo[global_ks] = arith.as_value(
                            vec_ext.shuffle(
                                q_wide, q_wide, [0, 1, 2, 3])
                        )
                        q_b_packs_hi[global_ks] = arith.as_value(
                            vec_ext.shuffle(
                                q_wide, q_wide, [4, 5, 6, 7])
                        )

                gpu.barrier()
                d_tile_start += batch_size

            # ---- Constants ----
            c_neg_inf = arith.constant(float("-inf"), type=compute_type)
            c_neg_large = arith.constant(-1e6, type=compute_type)
            c_zero_f = arith.constant(0.0, type=compute_type)
            c_one_f = arith.constant(1.0, type=compute_type)
            c_sm_scale = arith.constant(sm_scale, type=compute_type)
            c_log2e = arith.constant(1.4426950408889634, type=compute_type)
            c_zero_v4f32 = arith.as_value(
                arith.constant_vector(0.0, v4f32_type)
            )
            c0_idx = flir.const_index(0)
            c_kbuf = flir.const_index(K_BUF_ELEMS)

            N_DP = D_CHUNKS // 2

            def _vt_idx(dp):
                return (
                    (dp * 16 + arith.ArithValue(lane_mod_16))
                    * VT_STRIDE
                    + arith.ArithValue(lane_div_16) * 8
                ).value

            def vgpr_pin(val):
                """Opaque identity: prevent LLVM from rematerializing (LDS reload sinking)."""
                return arith.as_value(
                    llvm.InlineAsmOp(
                        res=arith.unwrap(val).type,
                        operands_=[arith.unwrap(val)],
                        asm_string="; vgpr_pin",
                        constraints="=v,0",
                        has_side_effects=True,
                        is_align_stack=False,
                    ).result
                )

            def softmax_and_pack(s_acc, m_old, l_old,
                                 kv_pos=None, do_causal=False):
                """Online softmax: s_acc → (m_new, l_new, p_pack, rescale).

                Updates running max m and sum l every tile.
                Cross-group max reduction via LDS (replaces ShuffleOp).
                """
                s_vals = [None] * 4
                for ii in range_constexpr(4):
                    s_val = arith.as_value(
                        vec_ext.extract(
                            s_acc,
                            static_position=[ii],
                            dynamic_position=[],
                        )
                    )
                    s_val = arith.as_value(
                        flir.arith.MulFOp(
                            s_val,
                            arith.as_value(c_sm_scale),
                            fastmath=fm_fast,
                        ).result
                    )
                    if CAUSAL and do_causal:
                        kv_abs = (
                            arith.ArithValue(kv_pos)
                            + arith.ArithValue(lane_div_16) * 4
                            + ii
                        ).value
                        q_abs_pos = (
                            arith.ArithValue(seqlen_kv_v)
                            - arith.ArithValue(seqlen_q_v)
                            + arith.ArithValue(q_row)
                        ).value
                        kv_abs_i64 = arith.as_value(
                            flir.arith.IndexCastOp(T.i64(), kv_abs).result
                        )
                        q_abs_i64 = arith.as_value(
                            flir.arith.IndexCastOp(T.i64(), q_abs_pos).result
                        )
                        is_masked = arith.as_value(
                            flir.arith.CmpIOp(
                                flir.arith.CmpIPredicate.ugt,
                                kv_abs_i64,
                                q_abs_i64,
                            ).result
                        )
                        s_val = arith.as_value(
                            flir.arith.SelectOp(
                                is_masked,
                                arith.as_value(c_neg_large),
                                s_val,
                            ).result
                        )
                    s_vals[ii] = s_val

                # Per-lane local max (max of 4 S values)
                local_max = s_vals[0]
                for ii in range_constexpr(3):
                    local_max = arith.as_value(
                        flir.arith.MaximumFOp(
                            local_max, s_vals[ii + 1]
                        ).result
                    )

                # Cross-group max reduction via LDS (transposed layout:
                # lane_mod_16*4 + lane_div_16 so each lane's 4 group
                # values are contiguous → ds_read_b128 instead of
                # ds_read2_b32 + 2×ds_read_b32).
                red_wr_idx = (
                    arith.ArithValue(wave_id) * 64
                    + arith.ArithValue(lane_mod_16) * 4
                    + arith.ArithValue(lane_div_16)
                ).value
                _memref.StoreOp(local_max, lds_red, [red_wr_idx])
                rocdl.s_waitcnt(lgkmcnt=0)

                red_rd_base = (
                    arith.ArithValue(wave_id) * 64
                    + arith.ArithValue(lane_mod_16) * 4
                ).value
                max_vec = arith.as_value(
                    vec_ext.load_op(v4f32_type, lds_red,
                                    [red_rd_base]))
                global_max = arith.as_value(c_neg_large)
                for g in range_constexpr(4):
                    max_g = arith.as_value(
                        vec_ext.extract(
                            max_vec,
                            static_position=[g],
                            dynamic_position=[],
                        )
                    )
                    global_max = arith.as_value(
                        flir.arith.MaximumFOp(global_max, max_g).result
                    )
                rocdl.s_waitcnt(lgkmcnt=0)

                m_new = arith.as_value(
                    flir.arith.MaximumFOp(m_old, global_max).result
                )

                # rescale = exp2((m_old - m_new) * log2e)
                m_diff = arith.as_value(
                    flir.arith.SubFOp(
                        m_old, m_new, fastmath=fm_fast
                    ).result
                )
                m_diff_scaled = arith.as_value(
                    flir.arith.MulFOp(
                        m_diff, arith.as_value(c_log2e), fastmath=fm_fast,
                    ).result
                )
                rescale = arith.as_value(
                    flir.math.exp2(m_diff_scaled, fastmath=fm_fast)
                )

                # P[i] = exp2((s[i] - m_new) * log2e), local_sum = sum(P)
                p_vals = [None] * 4
                local_sum = arith.as_value(c_zero_f)
                for ii in range_constexpr(4):
                    diff = arith.as_value(
                        flir.arith.SubFOp(
                            s_vals[ii], m_new, fastmath=fm_fast
                        ).result
                    )
                    diff_s = arith.as_value(
                        flir.arith.MulFOp(
                            diff, arith.as_value(c_log2e), fastmath=fm_fast,
                        ).result
                    )
                    p = arith.as_value(
                        flir.math.exp2(diff_s, fastmath=fm_fast)
                    )
                    p_vals[ii] = p
                    local_sum = arith.as_value(
                        flir.arith.AddFOp(
                            local_sum, p, fastmath=fm_fast
                        ).result
                    )

                # l_new = l_old * rescale + local_sum
                l_scaled = arith.as_value(
                    flir.arith.MulFOp(
                        l_old, rescale, fastmath=fm_fast
                    ).result
                )
                l_new = arith.as_value(
                    flir.arith.AddFOp(
                        l_scaled, local_sum, fastmath=fm_fast
                    ).result
                )

                p_f16 = []
                for ii in range_constexpr(4):
                    p_f16.append(
                        arith.as_value(
                            flir.arith.TruncFOp(
                                elem_type, p_vals[ii]
                            ).result
                        )
                    )
                p_pack = arith.as_value(
                    vec_ext.from_elements(v4f16_type, p_f16)
                )
                return m_new, l_new, p_pack, rescale

            def rescale_o_accs(o_accs, rescale):
                """Multiply all o_acc chunks by scalar rescale.

                Uses readfirstlane to create an opaque duplicate.
                readfirstlane is a convergent intrinsic that LLVM
                cannot fold, giving two distinct SSA values and
                preventing op_sel_hi on v_pk_mul_f32.
                """
                rescale_i32 = arith.as_value(
                    flir.arith.BitcastOp(T.i32(), arith.unwrap(rescale)).result
                )
                rescale_sgpr = rocdl.readfirstlane(T.i32(), rescale_i32)
                rescale_dup = arith.as_value(
                    flir.arith.BitcastOp(compute_type, arith.unwrap(rescale_sgpr)).result
                )
                rescale_vec = arith.as_value(
                    vec_ext.from_elements(
                        v4f32_type,
                        [rescale, rescale_dup,
                         rescale, rescale_dup])
                )
                for dc in range_constexpr(D_CHUNKS):
                    o_accs[dc] = arith.as_value(
                        flir.arith.MulFOp(
                            o_accs[dc], rescale_vec,
                            fastmath=fm_fast,
                        ).result
                    )

            def do_kv_tile(kv_pos, m_old, l_old, o_accs,
                           k_cur_buf, k_next_buf, k_buf,
                           load_knn=True, do_vt=True,
                           do_causal=False):
                """Process one KV tile (software-pipelined K/V prefetch).

                K data is already in k_buf (VGPRs), loaded by previous
                iteration's GEMM2 or by the prologue.

                Schedule per iteration:
                  page-table issue for k_next_next
                  → GEMM1: compute S from k_buf, interleave V loads
                  → resolve page-table + load k_next_next → k_cur (global→LDS)
                  → softmax
                  → GEMM2: compute O from v_buf, interleave K[next] loads
                  → barrier + V transpose k_next → VT
                """

                rocdl.sched_barrier(0)

                # ==== Issue page-table load early (before GEMM1) ====
                if load_knn:
                    nn_start = (arith.ArithValue(kv_pos)
                                + 2 * BLOCK_N).value
                    nn_phys_i32, nn_p_off = lookup_page_issue(nn_start)

                # ==== GEMM1: S = K @ Q^T ====
                # K data already in k_buf. Interleave V loads (for GEMM2).
                s_accs = [c_zero_v4f32]
                v_buf = [None] * N_DP
                for ks in range_constexpr(K_STEPS):
                    if ks < N_DP:
                        v_buf[ks] = arith.as_value(
                            vec_ext.load_op(
                                v8f16_type, lds_vt, [_vt_idx(ks)])
                        )
                    q_lo = q_b_packs_lo[ks]
                    q_hi = q_b_packs_hi[ks]
                    k_lo = arith.as_value(
                        vec_ext.shuffle(
                            k_buf[ks], k_buf[ks],
                            [0, 1, 2, 3])
                    )
                    k_hi = arith.as_value(
                        vec_ext.shuffle(
                            k_buf[ks], k_buf[ks],
                            [4, 5, 6, 7])
                    )
                    s_accs[0] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [k_lo, q_lo, s_accs[0], 0, 0, 0],
                        )
                    )
                    s_accs[0] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [k_hi, q_hi, s_accs[0], 0, 0, 0],
                        )
                    )

                if load_knn:
                    nn_page_base = lookup_page_resolve(nn_phys_i32)
                    coop_load_k(nn_page_base, nn_p_off, k_cur_buf)

                rocdl.sched_barrier(0)

                # ==== Softmax + P pack ====
                m_new, l_new, p_pack, rescale = softmax_and_pack(
                    s_accs[0], m_old, l_old,
                    kv_pos=kv_pos, do_causal=do_causal,
                )

                # ==== Rescale o_accs before GEMM2 ====
                rescale_o_accs(o_accs, rescale)
                  
                # ==== GEMM2: O^T = V^T @ P ====
                k_next_base_idx = (
                    arith.ArithValue(lane_mod_16) * K_STRIDE
                    + arith.ArithValue(lane_div_16) * 8
                ).value
                k_buf_next = [None] * K_STEPS
                EXTRA_K = K_STEPS - N_DP
                k_block = 0
                for dp in range_constexpr(N_DP):
                    k_next_idx = (
                        arith.ArithValue(k_next_base_idx)
                        + k_block * 32
                    ).value
                    k_buf_next[k_block] = arith.as_value(
                        vec_ext.load_op(
                            v8f16_type, k_next_buf, [k_next_idx])
                    )
                    k_block += 1
                    if dp < EXTRA_K:
                        k_extra_idx = (
                            arith.ArithValue(k_next_base_idx)
                            + k_block * 32
                        ).value
                        k_buf_next[k_block] = arith.as_value(
                            vec_ext.load_op(
                                v8f16_type, k_next_buf, [k_extra_idx])
                        )
                        k_block += 1
                    v_lo = arith.as_value(
                        vec_ext.shuffle(
                            v_buf[dp], v_buf[dp],
                            [0, 1, 2, 3])
                    )
                    v_hi = arith.as_value(
                        vec_ext.shuffle(
                            v_buf[dp], v_buf[dp],
                            [4, 5, 6, 7])
                    )
                    o_accs[dp * 2] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [v_lo, p_pack, o_accs[dp * 2], 0, 0, 0],
                        )
                    )
                    o_accs[dp * 2 + 1] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [v_hi, p_pack, o_accs[dp * 2 + 1], 0, 0, 0],
                        )
                    )

                if do_vt:
                    # rocdl.s_waitcnt(vmcnt=0)
                    # gpu.barrier()
                    coop_transpose_v_from_lds(k_next_buf)

                return m_new, l_new, o_accs, k_buf_next

            def do_kv_pair(kv_pos, m_old, l_old, o_accs,
                           k_cur_buf, k_next_buf, k_buf_a,
                           nn_a_phys_in=None,
                           load_knn=True, do_vt_final=True):
                """Process TWO KV tiles sequentially.

                Tiles A (kv_pos) and B (kv_pos + BLOCK_N).
                Pipeline (sequential per tile):
                  0. Page lookups (A+2 prefetched, B+2 issued, A+4 issued)
                  1. Resolve A+2, coop_load_k K[A+2] → k_cur_buf
                  -- Tile A --
                  2. GEMM1_A (k_buf_a @ Q → S_A) + V[A] VT loads
                  3. Softmax_A
                  4. Rescale o_accs *= rescale_a
                  5. GEMM2_A (v_buf_a @ P_A) + K[B] reads → k_buf_b
                  -- Transition --
                  6. coop_load_v(k_next_buf) + coop_perm_store_v → VT=V[B]^T
                  -- Tile B --
                  7. GEMM1_B (k_buf_b @ Q → S_B) + V[B] VT loads
                  8. Softmax_B
                  9. Rescale o_accs *= rescale_b
                  10. Resolve B+2, coop_load_k K[B+2] → k_next_buf
                  11. GEMM2_B (v_buf_b @ P_B) + K[A+2] reads → k_buf_a
                  12. coop_load_v + coop_perm_store_v → VT=V[A+2]^T
                """
                # ---- 0. Page lookups ----
                
                rocdl.sched_barrier(0)
                
                nn_a_phys_next = None
                if load_knn:
                    nn_a_phys = nn_a_phys_in
                    kv_pos_i32 = buffer_ops.index_cast_to_i32(kv_pos)
                    c_2bn_i32 = arith.constant(
                        2 * BLOCK_N, type=T.i32())
                    c_3bn_i32 = arith.constant(
                        3 * BLOCK_N, type=T.i32())
                    c_4bn_i32 = arith.constant(
                        4 * BLOCK_N, type=T.i32())
                    nn_a_pos_i32 = arith.as_value(
                        flir.arith.AddIOp(
                            arith.unwrap(kv_pos_i32),
                            arith.unwrap(c_2bn_i32),
                        ).result
                    )
                    _nn_a_and = flir.arith.AndIOp(
                        arith.unwrap(nn_a_pos_i32),
                        arith.unwrap(_c_pbs_mask_i32),
                    ).result
                    nn_a_poff = arith.as_value(
                        flir.arith.IndexCastOp(
                            T.index(), arith.unwrap(_nn_a_and)
                        ).result
                    )
                    nn_b_pos_i32 = arith.as_value(
                        flir.arith.AddIOp(
                            arith.unwrap(kv_pos_i32),
                            arith.unwrap(c_3bn_i32),
                        ).result
                    )
                    nn_b_phys, nn_b_poff = \
                        lookup_page_issue(
                            None, clamp=False, pos_i32=nn_b_pos_i32)
                    # nn_a_next is pos + 4*BLOCK_N = pos + PAGE_BLOCK_SIZE,
                    # so its block index is always (pos >> 6) + 1.
                    # Byte offset = (pos >> 6) << 2 + 4; LLVM extracts
                    # the +4 as buffer_load_dword offset:4.
                    _c4_i32 = arith.constant(4, type=T.i32())
                    kv_block_i32 = arith.as_value(
                        flir.arith.ShRUIOp(
                            arith.unwrap(kv_pos_i32),
                            arith.unwrap(_c6_i32),
                        ).result
                    )
                    kv_block_byte = arith.as_value(
                        flir.arith.ShLIOp(
                            arith.unwrap(kv_block_i32),
                            arith.unwrap(_c2_i32),
                        ).result
                    )
                    nn_a_next_byte = arith.as_value(
                        flir.arith.AddIOp(
                            arith.unwrap(kv_block_byte),
                            arith.unwrap(_c4_i32),
                        ).result
                    )
                    nn_a_phys_next = rocdl.raw_ptr_buffer_load(
                        i32_type, bt_rsrc, nn_a_next_byte,
                        arith.unwrap(_bt_soff), arith.unwrap(_bt_aux),
                    )

                # ---- 1. Resolve A+2, coop_load_k K[A+2] → k_cur_buf ----
                if load_knn:
                    nn_a_base = lookup_page_resolve(nn_a_phys)
                    coop_load_k(nn_a_base, nn_a_poff, k_cur_buf)

                # ============ TILE A ============

                # ---- 2. GEMM1_A: K[A] @ Q → S_A + V[A] VT loads ----
                s_a = [c_zero_v4f32]
                v_buf_a = [None] * N_DP
                for ks in range_constexpr(K_STEPS):
                    if ks < N_DP:
                        v_buf_a[ks] = arith.as_value(
                            vec_ext.load_op(
                                v8f16_type, lds_vt, [_vt_idx(ks)])
                        )
                    q_lo = q_b_packs_lo[ks]
                    q_hi = q_b_packs_hi[ks]
                    k_lo = arith.as_value(
                        vec_ext.shuffle(
                            k_buf_a[ks], k_buf_a[ks],
                            [0, 1, 2, 3])
                    )
                    k_hi = arith.as_value(
                        vec_ext.shuffle(
                            k_buf_a[ks], k_buf_a[ks],
                            [4, 5, 6, 7])
                    )
                    s_a[0] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [k_lo, q_lo, s_a[0], 0, 0, 0],
                        )
                    )
                    s_a[0] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [k_hi, q_hi, s_a[0], 0, 0, 0],
                        )
                    )

                rocdl.sched_barrier(0)

                # ---- 3. Softmax_A ----
                m_a, l_a, p_pack_a, rescale_a = softmax_and_pack(
                    s_a[0], m_old, l_old,
                )

                # ---- 4. Rescale o_accs *= rescale_a ----
                rescale_o_accs(o_accs, rescale_a)
                rocdl.sched_barrier(0)

                v_raw_b = coop_load_v(k_next_buf)

                # ---- 5. GEMM2_A: V[A] @ P_A → O + K[B] reads ----
                k_b_base = (
                    arith.ArithValue(lane_mod_16) * K_STRIDE
                    + arith.ArithValue(lane_div_16) * 8
                ).value
                k_buf_b = [None] * K_STEPS
                EXTRA_K = K_STEPS - N_DP
                k_block = 0
                for dp in range_constexpr(N_DP):
                    k_b_idx = (
                        arith.ArithValue(k_b_base) + k_block * 32
                    ).value
                    k_buf_b[k_block] = arith.as_value(
                        vec_ext.load_op(
                            v8f16_type, k_next_buf, [k_b_idx])
                    )
                    k_block += 1
                    if dp < EXTRA_K:
                        k_b_extra = (
                            arith.ArithValue(k_b_base)
                            + k_block * 32
                        ).value
                        k_buf_b[k_block] = arith.as_value(
                            vec_ext.load_op(
                                v8f16_type, k_next_buf, [k_b_extra])
                        )
                        k_block += 1

                    v_lo = arith.as_value(
                        vec_ext.shuffle(
                            v_buf_a[dp], v_buf_a[dp],
                            [0, 1, 2, 3])
                    )
                    v_hi = arith.as_value(
                        vec_ext.shuffle(
                            v_buf_a[dp], v_buf_a[dp],
                            [4, 5, 6, 7])
                    )
                    o_accs[dp * 2] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [v_lo, p_pack_a, o_accs[dp * 2],
                             0, 0, 0],
                        )
                    )
                    o_accs[dp * 2 + 1] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [v_hi, p_pack_a, o_accs[dp * 2 + 1],
                             0, 0, 0],
                        )
                    )

                # ---- 6. Transition: V[B] transpose → VT ----
                coop_perm_store_v(v_raw_b)
                rocdl.sched_barrier(0)

                # ============ TILE B ============

                # ---- 10. Resolve B+2, coop_load_k K[B+2] → k_next_buf ----
                if load_knn:
                    nn_b_base = lookup_page_resolve(nn_b_phys)
                    coop_load_k(nn_b_base, nn_b_poff, k_next_buf)

                # ---- 7. GEMM1_B: K[B] @ Q → S_B + V[B] VT loads ----
                s_b = [c_zero_v4f32]
                v_buf_b = [None] * N_DP
                for ks in range_constexpr(K_STEPS):
                    if ks < N_DP:
                        v_buf_b[ks] = arith.as_value(
                            vec_ext.load_op(
                                v8f16_type, lds_vt, [_vt_idx(ks)])
                        )
                    q_lo = q_b_packs_lo[ks]
                    q_hi = q_b_packs_hi[ks]
                    k_lo = arith.as_value(
                        vec_ext.shuffle(
                            k_buf_b[ks], k_buf_b[ks],
                            [0, 1, 2, 3])
                    )
                    k_hi = arith.as_value(
                        vec_ext.shuffle(
                            k_buf_b[ks], k_buf_b[ks],
                            [4, 5, 6, 7])
                    )
                    s_b[0] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [k_lo, q_lo, s_b[0], 0, 0, 0],
                        )
                    )
                    s_b[0] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [k_hi, q_hi, s_b[0], 0, 0, 0],
                        )
                    )

                rocdl.sched_barrier(0)
                # ---- 8. Softmax_B ----
                m_b, l_b, p_pack_b, rescale_b = softmax_and_pack(
                    s_b[0], m_a, l_a,
                )
                if do_vt_final:
                    v_raw_next = coop_load_v(k_cur_buf)

                # ---- 9. Rescale o_accs *= rescale_b ----
                rescale_o_accs(o_accs, rescale_b)
                rocdl.sched_barrier(0)

                # ---- 11. GEMM2_B: V[B] @ P_B → O + K[A+2] reads ----
                k_nn_base = (
                    arith.ArithValue(lane_mod_16) * K_STRIDE
                    + arith.ArithValue(lane_div_16) * 8
                ).value
                k_block = 0
                for dp in range_constexpr(N_DP):
                    k_nn_idx = (
                        arith.ArithValue(k_nn_base) + k_block * 32
                    ).value
                    k_buf_a[k_block] = arith.as_value(
                        vec_ext.load_op(
                            v8f16_type, k_cur_buf, [k_nn_idx])
                    )
                    k_block += 1
                    if dp < EXTRA_K:
                        k_nn_extra = (
                            arith.ArithValue(k_nn_base)
                            + k_block * 32
                        ).value
                        k_buf_a[k_block] = arith.as_value(
                            vec_ext.load_op(
                                v8f16_type, k_cur_buf, [k_nn_extra])
                        )
                        k_block += 1

                    v_lo = arith.as_value(
                        vec_ext.shuffle(
                            v_buf_b[dp], v_buf_b[dp],
                            [0, 1, 2, 3])
                    )
                    v_hi = arith.as_value(
                        vec_ext.shuffle(
                            v_buf_b[dp], v_buf_b[dp],
                            [4, 5, 6, 7])
                    )
                    o_accs[dp * 2] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [v_lo, p_pack_b, o_accs[dp * 2],
                             0, 0, 0],
                        )
                    )
                    o_accs[dp * 2 + 1] = arith.as_value(
                        rocdl.mfma_f32_16x16x16f16(
                            v4f32_type,
                            [v_hi, p_pack_b, o_accs[dp * 2 + 1],
                             0, 0, 0],
                        )
                    )

                # ---- 12. V[A+2] transpose → VT ----
                if do_vt_final:
                    coop_perm_store_v(v_raw_next)

                if nn_a_phys_next is not None:
                    nn_a_phys_next = rocdl.readfirstlane(
                        T.i32(), nn_a_phys_next)
                return (m_b, l_b, o_accs, k_buf_a,
                        nn_a_phys_next)

            has_kv_work = arith.as_value(
                flir.arith.CmpIOp(
                    flir.arith.CmpIPredicate.ult,
                    kv_split_start, kv_split_end,
                ).result
            )
            # ---- Pre-loop: double-prefetch K tiles 0 and 1 ----
            # Both tiles are loaded unconditionally.  lookup_page_issue
            # clamps to valid block indices, so speculative loads with
            # invalid positions are safe (data simply won't be consumed).
            page_base_0 = lookup_page_resolve(pf0_phys_i32)
            coop_load_k(page_base_0, pf0_p_off, lds_k_cur_buf)

            page_base_1 = lookup_page_resolve(pf1_phys_early)
            coop_load_k(page_base_1, pf1_p_off_early, lds_k_next_buf)
            rocdl.s_waitcnt(vmcnt=20)

            # Pre-transpose tile 0's V data so the peeled iteration's
            # GEMM2 can execute immediately without waiting.
            coop_transpose_v_from_lds(lds_k_cur_buf)
            
            rocdl.sched_barrier(0)

            # Prologue: pre-load all K[0] from LDS into AGPRs
            k_base_idx_init = (
                arith.ArithValue(lane_mod_16) * K_STRIDE
                + arith.ArithValue(lane_div_16) * 8
            ).value
            k_byte_base = (
                arith.ArithValue(lds_base_byte_idx)
                + arith.ArithValue(k_base_idx_init) * 2
            ).value
            k_byte_base_i32 = buffer_ops.index_cast_to_i32(k_byte_base)
            k_buf_init = []
            for ks in range_constexpr(K_STEPS):
                _off = ks * 64
                _ds_k = llvm.InlineAsmOp(
                    res=v8f16_type,
                    operands_=[arith.unwrap(k_byte_base_i32)],
                    asm_string=(
                        f"ds_read_b128 $0, $1 offset:{_off}"
                    ),
                    constraints="=a,v",
                    has_side_effects=True,
                    is_align_stack=False,
                )
                k_buf_init.append(arith.as_value(_ds_k.result))

            kv_if = scf.IfOp(has_kv_work, hasElse=False)
            with kv_if.then():
                # ---- Initial accumulators (no peeled first tile) ----
                o_accs_init = [arith.as_value(
                    arith.constant_vector(0.0, v4f32_type)
                ) for _ in range_constexpr(D_CHUNKS)]
                m_init = arith.as_value(c_neg_large)
                l_init = arith.as_value(c_zero_f)

                # ---- Loop bounds ----
                # The last 2 tiles are peeled: penultimate (V-transpose,
                # no k_next_next) and last (neither).  Their loops
                # collapse to 0 iterations when fewer tiles exist.

                # body_end: safe against unsigned underflow via select.
                three_tiles_end = (
                    arith.ArithValue(kv_split_start) + 3 * BLOCK_N
                ).value
                has_three = arith.as_value(
                    flir.arith.CmpIOp(
                        flir.arith.CmpIPredicate.ule,
                        three_tiles_end, kv_split_end,
                    ).result
                )
                body_end = arith.as_value(
                    flir.arith.SelectOp(
                        has_three,
                        (arith.ArithValue(kv_split_end)
                         - 2 * BLOCK_N).value,
                        kv_split_start,
                    ).result
                )

                # -- Pair loop (2-tile interleaved pipeline) --
                four_tiles = (
                    arith.ArithValue(kv_split_start) + 4 * BLOCK_N
                ).value
                has_four = arith.as_value(
                    flir.arith.CmpIOp(
                        flir.arith.CmpIPredicate.ule,
                        four_tiles, kv_split_end,
                    ).result
                )
                pair_ub = arith.as_value(
                    flir.arith.SelectOp(
                        has_four,
                        (arith.ArithValue(kv_split_end)
                         - 3 * BLOCK_N).value,
                        kv_split_start,
                    ).result
                )
                
                rocdl.sched_barrier(0)

                nn_a_pf_start = (
                    arith.ArithValue(kv_split_start) + 2 * BLOCK_N
                ).value
                nn_a_pf_phys_v, _ = lookup_page_issue(nn_a_pf_start)
                nn_a_pf_phys = rocdl.readfirstlane(
                    T.i32(), nn_a_pf_phys_v)

                _PKB = 5 + D_CHUNKS
                _PNN = _PKB + K_STEPS
                pair_init = ([kv_split_start, m_init, l_init]
                             + o_accs_init
                             + [lds_k_cur_buf, lds_k_next_buf]
                             + k_buf_init
                             + [nn_a_pf_phys])
                with scf.for_(kv_split_start, pair_ub,
                              2 * BLOCK_N,
                              iter_args=pair_init) as pair_loop:
                    pair_pos = arith.as_value(
                        pair_loop.induction_variable)
                    m_pair = arith.as_value(
                        pair_loop.inner_iter_args[1])
                    l_pair = arith.as_value(
                        pair_loop.inner_iter_args[2])
                    o_pair = [
                        arith.as_value(
                            pair_loop.inner_iter_args[3 + dc])
                        for dc in range_constexpr(D_CHUNKS)
                    ]
                    k_pair_cur = pair_loop.inner_iter_args[
                        3 + D_CHUNKS]
                    k_pair_next = pair_loop.inner_iter_args[
                        4 + D_CHUNKS]
                    k_pair_buf = [
                        arith.as_value(
                            pair_loop.inner_iter_args[_PKB + ks])
                        for ks in range_constexpr(K_STEPS)
                    ]
                    nn_a_p_in = pair_loop.inner_iter_args[_PNN]

                    (m_pr, l_pr, o_pr, kbuf_pr,
                     nn_a_p_out) = \
                        do_kv_pair(
                            pair_pos, m_pair, l_pair, o_pair,
                            k_pair_cur, k_pair_next,
                            k_pair_buf, nn_a_p_in,
                        )
                    next_pp = (
                        arith.ArithValue(pair_pos) + 2 * BLOCK_N
                    ).value
                    scf_yield([next_pp, m_pr, l_pr] + o_pr
                              + [k_pair_cur, k_pair_next]
                              + kbuf_pr
                              + [nn_a_p_out])

                pair_end = arith.as_value(pair_loop.results[0])
                m_from_pair = arith.as_value(
                    pair_loop.results[1])
                l_from_pair = arith.as_value(
                    pair_loop.results[2])
                o_from_pair = [
                    arith.as_value(pair_loop.results[3 + dc])
                    for dc in range_constexpr(D_CHUNKS)
                ]
                k_pc = pair_loop.results[3 + D_CHUNKS]
                k_pn = pair_loop.results[4 + D_CHUNKS]
                kbuf_fp = [
                    arith.as_value(pair_loop.results[_PKB + ks])
                    for ks in range_constexpr(K_STEPS)
                ]

                # -- Single-tile body: remainder after pairs --
                _KB = 4 + D_CHUNKS
                body_init = ([m_from_pair, l_from_pair] + o_from_pair
                             + [k_pc, k_pn] + kbuf_fp)
                with scf.for_(pair_end, body_end, BLOCK_N,
                              iter_args=body_init) as loop:
                    kv_pos = arith.as_value(loop.induction_variable)
                    m_running = arith.as_value(
                        loop.inner_iter_args[0])
                    l_running = arith.as_value(
                        loop.inner_iter_args[1])
                    o_accs = [
                        arith.as_value(loop.inner_iter_args[2 + dc])
                        for dc in range_constexpr(D_CHUNKS)
                    ]
                    k_cur_buf = loop.inner_iter_args[2 + D_CHUNKS]
                    k_next_buf = loop.inner_iter_args[3 + D_CHUNKS]
                    k_buf_loop = [
                        arith.as_value(loop.inner_iter_args[_KB + ks])
                        for ks in range_constexpr(K_STEPS)
                    ]

                    m_new, l_new, o_accs, k_buf_new = \
                        do_kv_tile(
                            kv_pos, m_running, l_running, o_accs,
                            k_cur_buf, k_next_buf, k_buf_loop,
                        )
                    scf_yield([m_new, l_new] + o_accs
                              + [k_next_buf, k_cur_buf]
                              + k_buf_new)

                m_body = arith.as_value(loop.results[0])
                l_body = arith.as_value(loop.results[1])
                o_body = [
                    arith.as_value(loop.results[2 + dc])
                    for dc in range_constexpr(D_CHUNKS)
                ]
                k_body_cur = loop.results[2 + D_CHUNKS]
                k_body_next = loop.results[3 + D_CHUNKS]
                k_buf_body = [
                    arith.as_value(loop.results[_KB + ks])
                    for ks in range_constexpr(K_STEPS)
                ]

                # -- Penultimate tile (0-or-1 iteration) --
                # V-transpose to prepare VT for the last tile's GEMM2;
                # no k_next_next load or page prefetch.
                two_tiles_end = (
                    arith.ArithValue(kv_split_start) + 2 * BLOCK_N
                ).value
                has_two = arith.as_value(
                    flir.arith.CmpIOp(
                        flir.arith.CmpIPredicate.ule,
                        two_tiles_end, kv_split_end,
                    ).result
                )
                penult_ub = arith.as_value(
                    flir.arith.SelectOp(
                        has_two,
                        (arith.ArithValue(kv_split_end)
                         - BLOCK_N).value,
                        body_end,
                    ).result
                )
                penult_init = ([m_body, l_body] + o_body
                               + [k_body_cur, k_body_next]
                               + k_buf_body)
                with scf.for_(body_end, penult_ub, BLOCK_N,
                              iter_args=penult_init) as penult_loop:
                    kv_pos_p = arith.as_value(
                        penult_loop.induction_variable)
                    m_p = arith.as_value(
                        penult_loop.inner_iter_args[0])
                    l_p = arith.as_value(
                        penult_loop.inner_iter_args[1])
                    o_p = [
                        arith.as_value(
                            penult_loop.inner_iter_args[2 + dc])
                        for dc in range_constexpr(D_CHUNKS)
                    ]
                    k_p_cur = penult_loop.inner_iter_args[
                        2 + D_CHUNKS]
                    k_p_next = penult_loop.inner_iter_args[
                        3 + D_CHUNKS]
                    k_buf_p = [
                        arith.as_value(
                            penult_loop.inner_iter_args[_KB + ks])
                        for ks in range_constexpr(K_STEPS)
                    ]

                    m_pn, l_pn, o_pn, k_buf_pn = do_kv_tile(
                        kv_pos_p, m_p, l_p, o_p,
                        k_p_cur, k_p_next, k_buf_p,
                        load_knn=False, do_vt=True,
                    )
                    scf_yield([m_pn, l_pn] + o_pn
                              + [k_p_next, k_p_cur]
                              + k_buf_pn)

                m_penult = arith.as_value(penult_loop.results[0])
                l_penult = arith.as_value(penult_loop.results[1])
                o_penult = [
                    arith.as_value(penult_loop.results[2 + dc])
                    for dc in range_constexpr(D_CHUNKS)
                ]
                k_last_cur = penult_loop.results[2 + D_CHUNKS]
                k_last_next = penult_loop.results[3 + D_CHUNKS]
                k_buf_last = [
                    arith.as_value(penult_loop.results[_KB + ks])
                    for ks in range_constexpr(K_STEPS)
                ]

                # -- Last tile (0-or-1 iteration) --
                # No k_next_next load, no V-transpose.
                last_init = ([m_penult, l_penult] + o_penult
                             + [k_last_cur, k_last_next]
                             + k_buf_last)
                with scf.for_(penult_ub, kv_split_end, BLOCK_N,
                              iter_args=last_init) as last_loop:
                    kv_pos_l = arith.as_value(
                        last_loop.induction_variable)
                    m_l = arith.as_value(
                        last_loop.inner_iter_args[0])
                    l_l = arith.as_value(
                        last_loop.inner_iter_args[1])
                    o_l = [
                        arith.as_value(
                            last_loop.inner_iter_args[2 + dc])
                        for dc in range_constexpr(D_CHUNKS)
                    ]
                    k_l_cur = last_loop.inner_iter_args[
                        2 + D_CHUNKS]
                    k_l_next = last_loop.inner_iter_args[
                        3 + D_CHUNKS]
                    k_buf_l = [
                        arith.as_value(
                            last_loop.inner_iter_args[_KB + ks])
                        for ks in range_constexpr(K_STEPS)
                    ]

                    m_ln, l_ln, o_ln, k_buf_ln = do_kv_tile(
                        kv_pos_l, m_l, l_l, o_l,
                        k_l_cur, k_l_next, k_buf_l,
                        load_knn=False, do_vt=False,
                        do_causal=True,
                    )
                    scf_yield([m_ln, l_ln] + o_ln
                              + [k_l_next, k_l_cur]
                              + k_buf_ln)

                m_final = arith.as_value(last_loop.results[0])
                l_partial = arith.as_value(last_loop.results[1])
                o_finals = [
                    arith.as_value(last_loop.results[2 + dc])
                    for dc in range_constexpr(D_CHUNKS)
                ]

                # Cross-group l reduction via LDS (transposed layout)
                red_wr_idx_epi = (
                    arith.ArithValue(wave_id) * 64
                    + arith.ArithValue(lane_mod_16) * 4
                    + arith.ArithValue(lane_div_16)
                ).value
                _memref.StoreOp(l_partial, lds_red, [red_wr_idx_epi])
                rocdl.s_waitcnt(lgkmcnt=0)

                red_rd_base_epi = (
                    arith.ArithValue(wave_id) * 64
                    + arith.ArithValue(lane_mod_16) * 4
                ).value
                l_vec = arith.as_value(
                    vec_ext.load_op(v4f32_type, lds_red,
                                    [red_rd_base_epi]))
                l_final = arith.as_value(c_zero_f)
                for g in range_constexpr(4):
                    l_g = arith.as_value(
                        vec_ext.extract(
                            l_vec,
                            static_position=[g],
                            dynamic_position=[],
                        )
                    )
                    l_final = arith.as_value(
                        flir.arith.AddFOp(
                            l_final, l_g, fastmath=fm_fast
                        ).result
                    )
                rocdl.s_waitcnt(lgkmcnt=0)

                l_gt_zero = arith.as_value(
                    flir.arith.CmpFOp(
                        flir.arith.CmpFPredicate.OGT,
                        l_final, arith.as_value(c_zero_f),
                    ).result
                )
                safe_inv_l = arith.as_value(
                    flir.arith.SelectOp(
                        l_gt_zero,
                        arith.as_value(
                            flir.arith.DivFOp(
                                arith.as_value(c_one_f), l_final,
                                fastmath=fm_fast,
                            ).result
                        ),
                        arith.as_value(c_zero_f),
                    ).result
                )
                inv_l_vec = arith.as_value(
                    vec_ext.broadcast(v4f32_type, safe_inv_l)
                )

                for dc in range_constexpr(D_CHUNKS):
                    o_norm = arith.as_value(
                        flir.arith.MulFOp(
                            o_finals[dc], inv_l_vec, fastmath=fm_fast
                        ).result
                    )
                    for ii in range_constexpr(4):
                        o_val = arith.as_value(
                            vec_ext.extract(
                                o_norm,
                                static_position=[ii],
                                dynamic_position=[],
                            )
                        )
                        d_col = (
                            flir.const_index((dc // 2) * 32 + dc % 2)
                            + arith.ArithValue(lane_div_16) * 8
                            + ii * 2
                        ).value
                        o_g_idx = mid_o_global_idx(q_row, d_col)
                        _memref.StoreOp(o_val, Mid_O, [o_g_idx])

                # lse = m + log(l)   (−inf for empty splits)
                log_l = arith.as_value(
                    flir.math.log(l_final, fastmath=fm_fast)
                )
                lse_val = arith.as_value(
                    flir.arith.AddFOp(
                        m_final, log_l, fastmath=fm_fast
                    ).result
                )
                lse_idx = mid_lse_global_idx(q_row)
                _memref.StoreOp(lse_val, Mid_lse, [lse_idx])

            # with kv_if.else_():
            #     for dc in range_constexpr(D_CHUNKS):
            #         for ii in range_constexpr(4):
            #             d_col = (
            #                 flir.const_index((dc // 2) * 32 + dc % 2)
            #                 + arith.ArithValue(lane_div_16) * 8
            #                 + ii * 2
            #             ).value
            #             o_g_idx = mid_o_global_idx(q_row, d_col)
            #             _memref.StoreOp(
            #                 arith.as_value(c_zero_f), Mid_O, [o_g_idx]
            #             )
            #     lse_idx = mid_lse_global_idx(q_row)
            #     _memref.StoreOp(
            #         arith.as_value(c_neg_inf), Mid_lse, [lse_idx]
            #     )

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Q: lambda: T.memref(DYN, _state["elem_type"]),
            KV: lambda: T.memref(DYN, _state["elem_type"]),
            Mid_O: lambda: T.memref(DYN, T.f32()),
            Mid_lse: lambda: T.memref(DYN, T.f32()),
            block_table: lambda: T.memref(DYN, T.i32()),
            batch_size: lambda: T.index(),
            seqlen_q: lambda: T.index(),
            kv_indptr: lambda: T.memref(DYN, T.i32()),
            max_num_blocks: lambda: T.index(),
            num_kv_splits: lambda: T.index(),
        ):
            c1 = arith.as_value(flir.arith_ext.index(1))
            c_nhg = arith.as_value(flir.arith_ext.index(NUM_HEAD_GROUPS))
            bs_val = arith.as_value(batch_size)
            nkv_splits_val = arith.as_value(num_kv_splits)
            bx = arith.as_value(flir.arith_ext.index(BLOCK_SIZE))
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, KERNEL_NAME],
                grid_size=(bs_val, c_nhg, nkv_splits_val),
                block_size=(bx, c1, c1),
                kernel_operands=[
                    Q, KV, Mid_O, Mid_lse, block_table,
                    batch_size,
                    seqlen_q, kv_indptr, max_num_blocks, num_kv_splits,
                ],
            )

    return _MLADecode()
