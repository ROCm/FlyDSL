# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE GEMM stage2 (down-projection MFMA) fp8/int8 kernel builder.

Layout-API pipeline: A2/W are fp8 (E4M3) or int8 on CDNA3/CDNA4 (gfx94*/gfx95*).
int8 shares the fp8 path via an i32 MFMA acc (converted to f32 in dequant).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

try:
    from flydsl.runtime.device import (
        bf16_global_atomics_arch_description,
        supports_bf16_global_atomics,
    )
except ImportError:
    # Fallback for runtime.device versions exposing only get_rocm_arch.
    def supports_bf16_global_atomics(arch: str) -> bool:
        return str(arch).startswith(("gfx94", "gfx95", "gfx12"))

    def bf16_global_atomics_arch_description() -> str:
        return "gfx94+/gfx95+/gfx12+"


from kernels.common import buffer_ops
from kernels.moe.moe_gemm_2stage import layout_helpers as fxh


def _build_moe_gemm2_fp8(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    out_dtype: str,
    accumulate: bool,
    in_dtype: str = "fp8",
):
    """Native stage2 down-projection (B-first MFMA, fp8/int8); out=(A2@W2^T)*a_scale*w_scale.
    Epilogue: atomic (f16/bf16/f32) or reduce (prefer reduce). int8 shares the fp8
    pipeline via an i32 MFMA acc converted to f32 in dequant. int8smooth bakes its
    smooth scale into A2 host-side, so stage2 sees plain int8 -- same path as int8.
    """
    is_int4 = in_dtype == "int4"  # W4A8: int8 activations x packed-int4 weights
    is_int8 = in_dtype in ("int8", "int8smooth") or is_int4
    elem_t = fx.Int8 if is_int8 else fx.Float8E4M3FNUZ
    acc_dtype = fx.Int32 if is_int8 else None
    MFMA_K = 32
    elem_bytes = elem_t.width // 8  # fp8/int8=1

    K = int(inter_dim)  # stage2 K dimension
    N = int(model_dim)  # stage2 output/N dimension
    BM = int(tile_m)
    BN = int(tile_n)
    TILE_K = int(tile_k)
    TOPK = int(topk)

    out_s = str(out_dtype).strip().lower()
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    out_elem = fx.Float32 if out_is_f32 else (fx.BFloat16 if out_is_bf16 else fx.Float16)
    out_bytes = 4 if out_is_f32 else 2

    # bf16 atomics: gfx95+/gfx12+ have buffer_atomic_pk_add_bf16; gfx942 has only
    # global_atomic_pk_add_bf16, so bf16+accumulate there uses GLOBAL (!llvm.ptr)
    # atomics. f16/f32 always use buffer atomics.
    _gpu_arch = get_rocm_arch()
    _has_buffer_atomic_bf16 = str(_gpu_arch).startswith(("gfx95", "gfx12"))
    _needs_global_atomic_bf16 = out_is_bf16 and accumulate and not _has_buffer_atomic_bf16
    if out_is_bf16 and accumulate and not supports_bf16_global_atomics(_gpu_arch):
        raise ValueError(
            f"out_dtype='bf16' with accumulate requires bf16 atomics "
            f"({bf16_global_atomics_arch_description()}), got arch={_gpu_arch!r}"
        )

    assert TILE_K in (128, 256), f"native stage2 needs tile_k in (128,256), got {TILE_K}"
    assert K % TILE_K == 0, f"K(inter_dim)={K} must be a multiple of TILE_K={TILE_K}"
    assert 64 <= BN <= 256 and BN % 64 == 0, f"tile_n must be in [64,256] multiple of 64, got {BN}"
    assert 16 <= BM <= 256 and BM % 16 == 0, f"tile_m must be a 16-multiple in [16,256], got {BM}"

    # B-first tiled_mma puts all 4 waves on the output (N) dim at 16 channels/wave,
    # so each block covers >=64 output channels (tile_n>=64 satisfies this).
    contiguous_n = BN
    assert contiguous_n % 64 == 0, f"tile_n={BN} must be a 64-multiple for the 4-wave B-first MMA"
    assert model_dim % contiguous_n == 0, f"model_dim={model_dim} must be divisible by tile_n={BN}"

    in_t = elem_t
    a_lds_bytes = BM * TILE_K * elem_bytes  # A LDS: BM*TILE_K activation elems
    cshuf_bytes = BM * BN * out_bytes  # CShuffle staging reuses the A LDS bytes
    # Single LDS region: A ping/pong (2*a_lds) in the loop, reused by the CShuffle
    # epilogue. max() must fit CDNA3 64KB / CDNA4 160KB. ping=region[0:], pong=+a_lds.
    region_bytes = max(2 * a_lds_bytes, cshuf_bytes)

    @fx.struct
    class GemmBuffers:
        region: fx.Array[fx.Int8, region_bytes, 16]

    @fx.union
    class SharedStorage:
        sorted_lds: fx.Array[fx.Int32, 256, 16]
        gemm: GemmBuffers

    _val_per_thr = 16 // elem_bytes  # elements per 128b buffer_load (fp8=16)
    _swz_params = (3, 4, 3)  # A-LDS swizzle, 8-bit fp8 (matches preshuffle_gemm)
    _thrs_k = TILE_K // _val_per_thr
    _thrs_m = 256 // _thrs_k
    _m_per_wave = _thrs_m // 4

    def _gemm_1x4(blk_n, arg_p_input, arg_p_weight, lds, M, expert_id):
        """B-first native-fp8 down-projection GEMM with A-gather + LDS ping-pong."""
        tid = gpu.thread_idx.x
        mma_atom, tiled_mma = fxh.make_1x4_tiled_mma(in_t, acc_dtype)

        a_tensor = fx.rocdl.make_buffer_tensor(
            arg_p_input, max_size=False, num_records_bytes=fx.Int64(M) * fx.Int64(K) * fx.Int64(elem_bytes)
        )
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)

        a_size_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((BM, K), (K, 1))), max_size=False
        )
        a_tile = fx.flat_divide(a_size_buf, fx.make_tile(BM, TILE_K))[None, None, 0, None]
        buf_cp_atom_r = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), in_t)
        g2r_tv_layout = fx.make_layout(
            ((_thrs_k, _thrs_m), (1, _val_per_thr)),
            ((_thrs_m * _val_per_thr, 1), (1, _thrs_m)),
        )
        a_mem_cp_g2r = fx.make_tiled_copy(buf_cp_atom_r, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K))
        cp_atom_sortid_a = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        tiled_copy_sortid_a = fx.make_tiled_copy(
            cp_atom_sortid_a,
            fx.make_layout(((_thrs_k, _m_per_wave, 4), 1), ((0, 1, _m_per_wave), 0)),
            fx.make_tile(_thrs_m),
        )
        a_index_frag = fxh.read_sorted_index(tiled_copy_sortid_a, tid, lds.sorted_lds, BM)
        a_idx = fxh.make_tensor_with_index(a_tensor, BM, TILE_K, a_index_frag, a_mem_cp_g2r, tid, TOPK)
        a_mem_thr = a_mem_cp_g2r.get_slice(tid).partition_S(a_tile)
        a_cp_frag = fx.make_fragment_like(a_mem_thr[None, None, None, 0])
        gpu.barrier()  # sorted_lds reads done before overwriting with A tile

        swz = fx.SwizzleType.get(*_swz_params)
        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), in_t)
        a_lds_bufs = []
        a_lds_w_bufs = []
        a_lds_r_bufs = []
        a_frag_bufs = []
        a_frag_retile_bufs = []
        for _buf_ptr in (lds.gemm.region.ptr, lds.gemm.region.ptr + a_lds_bytes):
            a_lds = fx.make_view(
                fx.recast_iter(in_t, _buf_ptr),
                fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((BM, TILE_K), order=(1, 0))),
            )
            a_r2s = fx.make_tiled_copy(uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)).get_slice(tid)
            a_lds_bufs.append(a_lds)
            a_lds_w_bufs.append(a_r2s.partition_D(a_lds))
            a_lds_r_bufs.append(fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(a_lds))
            a_frag = tiled_mma.make_fragment_B(a_lds)
            a_frag_bufs.append(a_frag)
            a_frag_retile_bufs.append(fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(a_frag))
        a_cp_frag_bufs = [a_cp_frag, fx.make_fragment_like(a_mem_thr[None, None, None, 0])]
        a_cp_frag_retile_bufs = [
            fx.make_tiled_copy(uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)).get_slice(tid).retile(f)
            for f in a_cp_frag_bufs
        ]

        # B (weight): single tile per block; global->register, prefetched one K-tile
        # ahead into a ping-pong pair of fragments.
        _ki_reps = TILE_K // 64  # 64-byte K super-steps per tile
        if const_expr(is_int4):
            # W4A8: packed-int4 weight via the ki-correct loader (load_weight_int4_frag).
            # Read as i32 dwords (align-4) so each buffer_load pulls 4 packed bytes.
            _w_i8_iter = fx.get_iter(arg_p_weight)
            _w_i32_ptr = fx.PointerType.get(fx.Int32.ir_type, _w_i8_iter.memspace, 4)
            b_raw = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(_w_i32_ptr, _w_i8_iter),
                    fx.make_layout((fx.Int32(experts * N * (K // 2) // 4),), (1,)),
                ),
                max_size=False,
            )
            b_layout_i4 = fxh.make_preshuffle_b_layout_int4(N, K)
            _expert_off = expert_id * fx.Int32((N * K) // 8)  # dwords per expert slab
            _col_base = blk_n * fx.Int32(contiguous_n)
            _wfake = fx.rocdl.make_buffer_tensor(
                fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((contiguous_n, TILE_K), (TILE_K, 1))),
                max_size=False,
            )
            _wtile = fx.flat_divide(_wfake, fx.make_tile(contiguous_n, TILE_K))[None, None, 0, 0]
            b_frag_bufs = [tiled_mma.make_fragment_A(_wtile) for _ in range(2)]
            _b_loads_per_tile = 0  # int4 loads are scalar buffer reads, not prefetched dwordx4
        else:
            b_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K))[None, None, blk_n, None]
            b_g2r = fx.make_tiled_copy_A(buf_cp_atom_r, tiled_mma).get_slice(tid)
            b_g2r_s = b_g2r.partition_S(b_tile)
            b_frag_bufs = [tiled_mma.make_fragment_A(b_tile[None, None, 0]) for _ in range(2)]
            b_ret_bufs = [b_g2r.retile(f) for f in b_frag_bufs]

            _b_loads_per_tile = fx.size(fx.get_shape(b_frag_bufs[0])).to_py_value() // _val_per_thr

        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((contiguous_n, BM), (BM, 1))), max_size=False
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(contiguous_n, BM))[None, None, 0, 0]
        c_frag = tiled_mma.make_fragment_C(c_fake)
        c_frag.fill(0)

        k_iters = TILE_K // (2 * MFMA_K)
        num_tiles = K // TILE_K

        def _load_gmem(kb, s):
            # kb is an fx.Int32 K-tile index (may be a runtime scf.for value).
            a_idx.copy(buf_cp_atom_r, kb, a_cp_frag_bufs[s])
            if const_expr(is_int4):
                fxh.load_weight_int4_frag(b_raw, b_layout_i4, b_frag_bufs[s], _expert_off, _col_base, kb, tid, _ki_reps)
            else:
                fx.copy(buf_cp_atom_r, b_g2r_s[None, None, None, kb], b_ret_bufs[s])

        def _write_a_lds(s):
            fx.copy(uni_cp_atom, a_cp_frag_retile_bufs[s], a_lds_w_bufs[s])

        def _read_a_lds(s):
            for ki in range_constexpr(k_iters):
                fx.copy(uni_cp_atom, a_lds_r_bufs[s][None, None, ki], a_frag_retile_bufs[s][None, None, ki])

        _m_reps = fxh.reps(c_frag, 1)
        _n_reps = fxh.reps(c_frag, 2)

        def _mfma(s):
            for ki in range_constexpr(k_iters):
                for n in range_constexpr(_n_reps):
                    for m in range_constexpr(_m_reps):
                        for k in range_constexpr(2):
                            fx.mma_atom_call(
                                mma_atom,
                                c_frag[None, m, n],
                                b_frag_bufs[s][None, m, (k, ki)],
                                a_frag_bufs[s][None, n, (k, ki)],
                                c_frag[None, m, n],
                            )

        # Rolled single-buffer scf.for: unrolled ping-pong kept both buffers live
        # (196 VGPR, 2 blocks/CU) and couldn't hide the atomic drain tail; rolling
        # drops to 130 VGPR (3 blocks/CU) while keeping intra-tile overlap.
        for iv in range(0, num_tiles, 1):
            kb = arith.index_cast(T.i32, iv)
            _load_gmem(kb, 0)
            rocdl.s_waitcnt(fxh._encode_waitcnt(vmcnt=_b_loads_per_tile))
            gpu.barrier()  # WAR: all waves finished reading the prior tile's A-LDS
            _write_a_lds(0)
            gpu.barrier()  # RAW: A-LDS write visible before the read below
            _read_a_lds(0)
            _mfma(0)
        return c_frag

    _gemm_1x4 = ASTRewriter.transform(_gemm_1x4)

    def _apply_dequant(c_frag, tid, expert_id, blk_n, asc_idx, M, arg_scale_w, arg_scale_x):
        # ptpc: per-channel (model_dim) weight scale, per-row act scale. Sentinel rows
        # get a_scale=0 (zero atomic contribution). fp8 folds in place; int8 i32->f32.
        m_reps = fxh.reps(c_frag, 1)
        n_reps = fxh.reps(c_frag, 2)
        out = fx.make_fragment_like(c_frag, dtype=fx.Float32) if const_expr(is_int8) else c_frag
        sw_ptr = fx.recast_iter(fx.Float32, fx.get_iter(arg_scale_w))
        scale_w = fx.make_view(sw_ptr + expert_id * N + blk_n * contiguous_n, fx.make_layout(contiguous_n, 1))
        cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        scale_copy = fx.make_tiled_copy(
            cp_atom_scale, fx.make_layout(((16, 4, 4), 4), ((0, 4, 16), 1)), fx.make_tile(64)
        )
        sw_thr = scale_copy.get_slice(tid).partition_S(scale_w)
        w_scale = fx.make_fragment_like(sw_thr)
        fx.copy(cp_atom_scale, sw_thr, w_scale)

        tokens = M // TOPK
        a_scale_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.recast_iter(fx.Float32, fx.get_iter(arg_scale_x)), fx.make_layout(M, 1)),
            max_size=False,
            num_records_bytes=fx.Int64(M) * fx.Int64(4),
        )
        # A2 scale row = token*TOPK + slot; sentinel (token>=tokens) -> scale 0.
        a_sc_n = []
        for n in range_constexpr(n_reps):
            packed = asc_idx[0, n]
            tok = packed & 0xFFFFFF
            slot = packed >> 24
            valid = tok < fx.Int32(tokens)
            row = valid.select(tok * fx.Int32(TOPK) + slot, fx.Int32(0))
            sc = valid.select(a_scale_tensor[row], fx.Float32(0.0))
            a_sc_n.append(sc)

        for m in range_constexpr(m_reps):
            sw_v = w_scale[None, m].load()
            for n in range_constexpr(n_reps):
                a_sc = a_sc_n[n]
                c = c_frag[None, m, n].load()
                items = []
                for v in range_constexpr(4):
                    cv = c[v].to(fx.Float32) if const_expr(is_int8) else c[v]
                    items.append(cv * sw_v[v] * a_sc)
                out[None, m, n].store(fxh.Vec.from_elements(items, fx.Float32))
        return out

    _apply_dequant = ASTRewriter.transform(_apply_dequant)

    def _apply_doweight(c_frag, tid, e_idx, arg_sorted_weights):
        # Per-sorted-row routed weight (one per token_rep n).
        m_reps = fxh.reps(c_frag, 1)
        n_reps = fxh.reps(c_frag, 2)
        sw_ptr = fx.recast_iter(fx.Float32, fx.get_iter(arg_sorted_weights) + e_idx * fx.Int32(BM))
        tw_view = fx.make_view(sw_ptr, fx.make_layout(BM, 1))
        tw_copy = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32),
            fx.make_layout(((16, 4, 4), 1), ((1, 0, 0), 0)),
            fx.make_tile(16),
        )
        tw_thr = tw_copy.get_slice(tid).partition_S(tw_view)
        tw_frag = fx.make_fragment_like(tw_thr)
        fx.copy(fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32), tw_thr, tw_frag)
        for n in range_constexpr(n_reps):
            tw = tw_frag[0, n]
            for m in range_constexpr(m_reps):
                c_frag[None, m, n].store(c_frag[None, m, n].load() * tw)

    _apply_doweight = ASTRewriter.transform(_apply_doweight)

    def _c_to_out_frag(c_frag):
        """Convert an f32 C fragment to an out_elem fragment (bf16 rounds via round_bit)."""
        round_bit = fx.Uint32(0x8000)
        out_frag = fx.make_fragment_like(c_frag, dtype=out_elem)
        m_reps = fxh.reps(c_frag, 1)
        n_reps = fxh.reps(c_frag, 2)
        for m in range_constexpr(m_reps):
            for n in range_constexpr(n_reps):
                acc = c_frag[None, m, n].load()
                if const_expr(out_is_f32):
                    pass
                elif const_expr(out_is_bf16):
                    acc = ((acc.bitcast(fx.Uint32) + round_bit) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
                else:
                    acc = acc.to(out_elem)
                out_frag[None, m, n].store(acc)
        return out_frag

    _c_to_out_frag = ASTRewriter.transform(_c_to_out_frag)

    @flyc.kernel
    def moe_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x  # tile along model_dim (output/N)
        e_idx = gpu.block_idx.y  # expert-block id (sorted M-block)

        tokens = i32_tokens_in
        M = tokens * fx.Int32(TOPK)

        in_ptr = fx.recast_iter(in_t, fx.get_iter(arg_x))
        # A2 is [tokens, topk, inter(K)] flattened. The A-gather decodes (token, slot),
        # so the view MUST be rank-3; a rank-2 view drops the slot and reads slot-0 of
        # every token. Row = token*topk + slot.
        arg_p_input = fx.make_view(
            in_ptr,
            fx.make_layout((tokens, fx.Int32(TOPK), fx.Int32(K)), (fx.Int32(TOPK * K), fx.Int32(K), 1)),
        )

        num_valid_id = fxh.view_as_torch_tensor(fx.get_iter(arg_num_valid_ids), (1,), fx.Int32)[0]

        if e_idx * fx.Int32(BM) < num_valid_id:
            lds = fx.SharedAllocator().allocate(SharedStorage)
            lds.sorted_lds = lds.sorted_lds.peek()
            lds.gemm = lds.gemm.peek()

            arg_p_sorted_ids = fx.make_view(
                fx.recast_iter(fx.Int32, fx.get_iter(arg_sorted_token_ids) + e_idx * fx.Int32(BM)),
                fx.make_layout(BM, 1),
            )
            expert_id = fxh.view_as_torch_tensor(fx.get_iter(arg_expert_ids), (1,), fx.Int32)[e_idx]

            w_ptr = fx.recast_iter(in_t, fx.get_iter(arg_w))
            if const_expr(is_int4):
                # Packed-int4: raw byte view; the ki-correct loader indexes per-expert.
                arg_p_weight = fx.make_view(w_ptr, fx.make_layout((fx.Int32(experts * N * (K // 2)),), (1,)))
            else:
                arg_p_weight = fxh.make_weight_view(w_ptr, expert_id, N, K)

            # Seed sorted ids into LDS (A-gather + output scatter index).
            sorted_ids_buf = fx.rocdl.make_buffer_tensor(arg_p_sorted_ids, max_size=False)
            if tid < fx.Int32(BM):
                lds_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # Output: reduce -> [tokens*topk, model_dim] buffer for BufferCopy128b
            # stores; atomic -> rank-2 [tokens, model_dim] buffer resource (scatter via
            # buffer_atomic_add at explicit element indices; out_tensor supplies shape).
            out_atomic_rsrc = None
            if const_expr(accumulate):
                out_atomic_rsrc = buffer_ops.create_buffer_resource(
                    arg_out,
                    max_size=False,
                    num_records_bytes=(fx.Index(tokens) * fx.Index(N * out_bytes)),
                )
                arg_p_output = fx.make_view(
                    fx.recast_iter(out_elem, fx.get_iter(arg_out)),
                    fx.make_layout((tokens, fx.Int32(N)), (fx.Int32(N), 1)),
                )
            else:
                arg_p_output = fx.make_view(
                    fx.recast_iter(out_elem, fx.get_iter(arg_out)),
                    fx.make_layout((tokens, fx.Int32(TOPK), fx.Int32(N)), (fx.Int32(TOPK * N), fx.Int32(N), 1)),
                )
            out_tensor = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False)
            _c_vec = 128 // out_elem.width  # values per 128b atom (f16/bf16=8, f32=4)
            # BUG GUARD (#2): CShuffle read/scatter TV layout MUST be CHANNEL-MAJOR --
            # the 32 lanes each walk _e_vec contiguous channels of ONE row (max
            # coalescing); a row-major lane map straddles 16 rows/wave and destroys
            # atomic coalescing (4x traffic).
            _cshuffle_nlane = 32
            _n_row_thr = 256 // _cshuffle_nlane  # 8 threads on the row (token) dim
            # BUG GUARD (#1): e_vec MUST be NARROW (2) on the atomic path --
            # _buffer_atomic_pk emits ONE pk-pair/instruction, so e_vec=2 keeps the 32
            # lanes on 64 CONTIGUOUS channels; wider strides pairs apart -> uncoalesced
            # atomics (4.25x TCC_ATOMIC, 4.8x regression). reduce -> 8 when
            # contiguous_n%256==0 (wide coalesced store), else 2.
            if const_expr(accumulate):
                _e_vec = 2
            else:
                _e_vec = _c_vec if (contiguous_n % (_cshuffle_nlane * _c_vec) == 0) else 2
            _n_chan_tile = _cshuffle_nlane * _e_vec  # contiguous channels per copy tile
            assert (
                contiguous_n % _n_chan_tile == 0
            ), f"tile_n={contiguous_n} must be a multiple of {_n_chan_tile} (32 lanes x {_e_vec} chan)"
            assert (
                BM % _n_row_thr == 0
            ), f"tile_m={BM} must be a multiple of {_n_row_thr} for the channel-major epilogue"
            # Copy atoms sized to _e_vec (full 128b vector, or a narrower atom for e_vec==2).
            _epi_bits = _e_vec * out_elem.width
            _buf_atom_ctor = {
                16: fx.rocdl.BufferCopy16b,
                32: fx.rocdl.BufferCopy32b,
                64: fx.rocdl.BufferCopy64b,
                128: fx.rocdl.BufferCopy128b,
            }[_epi_bits]
            _uni_atom_ctor = {
                16: fx.UniversalCopy16b,
                32: fx.UniversalCopy32b,
                64: fx.UniversalCopy64b,
                128: fx.UniversalCopy128b,
            }[_epi_bits]
            buf_atom_w128 = fx.make_copy_atom(_buf_atom_ctor(), out_elem)
            # tile = (rows=_n_row_thr, cols=_n_chan_tile); each lane steps e_vec columns,
            # each row-thr one row (channel-major, per BUG GUARD #2 above).
            c_rw_copy = fx.make_tiled_copy(
                buf_atom_w128,
                fx.make_layout(
                    ((_cshuffle_nlane, _n_row_thr), _e_vec),
                    ((_n_row_thr * _e_vec, 1), _n_row_thr),
                ),
                fx.make_tile(_n_row_thr, _n_chan_tile),
            )
            # Row (token) index copy: same row mapping as c_rw_copy; only the row-thread
            # part of tid selects the row, the 32 channel-lanes share it (stride 0).
            c_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((_cshuffle_nlane, _n_row_thr), 1), ((0, 1), 0)),
                fx.make_tile(_n_row_thr),
            )
            c_out_index_frag = fxh.read_sorted_index(c_index_copy, tid, lds.sorted_lds, BM)
            c_out = fxh.make_tensor_with_index(
                out_tensor, BM, contiguous_n, c_out_index_frag, c_rw_copy, tid, TOPK, is_read_from_mem=False
            )

            # Per-row activation scale index (ptpc): one packed id per token_rep.
            asc_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((16, 4, 4), 1), ((1, 0, 0), 0)),
                fx.make_tile(16),
            )
            asc_lds = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))
            asc_thr = asc_index_copy.get_slice(tid).partition_S(asc_lds)
            asc_idx = fx.make_fragment_like(asc_thr)
            fx.copy(fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32), asc_thr, asc_idx)

            c_frag = _gemm_1x4(blk_n, arg_p_input, arg_p_weight, lds, M, expert_id)

            # dequant: a_scale (per row) * w_scale (per channel); int8 also i32->f32 here.
            c_frag = _apply_dequant(c_frag, tid, expert_id, blk_n, asc_idx, M, arg_scale_w, arg_scale_x)
            if const_expr(doweight_stage2):
                _apply_doweight(c_frag, tid, e_idx, arg_sorted_weights)

            c_out_frag = _c_to_out_frag(c_frag)

            # CShuffle epilogue: stage output to LDS (transpose, swz 3,3,3 for 2B /
            # 3,2,3 for 4B), read back channel-contiguous, scatter via sorted-id index.
            _, _tiled_mma = fxh.make_1x4_tiled_mma(in_t, acc_dtype)
            _log2_vec = 3 if out_bytes == 2 else 2  # 8 (2B) or 4 (4B) elems per 128b
            cshuf_atom_w = fx.make_copy_atom(fx.UniversalCopy64b(), out_elem)
            cshuf_atom_r = fx.make_copy_atom(_uni_atom_ctor(), out_elem)
            cshuf_ptr = fx.recast_iter(out_elem, lds.gemm.region.ptr)
            swz_c = fx.SwizzleType.get(3, _log2_vec, 3)
            lds_c_store = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(fx.static(swz_c), fx.make_ordered_layout((contiguous_n, BM), order=(0, 1))),
            )
            lds_c = fx.make_view(
                cshuf_ptr,
                fx.make_composed_layout(fx.static(swz_c), fx.make_ordered_layout((BM, contiguous_n), order=(1, 0))),
            )
            gpu.barrier()
            store_c = fx.make_tiled_copy_C(cshuf_atom_w, _tiled_mma).get_slice(tid)
            fx.copy(cshuf_atom_w, store_c.retile(c_out_frag), store_c.partition_D(lds_c_store))
            gpu.barrier()
            rd = fx.make_fragment_like(c_rw_copy.get_slice(tid).partition_S(lds_c))
            fx.copy(cshuf_atom_r, c_rw_copy.get_slice(tid).partition_S(lds_c), rd)
            if const_expr(not accumulate):
                c_out.copy(buf_atom_w128, blk_n, rd)
            else:
                if const_expr(_needs_global_atomic_bf16):
                    _atomic_mode = "pk_global"
                elif const_expr(out_is_f32):
                    _atomic_mode = "f32"
                else:
                    _atomic_mode = "pk"
                c_out.copy(
                    buf_atom_w128,
                    blk_n,
                    rd,
                    atomic=_atomic_mode,
                    atomic_rsrc=out_atomic_rsrc,
                    out_bytes=out_bytes,
                    row_stride=N,
                    row_limit=tokens,
                    atomic_dst=arg_p_output,
                )

    @flyc.jit
    def launch_moe_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        n_in = arith.index_cast(T.index, i32_n_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        gx = n_in // fx.Index(contiguous_n)
        gy = size_expert_ids_in
        moe_gemm2(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(grid=(gx, gy, 1), block=(256, 1, 1), stream=stream)

    return launch_moe_gemm2


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    out_dtype: str = "f16",
    accumulate: bool = True,
    in_dtype: str = "fp8",
):
    """Compile stage2 down-projection kernel (``moe_gemm2``) and return it.

    A2/W are fp8 (E4M3), int8, int8smooth, or int4 (W4A8) on gfx94*/gfx95*; int8
    variants share the fp8 pipeline (i32 MFMA acc, f32 dequant), int4 swaps in the
    ki-correct packed-int4 loader. ``out_dtype``:
      - "f16": fp16 half2 atomics (fast, can overflow to +/-inf)
      - "bf16": bf16 atomics (buffer on gfx95+, global_atomic_pk_add_bf16 on gfx942)
      - "f32": fp32 scalar atomics (slower, avoids fp16 atomic overflow)

    ``accumulate=True`` uses atomics; ``accumulate=False`` writes per-(token,slot)
    partials for a separate reduce kernel and supports only f16/bf16 output.
    """
    _arch = get_rocm_arch()
    if not ("gfx94" in _arch or "gfx95" in _arch):
        raise ValueError(f"moe_gemm_2stage supports gfx94*/gfx95* (CDNA3/CDNA4); got arch={_arch!r}")
    _out_s = str(out_dtype).strip().lower()
    if _out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}")
    if in_dtype not in ("fp8", "int8", "int8smooth", "int4"):
        raise ValueError(f"in_dtype must be 'fp8', 'int8', 'int8smooth', or 'int4', got {in_dtype!r}")
    if (not bool(accumulate)) and _out_s in ("f32", "fp32", "float"):
        raise ValueError("compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}")
    return _build_moe_gemm2_fp8(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight_stage2,
        out_dtype=out_dtype,
        accumulate=accumulate,
        in_dtype=in_dtype,
    )
