# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE GEMM stage1 (gate-up MFMA) fp8/int8 kernel builder.

Layout-API pipeline: X/W are fp8 (E4M3) or int8 on CDNA3/CDNA4 (gfx94*/gfx95*).
int8 shares the fp8 path via an i32 MFMA acc (converted to f32 in dequant).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.moe.moe_gemm_2stage import layout_helpers as fxh


def _build_moe_gemm1_fp8_gateup(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    out_dtype: str,
    in_dtype: str = "fp8",
):
    """Native gate-up GEMM (B-first MFMA, fp8/int8): out[t,slot,inter] =
    silu(gate*sx*sw_g)*(up*sx*sw_u)[*routed], scattered by sorted token ids.

    int8 shares the fp8 pipeline; it only swaps the MFMA atom to an i32 accumulator
    and converts to f32 in dequant. ``int8smooth`` is int8 with slot-major
    activations (X/scale_x pre-expanded to [topk*tokens, ...], indexed
    row_ts = slot*tokens + token); it only changes the A-gather and act-scale load.
    """
    is_int4 = in_dtype == "int4"  # W4A8: int8 activations x packed-int4 weights
    is_int8smooth = in_dtype == "int8smooth"
    is_int8 = in_dtype == "int8" or is_int8smooth or is_int4
    elem_t = fx.Int8 if is_int8 else fx.Float8E4M3FNUZ
    acc_dtype = fx.Int32 if is_int8 else None
    MFMA_K = 32
    elem_bytes = elem_t.width // 8  # fp8/int8=1

    K = int(model_dim)
    N_e = int(2 * inter_dim)  # per-expert output cols (gate+up)
    BM = int(tile_m)
    BN = int(tile_n)
    TILE_K = int(tile_k)
    TOPK = int(topk)
    out_bf16 = out_dtype == "bf16"

    assert TILE_K in (128, 256), f"native gate-up needs tile_k in (128,256), got {TILE_K}"
    assert K % TILE_K == 0 and (K // TILE_K) % 2 == 0, f"K={K} must be an even multiple of TILE_K={TILE_K}"
    assert 64 <= BN <= 256 and BN % 64 == 0, f"tile_n must be in [64,256] multiple of 64, got {BN}"
    assert 16 <= BM <= 256 and BM % 16 == 0, f"tile_m must be a 16-multiple in [16,256], got {BM}"

    # 4 waves tile the N (channel) dim at 16 ch/wave/rep. fp8/int8 use a
    # 128-channel block (num_acc_n=2) instead of the older BN//2 tile: at the big
    # shape (tile_n=128) this halves the grid (128->64 blocks/row), cutting wave
    # count 2x and the barrier/LDS-sync overhead that made stage1 fp8 ~30% slower
    # than v0.3.0. Cap at 128 so tile_n=256 stays at 2 blocks/row (num_acc_n=2):
    # collapsing to a single 256-channel block regresses the inter_dim=256 shape
    # ~40% (grid too small, register pressure). int4 keeps the narrower BN//2 tile:
    # its single just-in-time weight buffer cannot feed a doubled per-block N
    # without spilling (measured +63% at BN=128), so it stays at 64 channels.
    contiguous_n = max(BN // 2, 64) if is_int4 else min(max(BN, 64), 128)
    assert inter_dim % contiguous_n == 0, (
        f"inter_dim={inter_dim} must be divisible by the effective channel tile "
        f"contiguous_n={contiguous_n} (from tile_n={BN})"
    )

    in_t = elem_t
    a_lds_size = BM * TILE_K

    @fx.struct
    class GemmBuffers:
        a_ping: fx.Array[in_t, a_lds_size, 16]
        a_pong: fx.Array[in_t, a_lds_size, 16]

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
        """B-first native-fp8 gate/up GEMM with A-gather + LDS ping-pong."""
        tid = gpu.thread_idx.x
        mma_atom, tiled_mma = fxh.make_1x4_tiled_mma(in_t, acc_dtype)

        # Record bound so sentinel-row gathers (token_id == M padding) read 0 via
        # hardware OOB. int8smooth covers all topk*M slot-major rows.
        a_rows = fx.Int64(TOPK) * fx.Int64(M) if const_expr(is_int8smooth) else fx.Int64(M)
        a_tensor = fx.rocdl.make_buffer_tensor(
            arg_p_input, max_size=False, num_records_bytes=a_rows * fx.Int64(K) * fx.Int64(elem_bytes)
        )
        b_tensor = fx.rocdl.make_buffer_tensor(arg_p_weight, max_size=False)

        # A (activation): static (BM,K) fake keeps flat_divide static; rows gathered via index.
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
        a_idx = fxh.make_tensor_with_index(
            a_tensor,
            BM,
            TILE_K,
            a_index_frag,
            a_mem_cp_g2r,
            tid,
            TOPK,
            token_slot_tokens=(M if is_int8smooth else None),
        )
        a_mem_thr = a_mem_cp_g2r.get_slice(tid).partition_S(a_tile)
        a_cp_frag = fx.make_fragment_like(a_mem_thr[None, None, None, 0])
        gpu.barrier()  # sorted_lds reads done before overwriting with A tile

        # 2-stage A LDS ping-pong: overlap the next K-tile's global load + LDS write
        # with the current tile's MFMA. a_ping/a_pong alternate each K-tile.
        swz = fx.SwizzleType.get(*_swz_params)
        uni_cp_atom = fx.make_copy_atom(fx.UniversalCopy128b(), in_t)
        a_lds_bufs = []
        a_lds_w_bufs = []
        a_lds_r_bufs = []
        a_frag_bufs = []
        a_frag_retile_bufs = []
        for _buf_ptr in (lds.gemm.a_ping.ptr, lds.gemm.a_pong.ptr):
            a_lds = fx.make_view(
                _buf_ptr,
                fx.make_composed_layout(fx.static(swz), fx.make_ordered_layout((BM, TILE_K), order=(1, 0))),
            )
            a_r2s = fx.make_tiled_copy(uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)).get_slice(tid)
            a_lds_bufs.append(a_lds)
            a_lds_w_bufs.append(a_r2s.partition_D(a_lds))
            a_lds_r_bufs.append(fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).partition_S(a_lds))
            a_frag = tiled_mma.make_fragment_B(a_lds)
            a_frag_bufs.append(a_frag)
            a_frag_retile_bufs.append(fx.make_tiled_copy_B(uni_cp_atom, tiled_mma).get_slice(tid).retile(a_frag))
        # Per-stage A gather staging fragments (one per in-flight buffer).
        a_cp_frag_bufs = [a_cp_frag, fx.make_fragment_like(a_mem_thr[None, None, None, 0])]
        a_cp_frag_retile_bufs = [
            fx.make_tiled_copy(uni_cp_atom, g2r_tv_layout, fx.make_tile(_thrs_m, TILE_K)).get_slice(tid).retile(f)
            for f in a_cp_frag_bufs
        ]

        # B (weight gate/up): direct global->register, prefetched one K-tile ahead
        # into a ping-pong pair of fragments so weight VMEM overlaps the MFMA.
        _ki_reps = TILE_K // 64  # 64-byte K super-steps per tile
        if const_expr(is_int4):
            # W4A8: packed int4 (2 nibbles/byte). The int8 MFMA A-fragment reads K in
            # ki-SEPARATED kpack groups that a generic half-width tiled_copy cannot
            # reproduce, so use the explicit ki-correct loader (load_weight_int4_frag).
            # Gate/up are separate channel ranges (col_gate, col_up = +inter_dim).
            # Read as i32 dwords (4 packed bytes = 8 int4); recast the align-1 int8
            # iter to an align-4 i32 pointer so the dword buffer_load is legal.
            _w_i8_iter = fx.get_iter(arg_p_weight)
            _w_i32_ptr = fx.PointerType.get(fx.Int32.ir_type, _w_i8_iter.memspace, 4)
            b_raw = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    fx.recast_iter(_w_i32_ptr, _w_i8_iter),
                    fx.make_layout((fx.Int32(experts * N_e * (K // 2) // 4),), (1,)),
                ),
                max_size=False,
            )
            b_layout_i4 = fxh.make_preshuffle_b_layout_int4(N_e, K)
            _expert_off = expert_id * fx.Int32((N_e * K) // 8)  # dwords per expert slab
            _col_gate = blk_n * fx.Int32(contiguous_n)
            _col_up = _col_gate + fx.Int32(inter_dim)
            # Fragment shape from a fake int8 (contiguous_n, TILE_K) weight tile.
            _wfake = fx.rocdl.make_buffer_tensor(
                fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((contiguous_n, TILE_K), (TILE_K, 1))),
                max_size=False,
            )
            _wtile = fx.flat_divide(_wfake, fx.make_tile(contiguous_n, TILE_K))[None, None, 0, 0]
            # Single weight buffer (no B ping-pong): a second buffer only inflates VGPR
            # (occupancy-bound) without overlap. A-LDS keeps its ping-pong; the tile's
            # weight is loaded just-in-time before its MFMA.
            bl_frag_bufs = [tiled_mma.make_fragment_A(_wtile)]
            br_frag_bufs = [tiled_mma.make_fragment_A(_wtile)]
            _b_loads_per_tile = 0  # A-gather-only wait (int4 weight loaded just-in-time)
        else:
            bl_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K))[None, None, blk_n * 2 + 0, None]
            br_tile = fx.flat_divide(b_tensor, fx.make_tile(contiguous_n, TILE_K))[None, None, blk_n * 2 + 1, None]
            b_g2r = fx.make_tiled_copy_A(buf_cp_atom_r, tiled_mma).get_slice(tid)
            bl_g2r = b_g2r.partition_S(bl_tile)
            br_g2r = b_g2r.partition_S(br_tile)
            bl_frag_bufs = [tiled_mma.make_fragment_A(bl_tile[None, None, 0]) for _ in range(2)]
            br_frag_bufs = [tiled_mma.make_fragment_A(br_tile[None, None, 0]) for _ in range(2)]
            bl_ret_bufs = [b_g2r.retile(f) for f in bl_frag_bufs]
            br_ret_bufs = [b_g2r.retile(f) for f in br_frag_bufs]

            # 128b B (gate+up) loads/thread; feeds the s_waitcnt that awaits only the
            # A-gather while B stays in flight to overlap the MFMA. gate+up => x2.
            _b_loads_per_tile = 2 * (fx.size(fx.get_shape(bl_frag_bufs[0])).to_py_value() // _val_per_thr)

        c_fake_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.get_iter(arg_p_input), fx.make_layout((contiguous_n, BM), (BM, 1))), max_size=False
        )
        c_fake = fx.flat_divide(c_fake_buf, fx.make_tile(contiguous_n, BM))[None, None, 0, 0]
        c_gate = tiled_mma.make_fragment_C(c_fake)
        c_up = tiled_mma.make_fragment_C(c_fake)
        c_gate.fill(0)
        c_up.fill(0)

        k_iters = TILE_K // (2 * MFMA_K)
        num_tiles = K // TILE_K

        def _load_gmem(kt, s):
            """A-gather (+ B gate/up loads for non-int4) for K-tile kt, stage s."""
            kb = fx.Int32(kt)
            a_idx.copy(buf_cp_atom_r, kb, a_cp_frag_bufs[s])
            if const_expr(not is_int4):
                fx.copy(buf_cp_atom_r, bl_g2r[None, None, None, kb], bl_ret_bufs[s])
                fx.copy(buf_cp_atom_r, br_g2r[None, None, None, kb], br_ret_bufs[s])

        def _load_weight_i4(kt):
            """ki-correct packed-int4 gate/up weight load for K-tile kt (int4 only)."""
            kb = fx.Int32(kt)
            fxh.load_weight_int4_frag(b_raw, b_layout_i4, bl_frag_bufs[0], _expert_off, _col_gate, kb, tid, _ki_reps)
            fxh.load_weight_int4_frag(b_raw, b_layout_i4, br_frag_bufs[0], _expert_off, _col_up, kb, tid, _ki_reps)

        def _write_a_lds(s):
            fx.copy(uni_cp_atom, a_cp_frag_retile_bufs[s], a_lds_w_bufs[s])

        def _read_a_lds(s):
            """LDS A-reads issued ahead of the next MFMA so ds_read latency hides."""
            for ki in range_constexpr(k_iters):
                fx.copy(uni_cp_atom, a_lds_r_bufs[s][None, None, ki], a_frag_retile_bufs[s][None, None, ki])

        # int4 at inter_dim=8192: a K-major serpentine traversal hides MFMA
        # read-after-write hazards slightly better than the default schedule --
        # 22.05ms vs 22.40ms (~1.6%, reproducible, non-overlapping ranges), and is
        # neutral at inter_dim=256. fp8/int8 measured best at the default on both
        # shape families, so only int4 overrides.
        _g1_trav = "kmn_serpentine" if is_int4 else None

        def _mfma(s):
            # int4 uses a single weight buffer (bs=0); A keeps its ping-pong (s).
            bs = 0 if const_expr(is_int4) else s
            fx.gemm(mma_atom, c_gate, bl_frag_bufs[bs], a_frag_bufs[s], c_gate, traversal_order=_g1_trav)
            fx.gemm(mma_atom, c_up, br_frag_bufs[bs], a_frag_bufs[s], c_up, traversal_order=_g1_trav)

        # Prologue: stage-0 loads + LDS write for K-tile 0. Wait only on the A-gather
        # (B stays in flight for the first MFMA); pre-read tile-0 A from LDS.
        _load_gmem(0, 0)
        rocdl.s_waitcnt(fxh._encode_waitcnt(vmcnt=_b_loads_per_tile))
        _write_a_lds(0)
        gpu.barrier()
        _read_a_lds(0)

        # Unrolled ping-pong: compute tile kt on `cur` while prefetching kt+1's A into
        # `nxt`. fp8/int8 also prefetch B; int4 loads its single-buffer weight JIT.
        for kt in range_constexpr(num_tiles):
            cur = kt % 2
            if kt + 1 < num_tiles:
                nxt = (kt + 1) % 2
                _load_gmem(kt + 1, nxt)
                rocdl.s_waitcnt(fxh._encode_waitcnt(vmcnt=_b_loads_per_tile))
                _write_a_lds(nxt)
                if const_expr(is_int4):
                    _load_weight_i4(kt)
                _mfma(cur)
                gpu.barrier()
                _read_a_lds(nxt)
            else:
                if const_expr(is_int4):
                    _load_weight_i4(kt)
                _mfma(cur)
        return c_gate, c_up

    _gemm_1x4 = ASTRewriter.transform(_gemm_1x4)

    def _apply_dequant(c_gate_frag, c_up_frag, tid, expert_id, blk_n, asc_idx, M, arg_scale_w, arg_scale_x):
        # ptpc: per-channel weight scale (gate [0,inter), up [inter,2inter)), per-token
        # act scale. fp8 folds in place; int8 converts the i32 acc to fresh f32 frags.
        m_reps = fxh.reps(c_gate_frag, 1)
        n_reps = fxh.reps(c_gate_frag, 2)
        if const_expr(is_int8):
            og = fx.make_fragment_like(c_gate_frag, dtype=fx.Float32)
            ou = fx.make_fragment_like(c_up_frag, dtype=fx.Float32)
        else:
            og = c_gate_frag
            ou = c_up_frag
        sw_ptr = fx.recast_iter(fx.Float32, fx.get_iter(arg_scale_w))
        scale_gate = fx.make_view(sw_ptr + expert_id * N_e + blk_n * contiguous_n, fx.make_layout(contiguous_n, 1))
        scale_up = fx.make_view(
            sw_ptr + expert_id * N_e + fx.Int32(inter_dim) + blk_n * contiguous_n, fx.make_layout(contiguous_n, 1)
        )
        cp_atom_scale = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
        scale_copy = fx.make_tiled_copy(
            cp_atom_scale, fx.make_layout(((16, 4, 4), 4), ((0, 4, 16), 1)), fx.make_tile(64)
        )
        sg_thr = scale_copy.get_slice(tid).partition_S(scale_gate)
        su_thr = scale_copy.get_slice(tid).partition_S(scale_up)
        gate_scale = fx.make_fragment_like(sg_thr)
        up_scale = fx.make_fragment_like(su_thr)
        fx.copy(cp_atom_scale, sg_thr, gate_scale)
        fx.copy(cp_atom_scale, su_thr, up_scale)

        # int8smooth: scale_x is [topk*M] slot-major (row = slot*tokens + token).
        asc_rows = fx.Int32(TOPK) * M if const_expr(is_int8smooth) else M
        a_scale_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(fx.recast_iter(fx.Float32, fx.get_iter(arg_scale_x)), fx.make_layout(asc_rows, 1)),
            max_size=False,
            num_records_bytes=fx.Int64(asc_rows) * fx.Int64(4),
        )
        if const_expr(is_int8smooth):
            a_sc_n = [
                a_scale_tensor[(asc_idx[0, n] >> 24) * M + (asc_idx[0, n] & 0xFFFFFF)] for n in range_constexpr(n_reps)
            ]
        else:
            a_sc_n = [a_scale_tensor[asc_idx[0, n] & 0xFFFFFF] for n in range_constexpr(n_reps)]
        for m in range_constexpr(m_reps):
            sg_v = gate_scale[None, m].load()
            su_v = up_scale[None, m].load()
            for n in range_constexpr(n_reps):
                a_sc = a_sc_n[n]
                cg = c_gate_frag[None, m, n].load()
                cu = c_up_frag[None, m, n].load()
                cg_items = []
                cu_items = []
                for v in range_constexpr(4):
                    g = cg[v].to(fx.Float32) if const_expr(is_int8) else cg[v]
                    u = cu[v].to(fx.Float32) if const_expr(is_int8) else cu[v]
                    cg_items.append(g * sg_v[v] * a_sc)
                    cu_items.append(u * su_v[v] * a_sc)
                og[None, m, n].store(fxh.Vec.from_elements(cg_items, fx.Float32))
                ou[None, m, n].store(fxh.Vec.from_elements(cu_items, fx.Float32))
        return og, ou

    _apply_dequant = ASTRewriter.transform(_apply_dequant)

    def _apply_doweight(c_gate_frag, c_up_frag, tid, e_idx, arg_sorted_weights):
        # Per-sorted-row routed weight (one per token_rep n), folded into gate.
        m_reps = fxh.reps(c_gate_frag, 1)
        n_reps = fxh.reps(c_gate_frag, 2)
        tw_frag = fxh.load_sorted_weight_frag(arg_sorted_weights, e_idx, BM, tid)
        for n in range_constexpr(n_reps):
            tw = tw_frag[0, n]
            for m in range_constexpr(m_reps):
                c_gate_frag[None, m, n].store(c_gate_frag[None, m, n].load() * tw)

    _apply_doweight = ASTRewriter.transform(_apply_doweight)

    @flyc.kernel
    def moe_gemm1_fp8_gateup(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        blk_n = gpu.block_idx.x  # tile along inter (channel/N)
        e_idx = gpu.block_idx.y  # expert-block id (sorted M-block)

        M = i32_tokens_in

        # Pointers / views.
        in_ptr = fx.recast_iter(in_t, fx.get_iter(arg_x))
        arg_p_input = fx.make_view(in_ptr, fx.make_layout((M, fx.Int32(K)), (fx.Int32(K), 1)))

        max_valid_id = fxh.view_as_torch_tensor(fx.get_iter(arg_max_token_ids), (1,), fx.Int32)[0]

        if e_idx * fx.Int32(BM) < max_valid_id:
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
                arg_p_weight = fx.make_view(w_ptr, fx.make_layout((fx.Int32(experts * N_e * (K // 2)),), (1,)))
            else:
                arg_p_weight = fxh.make_gateup_weight_view(w_ptr, expert_id, contiguous_n, N_e, K)

            # BUG GUARD: the A-gather/scatter TV layouts read up to 256/(tile_k/16)
            # M-rows (32 at tile_k=128), EXCEEDING BM on decode (BM=16). Un-seeded
            # slots decode to a VALID token 0 / slot 0, piling garbage onto token 0.
            # Seed every readable slot (full 256 range, not just BM) with sentinel id
            # == M (out of range -> hardware OOB clamp drops it), then write real rows.
            sorted_ids_buf = fx.rocdl.make_buffer_tensor(arg_p_sorted_ids, max_size=False)
            sentinel_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(256, 1))
            sentinel_view[tid] = M
            gpu.barrier()
            if tid < fx.Int32(BM):
                lds_view = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))
                lds_view[tid] = sorted_ids_buf[tid]
            gpu.barrier()

            # Output [M, TOPK, inter] fp16/bf16; scatter index from sorted_lds.
            out_elem = fx.BFloat16 if out_bf16 else fx.Float16
            arg_p_output = fx.make_view(
                fx.recast_iter(out_elem, fx.get_iter(arg_out)),
                fx.make_layout(
                    (M, fx.Int32(TOPK), fx.Int32(inter_dim)), (fx.Int32(TOPK * inter_dim), fx.Int32(inter_dim), 1)
                ),
            )
            out_tensor = fx.rocdl.make_buffer_tensor(arg_p_output, max_size=False)
            buf_atom_w128 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), out_elem)
            # CShuffle read/scatter: 4-wave 2x2 thread grid over (BM x contiguous_n).
            c_rw_copy = fx.make_tiled_copy(
                buf_atom_w128,
                fx.make_layout(((4, 16, 2, 2), 8), ((256, 1, 16, 1024), 32)),
                fx.make_tile(32, 64),
            )
            c_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((4, 16, 2, 2), 1), ((0, 1, 16, 0), 0)),
                fx.make_tile(32),
            )
            c_out_index_frag = fxh.read_sorted_index(c_index_copy, tid, lds.sorted_lds, BM)
            c_out = fxh.make_tensor_with_index(
                out_tensor, BM, contiguous_n, c_out_index_frag, c_rw_copy, tid, TOPK, is_read_from_mem=False
            )

            # Per-token activation scale index (ptpc): one id per token_rep.
            asc_index_copy = fx.make_tiled_copy(
                fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32),
                fx.make_layout(((16, 4, 4), 1), ((1, 0, 0), 0)),
                fx.make_tile(16),
            )
            asc_lds = fx.make_view(lds.sorted_lds.ptr, fx.make_layout(BM, 1))
            asc_thr = asc_index_copy.get_slice(tid).partition_S(asc_lds)
            asc_idx = fx.make_fragment_like(asc_thr)
            fx.copy(fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32), asc_thr, asc_idx)

            c_gate_frag, c_up_frag = _gemm_1x4(blk_n, arg_p_input, arg_p_weight, lds, M, expert_id)

            # dequant: sx (per token) * sw (per channel); int8 also i32->f32 here.
            c_gate_frag, c_up_frag = _apply_dequant(
                c_gate_frag, c_up_frag, tid, expert_id, blk_n, asc_idx, M, arg_scale_w, arg_scale_x
            )

            # Optional routed-weight scale (per sorted row).
            if const_expr(doweight_stage1):
                _apply_doweight(c_gate_frag, c_up_frag, tid, e_idx, arg_sorted_weights)

            # silu output dtype MUST match the CShuffle staging/store dtype (out_elem)
            # or the raw fragment bits are reinterpreted (bf16 0x4480 -> f16 4.5).
            c_out_bf16 = fxh.silu_pair_bf16(c_gate_frag, c_up_frag, out_dtype=out_elem)

            # CShuffle epilogue: stage silu to LDS (transpose, swz 3,3,3), read back
            # channel-contiguous, scatter to out[t, slot, inter] via sorted-id index.
            _, _tiled_mma = fxh.make_1x4_tiled_mma(in_t, acc_dtype)
            cshuf_atom_w = fx.make_copy_atom(fx.UniversalCopy64b(), out_elem)
            cshuf_atom_r = fx.make_copy_atom(fx.UniversalCopy128b(), out_elem)
            cshuf_ptr = fx.recast_iter(out_elem, lds.gemm.a_ping.ptr)
            swz_c = fx.SwizzleType.get(3, 3, 3)
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
            fx.copy(cshuf_atom_w, store_c.retile(c_out_bf16), store_c.partition_D(lds_c_store))
            gpu.barrier()
            rd = fx.make_fragment_like(c_rw_copy.get_slice(tid).partition_S(lds_c))
            fx.copy(cshuf_atom_r, c_rw_copy.get_slice(tid).partition_S(lds_c), rd)
            c_out.copy(buf_atom_w128, blk_n, rd)

    @flyc.jit
    def launch_moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        inter_in = arith.index_cast(T.index, i32_inter_in)
        size_expert_ids_in = arith.index_cast(T.index, i32_size_expert_ids_in)
        # Each block produces `contiguous_n` output channels, so gx = inter / contiguous_n.
        gx = inter_in // fx.Index(contiguous_n)
        gy = size_expert_ids_in
        moe_gemm1_fp8_gateup(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(grid=(gx, gy, 1), block=(256, 1, 1), stream=stream)

    return launch_moe_gemm1


@functools.lru_cache(maxsize=1024)
def compile_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    out_dtype: str = "f16",
    in_dtype: str = "fp8",
):
    """Compile stage1 gate-up kernel (``moe_gemm1``) and return the executable.

    X/W are fp8 (E4M3), int8, int8smooth, or int4 (W4A8) on gfx94*/gfx95*;
    ``out_dtype`` is f16/bf16. int8 variants share the fp8 pipeline (i32 MFMA acc,
    f32 dequant); int8smooth adds slot-major A/scale_x indexing; int4 swaps the
    weight load for the ki-correct packed-int4 loader (``load_weight_int4_frag``).
    """
    _arch = get_rocm_arch()
    if not ("gfx95" in _arch or "gfx94" in _arch):
        raise ValueError(f"moe_gemm_2stage supports gfx94*/gfx95* (CDNA3/CDNA4); got arch={_arch!r}")
    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")
    if in_dtype not in ("fp8", "int8", "int8smooth", "int4"):
        raise ValueError(f"in_dtype must be 'fp8', 'int8', 'int8smooth', or 'int4', got {in_dtype!r}")

    return _build_moe_gemm1_fp8_gateup(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight_stage1,
        out_dtype=out_dtype,
        in_dtype=in_dtype,
    )
