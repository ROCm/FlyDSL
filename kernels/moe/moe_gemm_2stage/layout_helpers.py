# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Portions Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout-API helper layer for the MoE 2-stage MFMA kernels (gemm1.py / gemm2.py)."""

import flydsl.expr as fx
from flydsl._mlir.dialects import rocdl, scf
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from kernels.common.kernels_common import _if_then


def reps(tensor, mode):
    """Static repeat count of ``tensor``'s ``mode`` (shape size, as a Python int)."""
    return fx.size(fx.get_shape(tensor)[mode]).to_py_value()


def _encode_waitcnt(vmcnt=63, expcnt=7, lgkmcnt=63):
    """Encode s_waitcnt bitfield for CDNA3 (gfx94x)."""
    vm_lo = vmcnt & 0xF
    vm_hi = (vmcnt >> 4) & 0x3
    return vm_lo | (expcnt << 4) | (lgkmcnt << 8) | (vm_hi << 14)


def _as_ptr(p, dtype=None):
    """Iterator for ``fx.make_view`` from a raw pointer or a runtime memref (opt. recast)."""
    try:
        p = fx.get_iter(p)
    finally:
        if dtype is not None and p.dtype != dtype:
            p = fx.recast_iter(dtype, p)
        return p  # noqa: B012


def torch_layout(*shape):
    if len(shape) == 1:
        return fx.make_layout(shape[0], 1)
    order = [i for i in range(len(shape) - 1, -1, -1)]
    return fx.make_ordered_layout(shape, order)


def view_as_torch_tensor(ptr, shape, dtype=None):
    ptr = _as_ptr(ptr, dtype)
    return fx.make_view(ptr, torch_layout(*shape))


def _buffer_atomic_pk(rsrc, elem_idx, reg_vec, elem_bytes):
    """Pairwise buffer atomic-add of an f16/bf16 vector into out[elem_idx..]
    (buffer rsrc + byte offset; OOB lanes dropped by hardware clamp).
    One pk-pair per instruction: keep the source vector NARROW (e_vec=2) or the
    lanes' pairs stride apart and every atomic goes uncoalesced (4.8x regression)."""
    from kernels.common.mem_ops import buffer_atomic_add

    _z = fx.Int32(0)
    for i in range_constexpr(reg_vec.numel // 2):
        pair = Vec.from_elements([reg_vec[i * 2], reg_vec[i * 2 + 1]], reg_vec.dtype)
        byte_off = (elem_idx + fx.Int32(i * 2)) * fx.Int32(elem_bytes)
        buffer_atomic_add(pair, rsrc, byte_off, _z, _z)


def _buffer_atomic_f32(rsrc, elem_idx, reg_vec):
    """Scalar buffer atomic-add of an f32 vector into out[elem_idx..]."""
    from kernels.common.mem_ops import buffer_atomic_add

    _z = fx.Int32(0)
    for i in range_constexpr(reg_vec.numel):
        byte_off = (elem_idx + fx.Int32(i)) * fx.Int32(4)
        buffer_atomic_add(reg_vec[i], rsrc, byte_off, _z, _z)


def _global_atomic_pk(dst, elem_idx, reg_vec, elem_bytes):
    """Pairwise GLOBAL atomic-add of an f16/bf16 vector into dst[elem_idx..]
    (raw !llvm.ptr atomicrmw fadd; lowers to global_atomic_pk_add_bf16 on gfx942,
    which lacks buffer_atomic_pk_add_bf16). NO hardware bounds-check (unlike buffer
    atomics), so callers MUST predicate out invalid lanes explicitly."""
    from kernels.common.mem_ops import atomic_add

    for i in range_constexpr(reg_vec.numel // 2):
        pair = Vec.from_elements([reg_vec[i * 2], reg_vec[i * 2 + 1]], reg_vec.dtype)
        atomic_add(dst, elem_idx + fx.Int32(i * 2), pair, dtype_bytes=elem_bytes, alignment=4)


def make_1x4_tiled_mma(weight_dtype, acc_dtype=None):
    """B-first 1x4 tiled_mma (weight=A, activation=B; 4 waves tile the M/channel dim).

    BUG GUARD (#5): int8 MUST pass ``acc_dtype=fx.Int32`` -- a default-f32 accumulator
    on the Int8 MFMA hard-aborts CDNA verification. fp8 leaves it at the atom default."""
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, weight_dtype, acc_dtype))
    k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
    tiled_mma = fx.make_tiled_mma(
        mma_atom,
        fx.make_layout((4, 1, 1), (1, 0, 0)),
        fx.make_tile(None, None, k_perm),
    )
    return mma_atom, tiled_mma


def make_gateup_weight_view(p_weight, expert_id, contiguous_n, N, K):
    """Per-expert (N,K) view over shuffle_weight-ordered weight, composed with the
    gate/up silu grouping (N = 2*inter_dim)."""
    group_layout_silu = fx.make_layout(
        ((contiguous_n, 2, N // (contiguous_n * 2)), K),
        ((1, N // 2, contiguous_n), N),
    )
    element_num = 16 // (p_weight.dtype.width // 8)
    return fx.make_view(
        p_weight + fx.Int64(expert_id * N * K),
        fx.composition(
            fx.make_layout(
                ((16, N // 16), (element_num, K // element_num)),
                ((element_num, 16 * K), (1, 16 * element_num)),
            ),
            group_layout_silu,
        ),
    )


def make_weight_view(p_weight, expert_id, N, K):
    """Per-expert (N,K) view over shuffle_weight-ordered weight, no gate/up grouping
    (stage2 analog of make_gateup_weight_view; N=model_dim, K=inter_dim)."""
    element_num = 16 // (p_weight.dtype.width // 8)
    return fx.make_view(
        p_weight + fx.Int64(expert_id * N * K),
        fx.make_layout(
            ((16, N // 16), (element_num, K // element_num)),
            ((element_num, 16 * K), (1, 16 * element_num)),
        ),
    )


def make_preshuffle_b_layout_int4(N_full, K):
    """W4A8 packed-int4 preshuffle address layout (per expert), shape
    ``(N_full/16, K_bytes/64, 4, 16, 8)``. Its ``crd2idx`` reproduces the int8 MFMA
    A-fragment's ki-separated kpack byte addressing (see ``load_weight_int4_frag``).
    ``N_full`` = per-expert output channels (2*inter_dim gemm1 / model_dim gemm2);
    ``K`` = contraction dim in int8 elems."""
    from flydsl.expr import arith as _lay_arith
    from kernels.common.mma.mfma_preshuffle_pipeline import make_preshuffle_b_layout

    return make_preshuffle_b_layout(_lay_arith, c_n=fx.Index(int(N_full)), c_k=fx.Index(int(K)), kpack_bytes=8).layout_b


def load_weight_int4_frag(bt_i32, b_layout, frag, expert_off_dwords, col_base, kb, tid, ki_reps):
    """Fill an int8 MFMA A-fragment from packed-int4 weight bytes, ki-correct.

    ``bt_i32``: buffer_tensor over the packed weight as i32 dwords, expert slab at
    ``expert_off_dwords``. ``b_layout`` = ``make_preshuffle_b_layout_int4``. ``frag``
    = make_fragment_A, shape ``(8, m_reps, (2, ki_reps))``. ``col_base`` = tile's
    first output channel (+inter_dim for the gemm1 up-half). ``kb`` = K-tile index.

    Address math (validated bit-exact vs the generic int8 A-load):
      col      = col_base + m*64 + wave*16 + lane%16    (channel this lane feeds)
      n_blk    = col//16 ; n_intra = col%16
      k0       = kb*ki_reps + ki                         (64-byte K super-step)
      k1       = lane//16 (0..3)                          (kpack lane group)
      byte_off = crd2idx((n_blk, k0, k1, n_intra, 0), b_layout)  (8-byte kpack base)
    The two MFMA k-halves are the two dwords of the 8-byte kpack (dword base =
    byte_off//4): dword 0 -> k=0, dword 1 -> k=1. 7-op nibble-unpack each
    (BUG GUARD #6: even={v0..v3}, odd={v4..v7}) into frag[None, m, (k, ki)]."""
    m_reps = fx.get_shape(frag)[1].to_py_value()
    lane = tid % fx.Int32(64)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_16 = lane // fx.Int32(16)
    wave = (tid // fx.Int32(64)) % fx.Int32(4)
    c_08 = fx.Int32(0x08080808)
    c_0f = fx.Int32(0x0F0F0F0F)
    c_1e = fx.Int32(0x1E)
    # expert_off_dwords may be a Python int or a runtime fx.Int32 (expert_id*slab).
    eoff = fx.Int32(expert_off_dwords) if isinstance(expert_off_dwords, int) else expert_off_dwords

    def _unpack(packed):
        even = (packed & c_0f) | ((packed & c_08) * c_1e)
        t = packed >> fx.Int32(4)
        odd = (t & c_0f) | ((t & c_08) * c_1e)
        return Vec.from_elements([even, odd], fx.Int32).bitcast(fx.Int8)

    for m in range_constexpr(m_reps):
        col = col_base + fx.Int32(m * 64) + wave * fx.Int32(16) + lane_mod_16
        n_blk = col // fx.Int32(16)
        n_intra = col % fx.Int32(16)
        for ki in range_constexpr(ki_reps):
            k0 = kb * fx.Int32(ki_reps) + fx.Int32(ki)
            coord = (n_blk, k0, lane_div_16, n_intra, fx.Int32(0))
            dw = (fx.get_scalar(fx.crd2idx(coord, b_layout)).to(fx.Int32) >> fx.Int32(2)) + eoff
            frag[None, m, (0, ki)].store(_unpack(bt_i32[dw]))
            frag[None, m, (1, ki)].store(_unpack(bt_i32[dw + fx.Int32(1)]))


def read_sorted_index(tiled_copy_index, tid, lds_index, index_size, index_offset=0):
    """Read the sorted M-row index from LDS into a per-thread fragment (read early,
    before the CShuffle epilogue overwrites sorted_lds)."""
    lds = fx.make_view(lds_index.ptr + index_offset, fx.make_layout(index_size, 1))
    cp_atom_lds = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
    lds_thr = tiled_copy_index.get_slice(tid).partition_S(lds)
    index_frag = fx.make_fragment_like(lds_thr)
    fx.copy(cp_atom_lds, lds_thr, index_frag)
    return index_frag


def load_sorted_weight_frag(arg_sorted_weights, e_idx, BM, tid):
    """Load the per-sorted-row routed weight tile (one f32 per token_rep) for the
    doweight epilogue, shared by gemm1/gemm2."""
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
    return tw_frag


def silu_pair_bf16(gate_frag, up_frag, gate_scale=None, up_scale=None, a_scale=None, out_dtype=fx.BFloat16):
    """silu(gate)*up -> out_dtype (optional fp8 weight/act scales folded in pre-silu).
    out_dtype MUST match the caller's CShuffle staging/store dtype: the fragment holds
    raw bits, so a mismatch silently reinterprets them (bf16 0x4480 == 1024.0 -> f16 4.5)."""
    log2_exp1 = -1.4426950408889634
    round_bit = fx.Uint32(0x8000)
    out_frag = fx.make_fragment_like(gate_frag, dtype=out_dtype)
    m_reps = reps(gate_frag, 1)
    n_reps = reps(gate_frag, 2)
    for m in range_constexpr(m_reps):
        if const_expr(a_scale is not None):
            a_sc = a_scale[m]
        for n in range_constexpr(n_reps):
            gate = gate_frag[None, m, n].load()
            up = up_frag[None, m, n].load()
            if const_expr(gate_scale is not None):
                sc_g = gate_scale[None, n].load()
                sc_u = up_scale[None, n].load()
            acc = []
            for j in range_constexpr(gate.numel):
                g = gate[j]
                u = up[j]
                if const_expr(gate_scale is not None):
                    g = g * sc_g[j]
                    u = u * sc_u[j]
                if const_expr(a_scale is not None):
                    g = g * a_sc
                    u = u * a_sc
                tmp = rocdl.exp2(T.f32, _raw(g * log2_exp1))
                acc.append((g * rocdl.rcp(T.f32, 1.0 + tmp)) * u)
            acc = Vec.from_elements(acc, fx.Float32)
            if const_expr(out_dtype == fx.BFloat16):
                acc = ((acc.bitcast(fx.Uint32) + round_bit) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
            else:
                acc = acc.to(out_dtype)
            out_frag[None, m, n].store(acc)
    return out_frag


def make_tensor_with_index(
    view, tile_m, tile_k, index_frag, tiled_copy, tid, topk, is_read_from_mem=True, token_slot_tokens=None
):
    """MoE gather/scatter helper: returns an object whose ``.copy(copy_atom, k_idx, frag)``
    gathers/scatters per-thread tiles by ``index_frag`` (packed token|slot ids).

    ``token_slot_tokens`` (int8smooth): when set, the rank-2 gather row is decoded
    slot-major as ``slot*token_slot_tokens + token`` instead of plain ``token`` (X
    pre-expanded to [topk*tokens, K])."""
    return _TensorWithIndex(
        view, tile_m, tile_k, index_frag, tiled_copy, tid, topk, is_read_from_mem, token_slot_tokens
    )


class _TensorWithIndex:
    def __init__(
        self, view, tile_m, tile_k, index_frag, tiled_copy, tid, topk, is_read_from_mem=True, token_slot_tokens=None
    ):
        self.view = view
        self.tile_m = tile_m
        self.tile_k = tile_k
        self.is_read_from_mem = is_read_from_mem
        self.TOPK = topk
        self.index_frag = index_frag
        # int8smooth slot-major row = slot*tokens + token; None -> plain token gather.
        self.token_slot_tokens = token_slot_tokens

        rank = fx.get_shape(self.view).rank
        dims = [1] * (rank - 1)
        self.tensor_blocks_in_k = fx.zipped_divide(view, fx.make_tile(*dims, tile_k))

        dtype = fx.PointerType.get(fx.Int8.ir_type, 1, 512)
        ptr = fx.inttoptr(dtype, fx.Int32(0))
        self.fake_tensor = fx.make_view(ptr, fx.make_layout((tile_m, tile_k), (1, tile_m)))
        self.fake_tensor_thr = (
            tiled_copy.get_slice(tid).partition_S(self.fake_tensor)
            if is_read_from_mem
            else tiled_copy.get_slice(tid).partition_D(self.fake_tensor)
        )
        offset_thread = fx.Int32(fx.ptrtoint(fx.get_iter(self.fake_tensor_thr)))
        self.offset_thread = offset_thread
        self.offset_thread_k = offset_thread // tile_m
        # Row-guard fake: a tall column-major tile whose row count exceeds any
        # tiled_copy grid, so the atomic epilogue can detect grid slots whose row is
        # outside [0, tile_m) (the plain-store path relies on buffer OOB instead).
        self._guard_rows = 256
        guard_fake = fx.make_view(ptr, fx.make_layout((self._guard_rows, tile_k), (1, self._guard_rows)))
        guard_thr = (
            tiled_copy.get_slice(tid).partition_S(guard_fake)
            if is_read_from_mem
            else tiled_copy.get_slice(tid).partition_D(guard_fake)
        )
        self.guard_offset = fx.Int32(fx.ptrtoint(fx.get_iter(guard_thr)))
        self.guard_layout = fx.get_layout(guard_thr)

    def copy(
        self,
        copy_atom,
        k_idx,
        frag,
        atomic=None,
        atomic_rsrc=None,
        out_bytes=None,
        row_stride=None,
        row_limit=None,
        atomic_dst=None,
    ):
        """Gather/scatter per-thread tiles: plain buffer-view store, or atomic-add at
        tok*row_stride+k_idx*tile_k+chan. ``atomic`` selects the mechanism:
          * "pk"/"f32" -> BUFFER atomic into atomic_rsrc; out-of-tile lanes go OOB
            (dropped by the buffer bounds-check).
          * "pk_global" -> GLOBAL (!llvm.ptr) bf16 pk atomic into atomic_dst (gfx942,
            no buffer_atomic_pk_add_bf16). No hardware bounds-check, so out-of-tile
            lanes are predicated off explicitly."""
        layout = fx.get_layout(self.fake_tensor_thr)
        rep_m = reps(self.fake_tensor_thr, 1)
        rep_k = reps(self.fake_tensor_thr, 2)
        value_size = fx.get_shape(frag)[0].to_py_value()
        stride_size = fx.get_stride(frag)[0].to_py_value()

        rank = fx.get_shape(self.view).rank
        block_cord = [None] * (rank - 1) + [k_idx]
        tensor_block = self.tensor_blocks_in_k[None, (*block_cord,)]
        for m in range_constexpr(rep_m):
            if const_expr(atomic is not None):
                tok = self.index_frag[0, m] & 0xFFFFFF
                # Skip padding rows (sentinel tok >= row_limit) to avoid their
                # dropped-but-still-counted L2 atomic traffic (~4.25x inflation on
                # prefill). Near-wave-uniform under the channel-major TV layout.
                row_valid = tok < fx.Int32(row_limit) if const_expr(row_limit is not None) else (tok >= fx.Int32(0))

                def _atomic_row():
                    row_base_i32 = tok * fx.Int32(row_stride) + fx.Int32(k_idx) * fx.Int32(self.tile_k)
                    for k in range_constexpr(rep_k):
                        offset_block = fx.crd2idx((0, m, k), layout).to_py_value()
                        offset_block_k = offset_block // self.tile_m
                        chan_off = offset_block_k + self.offset_thread_k
                        # valid = this grid slot maps inside the real (tile_m, tile_k) block.
                        guard_full = fx.crd2idx((0, m, k), self.guard_layout).to_py_value() + self.guard_offset
                        g_row = guard_full % fx.Int32(self._guard_rows)
                        g_col = guard_full // fx.Int32(self._guard_rows)
                        valid = (g_row < fx.Int32(self.tile_m)) & (g_col < fx.Int32(self.tile_k))
                        reg_vec = frag[None, m, k].load()
                        _va = reg_vec.numel
                        aligned = (row_base_i32 + chan_off) & fx.Int32(~(_va - 1))
                        if const_expr(atomic == "pk_global"):
                            # BUG GUARD #4: global atomics have NO bounds-check, so
                            # predicate on the in-tile `valid` guard (padding rows
                            # already dropped by `row_valid`).
                            def _global_pk():
                                _global_atomic_pk(atomic_dst, aligned, reg_vec, out_bytes)

                            _if_slot = scf.IfOp(fx.as_ir_value(valid))
                            with _if_then(_if_slot):
                                _global_pk()
                        else:
                            # Buffer path: out-of-tile slots -> OOB index, dropped by
                            # the buffer bounds-check.
                            elem_idx = valid.select(aligned, fx.Int32(row_limit) * fx.Int32(row_stride))
                            if const_expr(atomic == "f32"):
                                _buffer_atomic_f32(atomic_rsrc, elem_idx, reg_vec)
                            else:
                                _buffer_atomic_pk(atomic_rsrc, elem_idx, reg_vec, out_bytes)

                _if_row = scf.IfOp(fx.as_ir_value(row_valid))
                with _if_then(_if_row):
                    _atomic_row()
                continue
            if const_expr(rank == 2):
                if const_expr(self.token_slot_tokens is not None):
                    # int8smooth slot-major A-gather: row_ts = slot*tokens + token.
                    packed = self.index_frag[0, m]
                    row_ts = (packed >> 24) * fx.Int32(self.token_slot_tokens) + (packed & 0xFFFFFF)
                    tensor_sub_block = tensor_block[None, row_ts]
                else:
                    tensor_sub_block = tensor_block[None, self.index_frag[0, m] & 0xFFFFFF]
            else:
                tensor_sub_block = tensor_block[
                    None,
                    self.index_frag[0, m] & 0xFFFFFF,
                    (self.index_frag[0, m] >> 24),
                ]
            for k in range_constexpr(rep_k):
                offset_block = fx.crd2idx((0, m, k), layout).to_py_value()
                offset_block_k = offset_block // self.tile_m
                offset_k_in_tile = offset_block_k + self.offset_thread_k
                reg = frag[None, m, k]
                mem = fx.make_view(
                    fx.get_iter(tensor_sub_block) + offset_k_in_tile,
                    fx.make_layout(value_size, stride_size),
                )
                if const_expr(self.is_read_from_mem):
                    fx.copy(copy_atom, mem, reg)
                else:
                    fx.copy(copy_atom, reg, mem)


_TensorWithIndex.copy = ASTRewriter.transform(_TensorWithIndex.copy)
