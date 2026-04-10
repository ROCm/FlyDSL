# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused RoPE + KV Cache kernel builder using the @flyc.kernel API.

Fuses 3 operations into two kernel launches:
  Kernel 1 (Q RoPE):     Q → rotate → Q_out
  Kernel 2 (K+V cache):  K → rotate → K_out + key_cache;  V → value_cache

Input shapes:
  Q: [T, QH, D],  K: [T, KH, D],  V: [T, KH, D]
  CosCache/SinCache: [max_pos, D//2]  (must be 2-D contiguous)
  Positions: [T] int32,  SlotMapping: [T] int32

KV cache layouts:
  flash_layout=True:
    KeyCache:   [num_blocks, block_size, KH, D]
    ValueCache: [num_blocks, block_size, KH, D]
  flash_layout=False (ATOM default):
    KeyCache:   [num_blocks, KH, D//x, block_size, x]  (x=16, x-packed)
    ValueCache: [num_blocks, KH, D, block_size]         (dim-major)


"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.expr import vector, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr import buffer_ops
from kernels.kernels_common import dtype_to_elem_type
from kernels.mfma_preshuffle_pipeline import crd2idx


WARP_SIZE = 64
VEC_WIDTH = 8


def _layout_to_dword_off(coord, layout, elem_bytes):
    """Coordinate → dword offset for buffer_load/buffer_store.

    crd2idx(coord, layout) → element offset (index) → byte offset (i32) → dword offset (i32).
    """
    elem_off = ArithValue(crd2idx(coord, layout)).index_cast(T.i32)
    return (ArithValue(elem_off) * elem_bytes) >> fx.Int32(2)


def _make_rope_copy_helpers(elem_type, elem_bits):
    """Build copy atom and register types for RoPE vector loads/stores."""
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
    vec_reg_ty = fx.MemRefType.get(
        elem_type, fx.LayoutType.get(VEC_WIDTH, 1), fx.AddressSpace.Register
    )
    vec_reg_lay = fx.make_layout(VEC_WIDTH, 1)
    return copy_atom, vec_reg_ty, vec_reg_lay


def _load_vec_buf(copy_atom, vec_reg_ty, vec_reg_lay, div_tensor, idx):
    """Vector load via layout API: div_tensor[:, idx] → register vec."""
    r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
    fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return ArithValue(fx.memref_load_vec(r))


def _store_vec_buf(copy_atom, vec_reg_ty, vec_reg_lay, val, div_tensor, idx):
    """Vector store via layout API: register vec → div_tensor[:, idx]."""
    r = fx.memref_alloca(vec_reg_ty, vec_reg_lay)
    fx.memref_store_vec(val, r)
    fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def _apply_neox_rope(qk_div, cos_div, sin_div, pair_div,
                     qk_tid, cos_tid, pair_tid, is_first_half,
                     copy_atom, vec_reg_ty, vec_reg_lay):
    """Load, rotate (NeoX), and return the rotated vector.

    Performs:
      out[first_half]  = qk * cos - pair * sin
      out[second_half] = qk * cos + pair * sin

    Uses buffer-backed tensor layout API for vector loads.

    Args:
        qk_tid:   index into qk_div for current thread's vector
        cos_tid:  index into cos_div/sin_div (tid % vecs_per_half)
        pair_tid: index into pair_div for partner vector

    Returns:
        rot_e: rotated vector in element type
    """
    qk_e   = _load_vec_buf(copy_atom, vec_reg_ty, vec_reg_lay, qk_div, qk_tid)
    cos_e  = _load_vec_buf(copy_atom, vec_reg_ty, vec_reg_lay, cos_div, cos_tid)
    sin_e  = _load_vec_buf(copy_atom, vec_reg_ty, vec_reg_lay, sin_div, cos_tid)
    pair_e = _load_vec_buf(copy_atom, vec_reg_ty, vec_reg_lay, pair_div, pair_tid)

    # NeoX sign: first half uses -sin, second half uses +sin
    qk_cos   = ArithValue(qk_e) * ArithValue(cos_e)
    pair_sin = ArithValue(pair_e) * ArithValue(sin_e)
    sin_term = ArithValue(is_first_half).select(-pair_sin, pair_sin)
    rot_e    = ArithValue(qk_cos) + ArithValue(sin_term)

    return rot_e


def build_fused_rope_cache_module(
    head_dim: int = 64,
    rotary_dim: int = -1,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    block_size: int = 16,
    is_neox: bool = True,
    flash_layout: bool = True,
    dtype_str: str = "bf16",
):
    """Build fused RoPE + KV cache kernel.

    Args:
        head_dim: dimension per attention head
        rotary_dim: dimensions to rotate (== head_dim for full rotation)
        num_q_heads: query heads per rank
        num_kv_heads: KV heads per rank
        block_size: paged attention block size
        is_neox: True for NeoX-style rotation
        flash_layout: True for [num_blocks, block_size, KH, D] cache layout
        dtype_str: element dtype ("bf16" or "f16")

    Returns:
        launch_fn(Q, K, V, Positions, CosCache, SinCache, SlotMapping,
                  KeyCache, ValueCache, Q_out, K_out, num_tokens, stream)
    """
    if rotary_dim == -1:
        rotary_dim = head_dim
    if not is_neox:
        raise NotImplementedError("Only NeoX-style RoPE is supported")
    if rotary_dim != head_dim:
        raise NotImplementedError("Partial rotation not yet supported")
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(
            f"dtype_str must be 'bf16' or 'f16', got {dtype_str!r} "
            f"(f32 is not supported: kernel uses 2-byte elem_bytes and vec8 vectorization)"
        )
    half_dim = rotary_dim // 2
    elem_bytes = 2  # bf16 and f16 are both 2 bytes
    vec_dwords = (VEC_WIDTH * elem_bytes) // 4  # 4 dwords for vec8 of 2-byte elements
    vecs_per_half = half_dim // VEC_WIDTH   # number of VEC_WIDTH-wide vectors covering half_dim
    vecs_per_head = head_dim // VEC_WIDTH   # number of VEC_WIDTH-wide vectors covering head_dim
    x_size = 16  # x-packing factor for non-flash key_cache

    # Validate vectorization and layout assumptions to avoid silent truncation.
    if head_dim % VEC_WIDTH != 0:
        raise ValueError(
            f"head_dim must be a multiple of VEC_WIDTH ({VEC_WIDTH}), "
            f"got head_dim={head_dim}"
        )
    if rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be even so that half_dim=rotary_dim//2 is integral, "
            f"got rotary_dim={rotary_dim}"
        )
    if half_dim % VEC_WIDTH != 0:
        raise ValueError(
            f"half_dim (rotary_dim//2) must be a multiple of VEC_WIDTH "
            f"({VEC_WIDTH}), got half_dim={half_dim} (rotary_dim={rotary_dim})"
        )
    if not flash_layout and head_dim % x_size != 0:
        raise ValueError(
            f"With flash_layout=False, head_dim must be a multiple of the "
            f"key_cache packing factor x_size ({x_size}), got head_dim={head_dim}"
        )
    if vecs_per_head > WARP_SIZE:
        max_head_dim = WARP_SIZE * VEC_WIDTH
        raise ValueError(
            f"Unsupported head_dim={head_dim}: with WARP_SIZE={WARP_SIZE} and "
            f"VEC_WIDTH={VEC_WIDTH}, head_dim must satisfy "
            f"head_dim <= {max_head_dim} to avoid incomplete coverage "
            f"(got vecs_per_head={vecs_per_head} > WARP_SIZE)"
        )
    BLOCK_THREADS = WARP_SIZE

    # Layout shape/stride tuples (plain Python ints) — materialized as
    # fx.make_layout inside each kernel where an MLIR context is active.
    # None is used for dynamic/unknown extents (token count, position range,
    # block count) so the layout shape matches the actual indexing domain.
    _q_shape = (None, num_q_heads, vecs_per_head)
    _q_stride = (num_q_heads * head_dim, head_dim, VEC_WIDTH)
    _kv_shape = (None, num_kv_heads, vecs_per_head)
    _kv_stride = (num_kv_heads * head_dim, head_dim, VEC_WIDTH)
    _cos_shape = (None, vecs_per_half)
    _cos_stride = (half_dim, VEC_WIDTH)

    # ----- Kernel 1: Q RoPE -----
    # Grid: (T * QH, 1, 1), one program per (token, q_head)
    # Each program: vecs_per_head threads process head_dim elements
    @flyc.kernel
    def q_rope_kernel(
        Q: fx.Tensor,            # [T, QH, D]
        Positions: fx.Tensor,    # [T] int32
        CosCache: fx.Tensor,     # [max_pos, half_dim]
        SinCache: fx.Tensor,     # [max_pos, half_dim]
        Q_out: fx.Tensor,        # [T, QH, D]
    ):
        pid = fx.block_idx.x    # program id: 0..T*QH-1
        tid = fx.thread_idx.x   # 0..63

        elem_type = dtype_to_elem_type(dtype_str)
        elem_bits = 16  # bf16/f16 only

        # Buffer-backed tensors via layout API
        Q_buf = fx.rocdl.make_buffer_tensor(Q)
        Qo_buf = fx.rocdl.make_buffer_tensor(Q_out)
        Cos_buf = fx.rocdl.make_buffer_tensor(CosCache)
        Sin_buf = fx.rocdl.make_buffer_tensor(SinCache)
        pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)

        copy_atom, vec_reg_ty, vec_reg_lay = _make_rope_copy_helpers(elem_type, elem_bits)

        if tid < fx.Int32(vecs_per_head):
            pid_t = pid // num_q_heads
            pid_hq = pid % num_q_heads

            # Load position
            pos_val = buffer_ops.buffer_load(pos_rsrc, pid_t, vec_width=1, dtype=T.i32)

            # Q[pid_t, pid_hq, :] tiled by VEC_WIDTH
            q_row = fx.slice(Q_buf, (pid_t, fx.Int32(pid_hq), None))
            q_div = fx.logical_divide(q_row, fx.make_layout(VEC_WIDTH, 1))

            # Q_out[pid_t, pid_hq, :] tiled by VEC_WIDTH
            qo_row = fx.slice(Qo_buf, (pid_t, fx.Int32(pid_hq), None))
            qo_div = fx.logical_divide(qo_row, fx.make_layout(VEC_WIDTH, 1))

            # cos/sin[pos_val, :] tiled by VEC_WIDTH
            cos_row = fx.slice(Cos_buf, (pos_val, None))
            cos_div = fx.logical_divide(cos_row, fx.make_layout(VEC_WIDTH, 1))
            sin_row = fx.slice(Sin_buf, (pos_val, None))
            sin_div = fx.logical_divide(sin_row, fx.make_layout(VEC_WIDTH, 1))

            # NeoX rotation: pair with opposite half
            is_first_half = tid < fx.Int32(vecs_per_half)
            pair_tid = ArithValue(is_first_half).select(tid + vecs_per_half, tid - vecs_per_half)
            # tid % vecs_per_half wraps into cos/sin range
            cos_vec_idx = tid % vecs_per_half

            rot_e = _apply_neox_rope(
                q_div, cos_div, sin_div, q_div,
                tid, cos_vec_idx, pair_tid, is_first_half,
                copy_atom, vec_reg_ty, vec_reg_lay,
            )
            _store_vec_buf(copy_atom, vec_reg_ty, vec_reg_lay, rot_e, qo_div, tid)

    # ----- Kernel 2: K RoPE + KV cache write -----
    # Grid: (T * KH, 1, 1), one program per (token, kv_head)
    # Each program: vecs_per_head threads process head_dim elements
    @flyc.kernel
    def k_cache_kernel(
        K: fx.Tensor,            # [T, KH, D]
        V: fx.Tensor,            # [T, KH, D]
        Positions: fx.Tensor,    # [T] int32
        CosCache: fx.Tensor,     # [max_pos, half_dim]
        SinCache: fx.Tensor,     # [max_pos, half_dim]
        SlotMapping: fx.Tensor,  # [T] int32
        KeyCache: fx.Tensor,     # flash: [T_cache, BS, KH, D]
        ValueCache: fx.Tensor,   # flash: [T_cache, BS, KH, D]
        K_out: fx.Tensor,        # [T, KH, D]
    ):
        pid = fx.block_idx.x    # program id: 0..T*KH-1
        tid = fx.thread_idx.x   # 0..63

        elem_type = dtype_to_elem_type(dtype_str)
        vec_type_e = T.vec(VEC_WIDTH, elem_type)
        i32_vec_ty = T.vec(vec_dwords, T.i32)
        elem_bits = 16  # bf16/f16 only

        # Buffer-backed tensors via layout API
        K_buf = fx.rocdl.make_buffer_tensor(K)
        V_buf = fx.rocdl.make_buffer_tensor(V)
        Ko_buf = fx.rocdl.make_buffer_tensor(K_out)
        Cos_buf = fx.rocdl.make_buffer_tensor(CosCache)
        Sin_buf = fx.rocdl.make_buffer_tensor(SinCache)
        pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
        slot_rsrc = buffer_ops.create_buffer_resource(SlotMapping, max_size=True)
        # KV cache: keep buffer_ops for complex scattered writes
        kc_rsrc = buffer_ops.create_buffer_resource(KeyCache, max_size=True)
        vc_rsrc = buffer_ops.create_buffer_resource(ValueCache, max_size=True)

        copy_atom, vec_reg_ty, vec_reg_lay = _make_rope_copy_helpers(elem_type, elem_bits)

        # Layouts for KV cache (used in non-layout-API scatter paths)
        kv_layout = fx.make_layout(_kv_shape, _kv_stride)

        if tid < fx.Int32(vecs_per_head):
            pid_t = pid // num_kv_heads
            pid_hk = pid % num_kv_heads

            # Load position
            pos_val = buffer_ops.buffer_load(pos_rsrc, pid_t, vec_width=1, dtype=T.i32)

            # K[pid_t, pid_hk, :] tiled by VEC_WIDTH
            k_row = fx.slice(K_buf, (pid_t, fx.Int32(pid_hk), None))
            k_div = fx.logical_divide(k_row, fx.make_layout(VEC_WIDTH, 1))

            # K_out[pid_t, pid_hk, :] tiled by VEC_WIDTH
            ko_row = fx.slice(Ko_buf, (pid_t, fx.Int32(pid_hk), None))
            ko_div = fx.logical_divide(ko_row, fx.make_layout(VEC_WIDTH, 1))

            # cos/sin[pos_val, :] tiled by VEC_WIDTH
            cos_row = fx.slice(Cos_buf, (pos_val, None))
            cos_div = fx.logical_divide(cos_row, fx.make_layout(VEC_WIDTH, 1))
            sin_row = fx.slice(Sin_buf, (pos_val, None))
            sin_div = fx.logical_divide(sin_row, fx.make_layout(VEC_WIDTH, 1))

            # NeoX rotation
            is_first_half = tid < fx.Int32(vecs_per_half)
            pair_tid = ArithValue(is_first_half).select(tid + vecs_per_half, tid - vecs_per_half)
            cos_vec_idx = tid % vecs_per_half

            k_rot_e = _apply_neox_rope(
                k_div, cos_div, sin_div, k_div,
                tid, cos_vec_idx, pair_tid, is_first_half,
                copy_atom, vec_reg_ty, vec_reg_lay,
            )
            _store_vec_buf(copy_atom, vec_reg_ty, vec_reg_lay, k_rot_e, ko_div, tid)

            # --- KV Cache write ---
            slot_val = buffer_ops.buffer_load(slot_rsrc, pid_t, vec_width=1, dtype=T.i32)

            if slot_val >= fx.Int32(0):
                pid_t_slot = ArithValue(slot_val) // block_size
                pid_b = ArithValue(slot_val) % block_size

                # Load V via layout API
                v_row = fx.slice(V_buf, (pid_t, fx.Int32(pid_hk), None))
                v_div = fx.logical_divide(v_row, fx.make_layout(VEC_WIDTH, 1))
                v_e = _load_vec_buf(copy_atom, vec_reg_ty, vec_reg_lay, v_div, tid)

                # Bitcast for KV cache stores (buffer_ops needs i32 vecs)
                k_rot_i32 = vector.bitcast(i32_vec_ty, k_rot_e)
                v_raw = vector.bitcast(i32_vec_ty, v_e)

                # KV cache dword offset for stores (scattered, keep buffer_ops)
                kv_coord = (pid_t, fx.Int32(pid_hk), tid)
                k_dw = _layout_to_dword_off(kv_coord, kv_layout, elem_bytes)

                if flash_layout:
                    kc_flash_layout = fx.make_layout(
                        (None, block_size, num_kv_heads, vecs_per_head),
                        (block_size * num_kv_heads * head_dim,
                         num_kv_heads * head_dim,
                         head_dim,
                         VEC_WIDTH),
                    )
                    kc_coord = (pid_t_slot, pid_b, pid_hk, tid)
                    kc_dw = _layout_to_dword_off(kc_coord, kc_flash_layout, elem_bytes)

                    buffer_ops.buffer_store(k_rot_i32, kc_rsrc, kc_dw)
                    buffer_ops.buffer_store(v_raw, vc_rsrc, kc_dw)
                else:
                    # Non-flash key_cache: [num_blocks, KH, D//x, BS, x]
                    d_start = ArithValue(tid) * VEC_WIDTH
                    dim_group = d_start // x_size
                    dim_within = d_start % x_size

                    kc_nf_layout = fx.make_layout(
                        (None, num_kv_heads, head_dim // x_size, block_size, x_size),
                        (num_kv_heads * (head_dim // x_size) * block_size * x_size,
                         (head_dim // x_size) * block_size * x_size,
                         block_size * x_size,
                         x_size,
                         1),
                    )
                    kc_coord_nf = (pid_t_slot, pid_hk, dim_group, pid_b, dim_within)
                    kc_dw_nf = _layout_to_dword_off(kc_coord_nf, kc_nf_layout, elem_bytes)

                    buffer_ops.buffer_store(k_rot_i32, kc_rsrc, kc_dw_nf)

                    # Non-flash value_cache: scalar stores (non-contiguous layout)
                    vc_nf_layout = fx.make_layout(
                        (None, num_kv_heads, head_dim, block_size),
                        (num_kv_heads * head_dim * block_size,
                         head_dim * block_size,
                         block_size,
                         1),
                    )
                    for vi in range_constexpr(VEC_WIDTH):
                        v_scalar = vector.extract(v_e, static_position=[vi])
                        d_idx = ArithValue(tid) * VEC_WIDTH + vi
                        vc_coord = (pid_t_slot, pid_hk, d_idx, pid_b)
                        vc_elem_off = ArithValue(crd2idx(vc_coord, vc_nf_layout)).index_cast(T.i32)
                        buffer_ops.buffer_store(v_scalar, vc_rsrc, vc_elem_off)

    @flyc.jit
    def launch_fused_rope_cache(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Positions: fx.Tensor,
        CosCache: fx.Tensor,
        SinCache: fx.Tensor,
        SlotMapping: fx.Tensor,
        KeyCache: fx.Tensor,
        ValueCache: fx.Tensor,
        Q_out: fx.Tensor,
        K_out: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Kernel 1: Q RoPE
        n_q = ArithValue(num_tokens) * num_q_heads
        q_launcher = q_rope_kernel(Q, Positions, CosCache, SinCache, Q_out)
        q_launcher.launch(
            grid=(n_q, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

        # Kernel 2: K RoPE + KV cache write
        n_k = ArithValue(num_tokens) * num_kv_heads
        k_launcher = k_cache_kernel(
            K, V, Positions, CosCache, SinCache, SlotMapping,
            KeyCache, ValueCache, K_out,
        )
        k_launcher.launch(
            grid=(n_k, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_rope_cache
