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

Uses FlyDSL layout API (make_layout, crd2idx) for address computation,
replacing manual byte-offset arithmetic. Paired-half RoPE offset uses
arith (piecewise, not expressible as affine layout stride).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.expr import arith, vector, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr import buffer_ops
from flydsl._mlir import ir


WARP_SIZE = 64
VEC_WIDTH = 8


def _crd2idx_i32(coord, layout):
    """crd2idx → scalar i32 (unwraps int_tuple, casts to i32 for buffer offset math)."""
    result = fx.crd2idx(coord, layout)
    scalar = fx.get_scalar(result)
    if isinstance(scalar, ir.Value) and isinstance(scalar.type, ir.IndexType):
        scalar = arith.index_cast(T.i32, scalar)
    return scalar


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f16":
        return T.f16
    if dtype_str == "bf16":
        return T.bf16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'bf16' or 'f16')")


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
        raise NotImplementedError(
            f"Only bf16 and f16 are supported (got dtype_str={dtype_str!r}). "
            f"f32 requires vec_width=8 for buffer_load/store which exceeds "
            f"the hardware maximum of 4 dwords."
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
    _q_shape = (1, num_q_heads, vecs_per_head)
    _q_stride = (num_q_heads * head_dim, head_dim, VEC_WIDTH)
    _kv_shape = (1, num_kv_heads, vecs_per_head)
    _kv_stride = (num_kv_heads * head_dim, head_dim, VEC_WIDTH)
    _cos_shape = (1, vecs_per_half)
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
        vec_type_e = T.vec(VEC_WIDTH, elem_type)
        i32_vec_ty = T.vec(vec_dwords, T.i32)

        # Buffer resources at top level (not inside scf.if)
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True)
        pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
        cos_rsrc = buffer_ops.create_buffer_resource(CosCache, max_size=True)
        sin_rsrc = buffer_ops.create_buffer_resource(SinCache, max_size=True)
        qo_rsrc = buffer_ops.create_buffer_resource(Q_out, max_size=True)

        # Layouts (materialized inside kernel where MLIR context is active)
        q_layout = fx.make_layout(_q_shape, _q_stride)
        cos_sin_layout = fx.make_layout(_cos_shape, _cos_stride)

        if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_head)):
            pid_t = pid // num_q_heads
            pid_hq = pid % num_q_heads

            # Load position
            pos_val = buffer_ops.buffer_load(pos_rsrc, pid_t, vec_width=1, dtype=T.i32)

            # -- Q address via crd2idx --
            q_coord = (pid_t, fx.Int32(pid_hq), tid)
            q_elem_off = _crd2idx_i32(q_coord, q_layout)
            q_bytes = ArithValue(q_elem_off) * elem_bytes
            q_dw = q_bytes >> fx.Int32(2)

            # -- cos/sin address via crd2idx --
            cos_vec_idx = tid % vecs_per_half
            cos_coord = (pos_val, fx.Int32(cos_vec_idx))
            cos_elem_off = _crd2idx_i32(cos_coord, cos_sin_layout)
            cos_bytes = ArithValue(cos_elem_off) * elem_bytes
            cos_dw = cos_bytes >> fx.Int32(2)

            # Load Q, cos, sin
            q_raw = buffer_ops.buffer_load(q_rsrc, q_dw, vec_width=vec_dwords, dtype=T.i32)
            cos_raw = buffer_ops.buffer_load(cos_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
            sin_raw = buffer_ops.buffer_load(sin_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
            q_e = vector.bitcast(vec_type_e, q_raw) if vec_dwords != VEC_WIDTH else q_raw.bitcast(vec_type_e)
            cos_e = vector.bitcast(vec_type_e, cos_raw) if vec_dwords != VEC_WIDTH else cos_raw.bitcast(vec_type_e)
            sin_e = vector.bitcast(vec_type_e, sin_raw) if vec_dwords != VEC_WIDTH else sin_raw.bitcast(vec_type_e)

            # -- Paired-half load (arith — piecewise offset, not layout-expressible) --
            is_first_half = arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_half))
            pair_off_first = q_bytes + (half_dim * elem_bytes)
            pair_off_second = q_bytes - (half_dim * elem_bytes)
            pair_bytes = arith.select(is_first_half, pair_off_first, pair_off_second)
            pair_dw = pair_bytes >> fx.Int32(2)
            pair_raw = buffer_ops.buffer_load(q_rsrc, pair_dw, vec_width=vec_dwords, dtype=T.i32)
            pair_e = vector.bitcast(vec_type_e, pair_raw) if vec_dwords != VEC_WIDTH else pair_raw.bitcast(vec_type_e)

            # -- NeoX rotation (arith — real math) --
            # first_half:  out = q*cos - pair*sin
            # second_half: out = q*cos + pair*sin
            q_cos = ArithValue(q_e) * ArithValue(cos_e)
            pair_sin = ArithValue(pair_e) * ArithValue(sin_e)
            neg_pair_sin = arith.negf(pair_sin)
            sin_term = arith.select(is_first_half, neg_pair_sin, pair_sin)
            rot_e = ArithValue(q_cos) + ArithValue(sin_term)

            rot_i32 = vector.bitcast(i32_vec_ty, rot_e) if vec_dwords != VEC_WIDTH else rot_e.bitcast(i32_vec_ty)
            buffer_ops.buffer_store(rot_i32, qo_rsrc, q_dw)

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

        # Buffer resources at top level
        k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(V, max_size=True)
        pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
        cos_rsrc = buffer_ops.create_buffer_resource(CosCache, max_size=True)
        sin_rsrc = buffer_ops.create_buffer_resource(SinCache, max_size=True)
        slot_rsrc = buffer_ops.create_buffer_resource(SlotMapping, max_size=True)
        kc_rsrc = buffer_ops.create_buffer_resource(KeyCache, max_size=True)
        vc_rsrc = buffer_ops.create_buffer_resource(ValueCache, max_size=True)
        ko_rsrc = buffer_ops.create_buffer_resource(K_out, max_size=True)

        # Layouts (materialized inside kernel where MLIR context is active)
        kv_layout = fx.make_layout(_kv_shape, _kv_stride)
        cos_sin_layout = fx.make_layout(_cos_shape, _cos_stride)

        if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_head)):
            pid_t = pid // num_kv_heads
            pid_hk = pid % num_kv_heads

            # Load position
            pos_val = buffer_ops.buffer_load(pos_rsrc, pid_t, vec_width=1, dtype=T.i32)

            # -- K/V address via crd2idx --
            kv_coord = (pid_t, fx.Int32(pid_hk), tid)
            kv_elem_off = _crd2idx_i32(kv_coord, kv_layout)
            kv_bytes = ArithValue(kv_elem_off) * elem_bytes
            k_dw = kv_bytes >> fx.Int32(2)

            # -- cos/sin address via crd2idx --
            cos_vec_idx = tid % vecs_per_half
            cos_coord = (pos_val, fx.Int32(cos_vec_idx))
            cos_elem_off = _crd2idx_i32(cos_coord, cos_sin_layout)
            cos_bytes = ArithValue(cos_elem_off) * elem_bytes
            cos_dw = cos_bytes >> fx.Int32(2)

            # Load K, V, cos, sin
            k_raw = buffer_ops.buffer_load(k_rsrc, k_dw, vec_width=vec_dwords, dtype=T.i32)
            cos_raw = buffer_ops.buffer_load(cos_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
            sin_raw = buffer_ops.buffer_load(sin_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
            k_e = vector.bitcast(vec_type_e, k_raw) if vec_dwords != VEC_WIDTH else k_raw.bitcast(vec_type_e)
            cos_e = vector.bitcast(vec_type_e, cos_raw) if vec_dwords != VEC_WIDTH else cos_raw.bitcast(vec_type_e)
            sin_e = vector.bitcast(vec_type_e, sin_raw) if vec_dwords != VEC_WIDTH else sin_raw.bitcast(vec_type_e)

            # -- Paired-half load (arith — piecewise offset) --
            is_first_half = arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_half))
            pair_off_first = kv_bytes + (half_dim * elem_bytes)
            pair_off_second = kv_bytes - (half_dim * elem_bytes)
            pair_bytes = arith.select(is_first_half, pair_off_first, pair_off_second)
            pair_dw = pair_bytes >> fx.Int32(2)
            pair_raw = buffer_ops.buffer_load(k_rsrc, pair_dw, vec_width=vec_dwords, dtype=T.i32)
            pair_e = vector.bitcast(vec_type_e, pair_raw) if vec_dwords != VEC_WIDTH else pair_raw.bitcast(vec_type_e)

            # -- K RoPE rotation (arith — real math) --
            k_cos = ArithValue(k_e) * ArithValue(cos_e)
            pair_sin = ArithValue(pair_e) * ArithValue(sin_e)
            neg_pair_sin = arith.negf(pair_sin)
            sin_term = arith.select(is_first_half, neg_pair_sin, pair_sin)
            k_rot_e = ArithValue(k_cos) + ArithValue(sin_term)

            # Store k_out
            k_rot_i32 = vector.bitcast(i32_vec_ty, k_rot_e) if vec_dwords != VEC_WIDTH else k_rot_e.bitcast(i32_vec_ty)
            buffer_ops.buffer_store(k_rot_i32, ko_rsrc, k_dw)

            # --- KV Cache write ---
            slot_val = buffer_ops.buffer_load(slot_rsrc, pid_t, vec_width=1, dtype=T.i32)

            if arith.cmpi(arith.CmpIPredicate.sge, slot_val, fx.Int32(0)):
                pid_t_slot = ArithValue(slot_val) // block_size
                pid_b = ArithValue(slot_val) % block_size

                # Load V for cache write (same offset as K)
                v_raw = buffer_ops.buffer_load(v_rsrc, k_dw, vec_width=vec_dwords, dtype=T.i32)

                if flash_layout:
                    # -- Flash KV cache: [num_blocks, BS, KH, D] via crd2idx --
                    kc_flash_layout = fx.make_layout(
                        (1, block_size, num_kv_heads, vecs_per_head),
                        (block_size * num_kv_heads * head_dim,
                         num_kv_heads * head_dim,
                         head_dim,
                         VEC_WIDTH),
                    )
                    kc_coord = (pid_t_slot, pid_b, pid_hk, tid)
                    kc_elem_off = _crd2idx_i32(kc_coord, kc_flash_layout)
                    kc_bytes = ArithValue(kc_elem_off) * elem_bytes
                    kc_dw = kc_bytes >> fx.Int32(2)

                    buffer_ops.buffer_store(k_rot_i32, kc_rsrc, kc_dw)
                    # value_cache: same layout
                    buffer_ops.buffer_store(v_raw, vc_rsrc, kc_dw)
                else:
                    # -- Non-flash key_cache: [num_blocks, KH, D//x, BS, x] via crd2idx --
                    d_start = ArithValue(tid) * VEC_WIDTH
                    dim_group = d_start // x_size
                    dim_within = d_start % x_size

                    kc_nf_layout = fx.make_layout(
                        (1, num_kv_heads, head_dim // x_size, block_size, x_size),
                        (num_kv_heads * (head_dim // x_size) * block_size * x_size,
                         (head_dim // x_size) * block_size * x_size,
                         block_size * x_size,
                         x_size,
                         1),
                    )
                    kc_coord_nf = (pid_t_slot, pid_hk, dim_group, pid_b, dim_within)
                    kc_elem_off_nf = _crd2idx_i32(kc_coord_nf, kc_nf_layout)
                    kc_bytes_nf = ArithValue(kc_elem_off_nf) * elem_bytes
                    kc_dw_nf = kc_bytes_nf >> fx.Int32(2)

                    buffer_ops.buffer_store(k_rot_i32, kc_rsrc, kc_dw_nf)

                    # -- Non-flash value_cache: [num_blocks, KH, D, BS] scalar stores --
                    v_e = vector.bitcast(vec_type_e, v_raw) if vec_dwords != VEC_WIDTH else v_raw.bitcast(vec_type_e)

                    vc_nf_layout = fx.make_layout(
                        (1, num_kv_heads, head_dim, block_size),
                        (num_kv_heads * head_dim * block_size,
                         head_dim * block_size,
                         block_size,
                         1),
                    )
                    for vi in range_constexpr(VEC_WIDTH):
                        v_scalar = vector.extract(v_e, static_position=[vi])
                        d_idx = ArithValue(tid) * VEC_WIDTH + vi
                        vc_coord = (pid_t_slot, pid_hk, d_idx, pid_b)
                        vc_elem_off = _crd2idx_i32(vc_coord, vc_nf_layout)
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
