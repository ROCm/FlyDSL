# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x linear attention. The state is the prefix of outer products.

``h_t = h_{t-1} + k_t v_t^T`` and ``o_t = q_t h_t``, with ``h_{-1} = 0``.
Quant inputs are scaled into that recurrence. The output of a quant launch is bf16.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch


def _elem(dtype_str: str):
    return {
        "bf16": fx.BFloat16,
        "f16": fx.Float16,
        "fp8_e4m3fn": fx.Float8E4M3FN,
        "fp8_e5m2": fx.Float8E5M2,
        "int8": fx.Int8,
    }[dtype_str]


def build_linear_attn_module(
    head_dim: int,
    head_dim_v: int,
    dtype_str: str,
) -> Callable[..., None]:
    """Build the recurrence. ``head_dim_v`` is the block size."""
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"build_linear_attn_module requires gfx120x, got {arch}"
    if dtype_str not in ("bf16", "f16", "fp8_e4m3fn", "fp8_e5m2", "int8"):
        raise ValueError(f"linear attention dtype {dtype_str!r} is not supported")
    if head_dim < 1 or head_dim_v < 1:
        raise ValueError("linear attention head sizes must be positive")
    elem = _elem(dtype_str)
    out_elem = elem if dtype_str in ("bf16", "f16") else fx.BFloat16
    quant = dtype_str not in ("bf16", "f16")
    # A typed f8 load does not legalize. Quant values are read as i8 bytes.
    load_elem = fx.Int8 if quant else elem
    d_q = int(head_dim)
    block = int(head_dim_v)

    @flyc.kernel(known_block_size=[block, 1, 1])
    def linear_attn_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32,
        num_heads: fx.Int32,
        q_scale: fx.Float32,
        k_scale: fx.Float32,
        v_scale: fx.Float32,
    ) -> None:
        q_ptr = fx.recast_iter(fx.PointerType.get(load_elem.ir_type, Q.address_space), Q)
        k_ptr = fx.recast_iter(fx.PointerType.get(load_elem.ir_type, K.address_space), K)
        v_ptr = fx.recast_iter(fx.PointerType.get(load_elem.ir_type, V.address_space), V)
        o_ptr = fx.recast_iter(fx.PointerType.get(out_elem.ir_type, O.address_space), O)

        def _as_f32(ptr, idx):
            loaded = fx.ptr_load(ptr + idx)
            if const_expr(not quant):
                return fx.Float32(loaded)
            if const_expr(dtype_str == "int8"):
                return fx.Float32(fx.Int8(loaded))
            bits = fx.Int32(loaded) & fx.Int32(255)
            if const_expr(dtype_str == "fp8_e5m2"):
                return fx.Float32(fx.rocdl.cvt_f32_bf8(T.f32, bits.ir_value(), 0))
            return fx.Float32(fx.rocdl.cvt_f32_fp8(T.f32, bits.ir_value(), 0))
        seq_i = fx.Int32(seq_len)
        heads_i = fx.Int32(num_heads)
        linear = fx.Int32(gpu.block_idx.x)
        t = linear % seq_i
        bh = linear // seq_i
        head = bh % heads_i
        batch = bh // heads_i
        j = fx.Int32(gpu.thread_idx.x)
        acc = fx.Float32(0.0)
        s = fx.Int32(0)
        for s in range(fx.Int32(0), t + fx.Int32(1), fx.Int32(1)):
            dot = fx.Float32(0.0)
            q_base = ((batch * seq_i + t) * heads_i + head) * fx.Int32(d_q)
            k_base = ((batch * seq_i + s) * heads_i + head) * fx.Int32(d_q)
            for i in range_constexpr(d_q):
                qv = _as_f32(q_ptr, q_base + fx.Int32(i))
                kv = _as_f32(k_ptr, k_base + fx.Int32(i))
                if const_expr(quant):
                    qv = qv * q_scale
                    kv = kv * k_scale
                dot = dot + qv * kv
            v_base = ((batch * seq_i + s) * heads_i + head) * fx.Int32(block)
            vv = _as_f32(v_ptr, v_base + j)
            if const_expr(quant):
                vv = vv * v_scale
            acc = acc + dot * vv
        o_base = ((batch * seq_i + t) * heads_i + head) * fx.Int32(block)
        fx.ptr_store(acc.to(out_elem), o_ptr + o_base + j)

    @flyc.jit
    def launch(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        batch: fx.Int32,
        seq_len: fx.Int32,
        num_heads: fx.Int32,
        q_scale: fx.Float32,
        k_scale: fx.Float32,
        v_scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid = fx.Uint64(batch) * fx.Uint64(num_heads) * fx.Uint64(seq_len)
        linear_attn_kernel(Q, K, V, O, seq_len, num_heads, q_scale, k_scale, v_scale).launch(
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    return launch


def linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_scale: float = 1.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    stream=None,
) -> torch.Tensor:
    """Run the recurrence. ``q`` and ``k`` share a head, ``v`` sets the output head."""
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"linear_attn requires gfx120x, got {arch}"
    if q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError(f"linear attention shape q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("linear attention q, k, and v must share a dtype")
    names = {
        torch.bfloat16: "bf16",
        torch.float16: "f16",
        torch.float8_e4m3fn: "fp8_e4m3fn",
        torch.float8_e5m2: "fp8_e5m2",
        torch.int8: "int8",
    }
    if q.dtype not in names:
        raise ValueError(f"linear attention dtype {q.dtype} is not supported")
    dtype_str = names[q.dtype]
    batch, seq, heads, head_dim = (int(x) for x in q.shape)
    head_dim_v = int(v.shape[-1])
    out_dtype = q.dtype if dtype_str in ("bf16", "f16") else torch.bfloat16
    out = torch.empty(batch, seq, heads, head_dim_v, device=q.device, dtype=out_dtype)
    launch = build_linear_attn_module(head_dim, head_dim_v, dtype_str)
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    in_ty = _elem(dtype_str)
    out_ty = in_ty if dtype_str in ("bf16", "f16") else fx.BFloat16
    launch(
        flyc.from_c_void_p(in_ty, q.data_ptr()),
        flyc.from_c_void_p(in_ty, k.data_ptr()),
        flyc.from_c_void_p(in_ty, v.data_ptr()),
        flyc.from_c_void_p(out_ty, out.data_ptr()),
        batch,
        seq,
        heads,
        float(q_scale),
        float(k_scale),
        float(v_scale),
        launch_stream,
    )
    return out
