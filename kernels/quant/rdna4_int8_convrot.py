# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X INT8 ConvRot weight/act quantize (Hadamard rotate + rowwise INT8).

Matches HIP ``quantize_int8_convrot_weight`` / fused
``quantize_and_rotate_rowwise`` semantics:

* Regular Hadamard of size ``group_size`` in {16, 64, 256} via radix-4 FWHT
  (Kronecker of H4); normalize by ``1/sqrt(G)``.
* Per-row scale ``amax/127`` (floor ``1e-30``); ``q = roundeven(x * rcp(scale))``.

Weights are rotated offline (``W @ H``; H is symmetric). Activations use the
same kernel online before ``int8_linear``. Dequant is the inverse kernel:
int8 times the row scale, then the same FWHT. Packed ``convrot_w4a4`` /
``asym_w4a8`` live in ``rdna4_convrot_w4a4`` / ``rdna4_asym_w4a8``.

"""

import math
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor


def _kernel_signature(**params: object) -> str:
    """Specialization suffix for kernel/launch names (matches kernels.common.gfx120x_buf_helpers)."""
    return "_".join(
        f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()
    ).replace("-", "_")


KERNEL_NAME = "int8_convrot_quant_gfx120x"
WARP = 32
_SUPPORTED_GROUPS = (16, 64, 256)
# One Hadamard group is staged in LDS as f32. Groups do not mix, so the row
# length is not an LDS limit.
_INV_127 = 1.0 / 127.0
TUNING_SCHEMA = 1
_BLOCK_CHOICES = (32, 64, 128, 256, 512, 1024)


def _block_threads(k: int) -> int:
    if k <= 256:
        return 32
    if k <= 512:
        return 64
    if k <= 2048:
        return 128
    return 256


def _dtype_name(dtype: torch.dtype) -> str:
    import torch

    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3fn",
        torch.float8_e5m2: "float8_e5m2",
    }[dtype]


@lru_cache(maxsize=64)
def build_int8_convrot_quant_module(
    in_dtype: str = "bfloat16",
    group_size: int = 256,
    k: int = 256,
    block_threads: int | None = None,
    stochastic: bool = False,
    logical_k: int | None = None,
) -> Callable[..., None]:
    """Compile one-row-per-block ConvRot rotate + rowwise INT8 quant kernel.

    ``k`` is the Hadamard width and the stored width. ``logical_k`` is the
    caller's K. A shorter row zero-fills the group on load and still stores
    every rotated column. The tail is part of the orthogonal product.
    """
    if logical_k is None:
        logical_k = k
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    if k <= 0 or k % group_size != 0:
        raise ValueError(f"K={k} must be positive and divisible by group_size={group_size}")
    if logical_k <= 0 or logical_k > k:
        raise ValueError(f"logical_k={logical_k} must be in 1..K={k}")

    InTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[in_dtype]
    in_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[in_dtype]
    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")

    n_stages = int(math.log(group_size, 4))
    n_groups = k // group_size
    # Butterflies inside one group. The LDS tile is that group, not the row.
    g_butterflies = group_size // 4
    g_bf_steps = (g_butterflies + block_threads - 1) // block_threads
    g_load_steps = (group_size + block_threads - 1) // block_threads
    reduction_slots = (block_threads + WARP - 1) // WARP
    inv_sqrt_g = 1.0 / math.sqrt(float(group_size))
    sig = _kernel_signature(
        block=block_threads,
        dtype=in_dtype,
        group_size=group_size,
        k=k,
        op="int8_convrot",
        stoch=int(stochastic),
    )

    @fx.struct
    class SharedStorage:
        row: fx.Array[fx.Float32, group_size, 16]
        red: fx.Array[fx.Float32, reduction_slots, 16]

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def convrot_quant_kernel(
        In: fx.Tensor,
        OutQ: fx.Tensor,
        OutScale: fx.Tensor,
        n_rows: fx.Int32,
        seed: fx.Int32,
    ) -> None:
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        c0 = fx.Float32(0.0)
        # Grid is exactly n_rows (one block per row).
        active_row = bid < n_rows

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        row_v = lds.row.view(fx.make_layout(group_size, 1))
        red_v = lds.red.view(fx.make_layout(reduction_slots, 1))

        def lds_load(idx: object) -> object:
            safe = (idx >= fx.Int32(0)) & (idx < fx.Int32(group_size))
            off = safe.select(idx, fx.Int32(0))
            return safe.select(fx.Float32(row_v[off]), c0)

        def lds_store(idx: object, val: object) -> None:
            safe = (idx >= fx.Int32(0)) & (idx < fx.Int32(group_size))
            off = safe.select(idx, fx.Int32(0))
            fx.memref_store(fx.Float32(val), row_v, off)

        def wave_reduce_max(val: object) -> fx.Float32:
            w = val
            for _sh in range_constexpr(int(math.log2(WARP))):
                off = WARP // (2 << _sh)
                w = fx.max(w, gpu.shuffle_xor(w, off, WARP))
            return w

        def block_reduce_max(val: object) -> fx.Float32:
            lane = tid % WARP
            wave = tid // WARP
            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, red_v, wave)
            gpu.barrier()
            if wave == 0:
                in_range = lane < fx.Int32(reduction_slots)
                lane_safe = in_range.select(lane, fx.Int32(0))
                vv = red_v[lane_safe]
                ww = in_range.select(fx.Float32(vv), c0)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, red_v, fx.Int32(0))
            gpu.barrier()
            return fx.Float32(red_v[fx.Int32(0)])

        # One group in LDS. Pass 1 finds the row amax. Pass 2 repeats the
        # same f32 FWHT and quantizes. Repeating avoids a full-row scratch.
        in_buf = ptr_buf_tensor(
            In,
            elem=InTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(logical_k) * fx.Int64(in_bytes),
        )

        def load_group(group: object) -> None:
            base = fx.Int32(group) * fx.Int32(group_size)
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                col = base + local
                inb = active_row & (local < fx.Int32(group_size))
                if const_expr(logical_k != k):
                    take = inb & (col < fx.Int32(logical_k))
                    gidx = fx.Int64(bid) * fx.Int64(logical_k) + fx.Int64(col)
                    safe_g = take.select(gidx, fx.Int64(0))
                    raw = buf_copy_load(in_buf, safe_g, elem=InTy, unit_elems=1)
                    if const_expr(in_dtype != "float32"):
                        val = fx.Float32(InTy(raw).to(fx.Float32))
                    else:
                        val = fx.Float32(raw)
                    val = take.select(val, c0)
                else:
                    gidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(col)
                    safe_g = inb.select(gidx, fx.Int64(0))
                    raw = buf_copy_load(in_buf, safe_g, elem=InTy, unit_elems=1)
                    if const_expr(in_dtype != "float32"):
                        val = fx.Float32(InTy(raw).to(fx.Float32))
                    else:
                        val = fx.Float32(raw)
                    val = inb.select(val, c0)
                if local < fx.Int32(group_size):
                    lds_store(local, val)

        def fwht_tile() -> None:
            for stage in range_constexpr(n_stages):
                stride = 4**stage
                block = 4 * stride
                for step in range_constexpr(g_bf_steps):
                    bf = tid + fx.Int32(step * block_threads)
                    inb = active_row & (bf < fx.Int32(g_butterflies))
                    base_idx = (bf // fx.Int32(stride)) * fx.Int32(block)
                    i = bf % fx.Int32(stride)
                    idx0 = base_idx + i
                    idx1 = idx0 + fx.Int32(stride)
                    idx2 = idx0 + fx.Int32(2 * stride)
                    idx3 = idx0 + fx.Int32(3 * stride)
                    a = lds_load(idx0)
                    b = lds_load(idx1)
                    c = lds_load(idx2)
                    d = lds_load(idx3)
                    y0 = a + b + c - d
                    y1 = a + b - c + d
                    y2 = a - b + c + d
                    y3 = -a + b + c + d
                    if inb:
                        lds_store(idx0, y0)
                        lds_store(idx1, y1)
                        lds_store(idx2, y2)
                        lds_store(idx3, y3)
                gpu.barrier()

        def norm_tile() -> None:
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                if local < fx.Int32(group_size):
                    lds_store(local, lds_load(local) * fx.Float32(inv_sqrt_g))
            gpu.barrier()

        thread_amax = c0
        for group in range(fx.Int32(n_groups)):
            load_group(group)
            gpu.barrier()
            fwht_tile()
            norm_tile()
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                inb = active_row & (local < fx.Int32(group_size))
                v = lds_load(local)
                thread_amax = fx.max(thread_amax, inb.select(fmath.absf(v), c0))
            gpu.barrier()

        amax = block_reduce_max(thread_amax)
        scale = fx.max(amax * fx.Float32(_INV_127), fx.Float32(1e-30))
        inv = fx.Float32(fx.rocdl.rcp(T.f32, scale))

        if active_row & (tid == fx.Int32(0)):
            sc_buf = ptr_buf_tensor(
                OutScale,
                elem=fx.Float32,
                n=0x3FFFFFFF,
                unit_elems=1,
                num_records_bytes=fx.Int64(n_rows) * fx.Int64(4),
            )
            buf_copy_store(sc_buf, fx.Int64(bid), scale, elem=fx.Float32, unit_elems=1)

        gpu.barrier()

        q_buf = ptr_buf_tensor(
            OutQ,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(k),
        )
        for group in range(fx.Int32(n_groups)):
            load_group(group)
            gpu.barrier()
            fwht_tile()
            norm_tile()
            base = fx.Int32(group) * fx.Int32(group_size)
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                col = base + local
                inb = active_row & (local < fx.Int32(group_size))
                v = lds_load(local)
                # seed > 0: floor(v/scale + U). U is an in-kernel hash in [0, 1), not a torch buffer.
                qf = v * inv
                if const_expr(stochastic):
                    h = (fx.Int32(bid) * fx.Int32(2246822519)) ^ (col * fx.Int32(668265263)) ^ seed
                    h = h ^ (h >> fx.Int32(16))
                    h = h * fx.Int32(2246822519)
                    h = h ^ (h >> fx.Int32(13))
                    h = h * fx.Int32(668265263)
                    h = h ^ (h >> fx.Int32(16))
                    u = fx.Float32(h & fx.Int32(16777215)) * fx.Float32(5.960464477539063e-08)
                    u = inb.select(u, c0)
                    qf = fmath.floor(qf + u)
                else:
                    qf = fmath.roundeven(qf)
                qf = fx.max(fx.min(qf, fx.Float32(127.0)), fx.Float32(-128.0))
                qi = qf.to(fx.Int8)
                gidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(col)
                safe_g = inb.select(gidx, fx.Int64(0))
                if inb:
                    buf_copy_store(q_buf, safe_g, qi, elem=fx.Int8, unit_elems=1)
            gpu.barrier()

    convrot_quant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        In: fx.Tensor,
        OutQ: fx.Tensor,
        OutScale: fx.Tensor,
        n_rows: fx.Int32,
        seed: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        convrot_quant_kernel(In, OutQ, OutScale, n_rows, seed).launch(
            grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


DEQUANT_KERNEL_NAME = "int8_convrot_dequant_gfx120x"
FWHT_KERNEL_NAME = "convrot_fwht_gfx120x"
# comfy_kitchen eager quantization.DTYPE_CODE_TO_DTYPE, for the dtype-coded entry.
_DTYPE_CODE_TO_DTYPE = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
    5: torch.float8_e4m3fn,
    6: torch.float8_e5m2,
}


def _out_type(name: str) -> tuple[object, int]:
    table = {
        "float32": (fx.Float32, 4),
        "float16": (fx.Float16, 2),
        "bfloat16": (fx.BFloat16, 2),
        "float8_e4m3fn": (fx.Float8E4M3FN, 1),
        "float8_e5m2": (fx.Float8E5M2, 1),
    }
    if name not in table:
        raise ValueError(f"unsupported out dtype {name}")
    return table[name]


@lru_cache(maxsize=64)
def build_int8_convrot_dequant_module(
    out_dtype: str = "bfloat16",
    group_size: int = 256,
    k: int = 256,
    block_threads: int | None = None,
    logical_k: int | None = None,
) -> Callable[..., None]:
    """One row per block: int8 * row scale, then the same FWHT (H is its own inverse)."""
    if logical_k is None:
        logical_k = k
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    if k <= 0 or k % group_size != 0:
        raise ValueError(f"K={k} must be positive and divisible by group_size={group_size}")
    if logical_k <= 0 or logical_k > k:
        raise ValueError(f"logical_k={logical_k} must be in 1..K={k}")
    OutTy, out_bytes = _out_type(out_dtype)
    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")
    n_stages = int(math.log(group_size, 4))
    n_groups = k // group_size
    g_butterflies = group_size // 4
    g_bf_steps = (g_butterflies + block_threads - 1) // block_threads
    g_load_steps = (group_size + block_threads - 1) // block_threads
    inv_sqrt_g = 1.0 / math.sqrt(float(group_size))
    sig = _kernel_signature(block=block_threads, dtype=out_dtype, group_size=group_size, k=k, op="dequant")

    @fx.struct
    class SharedStorage:
        row: fx.Array[fx.Float32, group_size, 16]

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def convrot_dequant_kernel(
        Q: fx.Tensor,
        Scale: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
    ) -> None:
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        c0 = fx.Float32(0.0)
        active_row = bid < n_rows
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        row_v = lds.row.view(fx.make_layout(group_size, 1))

        def lds_load(idx: object) -> object:
            safe = (idx >= fx.Int32(0)) & (idx < fx.Int32(group_size))
            off = safe.select(idx, fx.Int32(0))
            return safe.select(fx.Float32(row_v[off]), c0)

        def lds_store(idx: object, val: object) -> None:
            safe = (idx >= fx.Int32(0)) & (idx < fx.Int32(group_size))
            off = safe.select(idx, fx.Int32(0))
            fx.memref_store(fx.Float32(val), row_v, off)

        q_buf = ptr_buf_tensor(
            Q,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(logical_k),
        )
        sc_buf = ptr_buf_tensor(
            Scale,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(4),
        )
        out_buf = ptr_buf_tensor(
            Out,
            elem=OutTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(logical_k) * fx.Int64(out_bytes),
        )
        sc = fx.Float32(buf_copy_load(sc_buf, fx.Int64(bid), elem=fx.Float32, unit_elems=1))
        sc = active_row.select(sc, c0)

        def fwht_tile() -> None:
            for stage in range_constexpr(n_stages):
                stride = 4**stage
                block = 4 * stride
                for step in range_constexpr(g_bf_steps):
                    bf = tid + fx.Int32(step * block_threads)
                    inb = active_row & (bf < fx.Int32(g_butterflies))
                    base_idx = (bf // fx.Int32(stride)) * fx.Int32(block)
                    i = bf % fx.Int32(stride)
                    idx0 = base_idx + i
                    idx1 = idx0 + fx.Int32(stride)
                    idx2 = idx0 + fx.Int32(2 * stride)
                    idx3 = idx0 + fx.Int32(3 * stride)
                    a = lds_load(idx0)
                    b = lds_load(idx1)
                    c = lds_load(idx2)
                    d = lds_load(idx3)
                    if inb:
                        lds_store(idx0, a + b + c - d)
                        lds_store(idx1, a + b - c + d)
                        lds_store(idx2, a - b + c + d)
                        lds_store(idx3, -a + b + c + d)
                gpu.barrier()

        def norm_tile() -> None:
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                if local < fx.Int32(group_size):
                    lds_store(local, lds_load(local) * fx.Float32(inv_sqrt_g))
            gpu.barrier()

        for group in range(fx.Int32(n_groups)):
            base = fx.Int32(group) * fx.Int32(group_size)
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                col = base + local
                inb = active_row & (local < fx.Int32(group_size))
                if const_expr(logical_k != k):
                    take = inb & (col < fx.Int32(logical_k))
                    gidx = fx.Int64(bid) * fx.Int64(logical_k) + fx.Int64(col)
                    safe_g = take.select(gidx, fx.Int64(0))
                    raw = buf_copy_load(q_buf, safe_g, elem=fx.Int8, unit_elems=1)
                    val = fx.Float32(fx.Int8(raw).to(fx.Float32)) * sc
                    val = take.select(val, c0)
                else:
                    gidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(col)
                    safe_g = inb.select(gidx, fx.Int64(0))
                    raw = buf_copy_load(q_buf, safe_g, elem=fx.Int8, unit_elems=1)
                    val = fx.Float32(fx.Int8(raw).to(fx.Float32)) * sc
                    val = inb.select(val, c0)
                if local < fx.Int32(group_size):
                    lds_store(local, val)
            gpu.barrier()
            fwht_tile()
            norm_tile()
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                col = base + local
                inb = active_row & (local < fx.Int32(group_size))
                v = lds_load(local)
                if const_expr(out_dtype != "float32"):
                    stored = fx.Float32(v).to(OutTy)
                else:
                    stored = v
                if const_expr(logical_k != k):
                    keep = inb & (col < fx.Int32(logical_k))
                    gidx = fx.Int64(bid) * fx.Int64(logical_k) + fx.Int64(col)
                    safe_g = keep.select(gidx, fx.Int64(0))
                    if keep:
                        buf_copy_store(out_buf, safe_g, stored, elem=OutTy, unit_elems=1)
                else:
                    gidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(col)
                    safe_g = inb.select(gidx, fx.Int64(0))
                    if inb:
                        buf_copy_store(out_buf, safe_g, stored, elem=OutTy, unit_elems=1)
            gpu.barrier()

    convrot_dequant_kernel.__name__ = f"{DEQUANT_KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        Scale: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        convrot_dequant_kernel(Q, Scale, Out, n_rows).launch(
            grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{DEQUANT_KERNEL_NAME}_{sig}"
    return launch


@lru_cache(maxsize=64)
def build_convrot_fwht_module(
    dtype: str = "float32",
    group_size: int = 256,
    k: int = 256,
    block_threads: int | None = None,
    logical_k: int | None = None,
) -> Callable[..., None]:
    """In-place-shaped ``W @ H`` for one ConvRot group size. Device FWHT, not a matmul."""
    if logical_k is None:
        logical_k = k
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    if k <= 0 or k % group_size != 0:
        raise ValueError(f"K={k} must be positive and divisible by group_size={group_size}")
    if logical_k <= 0 or logical_k > k:
        raise ValueError(f"logical_k={logical_k} must be in 1..K={k}")
    InTy, in_bytes = _out_type(dtype)
    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")
    n_stages = int(math.log(group_size, 4))
    n_groups = k // group_size
    g_butterflies = group_size // 4
    g_bf_steps = (g_butterflies + block_threads - 1) // block_threads
    g_load_steps = (group_size + block_threads - 1) // block_threads
    inv_sqrt_g = 1.0 / math.sqrt(float(group_size))
    sig = _kernel_signature(block=block_threads, dtype=dtype, group_size=group_size, k=k, op="fwht")

    @fx.struct
    class SharedStorage:
        row: fx.Array[fx.Float32, group_size, 16]

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def convrot_fwht_kernel(
        In: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
    ) -> None:
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        c0 = fx.Float32(0.0)
        active_row = bid < n_rows
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        row_v = lds.row.view(fx.make_layout(group_size, 1))

        def lds_load(idx: object) -> object:
            safe = (idx >= fx.Int32(0)) & (idx < fx.Int32(group_size))
            off = safe.select(idx, fx.Int32(0))
            return safe.select(fx.Float32(row_v[off]), c0)

        def lds_store(idx: object, val: object) -> None:
            safe = (idx >= fx.Int32(0)) & (idx < fx.Int32(group_size))
            off = safe.select(idx, fx.Int32(0))
            fx.memref_store(fx.Float32(val), row_v, off)

        in_buf = ptr_buf_tensor(
            In,
            elem=InTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(logical_k) * fx.Int64(in_bytes),
        )
        out_buf = ptr_buf_tensor(
            Out,
            elem=InTy,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(logical_k) * fx.Int64(in_bytes),
        )

        def fwht_tile() -> None:
            for stage in range_constexpr(n_stages):
                stride = 4**stage
                block = 4 * stride
                for step in range_constexpr(g_bf_steps):
                    bf = tid + fx.Int32(step * block_threads)
                    inb = active_row & (bf < fx.Int32(g_butterflies))
                    base_idx = (bf // fx.Int32(stride)) * fx.Int32(block)
                    i = bf % fx.Int32(stride)
                    idx0 = base_idx + i
                    idx1 = idx0 + fx.Int32(stride)
                    idx2 = idx0 + fx.Int32(2 * stride)
                    idx3 = idx0 + fx.Int32(3 * stride)
                    a = lds_load(idx0)
                    b = lds_load(idx1)
                    c = lds_load(idx2)
                    d = lds_load(idx3)
                    if inb:
                        lds_store(idx0, a + b + c - d)
                        lds_store(idx1, a + b - c + d)
                        lds_store(idx2, a - b + c + d)
                        lds_store(idx3, -a + b + c + d)
                gpu.barrier()

        for group in range(fx.Int32(n_groups)):
            base = fx.Int32(group) * fx.Int32(group_size)
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                col = base + local
                inb = active_row & (local < fx.Int32(group_size))
                if const_expr(logical_k != k):
                    take = inb & (col < fx.Int32(logical_k))
                    gidx = fx.Int64(bid) * fx.Int64(logical_k) + fx.Int64(col)
                    safe_g = take.select(gidx, fx.Int64(0))
                    raw = buf_copy_load(in_buf, safe_g, elem=InTy, unit_elems=1)
                    if const_expr(dtype != "float32"):
                        val = fx.Float32(InTy(raw).to(fx.Float32))
                    else:
                        val = fx.Float32(raw)
                    val = take.select(val, c0)
                else:
                    gidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(col)
                    safe_g = inb.select(gidx, fx.Int64(0))
                    raw = buf_copy_load(in_buf, safe_g, elem=InTy, unit_elems=1)
                    if const_expr(dtype != "float32"):
                        val = fx.Float32(InTy(raw).to(fx.Float32))
                    else:
                        val = fx.Float32(raw)
                    val = inb.select(val, c0)
                if local < fx.Int32(group_size):
                    lds_store(local, val)
            gpu.barrier()
            fwht_tile()
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                if local < fx.Int32(group_size):
                    lds_store(local, lds_load(local) * fx.Float32(inv_sqrt_g))
            gpu.barrier()
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                col = base + local
                inb = active_row & (local < fx.Int32(group_size))
                v = lds_load(local)
                if const_expr(dtype != "float32"):
                    stored = fx.Float32(v).to(InTy)
                else:
                    stored = v
                if const_expr(logical_k != k):
                    keep = inb & (col < fx.Int32(logical_k))
                    gidx = fx.Int64(bid) * fx.Int64(logical_k) + fx.Int64(col)
                    safe_g = keep.select(gidx, fx.Int64(0))
                    if keep:
                        buf_copy_store(out_buf, safe_g, stored, elem=InTy, unit_elems=1)
                else:
                    gidx = fx.Int64(bid) * fx.Int64(k) + fx.Int64(col)
                    safe_g = inb.select(gidx, fx.Int64(0))
                    if inb:
                        buf_copy_store(out_buf, safe_g, stored, elem=InTy, unit_elems=1)
            gpu.barrier()

    convrot_fwht_kernel.__name__ = f"{FWHT_KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        In: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        convrot_fwht_kernel(In, Out, n_rows).launch(grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream)

    launch.__name__ = f"launch_{FWHT_KERNEL_NAME}_{sig}"
    return launch


def _default_block(*_args, **_kwargs) -> Config:
    return Config(BLOCK_THREADS=128)


def _stream_kw(stream: torch.cuda.Stream | None) -> dict:
    return {} if stream is None else {"stream": stream}


@contextmanager
def _validate_scale(sig_args):
    import torch

    scale = sig_args["OutScale"]
    scale.fill_(float("nan"))
    yield
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("candidate produced non-finite output")


@contextmanager
def _validate_out(sig_args):
    import torch

    out = sig_args["Out"]
    if out.dtype.is_floating_point:
        out.fill_(float("nan"))
    else:
        out.fill_(0)
    yield
    check = out if out.dtype.is_floating_point else out.float()
    if not bool(torch.isfinite(check).all()):
        raise ValueError("candidate produced non-finite output")


@flyc.jit
def int8_convrot_quant_direct(
    In: fx.Tensor,
    OutQ: fx.Tensor,
    OutScale: fx.Tensor,
    n_rows: fx.Int32,
    seed: fx.Int32,
    in_dtype: fx.Constexpr[str],
    group_size: fx.Constexpr[int],
    K: fx.Constexpr[int],
    stochastic: fx.Constexpr[bool],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    logical_k: fx.Constexpr[int] = 0,
    stream: fx.Stream = fx.Stream(None),
):
    lk = K if logical_k == 0 else logical_k
    launch = build_int8_convrot_quant_module(
        in_dtype,
        group_size,
        K,
        block_threads=BLOCK_THREADS,
        stochastic=bool(stochastic),
        logical_k=lk,
    )
    launch(In, OutQ, OutScale, n_rows, seed, stream)


@flyc.jit
def int8_convrot_dequant_direct(
    Q: fx.Tensor,
    Scale: fx.Tensor,
    Out: fx.Tensor,
    n_rows: fx.Int32,
    out_dtype: fx.Constexpr[str],
    group_size: fx.Constexpr[int],
    K: fx.Constexpr[int],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    logical_k: fx.Constexpr[int] = 0,
    stream: fx.Stream = fx.Stream(None),
):
    lk = K if logical_k == 0 else logical_k
    launch = build_int8_convrot_dequant_module(out_dtype, group_size, K, block_threads=BLOCK_THREADS, logical_k=lk)
    launch(Q, Scale, Out, n_rows, stream)


@flyc.jit
def convrot_fwht_direct(
    In: fx.Tensor,
    Out: fx.Tensor,
    n_rows: fx.Int32,
    dtype: fx.Constexpr[str],
    group_size: fx.Constexpr[int],
    K: fx.Constexpr[int],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    logical_k: fx.Constexpr[int] = 0,
    stream: fx.Stream = fx.Stream(None),
):
    lk = K if logical_k == 0 else logical_k
    launch = build_convrot_fwht_module(dtype, group_size, K, block_threads=BLOCK_THREADS, logical_k=lk)
    launch(In, Out, n_rows, stream)


_int8_convrot_quant_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["in_dtype", "group_size", "K", "logical_k", "stochastic", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_int8_convrot_quant",
    validate_hook=_validate_scale,
)(int8_convrot_quant_direct)

_int8_convrot_dequant_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["out_dtype", "group_size", "K", "logical_k", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_int8_convrot_dequant",
    validate_hook=_validate_out,
)(int8_convrot_dequant_direct)

_convrot_fwht_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["dtype", "group_size", "K", "logical_k", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_convrot_fwht",
    validate_hook=_validate_out,
)(convrot_fwht_direct)


def convrot_fwht(
    weight: torch.Tensor,
    group_size: int = 256,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Device ``W @ H`` per ConvRot group. Replaces a host Hadamard matmul.

    Odd ``K`` is zero-padded to ``group_size`` then cropped (never rejected).
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(weight.device, what="convrot_fwht (gfx120x)")

    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {weight.dtype}")
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    orig = tuple(weight.shape)
    k = int(orig[-1])
    w2 = ensure_contiguous(weight.reshape(-1, k), stream=stream)
    k_pad = k if k % group_size == 0 else ((k + group_size - 1) // group_size) * group_size
    out = torch.empty((int(w2.shape[0]), k), device=w2.device, dtype=w2.dtype)
    _convrot_fwht_tuned(
        w2,
        out,
        int(w2.shape[0]),
        dtype=_dtype_name(w2.dtype),
        group_size=group_size,
        K=k_pad,
        logical_k=k,
        tuning_schema=TUNING_SCHEMA,
        **_stream_kw(stream),
    )
    return out.reshape(orig)


def quantize_int8_convrot_weight(
    weight: torch.Tensor,
    group_size: int = 256,
    *,
    stochastic_rounding: int | None = 0,
    stream: torch.cuda.Stream | None = None,
) -> tuple[object, object]:
    """Offline ConvRot weight rotation + rowwise INT8 quantize (HIP-compatible API).

    Args:
        weight: floating weight ``[..., K]`` (bf16/fp16/fp32) on CUDA.
        group_size: Hadamard size ∈ {16, 64, 256}. Odd ``K`` is stored at the
            next multiple of ``group_size`` so the rotated tail is kept.
        stochastic_rounding: seed. ``<= 0`` is round-even. ``> 0`` is
            ``floor(v / scale + U)`` where ``U`` is an in-kernel hash of
            ``(seed, row, col)`` in ``[0, 1)``.

    Returns:
        ``(q_int8, scale_f32)``. ``q`` is ``[..., K_pad]`` with ``K_pad`` the
        next multiple of ``group_size``. ``scale`` is ``[*weight.shape[:-1], 1]``.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(weight.device, what="quantize_int8_convrot_weight (gfx120x)")

    seed = 0 if stochastic_rounding is None else int(stochastic_rounding)
    if seed < 0:
        raise ValueError(f"stochastic_rounding seed must be >= 0, got {seed}")
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    if weight.dim() < 1:
        raise ValueError("weight must have at least 1 dim")
    if weight.device.type != "cuda":
        raise ValueError("weight must be on CUDA/ROCm")
    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {weight.dtype}")

    orig_shape = tuple(weight.shape)
    k = int(orig_shape[-1])
    w2d = ensure_contiguous(weight.reshape(-1, k), stream=stream)
    k_pad = k if k % group_size == 0 else ((k + group_size - 1) // group_size) * group_size
    m = w2d.shape[0]
    q = torch.empty((m, k_pad), device=w2d.device, dtype=torch.int8)
    scales = torch.empty((m,), device=w2d.device, dtype=torch.float32)
    _int8_convrot_quant_tuned(
        w2d,
        q,
        scales,
        m,
        seed,
        in_dtype=_dtype_name(w2d.dtype),
        group_size=group_size,
        K=k_pad,
        logical_k=k,
        stochastic=seed > 0,
        tuning_schema=TUNING_SCHEMA,
        **_stream_kw(stream),
    )
    scale_out = scales.reshape(*orig_shape[:-1], 1)
    return q.reshape(*orig_shape[:-1], k_pad), scale_out


def quantize_and_rotate_rowwise(
    x: object,
    group_size: int = 256,
    *,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online activation ConvRot rotate + rowwise INT8 (same kernel as weights).

    Returns ``(q, scale)`` — same ABI as :func:`quantize_int8_convrot_weight`.
    """
    return quantize_int8_convrot_weight(x, group_size=group_size, stream=stream)


def dequantize_int8_convrot_weight(
    q: torch.Tensor,
    scale: torch.Tensor | float | None,
    group_size: int = 256,
    out_dtype: torch.dtype | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Dequant INT8 ConvRot weights and un-rotate on the gfx120x FWHT kernel.

    ``scale`` is per row (or a scalar). Output is the original basis, same
    shape as ``q``.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(q.device, what="dequantize_int8_convrot_weight (gfx120x)")

    if out_dtype is None:
        out_dtype = torch.bfloat16
    if out_dtype not in _DTYPE_CODE_TO_DTYPE.values():
        raise ValueError(f"unsupported out dtype {out_dtype}")
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    if q.dtype != torch.int8:
        raise ValueError(f"q must be int8, got {q.dtype}")
    if q.device.type != "cuda":
        raise ValueError("q must be on CUDA/ROCm")
    orig = tuple(q.shape)
    k = int(orig[-1])
    q2 = ensure_contiguous(q.reshape(-1, k), stream=stream)
    k_pad = k if k % group_size == 0 else ((k + group_size - 1) // group_size) * group_size
    m = int(q2.shape[0])
    if isinstance(scale, (int, float)):
        sc = torch.full((m,), float(scale), device=q.device, dtype=torch.float32)
    else:
        sc = scale.reshape(-1)
        if sc.dtype != torch.float32 or sc.device != q.device:
            raise ValueError("scale must be float32 on the q device")
        if not sc.is_contiguous():
            sc = ensure_contiguous(sc, stream=stream)
        if sc.numel() == 1 and m != 1:
            sc = ensure_contiguous(sc.expand(m), stream=stream)
        if sc.numel() != m:
            raise ValueError(f"scale must be scalar or [rows]={m}, got {sc.numel()}")
    out = torch.empty((m, k), device=q.device, dtype=out_dtype)
    _int8_convrot_dequant_tuned(
        q2,
        sc,
        out,
        m,
        out_dtype=_dtype_name(out_dtype),
        group_size=group_size,
        K=k_pad,
        logical_k=k,
        tuning_schema=TUNING_SCHEMA,
        **_stream_kw(stream),
    )
    return out.reshape(orig)


def dequantize_int8_convrot_weight_dtype(
    q: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    output_dtype_code: int,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Kitchen entry: ``output_dtype_code`` matches ``DTYPE_CODE_TO_DTYPE``."""
    if output_dtype_code not in _DTYPE_CODE_TO_DTYPE:
        raise ValueError(f"unsupported output_dtype_code {output_dtype_code}")
    return dequantize_int8_convrot_weight(
        q,
        scale,
        group_size,
        out_dtype=_DTYPE_CODE_TO_DTYPE[output_dtype_code],
        stream=stream,
    )


def int8_linear_convrot(
    x: object,
    weight_q: object,
    weight_scale: object,
    *,
    group_size: int = 256,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """ConvRot INT8 linear: online act rotate+quant + iu8 WMMA GEMM.

    ``weight_q`` must already be offline-ConvRot-quantized (same ``group_size``).
    Requires the gfx120x iu8 WMMA atom (iu8 / kit 13).

    Host path is intentionally lean for short dual-launch shapes (FlyDSL
    ``do_bench`` / HIP launch-overhead guidance): one act-quant launch then one
    iu8 GEMM, with flat scale buffers and no redundant ``[..., 1]`` round-trips.
    Allocates act-q / scales / out each call (same contract as HIP).
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(x.device, what="int8_linear_convrot (gfx120x)")

    from kernels.gemm.rdna4_int8_linear import int8_linear

    if out_dtype is None:
        out_dtype = torch.bfloat16
    if x.dim() < 1:
        raise ValueError("x must be at least 1D")
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    orig_shape = tuple(x.shape)
    k = int(orig_shape[-1])
    if weight_q.dim() != 2:
        raise ValueError("weight_q must be 2D [N, K]")
    if weight_q.dtype != torch.int8:
        raise ValueError(f"weight_q must be int8, got {weight_q.dtype}")
    if weight_q.device != x.device:
        raise ValueError(f"weight_q device mismatch: {weight_q.device} vs {x.device}")
    n = int(weight_q.shape[0])
    wk = int(weight_q.shape[1])
    k_pad = k if k % int(group_size) == 0 else ((k + int(group_size) - 1) // int(group_size)) * int(group_size)
    if wk % int(group_size) != 0 or wk < k_pad:
        raise ValueError(
            f"ConvRot int8 weight K={wk} must be a multiple of group_size={group_size} "
            f"and at least the padded activation K={k_pad}"
        )
    if x.device.type != "cuda":
        raise ValueError("x must be on CUDA/ROCm")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {x.dtype}")

    # Contiguous 2D view without an extra copy when already packed.
    x2d = x.reshape(-1, k)
    if not x2d.is_contiguous():
        x2d = ensure_contiguous(x2d, stream=stream)
    # The GEMM K is the weight width, including the Hadamard tail.
    m = int(x2d.shape[0])
    aq = torch.empty((m, wk), device=x2d.device, dtype=torch.int8)
    a_scale = torch.empty((m,), device=x2d.device, dtype=torch.float32)
    _int8_convrot_quant_tuned(
        x2d,
        aq,
        a_scale,
        m,
        0,
        in_dtype=_dtype_name(x2d.dtype),
        group_size=group_size,
        K=wk,
        logical_k=k,
        stochastic=False,
        tuning_schema=TUNING_SCHEMA,
        **_stream_kw(stream),
    )

    if weight_q.is_contiguous():
        wq = weight_q
    else:
        wq = ensure_contiguous(weight_q, stream=stream)
    out = int8_linear(
        aq,
        wq,
        a_scale,
        weight_scale,
        out_dtype=out_dtype,
        stream=stream,
    )
    if bias is not None:
        from kernels.common.gfx120x_row_bias import add_row_bias

        out = add_row_bias(out, bias, stream=stream)
    return out.reshape(*orig_shape[:-1], n)
