# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X ConvRot W4A4 weight/act quantize (Hadamard + signed INT4 pack).

Hadamard of size ``convrot_groupsize`` in {16, 64, 256} (radix-4 FWHT), then
per-row scale ``amax/7``, round-to-nearest, clamp to signed ``[-7, 7]``, and
row-major nibble pack ``qdata[N, K//2]`` (low nibble = even column).

**Linear default is native int4:** ``convrot_w4a4_linear`` with
``linear_dtype="int4"`` zero-fills a short K up to the Hadamard / quant-group /
native iu4 K contract (multiples of ``max(convrot_groupsize, 64)``, which is
also a multiple of 32), then runs device act ConvRot-i4 + ``rdna4_iu4_gemm``.
Real ABI failures (empty MNK, bad wscales, unsupported group, …) raise.
Pass ``linear_dtype="int8"`` only for the explicit unpack→iu8 path.

``quant_group_size`` must be 64. AWQ and SVDQuant stay on their own modules.

Call examples: ``docs/prebuilt_kernels_guide.md`` (gfx120x ConvRot / iu4).
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

# Reference int4 MMA group (layout contract); quantize itself is rowwise.
_INT4_GROUP_SIZE = 64
_INT4_MAX = 7
_INV_7 = 1.0 / float(_INT4_MAX)
_SCALE_FLOOR = 1e-10


def _kernel_signature(**params: object) -> str:
    """Specialization suffix for kernel/launch names (matches kernels.common.gfx120x_buf_helpers)."""
    return "_".join(
        f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()
    ).replace("-", "_")


KERNEL_NAME = "convrot_w4a4_quant_gfx120x"
WARP = 32
_SUPPORTED_GROUPS = (16, 64, 256)
TUNING_SCHEMA = 1
_BLOCK_CHOICES = (32, 64, 128, 256, 512, 1024)


def _native_iu4_k_multiple(convrot_groupsize: int) -> int:
    """Soft-pad K multiple for native iu4 + ConvRot Hadamard.

    Covers Hadamard ``convrot_groupsize``, INT4 quant groups (64), and the
    native iu4 WMMA K=32 issue. Supported groups are {16, 64, 256}; bumping
    16 → 64 yields a multiple of 32 in every case.
    """
    if convrot_groupsize not in _SUPPORTED_GROUPS:
        raise ValueError(f"convrot_groupsize must be one of {_SUPPORTED_GROUPS}, got {convrot_groupsize}")
    return convrot_groupsize if convrot_groupsize >= _INT4_GROUP_SIZE else _INT4_GROUP_SIZE


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
    }[dtype]


@lru_cache(maxsize=64)
def build_convrot_w4a4_quant_module(
    in_dtype: str = "bfloat16",
    group_size: int = 256,
    k: int = 256,
    block_threads: int | None = None,
    logical_k: int | None = None,
) -> Callable[..., None]:
    """Compile one-row-per-block ConvRot rotate + rowwise signed INT4 pack.

    ``k`` is the Hadamard width (a multiple of ``group_size``). ``logical_k``
    is the caller's K. A shorter row zero-fills lanes past that K inside the
    group. Equal values keep the previous load.
    """
    if logical_k is None:
        logical_k = k
    if group_size not in _SUPPORTED_GROUPS:
        raise ValueError(f"group_size must be one of {_SUPPORTED_GROUPS}, got {group_size}")
    if k <= 0 or k % group_size != 0:
        raise ValueError(f"K={k} must be positive and divisible by group_size={group_size}")
    if logical_k <= 0 or logical_k > k:
        raise ValueError(f"logical_k={logical_k} must be in 1..K={k}")
    if k % 2 != 0:
        raise ValueError(f"K={k} must be even for INT4 packing")

    InTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[in_dtype]
    in_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[in_dtype]
    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")

    n_stages = int(math.log(group_size, 4))
    n_groups = k // group_size
    g_butterflies = group_size // 4
    g_bf_steps = (g_butterflies + block_threads - 1) // block_threads
    g_load_steps = (group_size + block_threads - 1) // block_threads
    g_pack = group_size // 2
    g_pack_steps = (g_pack + block_threads - 1) // block_threads
    packed_k = k // 2
    reduction_slots = (block_threads + WARP - 1) // WARP
    inv_sqrt_g = 1.0 / math.sqrt(float(group_size))
    sig = _kernel_signature(
        block=block_threads,
        dtype=in_dtype,
        group_size=group_size,
        k=k,
        op="convrot_w4a4",
    )

    @fx.struct
    class SharedStorage:
        row: fx.Array[fx.Float32, group_size, 16]
        red: fx.Array[fx.Float32, reduction_slots, 16]

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def convrot_w4a4_quant_kernel(
        In: fx.Tensor,
        OutQ: fx.Tensor,
        OutScale: fx.Tensor,
        n_rows: fx.Int32,
    ) -> None:
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        c0 = fx.Float32(0.0)
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

        # One group in LDS. Pass 1 is the row amax. Pass 2 repeats the FWHT
        # and packs signed nibbles. Groups are even, so pairs stay inside a group.
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
        scale = fx.max(amax * fx.Float32(_INV_7), fx.Float32(_SCALE_FLOOR))
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
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(packed_k),
        )
        for group in range(fx.Int32(n_groups)):
            load_group(group)
            gpu.barrier()
            fwht_tile()
            norm_tile()
            for step in range_constexpr(g_load_steps):
                local = tid + fx.Int32(step * block_threads)
                inb = active_row & (local < fx.Int32(group_size))
                v = lds_load(local)
                qf = fmath.roundeven(v * inv)
                qf = fx.max(fx.min(qf, fx.Float32(float(_INT4_MAX))), fx.Float32(float(-_INT4_MAX)))
                if local < fx.Int32(group_size):
                    lds_store(local, inb.select(qf, c0))
            gpu.barrier()
            for step in range_constexpr(g_pack_steps):
                pc = tid + fx.Int32(step * block_threads)
                inb = active_row & (pc < fx.Int32(g_pack))
                c0i = pc * fx.Int32(2)
                c1i = c0i + fx.Int32(1)
                lo_i = lds_load(c0i).to(fx.Int32) & fx.Int32(0xF)
                hi_i = lds_load(c1i).to(fx.Int32) & fx.Int32(0xF)
                packed = (lo_i | (hi_i << fx.Int32(4))).to(fx.Int8)
                gidx = fx.Int64(bid) * fx.Int64(packed_k) + fx.Int64(group) * fx.Int64(g_pack) + fx.Int64(pc)
                safe_g = inb.select(gidx, fx.Int64(0))
                if inb:
                    buf_copy_store(q_buf, safe_g, packed, elem=fx.Int8, unit_elems=1)
            gpu.barrier()

    convrot_w4a4_quant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        In: fx.Tensor,
        OutQ: fx.Tensor,
        OutScale: fx.Tensor,
        n_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        convrot_w4a4_quant_kernel(In, OutQ, OutScale, n_rows).launch(
            grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


def _default_block(*_args, **_kwargs) -> Config:
    return Config(BLOCK_THREADS=128)


def _stream_kw(stream: torch.cuda.Stream | None) -> dict:
    return {} if stream is None else {"stream": stream}


@contextmanager
def _validate_f32_scale(sig_args):
    import torch

    scale = sig_args["OutScale"]
    scale.fill_(float("nan"))
    yield
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("candidate produced non-finite output")


@contextmanager
def _validate_i8_out(sig_args):
    import torch

    out = sig_args["Out"]
    out.fill_(0)
    yield
    if not bool(torch.isfinite(out.float()).all()):
        raise ValueError("candidate produced non-finite output")


@contextmanager
def _validate_f32_out(sig_args):
    import torch

    out = sig_args["Out"]
    out.fill_(float("nan"))
    yield
    if not bool(torch.isfinite(out).all()):
        raise ValueError("candidate produced non-finite output")


@flyc.jit
def convrot_w4a4_quant_direct(
    In: fx.Tensor,
    OutQ: fx.Tensor,
    OutScale: fx.Tensor,
    n_rows: fx.Int32,
    in_dtype: fx.Constexpr[str],
    group_size: fx.Constexpr[int],
    K: fx.Constexpr[int],
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    logical_k: fx.Constexpr[int] = 0,
    stream: fx.Stream = fx.Stream(None),
):
    """Direct JIT entry. ``tuning_schema`` partitions the autotune cache."""
    lk = K if logical_k == 0 else logical_k
    launch = build_convrot_w4a4_quant_module(in_dtype, group_size, K, block_threads=BLOCK_THREADS, logical_k=lk)
    launch(In, OutQ, OutScale, n_rows, stream)


_convrot_w4a4_quant_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["in_dtype", "group_size", "K", "logical_k", "tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_convrot_w4a4_quant",
    validate_hook=_validate_f32_scale,
)(convrot_w4a4_quant_direct)


def _build_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    import torch

    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    return h / (size**0.5)


def _pack_int4_row_major(values: object) -> torch.Tensor:
    """Wire format: low nibble = even column, high = odd."""
    import torch

    if values.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _unpack_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Signed-nibble unpack (full ``[-8, 7]`` storage range).

    Offline/oracle helper. CUDA uses :func:`expand_signed_i4`; product linear
    paths already call that device helper directly.
    """
    import torch

    if packed.is_cuda:
        return expand_signed_i4(packed).reshape(*packed.shape[:-1], packed.shape[-1] * 2)

    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    *,
    stochastic_rounding: int | None = 0,
    stream: torch.cuda.Stream | None = None,
) -> tuple[object, object]:
    """Offline ConvRot + signed INT4 pack (HIP-compatible API).

    Returns:
        ``(qdata[N, K//2] int8, scales[N] float32)``.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(weight.device, what="quantize_convrot_w4a4_weight (gfx120x)")

    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"int4 MMA contract requires quant_group_size {_INT4_GROUP_SIZE}, got {quant_group_size}")
    if stochastic_rounding:
        raise ValueError("quantize_convrot_w4a4_weight FlyDSL path does not support stochastic_rounding")
    if convrot_groupsize not in _SUPPORTED_GROUPS:
        raise ValueError(f"convrot_groupsize must be one of {_SUPPORTED_GROUPS}, got {convrot_groupsize}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D [N, K], got shape {tuple(weight.shape)}")
    if weight.device.type != "cuda":
        raise ValueError("weight must be on CUDA/ROCm")
    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported dtype {weight.dtype}")

    n, k = int(weight.shape[0]), int(weight.shape[1])
    # Hadamard width stays a group multiple. Lanes past the caller's K are
    # zero inside the kernel, so the weight is not cloned.
    multiple = _native_iu4_k_multiple(convrot_groupsize)
    w2d = ensure_contiguous(weight, stream=stream)
    k_logical = k
    if k % multiple != 0:
        k = ((k + multiple - 1) // multiple) * multiple
    q = torch.empty((n, k // 2), device=w2d.device, dtype=torch.int8)
    scales = torch.empty((n,), device=w2d.device, dtype=torch.float32)
    _convrot_w4a4_quant_tuned(
        w2d,
        q,
        scales,
        n,
        in_dtype=_dtype_name(w2d.dtype),
        group_size=convrot_groupsize,
        K=k,
        logical_k=k_logical,
        tuning_schema=TUNING_SCHEMA,
        **_stream_kw(stream),
    )
    return q, scales


@lru_cache(maxsize=8)
def build_expand_signed_i4_module(block_threads: int = 256) -> Callable[..., None]:
    """Packed signed nibbles → int8 codes, one byte per thread."""
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")
    block = block_threads

    @flyc.kernel(known_block_size=[block, 1, 1])
    def expand_kernel(Packed: fx.Tensor, Out: fx.Tensor, n_packed: fx.Int32) -> None:
        idx = fx.Int32(fx.block_idx.x) * fx.Int32(block) + fx.Int32(fx.thread_idx.x)
        inb = idx < n_packed
        src = ptr_buf_tensor(Packed, elem=fx.Int8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_packed))
        dst = ptr_buf_tensor(
            Out, elem=fx.Int8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_packed) * fx.Int64(2)
        )
        raw = buf_copy_load(src, inb.select(idx, fx.Int32(0)), elem=fx.Int8, unit_elems=1)
        byte = fx.Int32(fx.Int8(raw))
        lo = byte & fx.Int32(15)
        hi = (byte >> fx.Int32(4)) & fx.Int32(15)
        lo = (lo >= fx.Int32(8)).select(lo - fx.Int32(16), lo)
        hi = (hi >= fx.Int32(8)).select(hi - fx.Int32(16), hi)
        if inb:
            buf_copy_store(dst, idx * fx.Int32(2), lo.to(fx.Int8), elem=fx.Int8, unit_elems=1)
            buf_copy_store(dst, idx * fx.Int32(2) + fx.Int32(1), hi.to(fx.Int8), elem=fx.Int8, unit_elems=1)

    expand_kernel.__name__ = "expand_signed_i4"

    @flyc.jit
    def launch(Packed: fx.Tensor, Out: fx.Tensor, n_packed: fx.Int32, stream: fx.Stream = fx.Stream(None)) -> None:
        grid = (fx.Int64(n_packed) + fx.Int64(block - 1)) // fx.Int64(block)
        expand_kernel(Packed, Out, n_packed).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


@flyc.jit
def expand_signed_i4_direct(
    Packed: fx.Tensor,
    Out: fx.Tensor,
    n_packed: fx.Int32,
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    launch = build_expand_signed_i4_module(BLOCK_THREADS)
    launch(Packed, Out, n_packed, stream)


_expand_signed_i4_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_expand_signed_i4",
    validate_hook=_validate_i8_out,
)(expand_signed_i4_direct)


def expand_signed_i4(packed: torch.Tensor, *, stream: torch.cuda.Stream | None = None) -> torch.Tensor:
    """Device signed-nibble unpack. ``packed[..., K//2]`` → ``[..., K]`` int8."""
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(packed.device, what="expand_signed_i4 (gfx120x)")

    if packed.dtype != torch.int8:
        raise ValueError(f"packed must be int8, got {packed.dtype}")
    orig = tuple(packed.shape)
    k_half = int(orig[-1])
    flat = ensure_contiguous(packed.reshape(-1), stream=stream)
    out = torch.empty((flat.numel() * 2,), device=packed.device, dtype=torch.int8)
    _expand_signed_i4_tuned(flat, out, int(flat.numel()), tuning_schema=TUNING_SCHEMA, **_stream_kw(stream))
    return out.reshape(*orig[:-1], k_half * 2)


@lru_cache(maxsize=8)
def build_i4_row_scale_module(block_threads: int = 256) -> Callable[..., None]:
    """Signed nibble × per-row scale → f32, one packed byte per thread."""
    if block_threads not in _BLOCK_CHOICES:
        raise ValueError(f"block_threads={block_threads} is not a legal wave32 block")
    block = block_threads

    @flyc.kernel(known_block_size=[block, 1, 1])
    def scale_kernel(Packed: fx.Tensor, Scale: fx.Tensor, Out: fx.Tensor, n_rows: fx.Int32, k_half: fx.Int32) -> None:
        idx = fx.Int32(fx.block_idx.x) * fx.Int32(block) + fx.Int32(fx.thread_idx.x)
        n_packed = n_rows * k_half
        inb = idx < n_packed
        src = ptr_buf_tensor(Packed, elem=fx.Int8, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_packed))
        scb = ptr_buf_tensor(
            Scale, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_rows) * fx.Int64(4)
        )
        dst = ptr_buf_tensor(
            Out, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_packed) * fx.Int64(8)
        )
        safe = inb.select(idx, fx.Int32(0))
        raw = buf_copy_load(src, safe, elem=fx.Int8, unit_elems=1)
        row = safe // k_half
        sc = fx.Float32(buf_copy_load(scb, fx.Int64(row), elem=fx.Float32, unit_elems=1))
        byte = fx.Int32(fx.Int8(raw))
        lo = byte & fx.Int32(15)
        hi = (byte >> fx.Int32(4)) & fx.Int32(15)
        lo = (lo >= fx.Int32(8)).select(lo - fx.Int32(16), lo)
        hi = (hi >= fx.Int32(8)).select(hi - fx.Int32(16), hi)
        if inb:
            base = idx * fx.Int32(2)
            buf_copy_store(dst, base, fx.Float32(lo) * sc, elem=fx.Float32, unit_elems=1)
            buf_copy_store(dst, base + fx.Int32(1), fx.Float32(hi) * sc, elem=fx.Float32, unit_elems=1)

    @flyc.jit
    def launch(
        Packed: fx.Tensor,
        Scale: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
        k_half: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        n_packed = fx.Int64(n_rows) * fx.Int64(k_half)
        grid = (n_packed + fx.Int64(block - 1)) // fx.Int64(block)
        scale_kernel(Packed, Scale, Out, n_rows, k_half).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


@flyc.jit
def i4_row_scale_direct(
    Packed: fx.Tensor,
    Scale: fx.Tensor,
    Out: fx.Tensor,
    n_rows: fx.Int32,
    k_half: fx.Int32,
    BLOCK_THREADS: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    launch = build_i4_row_scale_module(BLOCK_THREADS)
    launch(Packed, Scale, Out, n_rows, k_half, stream)


_i4_row_scale_tuned = autotune(
    configs=[Config(BLOCK_THREADS=block) for block in _BLOCK_CHOICES],
    key=["tuning_schema"],
    default=_default_block,
    artifact_name="rdna4_i4_row_scale",
    validate_hook=_validate_f32_out,
)(i4_row_scale_direct)


def dequantize_convrot_w4a4_weight(
    qdata: object,
    scales: object,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    output_dtype: object = None,
    *,
    logical_k: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Dequant packed W4 and un-rotate. Nibble expand and FWHT are both device kernels.

    When the packed width is the Hadamard multiple and the caller K was shorter,
    pass ``logical_k`` to crop the reconstructed matrix. Otherwise the full
    packed K is returned.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous
    from kernels.quant.rdna4_int8_convrot import convrot_fwht

    require_gfx120x(qdata.device, what="dequantize_convrot_w4a4_weight (gfx120x)")
    if output_dtype is None:
        output_dtype = torch.float32
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"quant_group_size must be {_INT4_GROUP_SIZE}")
    if qdata.dtype != torch.int8:
        raise ValueError(f"qdata must be int8, got {qdata.dtype}")
    q2 = ensure_contiguous(qdata.reshape(-1, qdata.shape[-1]), stream=stream)
    n, k_half = int(q2.shape[0]), int(q2.shape[1])
    sc = scales.reshape(-1)
    if sc.dtype != torch.float32 or sc.device != qdata.device:
        raise ValueError("scales must be float32 on the qdata device")
    if not sc.is_contiguous():
        sc = ensure_contiguous(sc, stream=stream)
    if sc.numel() == 1 and n != 1:
        sc = ensure_contiguous(sc.expand(n), stream=stream)
    elif int(sc.numel()) != n:
        raise ValueError(f"scales length {int(sc.numel())} must be 1 or N={n}")
    scaled = torch.empty((n, k_half * 2), device=qdata.device, dtype=torch.float32)
    _i4_row_scale_tuned(q2, sc, scaled, n, k_half, tuning_schema=TUNING_SCHEMA, **_stream_kw(stream))
    rotated = convrot_fwht(scaled, convrot_groupsize, stream=stream)
    if logical_k is not None:
        lk = int(logical_k)
        if lk < 0 or lk > int(rotated.shape[-1]):
            raise ValueError(f"logical_k={lk} out of range for dequant K={rotated.shape[-1]}")
        rotated = ensure_contiguous(rotated[..., :lk], stream=stream)
    if output_dtype == torch.float32:
        return rotated
    return rotated.to(dtype=output_dtype)


def _convrot_w4a4_native_iu4(
    x: object,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    *,
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Native iu4 path: device act ConvRot-i4 + ``rdna4_iu4_gemm``.

    A short K is zero-filled up to the Hadamard width. The activation is not
    cloned. Empty M/N/K and other ABI failures raise (no unpack→iu8 demotion).
    """
    import torch

    from kernels.common.gfx120x_pad import ensure_contiguous
    from kernels.gemm.rdna4_iu4_gemm import iu4_gemm

    if out_dtype is None:
        out_dtype = x.dtype
    orig = tuple(x.shape)
    k = int(orig[-1])
    n = int(qweight.shape[0])
    qw_k = int(qweight.shape[1]) * 2
    from kernels.common.gfx120x_pad import ceil_to_multiple

    # Act quant zero-fills up to the Hadamard width. The packed weight must
    # already be that width from quantize_convrot_w4a4_weight.
    multiple = _native_iu4_k_multiple(convrot_groupsize)
    k_target = ceil_to_multiple(k, multiple)
    if qw_k != k and qw_k != k_target:
        raise ValueError(f"Input K={k} does not match qweight K={qw_k}")
    x2d = x.reshape(-1, k)
    if not x2d.is_contiguous():
        x2d = ensure_contiguous(x2d, stream=stream)
    m = int(x2d.shape[0])
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"native iu4 needs positive MNK, got ({m},{n},{k})")
    qw = ensure_contiguous(qweight, stream=stream)

    # Device act path: same ConvRot rotate+i4 pack kernel as weights (reference
    # HIP uses convrot_quant_int4). Avoid host Hadamard tax that slowed the idle microbench.
    a_packed, a_scale = quantize_convrot_w4a4_weight(x2d, convrot_groupsize=convrot_groupsize, stream=stream)
    if int(a_packed.shape[1]) != int(qw.shape[1]):
        raise ValueError(
            f"qweight packed K {int(qw.shape[1])} != activation packed K {int(a_packed.shape[1])}. "
            "Quantize the weight with quantize_convrot_w4a4_weight."
        )
    w_scale = wscales.reshape(-1)
    if w_scale.dtype != torch.float32 or w_scale.device != x.device:
        raise ValueError("wscales must be float32 on the activation device")
    if not w_scale.is_contiguous():
        w_scale = ensure_contiguous(w_scale, stream=stream)
    out2d = iu4_gemm(
        a_packed,
        qw,
        ensure_contiguous(a_scale, stream=stream),
        w_scale,
        out_dtype=out_dtype,
        prefer_native=True,
        stream=stream,
    )
    if bias is not None:
        from kernels.common.gfx120x_row_bias import add_row_bias

        out2d = add_row_bias(out2d, bias, stream=stream)
    return out2d.reshape(*orig[:-1], n)


def convrot_w4a4_linear(
    x: object,
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
    linear_dtype: str = "int4",
    *,
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """ConvRot W4A4 linear: bf16/fp16 ``x`` × packed W4 weights → ``[M, N]``.

    Typical call (default = native int4)::

        qweight, wscales = quantize_convrot_w4a4_weight(w, convrot_groupsize=64)
        y = convrot_w4a4_linear(x, qweight, wscales, convrot_groupsize=64)

    Args:
        x: ``[..., K]`` activations (bf16/fp16). On ``linear_dtype='int4'``
            the kernel zero-fills K up to the Hadamard width and quantizes
            acts to packed int4. The activation tensor is not cloned.
        qweight: ``[N, K//2]`` int8 from ``quantize_convrot_w4a4_weight``.
        wscales: ``[N]`` float32 weight scales from the same quantize call.
        convrot_groupsize: Hadamard group in ``{16, 64, 256}``. A short K is
            zero-filled in the kernel. The packed width is that multiple.
        linear_dtype: ``'int4'`` (default) = native iu4 (act ConvRot-i4 +
            ``iu4_gemm``). ``'int8'`` = explicit unpack weight
            nibbles → int8-act ConvRot linear. ABI failures raise (no silent
            int4→iu8 demotion).
        out_dtype: output dtype (default = ``x.dtype``).

    See ``docs/prebuilt_kernels_guide.md`` for call examples.
    Requires gfx120x iu4 WMMA (and iu8 when ``linear_dtype='int8'``).
    """
    require_gfx120x(x.device, what="convrot_w4a4_linear (gfx120x)")

    from kernels.quant.rdna4_int8_convrot import int8_linear_convrot

    if linear_dtype not in {"int4", "int8"}:
        raise ValueError(f"ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got {linear_dtype!r}")
    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"quant_group_size must be {_INT4_GROUP_SIZE}")
    if out_dtype is None:
        out_dtype = x.dtype
    x_k = int(x.shape[-1])
    qw_k = int(qweight.shape[-1]) * 2
    if x_k != qw_k:
        from kernels.common.gfx120x_pad import ceil_to_multiple

        multiple = _native_iu4_k_multiple(convrot_groupsize)
        if qw_k != ceil_to_multiple(x_k, multiple):
            raise ValueError(f"Input K={x_k} does not match qweight K={qw_k}")

    if linear_dtype == "int4":
        # Always stay on native iu4. A short K is zero-filled. Do not demote to iu8.
        return _convrot_w4a4_native_iu4(
            x,
            qweight,
            wscales,
            bias=bias,
            convrot_groupsize=convrot_groupsize,
            out_dtype=out_dtype,
            stream=stream,
        )

    # Explicit unpack→iu8 path (linear_dtype="int8" only).
    w_int8 = expand_signed_i4(qweight, stream=stream)
    return int8_linear_convrot(
        x,
        w_int8,
        wscales,
        group_size=convrot_groupsize,
        bias=bias,
        out_dtype=out_dtype,
        stream=stream,
    )
