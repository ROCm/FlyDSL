# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Constant, replicate, reflect, and circular pad for gfx120x.

Same layout contract as ``torch.nn.functional.pad``: ``pad`` is
``(left, right)`` pairs starting at the last dimension. The value
written into constant padding is zero. Rank is at most 6.
"""

from contextlib import contextmanager
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from flydsl.expr import const_expr
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor

_RANK = 6
_BLOCKS = (32, 64, 128, 256, 512, 1024)
_TUNING_SCHEMA = 1
_MODES = ("constant", "replicate", "reflect", "circular")
_ELEM = {
    "float32": (fx.Float32, 4),
    "float16": (fx.Float16, 2),
    "bfloat16": (fx.BFloat16, 2),
    "uint8": (fx.Uint8, 1),
}


def _elem_name(dtype: torch.dtype) -> str:
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8, torch.uint8):
        return "uint8"
    return {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }[dtype]


@lru_cache(maxsize=64)
def build_pad_module(elem_name: str, mode: str, block_threads: int = 256):
    ty, nbytes = _ELEM[elem_name]
    block = int(block_threads)
    if block not in _BLOCKS:
        raise ValueError(f"pad block_threads={block} is not a wave32 block in {_BLOCKS}")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def pad_kernel(
        Src: fx.Tensor,
        Dst: fx.Tensor,
        n_in: fx.Int64,
        n_out: fx.Int64,
        s0: fx.Int64,
        s1: fx.Int64,
        s2: fx.Int64,
        s3: fx.Int64,
        s4: fx.Int64,
        s5: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        i0: fx.Int64,
        i1: fx.Int64,
        i2: fx.Int64,
        i3: fx.Int64,
        i4: fx.Int64,
        i5: fx.Int64,
    ) -> None:
        idx = fx.Int64(fx.block_idx.x) * fx.Int64(block) + fx.Int64(fx.thread_idx.x)
        inb = idx < n_out
        safe = inb.select(idx, fx.Int64(0))
        src = ptr_buf_tensor(Src, elem=ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=n_in * fx.Int64(nbytes))
        dst = ptr_buf_tensor(Dst, elem=ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=n_out * fx.Int64(nbytes))

        def _map(coord, pad, n):
            ic = coord - pad
            if const_expr(mode == "constant"):
                lo = (ic < fx.Int64(0)).select(fx.Int32(1), fx.Int32(0))
                hi = (ic >= n).select(fx.Int32(1), fx.Int32(0))
                oob = lo | hi
                ic = (oob != fx.Int32(0)).select(fx.Int64(0), ic)
                return ic, oob
            if const_expr(mode == "replicate"):
                ic = (ic < fx.Int64(0)).select(fx.Int64(0), ic)
                last = n - fx.Int64(1)
                last = (last < fx.Int64(0)).select(fx.Int64(0), last)
                ic = (ic > last).select(last, ic)
                return ic, fx.Int32(0)
            if const_expr(mode == "reflect"):
                period = (n - fx.Int64(1)) * fx.Int64(2)
                small = (period == fx.Int64(0)).select(fx.Int32(1), fx.Int32(0))
                period_safe = (small != fx.Int32(0)).select(fx.Int64(1), period)
                shifted = (ic < fx.Int64(0)).select(ic + period_safe, ic)
                shifted = shifted - (shifted // period_safe) * period_safe
                folded = (shifted >= n).select(period_safe - shifted, shifted)
                ic = (small != fx.Int32(0)).select(fx.Int64(0), folded)
                return ic, fx.Int32(0)
            n_safe = (n == fx.Int64(0)).select(fx.Int64(1), n)
            shifted = (ic < fx.Int64(0)).select(ic + n_safe, ic)
            shifted = (shifted < fx.Int64(0)).select(shifted + n_safe, shifted)
            shifted = shifted - (shifted // n_safe) * n_safe
            return shifted, fx.Int32(0)

        rem = safe
        c5 = rem - (rem // s5) * s5
        rem = rem // s5
        c4 = rem - (rem // s4) * s4
        rem = rem // s4
        c3 = rem - (rem // s3) * s3
        rem = rem // s3
        c2 = rem - (rem // s2) * s2
        rem = rem // s2
        c1 = rem - (rem // s1) * s1
        rem = rem // s1
        c0 = rem
        ic5, o5 = _map(c5, p5, i5)
        ic4, o4 = _map(c4, p4, i4)
        ic3, o3 = _map(c3, p3, i3)
        ic2, o2 = _map(c2, p2, i2)
        ic1, o1 = _map(c1, p1, i1)
        ic0, o0 = _map(c0, p0, i0)
        oob = (((o0 | o1) | (o2 | o3)) | (o4 | o5)) != fx.Int32(0)
        stride = i5
        in_index = ic5
        in_index = in_index + ic4 * stride
        stride = stride * i4
        in_index = in_index + ic3 * stride
        stride = stride * i3
        in_index = in_index + ic2 * stride
        stride = stride * i2
        in_index = in_index + ic1 * stride
        stride = stride * i1
        in_index = in_index + ic0 * stride
        loaded = buf_copy_load(src, oob.select(fx.Int64(0), in_index), elem=ty, unit_elems=1)
        if const_expr(elem_name == "uint8"):
            stored = oob.select(fx.Uint8(0), loaded)
        elif const_expr(elem_name == "float32"):
            stored = oob.select(fx.Float32(0), fx.Float32(loaded))
        else:
            stored = oob.select(fx.Float32(0).to(ty), loaded)
        if inb:
            buf_copy_store(dst, safe, stored, elem=ty, unit_elems=1)

    @flyc.jit
    def launch(
        Src: fx.Tensor,
        Dst: fx.Tensor,
        n_in: fx.Int64,
        n_out: fx.Int64,
        s0: fx.Int64,
        s1: fx.Int64,
        s2: fx.Int64,
        s3: fx.Int64,
        s4: fx.Int64,
        s5: fx.Int64,
        p0: fx.Int64,
        p1: fx.Int64,
        p2: fx.Int64,
        p3: fx.Int64,
        p4: fx.Int64,
        p5: fx.Int64,
        i0: fx.Int64,
        i1: fx.Int64,
        i2: fx.Int64,
        i3: fx.Int64,
        i4: fx.Int64,
        i5: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        grid = (n_out + fx.Int64(block - 1)) // fx.Int64(block)
        pad_kernel(
            Src, Dst, n_in, n_out, s0, s1, s2, s3, s4, s5, p0, p1, p2, p3, p4, p5, i0, i1, i2, i3, i4, i5
        ).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


@flyc.jit
def _pad_direct(
    Src: fx.Tensor,
    Dst: fx.Tensor,
    n_in: fx.Int64,
    n_out: fx.Int64,
    s0: fx.Int64,
    s1: fx.Int64,
    s2: fx.Int64,
    s3: fx.Int64,
    s4: fx.Int64,
    s5: fx.Int64,
    p0: fx.Int64,
    p1: fx.Int64,
    p2: fx.Int64,
    p3: fx.Int64,
    p4: fx.Int64,
    p5: fx.Int64,
    i0: fx.Int64,
    i1: fx.Int64,
    i2: fx.Int64,
    i3: fx.Int64,
    i4: fx.Int64,
    i5: fx.Int64,
    elem_name: fx.Constexpr[str],
    mode: fx.Constexpr[str],
    BLOCK: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    del tuning_schema
    launch = build_pad_module(str(elem_name), str(mode), int(BLOCK))
    launch(Src, Dst, n_in, n_out, s0, s1, s2, s3, s4, s5, p0, p1, p2, p3, p4, p5, i0, i1, i2, i3, i4, i5, stream)


@contextmanager
def _validate_pad(sig_args):
    dst = sig_args["Dst"]
    if dst.dtype.is_floating_point:
        dst.fill_(float("nan"))
    yield
    if dst.dtype.is_floating_point and not bool(torch.isfinite(dst).all()):
        raise ValueError("pad candidate left a non-finite output")


def _default_pad(*_args, **_kwargs):
    return Config(BLOCK=256)


_pad = autotune(
    configs=[Config(BLOCK=block) for block in _BLOCKS],
    key=["elem_name", "mode", "tuning_schema"],
    default=_default_pad,
    artifact_name="pad_gfx120x",
    validate_hook=_validate_pad,
)(_pad_direct)


def ensure_contiguous(
    x: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Return ``x`` contiguous; if a copy is needed, run it on ``stream``.

    Soft-contig for pointer ABI must not race a following kernel launched on a
    non-default ``torch.cuda.Stream``.
    """
    if x.is_contiguous():
        return x
    if stream is None:
        return x.contiguous()
    with torch.cuda.stream(stream):
        return x.contiguous()


def device_pad(
    x: torch.Tensor,
    pad,
    mode: str = "constant",
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Pad ``x`` on device. ``mode`` matches ``F.pad`` (``zeros`` is constant).

    Pass ``stream`` so soft-pad runs on the same queue as a preceding/following kernel.
    """
    require_gfx120x(x.device, what="device_pad (gfx120x)")
    if mode == "zeros":
        mode = "constant"
    if mode not in _MODES:
        raise ValueError(f"device_pad mode must be one of {_MODES}, got {mode!r}")
    try:
        name = _elem_name(x.dtype)
    except KeyError as error:
        raise ValueError(f"device_pad dtype {x.dtype} is not f32/f16/bf16/fp8(e4m3fn|e5m2)/int8/uint8") from error
    pad = tuple(int(v) for v in pad)
    if len(pad) % 2:
        raise ValueError(f"pad must be an even number of (left, right) pairs, got {pad}")
    if any(v < 0 for v in pad):
        raise ValueError(f"device_pad does not crop, got {pad}")
    rank = x.dim()
    npairs = len(pad) // 2
    if npairs > rank:
        raise ValueError(f"pad rank {npairs} exceeds tensor rank {rank}")
    if rank > _RANK:
        raise ValueError(f"device_pad supports rank <= {_RANK}, got {rank}")
    lo = [0] * rank
    hi = [0] * rank
    for pair in range(npairs):
        lo[rank - 1 - pair] = pad[2 * pair]
        hi[rank - 1 - pair] = pad[2 * pair + 1]
    if mode == "reflect":
        for axis, (left, right, size) in enumerate(zip(lo, hi, x.shape)):
            size = int(size)
            # Empty axis + zero pad is a no-op; nonzero reflect pad needs size > pad.
            if left or right:
                if size == 0 or max(left, right) >= size:
                    raise ValueError(f"reflect pad on axis {axis} must be < size {size}")
    if mode == "circular":
        for axis, (left, right, size) in enumerate(zip(lo, hi, x.shape)):
            size = int(size)
            if left or right:
                if size == 0 or max(left, right) > size:
                    raise ValueError(f"circular pad on axis {axis} must be <= size {size}")
    if mode == "replicate":
        for axis, size in enumerate(x.shape):
            if int(size) == 0 and (lo[axis] or hi[axis]):
                raise ValueError(f"replicate pad needs a non-empty axis {axis}")
    lead = _RANK - rank
    in_sizes = (1,) * lead + tuple(int(v) for v in x.shape)
    pad_lo = (0,) * lead + tuple(lo)
    out_sizes = tuple(n + left + right for n, left, right in zip(in_sizes, pad_lo, (0,) * lead + tuple(hi)))
    out_shape = tuple(int(v) + lo[i] + hi[i] for i, v in enumerate(x.shape))
    n_out = 1
    for v in out_shape:
        n_out *= int(v)
    src = ensure_contiguous(x, stream=stream)
    if n_out == 0:
        return torch.empty(out_shape, device=x.device, dtype=x.dtype)
    dst = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    kw = dict(
        elem_name=name,
        mode=mode,
        tuning_schema=_TUNING_SCHEMA,
    )
    if stream is not None:
        kw["stream"] = stream
    _pad(
        src,
        dst,
        int(src.numel()),
        int(n_out),
        *[int(v) for v in out_sizes],
        *[int(v) for v in pad_lo],
        *[int(v) for v in in_sizes],
        **kw,
    )
    return dst


def ceil_to_multiple(n: int, multiple: int) -> int:
    """Smallest multiple of ``multiple`` that is >= ``n`` (``n`` must be > 0)."""
    if multiple <= 0:
        raise ValueError(f"ceil_to_multiple multiple must be > 0, got {multiple}")
    n = int(n)
    if n <= 0:
        raise ValueError(f"ceil_to_multiple n must be > 0, got {n}")
    rem = n % multiple
    return n if rem == 0 else n + (multiple - rem)
