# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Row bias add for gfx120x. ``out[..., N] + bias[N]``, accumulated in f32."""

from contextlib import contextmanager
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.autotune import Config, autotune
from flydsl.expr import const_expr
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor

_DTYPES = {
    torch.float32: ("float32", fx.Float32, 4),
    torch.float16: ("float16", fx.Float16, 2),
    torch.bfloat16: ("bfloat16", fx.BFloat16, 2),
}
_TYPES = {"float32": (fx.Float32, 4), "float16": (fx.Float16, 2), "bfloat16": (fx.BFloat16, 2)}
_BLOCKS = (32, 64, 128, 256, 512, 1024)
_TUNING_SCHEMA = 1


def _as_f32(raw, name: str):
    if const_expr(name == "float32"):
        return fx.Float32(raw)
    return raw.to(fx.Float32)


@lru_cache(maxsize=64)
def build_row_bias_module(in_name: str, bias_name: str, out_name: str, block_threads: int = 256):
    in_ty, in_bytes = _TYPES[in_name]
    bias_ty, bias_bytes = _TYPES[bias_name]
    out_ty, out_bytes = _TYPES[out_name]
    block = int(block_threads)
    if block not in _BLOCKS:
        raise ValueError(f"row-bias block_threads={block} is not a wave32 block in {_BLOCKS}")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def bias_kernel(In: fx.Tensor, Bias: fx.Tensor, Out: fx.Tensor, n_rows: fx.Int32, n_cols: fx.Int32) -> None:
        idx = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        n_elem = n_rows * n_cols
        inb = idx < n_elem
        safe = inb.select(idx, fx.Int32(0))
        col = safe - (safe // n_cols) * n_cols
        src = ptr_buf_tensor(
            In, elem=in_ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_elem) * fx.Int64(in_bytes)
        )
        bb = ptr_buf_tensor(
            Bias,
            elem=bias_ty,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_cols) * fx.Int64(bias_bytes),
        )
        dst = ptr_buf_tensor(
            Out, elem=out_ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(n_elem) * fx.Int64(out_bytes)
        )
        val = _as_f32(buf_copy_load(src, fx.Int64(safe), elem=in_ty, unit_elems=1), in_name)
        b = _as_f32(buf_copy_load(bb, fx.Int64(col), elem=bias_ty, unit_elems=1), bias_name)
        y = val + b
        if const_expr(out_name == "float32"):
            stored = y
        else:
            stored = y.to(out_ty)
        if inb:
            buf_copy_store(dst, fx.Int64(safe), stored, elem=out_ty, unit_elems=1)

    @flyc.jit
    def launch(
        In: fx.Tensor,
        Bias: fx.Tensor,
        Out: fx.Tensor,
        n_rows: fx.Int32,
        n_cols: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ) -> None:
        n_elem = fx.Int64(n_rows) * fx.Int64(n_cols)
        grid = (n_elem + fx.Int64(block - 1)) // fx.Int64(block)
        bias_kernel(In, Bias, Out, n_rows, n_cols).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


@flyc.jit
def _row_bias_direct(
    In: fx.Tensor,
    Bias: fx.Tensor,
    Out: fx.Tensor,
    n_rows: fx.Int32,
    n_cols: fx.Int32,
    in_name: fx.Constexpr[str],
    bias_name: fx.Constexpr[str],
    out_name: fx.Constexpr[str],
    BLOCK: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    del tuning_schema
    launch = build_row_bias_module(str(in_name), str(bias_name), str(out_name), int(BLOCK))
    launch(In, Bias, Out, n_rows, n_cols, stream)


@contextmanager
def _validate_row_bias(sig_args):
    out = sig_args["Out"]
    inp = sig_args.get("In")
    # In-place same-dtype path: Out aliases In — do not nan-fill the live input.
    if inp is None or out.data_ptr() != inp.data_ptr():
        out.fill_(float("nan"))
    yield
    if not bool(torch.isfinite(out).all()):
        raise ValueError("row-bias candidate left a non-finite output")


def _default_block(*_args, **_kwargs):
    return Config(BLOCK=256)


_row_bias = autotune(
    configs=[Config(BLOCK=block) for block in _BLOCKS],
    key=["in_name", "bias_name", "out_name", "tuning_schema"],
    default=_default_block,
    artifact_name="row_bias_gfx120x",
    validate_hook=_validate_row_bias,
    # Same-dtype API aliases In/Out; nan-fill would poison In without restore.
    restore_value=["In"],
)(_row_bias_direct)


def add_row_bias(
    out: torch.Tensor,
    bias: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """``out[..., N] + bias[N]``. The add is a gfx120x kernel, accumulated in f32.

    ``bias`` stays in its wire dtype (f32, f16, or bf16). The kernel widens it.
    Pass ``stream`` so the add runs on the same stream as a preceding GEMM.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(out.device, what="add_row_bias (gfx120x)")
    if out.dtype not in _DTYPES:
        raise ValueError(f"add_row_bias input dtype {out.dtype} is not f32/f16/bf16")
    if bias.dtype not in _DTYPES:
        raise ValueError(f"add_row_bias bias dtype {bias.dtype} is not f32/f16/bf16")
    if bias.device != out.device:
        raise ValueError(f"bias device {bias.device} does not match {out.device}")
    dst_dtype = out.dtype if out_dtype is None else out_dtype
    if dst_dtype not in _DTYPES:
        raise ValueError(f"add_row_bias output dtype {dst_dtype} is not f32/f16/bf16")
    n_cols = int(out.shape[-1])
    b = bias.reshape(-1)
    if int(b.numel()) != n_cols:
        raise ValueError(f"bias must have {n_cols} elements, got {int(b.numel())}")
    if not b.is_contiguous():
        b = ensure_contiguous(b, stream=stream)
    flat = out.reshape(-1, n_cols)
    if not flat.is_contiguous():
        flat = ensure_contiguous(flat, stream=stream)
    n_rows = int(flat.shape[0])
    dst = flat if dst_dtype == flat.dtype else torch.empty((n_rows, n_cols), device=out.device, dtype=dst_dtype)
    if n_rows:
        kw = dict(
            in_name=_DTYPES[flat.dtype][0],
            bias_name=_DTYPES[b.dtype][0],
            out_name=_DTYPES[dst_dtype][0],
            tuning_schema=_TUNING_SCHEMA,
        )
        if stream is not None:
            kw["stream"] = stream
        _row_bias(flat, b, dst, n_rows, n_cols, **kw)
    return dst.reshape(out.shape)


@lru_cache(maxsize=32)
def build_add_same_module(name: str, block_threads: int = 256):
    ty, nbytes = _TYPES[name]
    block = int(block_threads)
    if block not in _BLOCKS:
        raise ValueError(f"add_same block_threads={block} is not a wave32 block in {_BLOCKS}")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def add_kernel(A: fx.Tensor, B: fx.Tensor, Out: fx.Tensor, n_elem: fx.Int32) -> None:
        idx = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        inb = idx < n_elem
        safe = inb.select(idx, fx.Int32(0))
        bytes_n = fx.Int64(n_elem) * fx.Int64(nbytes)
        ab = ptr_buf_tensor(A, elem=ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=bytes_n)
        bb = ptr_buf_tensor(B, elem=ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=bytes_n)
        ob = ptr_buf_tensor(Out, elem=ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=bytes_n)
        av = _as_f32(buf_copy_load(ab, fx.Int64(safe), elem=ty, unit_elems=1), name)
        bv = _as_f32(buf_copy_load(bb, fx.Int64(safe), elem=ty, unit_elems=1), name)
        y = av + bv
        if const_expr(name == "float32"):
            stored = y
        else:
            stored = y.to(ty)
        if inb:
            buf_copy_store(ob, fx.Int64(safe), stored, elem=ty, unit_elems=1)

    @flyc.jit
    def launch(
        A: fx.Tensor, B: fx.Tensor, Out: fx.Tensor, n_elem: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ) -> None:
        grid = (fx.Int64(n_elem) + fx.Int64(block - 1)) // fx.Int64(block)
        add_kernel(A, B, Out, n_elem).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


@flyc.jit
def _add_same_direct(
    A: fx.Tensor,
    B: fx.Tensor,
    Out: fx.Tensor,
    n_elem: fx.Int32,
    name: fx.Constexpr[str],
    BLOCK: fx.Constexpr[int],
    tuning_schema: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    del tuning_schema
    launch = build_add_same_module(str(name), int(BLOCK))
    launch(A, B, Out, n_elem, stream)


_add_same = autotune(
    configs=[Config(BLOCK=block) for block in _BLOCKS],
    key=["name", "tuning_schema"],
    default=_default_block,
    artifact_name="add_same_gfx120x",
    validate_hook=_validate_row_bias,
)(_add_same_direct)


def add_same(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Elementwise ``a + b`` on gfx120x. Both tensors must match shape and dtype.

    Pass ``stream`` so the add runs on the same queue as a preceding GEMM/LoRA.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(a.device, what="add_same (gfx120x)")
    if a.shape != b.shape or a.dtype != b.dtype or a.dtype not in _DTYPES:
        raise ValueError(
            f"add_same expects matching f32/f16/bf16 tensors, got {a.shape}/{a.dtype} and {b.shape}/{b.dtype}"
        )
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    if not flat_a.is_contiguous():
        flat_a = ensure_contiguous(flat_a, stream=stream)
    if not flat_b.is_contiguous():
        flat_b = ensure_contiguous(flat_b, stream=stream)
    out = torch.empty_like(flat_a)
    n_elem = int(flat_a.numel())
    if n_elem:
        kw = dict(name=_DTYPES[a.dtype][0], tuning_schema=_TUNING_SCHEMA)
        if stream is not None:
            kw["stream"] = stream
        _add_same(flat_a, flat_b, out, n_elem, **kw)
    return out.reshape(a.shape)


@lru_cache(maxsize=32)
def build_mul_by_scale1_module(name: str, block_threads: int = 256):
    """``out[i] = in[i] * scale[0]`` for bf16/fp16/fp32 (scale is 1xf32)."""
    ty, nbytes = _TYPES[name]
    block = int(block_threads)
    if block not in _BLOCKS:
        raise ValueError(f"mul_by_scale1 block_threads={block} is not a wave32 block in {_BLOCKS}")

    @flyc.kernel(known_block_size=[block, 1, 1])
    def mul_kernel(In: fx.Tensor, Scale: fx.Tensor, Out: fx.Tensor, n_elem: fx.Int32) -> None:
        idx = fx.block_idx.x * fx.Int32(block) + fx.thread_idx.x
        inb = idx < n_elem
        safe = inb.select(idx, fx.Int32(0))
        bytes_n = fx.Int64(n_elem) * fx.Int64(nbytes)
        ib = ptr_buf_tensor(In, elem=ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=bytes_n)
        sb = ptr_buf_tensor(Scale, elem=fx.Float32, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=fx.Int64(4))
        ob = ptr_buf_tensor(Out, elem=ty, n=0x3FFFFFFF, unit_elems=1, num_records_bytes=bytes_n)
        av = _as_f32(buf_copy_load(ib, fx.Int64(safe), elem=ty, unit_elems=1), name)
        scale = fx.Float32(buf_copy_load(sb, fx.Int64(0), elem=fx.Float32, unit_elems=1))
        y = av * scale
        if const_expr(name == "float32"):
            stored = y
        else:
            stored = y.to(ty)
        if inb:
            buf_copy_store(ob, fx.Int64(safe), stored, elem=ty, unit_elems=1)

    @flyc.jit
    def launch(
        In: fx.Tensor, Scale: fx.Tensor, Out: fx.Tensor, n_elem: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ) -> None:
        grid = (fx.Int64(n_elem) + fx.Int64(block - 1)) // fx.Int64(block)
        mul_kernel(In, Scale, Out, n_elem).launch(grid=(grid, 1, 1), block=(block, 1, 1), stream=stream)

    return launch


def mul_by_scale1(
    x: torch.Tensor,
    scale: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Elementwise ``x * scale[0]`` on gfx120x. ``scale`` is a 1-element fp32 CUDA tensor.

    Pass ``stream`` so the mul runs on the same queue as a preceding/following kernel.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(x.device, what="mul_by_scale1 (gfx120x)")
    if x.dtype not in _DTYPES:
        raise ValueError(f"mul_by_scale1 expects f32/f16/bf16, got {x.dtype}")
    if scale.dtype != torch.float32 or scale.device != x.device or scale.numel() != 1:
        raise ValueError(
            f"mul_by_scale1 scale must be 1xf32 on {x.device}, got {tuple(scale.shape)}/{scale.dtype}/{scale.device}"
        )
    flat = x.reshape(-1)
    if not flat.is_contiguous():
        flat = ensure_contiguous(flat, stream=stream)
    scale_f = ensure_contiguous(scale, stream=stream)
    out = torch.empty_like(flat)
    n_elem = int(flat.numel())
    if n_elem:
        launch = build_mul_by_scale1_module(_DTYPES[x.dtype][0], 256)
        if stream is None:
            launch(flat, scale_f, out, n_elem)
        else:
            launch(flat, scale_f, out, n_elem, stream)
    return out.reshape(x.shape)
