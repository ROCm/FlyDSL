#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""AOT compile + ``export_to_c`` example for gfx120x (RDNA4) ship kernels.

Compiles FlyDSL gfx120x launchers with pointer placeholders (no GPU required
for most compile/export paths) and writes self-contained ``.o`` / ``.h`` artifacts.

Usage:
    ARCH=gfx1201 python tests/python/examples/aot_gfx120x_example.py
    ARCH=gfx1201 python tests/python/examples/aot_gfx120x_example.py --out ./aot_gfx120x_out
    ARCH=gfx1201 python tests/python/examples/aot_gfx120x_example.py --all

``--all`` exports the gfx120x product kernels. Pointer launches compile
without a device. Tensor launches (AdaLN, RMS+RoPE, MXFP quant, ALiBi, the
bool mask, AWQ dequant, int8 ConvRot dequant, fused MLP, rowwise) need a
live CUDA device and are skipped with a printed reason when it is absent.

Environment:
    ARCH / FLYDSL_GPU_ARCH   Target arch (e.g. gfx1201, gfx1200). ``ARCH`` wins
    when both are set. The missing compile var is filled, and
    ``FLYDSL_GPU_ARCH`` is pointed at that gfx120x base for the pack, then
    restored. Default: gfx1201.

Tutorial sibling (live GPU tiled WMMA, not AOT):
    ``examples/07-tiledMma_gfx120x.py`` — gfx120x counterpart to ``examples/03-tiledMma.py``.

"""

import argparse
import os
import sys
import tempfile
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.compiler.jit_argument import PointerJitArg  # noqa: E402


def _ensure_gfx120x_arch() -> str:
    """Point this process at a gfx120x compile target. Caller restores env.

    Compiler target is ``ARCH`` (``env.compile.arch``, read lazily). WMMA
    dispatch uses ``get_rocm_arch()``, which prefers ``FLYDSL_GPU_ARCH``.
    ``ARCH`` wins when both are set, so a machine-wide ``FLYDSL_GPU_ARCH=gfx950``
    does not retarget this example. The unset compile var is filled in.
    ``FLYDSL_GPU_ARCH`` is set to the gfx120x base for the compile window
    (a ``:suffix`` already in ``ARCH`` is left untouched).
    """
    arch_env = os.environ.get("ARCH")
    gpu_env = os.environ.get("FLYDSL_GPU_ARCH")
    raw = arch_env or gpu_env or "gfx1201"
    arch_base = raw.lower().split(":")[0]
    if not arch_base.startswith("gfx120"):
        raise SystemExit(f"aot_gfx120x_example requires gfx120x family ARCH, got {raw!r}")
    if not arch_env:
        os.environ["ARCH"] = arch_base
    os.environ["FLYDSL_GPU_ARCH"] = arch_base
    return arch_base


def _ptr(value: int = 0) -> PointerJitArg:
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    return flyc.from_c_void_p(fx.Uint8, value)


def compile_and_export_rope(out_dir: Path) -> Path:
    """Compile gfx120x RoPE and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.norm.rope_gfx120x import build_rope_module

    rows, hd, block = 2, 64, 256
    pairs = hd // 2
    n_pairs_total = rows * pairs
    launch = build_rope_module("bfloat16", block)
    args = (
        _ptr(),
        _ptr(),
        _ptr(),
        n_pairs_total,
        pairs,
        1,
        rows,
        1,
        rows,
        1,
        rows * pairs,
        fx.Stream(None),
    )
    compiled = flyc.compile(launch, *args)
    if compiled is None:
        raise RuntimeError("flyc.compile returned None (is COMPILE_ONLY set?)")
    name = "rope_gfx120x_bf16"
    compiled.export_to_c(out_dir, name, f"aoti_{name}")
    obj = out_dir / f"{name}.o"
    hdr = out_dir / f"{name}.h"
    if not obj.is_file() or not hdr.is_file():
        raise RuntimeError(f"export_to_c did not write {obj} / {hdr}")
    return obj


def compile_and_export_w8a16(out_dir: Path) -> Path:
    """Compile gfx120x W8A16 linear GEMM and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.gemm.rdna4_w8a16_linear import _CFG_64_64_64, build_w8a16_linear_module

    cfg = _CFG_64_64_64
    launch = build_w8a16_linear_module(
        act_name="bfloat16",
        out_name="bfloat16",
        cfg=cfg,
        skip_bounds=True,
        w_scale_per_n=True,
        w_dtype_name="int8",
    )
    M, N, K = 64, 64, 64
    compiled = flyc.compile(
        launch,
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        M,
        N,
        K,
        fx.Stream(None),
    )
    if compiled is None:
        raise RuntimeError("flyc.compile returned None (is COMPILE_ONLY set?)")
    name = "w8a16_gfx120x_bf16"
    compiled.export_to_c(out_dir, name, f"aoti_{name}")
    obj = out_dir / f"{name}.o"
    if not obj.is_file():
        raise RuntimeError(f"export_to_c did not write {obj}")
    return obj


def compile_and_export_flash_attn(out_dir: Path) -> Path:
    """Compile gfx120x bf16 FlashAttention and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.attention.flash_attn_gfx120x import build_flash_attn_func_module

    host = build_flash_attn_func_module(
        num_heads=4,
        head_dim=64,
        causal=True,
        dtype_str="bf16",
        block_m=64,
        block_n=32,
    )
    jit = getattr(host, "jit_function", None)
    if jit is None:
        raise RuntimeError("flash_attn_gfx120x launch missing jit_function (AOT hook)")
    batch, seq = 1, 128
    compiled = flyc.compile(
        jit,
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        batch,
        seq,
        seq,
        seq,
        _ptr(),  # Bias
        _ptr(),  # Slopes
        _ptr(),  # LSE
        _ptr(),  # Sink
        _ptr(),  # CuSeqlensQ
        _ptr(),  # CuSeqlensKV
        _ptr(),  # BlockTable
        _ptr(),  # SeqlenK
        0,  # block_table_stride
        1,  # num_splits
        _ptr(),  # WsM
        _ptr(),  # WsL
        _ptr(),  # WsO
        0,  # ws_rows
        fx.Stream(None),
    )
    if compiled is None:
        raise RuntimeError("flyc.compile returned None (is COMPILE_ONLY set?)")
    name = "flash_attn_gfx120x_bf16"
    compiled.export_to_c(out_dir, name, f"aoti_{name}")
    obj = out_dir / f"{name}.o"
    if not obj.is_file():
        raise RuntimeError(f"export_to_c did not write {obj}")
    return obj


def compile_and_export_flash_attn_fp8(out_dir: Path) -> Path:
    """Compile gfx120x FP8 (e4m3fn) FlashAttention and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.attention.flash_attn_fp8_gfx120x import build_flash_attn_func_fp8_module

    host = build_flash_attn_func_fp8_module(
        num_heads=4, head_dim=64, causal=True, dtype_str="fp8", block_m=64, block_n=32
    )
    jit = getattr(host, "jit_function", None)
    if jit is None:
        raise RuntimeError("flash_attn_fp8_gfx120x launch missing jit_function")
    batch, seq = 1, 64
    compiled = flyc.compile(
        jit,
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        batch,
        seq,
        seq,
        seq,
        _ptr(),  # q_descale
        _ptr(),  # k_descale
        _ptr(),  # v_descale
        _ptr(),  # Bias
        _ptr(),  # Slopes
        _ptr(),  # LSE
        _ptr(),  # Sink
        _ptr(),  # CuSeqlensQ
        _ptr(),  # CuSeqlensKV
        _ptr(),  # BlockTable
        _ptr(),  # SeqlenK
        0,  # block_table_stride
        1,  # num_splits
        _ptr(),  # WsM
        _ptr(),  # WsL
        _ptr(),  # WsO
        0,  # ws_rows
        fx.Stream(None),
    )
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    name = "flash_attn_gfx120x_fp8"
    compiled.export_to_c(out_dir, name, f"aoti_{name}")
    obj = out_dir / f"{name}.o"
    if not obj.is_file():
        raise RuntimeError(f"export_to_c did not write {obj}")
    return obj


def compile_and_export_flash_attn_int8(out_dir: Path) -> Path:
    """Compile gfx120x int8 FlashAttention and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.attention.flash_attn_int8_gfx120x import build_flash_attn_func_int8_module

    host = build_flash_attn_func_int8_module(
        num_heads=4, head_dim=64, causal=True, dtype_str="int8", block_m=64, block_n=32
    )
    jit = getattr(host, "jit_function", None)
    if jit is None:
        raise RuntimeError("flash_attn_int8_gfx120x launch missing jit_function")
    batch, seq = 1, 64
    compiled = flyc.compile(
        jit,
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        batch,
        seq,
        seq,
        seq,
        _ptr(),  # q_descale
        _ptr(),  # k_descale
        _ptr(),  # v_descale
        _ptr(),  # Bias
        _ptr(),  # Slopes
        _ptr(),  # LSE
        _ptr(),  # Sink
        _ptr(),  # CuSeqlensQ
        _ptr(),  # CuSeqlensKV
        _ptr(),  # BlockTable
        _ptr(),  # SeqlenK
        0,  # block_table_stride
        1,  # num_splits
        _ptr(),  # WsM
        _ptr(),  # WsL
        _ptr(),  # WsO
        0,  # ws_rows
        fx.Stream(None),
    )
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    name = "flash_attn_gfx120x_int8"
    compiled.export_to_c(out_dir, name, f"aoti_{name}")
    obj = out_dir / f"{name}.o"
    if not obj.is_file():
        raise RuntimeError(f"export_to_c did not write {obj}")
    return obj


def compile_and_export_iu4_gemm(out_dir: Path) -> Path:
    """Compile bare native iu4 GEMM and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.gemm.rdna4_iu4_gemm import _CFG_64_64_64, build_iu4_gemm_module

    launch = build_iu4_gemm_module(out_name="bfloat16", cfg=_CFG_64_64_64, skip_bounds=True)
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), _ptr(), _ptr(), 64, 64, 64, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    name = "iu4_gemm_gfx120x"
    compiled.export_to_c(out_dir, name, f"aoti_{name}")
    obj = out_dir / f"{name}.o"
    if not obj.is_file():
        raise RuntimeError(f"export_to_c did not write {obj}")
    return obj


def compile_and_export_awq_gemv(out_dir: Path) -> Path:
    """Compile AWQ W4A16 fused GEMV and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_awq_w4a16 import build_awq_gemv_module

    launch = build_awq_gemv_module(k=256, group_size=64, act_dtype="bfloat16", max_m=1, n_tile=1)
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), _ptr(), _ptr(), _ptr(), 1, 32, 0, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    name = "awq_gemv_gfx120x"
    compiled.export_to_c(out_dir, name, f"aoti_{name}")
    obj = out_dir / f"{name}.o"
    if not obj.is_file():
        raise RuntimeError(f"export_to_c did not write {obj}")
    return obj


def compile_and_export_svdquant(out_dir: Path) -> Path:
    """Compile SVDQuant fused scaled_mm and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_svdquant_w4a4 import build_svdquant_scaled_mm_fused_module

    launch = build_svdquant_scaled_mm_fused_module(
        k=256, group_size=64, out_dtype="bfloat16", act_unsigned=False, n_tile=1
    )
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), _ptr(), _ptr(), 1, 64, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    name = "svdquant_fused_gfx120x"
    compiled.export_to_c(out_dir, name, f"aoti_{name}")
    obj = out_dir / f"{name}.o"
    if not obj.is_file():
        raise RuntimeError(f"export_to_c did not write {obj}")
    return obj


def compile_and_export_convrot_quant(out_dir: Path) -> Path:
    """Compile ConvRot W4A4 act-quant launcher and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_convrot_w4a4 import build_convrot_w4a4_quant_module

    launch = build_convrot_w4a4_quant_module(in_dtype="bfloat16", group_size=256, k=256)
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), 16, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    name = "convrot_quant_gfx120x"
    compiled.export_to_c(out_dir, name, f"aoti_{name}")
    obj = out_dir / f"{name}.o"
    if not obj.is_file():
        raise RuntimeError(f"export_to_c did not write {obj}")
    return obj


def _export(compiled: object, out_dir: Path, name: str) -> Path:
    compiled.export_to_c(out_dir, name, f"aoti_{name}")
    obj = out_dir / f"{name}.o"
    if not obj.is_file():
        raise RuntimeError(f"export_to_c did not write {obj}")
    return obj


def _cuda_tensor(*shape, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    """Small CUDA placeholder for fx.Tensor launch compile; None if unavailable."""
    try:
        import torch  # noqa: E402
    except Exception:
        return None
    if dtype is None:
        dtype = torch.bfloat16
    if not torch.cuda.is_available():
        return None
    return torch.empty(*shape, device="cuda", dtype=dtype)


def compile_and_export_adaln(out_dir: Path) -> Path:
    """Compile AdaLN and export C object (needs CUDA placeholder tensors)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.norm.adaln_gfx120x import build_adaln_module

    x = _cuda_tensor(2, 64)
    if x is None:
        raise RuntimeError("SKIP adaln: needs live CUDA for fx.Tensor launch compile")
    # Scale and shift are [group, N]. The kernel indexes row bid // group.
    scale = _cuda_tensor(2, 64)
    shift = _cuda_tensor(2, 64)
    out = _cuda_tensor(2, 64)
    launch = build_adaln_module(64, "bfloat16", True, block_threads=32)
    stream = fx.Stream(None)
    compiled = flyc.compile(launch, x, scale, shift, out, 2, 1, 1, 1e-6, stream)
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "adaln_gfx120x_bf16")


def compile_and_export_rms_rope(out_dir: Path) -> Path:
    """Compile RMS-RoPE and export C object (needs CUDA placeholder tensors)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.norm.rms_rope_gfx120x import build_rms_rope_module

    hd = 64
    x = _cuda_tensor(2, hd)
    if x is None:
        raise RuntimeError("SKIP rms_rope: needs live CUDA for fx.Tensor launch compile")
    scale = _cuda_tensor(hd)
    # Freqs are [rows, pairs, 2, 2], the layout the kernel indexes.
    freqs = _cuda_tensor(2, hd // 2, 2, 2, dtype=__import__("torch").float32)
    out = _cuda_tensor(2, hd)
    launch = build_rms_rope_module(hd, "bfloat16", block_threads=64)
    stream = fx.Stream(None)
    compiled = flyc.compile(launch, x, scale, freqs, out, 2, 1, 1e-6, stream)
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "rms_rope_gfx120x_bf16")


def compile_and_export_swiglu(out_dir: Path) -> Path:
    """Compile SwiGLU chunk and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.common.gfx120x_swiglu import build_swiglu_chunk_module

    launch = build_swiglu_chunk_module("bfloat16")
    compiled = flyc.compile(launch, _ptr(), _ptr(), 64, 64, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "swiglu_chunk_gfx120x_bf16")


def compile_and_export_stoch_fp8(out_dir: Path) -> Path:
    """Compile stochastic FP8 (bitcast path) and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_stoch_fp8 import build_stoch_fp8_module

    launch = build_stoch_fp8_module("bfloat16", e5m2=False, path="bitcast")
    compiled = flyc.compile(launch, _ptr(), _ptr(), 64, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "stoch_fp8_gfx120x_bf16")


def compile_and_export_fp8_quant(out_dir: Path) -> Path:
    """Compile FP8 e4m3fn quant and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_fp8_quant import build_fp8_quant_module

    launch = build_fp8_quant_module("bfloat16", e5m2=False)
    # lp_max is constexpr float — pass Python float
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), 64, 448.0, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "fp8_quant_gfx120x_e4m3fn")


def compile_and_export_int8_linear(out_dir: Path) -> Path:
    """Compile int8 iu8 linear GEMM and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.gemm.rdna4_int8_linear import _CFG_64_64_64, build_int8_linear_module

    launch = build_int8_linear_module(out_name="bfloat16", cfg=_CFG_64_64_64, skip_bounds=True)
    M = N = K = 64
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), _ptr(), _ptr(), M, N, K, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "int8_linear_gfx120x_bf16")


def compile_and_export_scaled_mm_fp8(out_dir: Path) -> Path:
    """Compile scaled_mm_fp8 (e4m3fn) and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.gemm.rdna4_scaled_mm_fp8 import _CFG_64_64_64, build_scaled_mm_fp8_module

    launch = build_scaled_mm_fp8_module(out_name="bfloat16", cfg=_CFG_64_64_64, skip_bounds=True)
    M = N = K = 64
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), _ptr(), _ptr(), M, N, K, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "scaled_mm_fp8_gfx120x_bf16")


def compile_and_export_fused_mlp(out_dir: Path) -> Path:
    """Compile fused SwiGLU MLP tile and export C object (needs CUDA tensors)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.gemm.rdna4_fused_mlp_nmajor import build_fused_swiglu_mlp_module

    a0 = _cuda_tensor(16, 16)
    if a0 is None:
        raise RuntimeError("SKIP fused_mlp: needs live CUDA for fx.Tensor launch compile")
    import torch  # noqa: E402

    bg = torch.empty(16, 16, device="cuda", dtype=torch.bfloat16)
    bu = torch.empty(16, 16, device="cuda", dtype=torch.bfloat16)
    bd = torch.empty(16, 16, device="cuda", dtype=torch.bfloat16)
    c1 = torch.empty(16, 16, device="cuda", dtype=torch.bfloat16)
    launch = build_fused_swiglu_mlp_module("bfloat16")
    compiled = flyc.compile(launch, a0, bg, bu, bd, c1, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "fused_swiglu_mlp_gfx120x_bf16")


def compile_and_export_asym_w4a8(out_dir: Path) -> Path:
    """Compile asym W4A8 dequant-to-int8 and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_asym_w4a8 import build_w4a8_dequant_int4_to_int8_module

    launch = build_w4a8_dequant_int4_to_int8_module(k=128, group_size=16, use_codebook=True)
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), _ptr(), 16, 128, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "asym_w4a8_dequant_gfx120x")


def compile_and_export_int8_fused(out_dir: Path) -> Path:
    """Compile fused int8 act-quant+mm and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.gemm.rdna4_int8_linear import _CFG_64_64_64
    from kernels.gemm.rdna4_int8_linear_fused import build_int8_linear_fused_module

    launch = build_int8_linear_fused_module(
        out_name="bfloat16", in_name="bfloat16", cfg=_CFG_64_64_64, skip_bounds=True, lora_rank=0
    )
    M = N = K = 64
    # Af Bnk C ScaleA ScaleB LoraDown LoraUp LoraScale M N K stream
    compiled = flyc.compile(
        launch,
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        M,
        N,
        K,
        fx.Stream(None),
    )
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "int8_linear_fused_gfx120x_bf16")


def compile_and_export_fp8_fused(out_dir: Path) -> Path:
    """Compile fused FP8 act-quant+scaled_mm and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.gemm.rdna4_scaled_mm_fp8 import _CFG_64_64_64
    from kernels.gemm.rdna4_scaled_mm_fp8_fused import build_scaled_mm_fp8_fused_module

    launch = build_scaled_mm_fp8_fused_module(
        out_name="bfloat16", in_name="bfloat16", cfg=_CFG_64_64_64, skip_bounds=True, lora_rank=0
    )
    M = N = K = 64
    compiled = flyc.compile(
        launch,
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        M,
        N,
        K,
        fx.Stream(None),
    )
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "scaled_mm_fp8_fused_gfx120x_bf16")


def compile_and_export_rowwise(out_dir: Path) -> Path:
    """Compile int8 rowwise quant and export C object (needs CUDA tensors)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_quantize_int8_rowwise import build_quantize_int8_rowwise_module

    k = 64
    x = _cuda_tensor(2, k)
    if x is None:
        raise RuntimeError("SKIP rowwise: needs live CUDA for fx.Tensor launch compile")
    import torch  # noqa: E402

    q = torch.empty(2, k, device="cuda", dtype=torch.int8)
    scale = torch.empty(2, 1, device="cuda", dtype=torch.float32)
    launch = build_quantize_int8_rowwise_module(k, "bfloat16", block_threads=32)
    compiled = flyc.compile(launch, x, q, scale, 2, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "quantize_int8_rowwise_gfx120x")


def compile_and_export_mxfp8_gemm(out_dir: Path) -> Path:
    """Compile MXFP8 block GEMM and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.gemm.rdna4_mxfp8_block_gemm import build_mxfp8_block_gemm_module

    launch = build_mxfp8_block_gemm_module("bfloat16", skip_bounds=True, k_tail=0)
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), _ptr(), _ptr(), 64, 64, 64, 2, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "mxfp8_block_gemm_gfx120x_bf16")


def compile_and_export_mxfp4_gemm(out_dir: Path) -> Path:
    """Compile MXFP4 block GEMM and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.gemm.rdna4_mxfp4_block_gemm import build_mxfp4_block_gemm_module

    launch = build_mxfp4_block_gemm_module("bfloat16", skip_bounds=True)
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), _ptr(), _ptr(), 64, 64, 64, 2, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "mxfp4_block_gemm_gfx120x_bf16")


def compile_and_export_silu(out_dir: Path) -> Path:
    """Compile SiLU multiply and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.common.gfx120x_swiglu import build_silu_mul_module

    launch = build_silu_mul_module("bfloat16")
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), 64, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "silu_mul_gfx120x_bf16")


def compile_and_export_fp8_dequant(out_dir: Path) -> Path:
    """Compile per-tensor FP8 dequant and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_fp8_quant import build_fp8_dequant_module

    launch = build_fp8_dequant_module("bfloat16", e5m2=False)
    compiled = flyc.compile(launch, _ptr(), _ptr(), _ptr(), 64, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "fp8_dequant_gfx120x_bf16")


def compile_and_export_splitk(out_dir: Path) -> Path:
    """Compile the bf16 split-K combine and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.attention.flash_attn_gfx120x_splitk import build_splitk_combine_module

    launch = build_splitk_combine_module(4, 64, "bf16").jit_function
    compiled = flyc.compile(
        launch,
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        _ptr(),
        1,
        64,
        2,
        128,
        fx.Stream(None),
    )
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "splitk_combine_gfx120x_bf16")


def _need_cuda(label: str):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"SKIP {label}: needs live CUDA for fx.Tensor launch compile")


def compile_and_export_alibi(out_dir: Path) -> Path:
    """Compile the ALiBi bias fill and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.attention.gfx120x_alibi_bias import build_alibi_bias_module

    _need_cuda("alibi")
    import torch

    slopes = torch.empty(1, device="cuda", dtype=torch.float32)
    out = torch.empty(16 * 16, device="cuda", dtype=torch.float32)
    launch = build_alibi_bias_module(block_threads=256, per_head=False)
    compiled = flyc.compile(launch, slopes, out, 16, 16, 1, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "alibi_bias_gfx120x")


def compile_and_export_attn_mask(out_dir: Path) -> Path:
    """Compile the bool-mask conversion and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.attention.gfx120x_attn_mask import build_bool_mask_to_bias_module

    _need_cuda("attn_mask")
    import torch

    mask = torch.empty(64, device="cuda", dtype=torch.uint8)
    out = torch.empty(64, device="cuda", dtype=torch.float32)
    launch = build_bool_mask_to_bias_module(256)
    compiled = flyc.compile(launch, mask, out, 64, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "attn_mask_gfx120x")


def compile_and_export_mxfp8_quant(out_dir: Path) -> Path:
    """Compile aligned MXFP8 quant and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_mxfp8_e8m0 import build_mxfp8_quant_module

    _need_cuda("mxfp8_quant")
    import torch

    x = torch.empty(2, 32, device="cuda", dtype=torch.bfloat16)
    q = torch.empty(2, 8, device="cuda", dtype=torch.int32)
    scale = torch.empty(2, device="cuda", dtype=torch.uint8)
    launch = build_mxfp8_quant_module("bfloat16")
    compiled = flyc.compile(launch, x, q, scale, 2, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "mxfp8_quant_gfx120x")


def compile_and_export_mxfp8_dequant(out_dir: Path) -> Path:
    """Compile aligned MXFP8 dequant and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_mxfp8_e8m0 import build_mxfp8_dequant_module

    _need_cuda("mxfp8_dequant")
    import torch

    q = torch.empty(2, 8, device="cuda", dtype=torch.int32)
    scale = torch.empty(2, device="cuda", dtype=torch.uint8)
    out = torch.empty(2, 32, device="cuda", dtype=torch.float32)
    launch = build_mxfp8_dequant_module(False)
    compiled = flyc.compile(launch, q, scale, out, 2, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "mxfp8_dequant_gfx120x")


def compile_and_export_mxfp4_quant(out_dir: Path) -> Path:
    """Compile aligned MXFP4 quant and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_mxfp4_e2m1 import build_mxfp4_quant_module

    _need_cuda("mxfp4_quant")
    import torch

    x = torch.empty(2, 32, device="cuda", dtype=torch.bfloat16)
    q = torch.empty(2, 16, device="cuda", dtype=torch.uint8)
    scale = torch.empty(2, device="cuda", dtype=torch.uint8)
    launch = build_mxfp4_quant_module("bfloat16")
    compiled = flyc.compile(launch, x, q, scale, 2, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "mxfp4_quant_gfx120x")


def compile_and_export_mxfp4_dequant(out_dir: Path) -> Path:
    """Compile aligned MXFP4 dequant and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_mxfp4_e2m1 import build_mxfp4_dequant_module

    _need_cuda("mxfp4_dequant")
    import torch

    q = torch.empty(2, 16, device="cuda", dtype=torch.uint8)
    scale = torch.empty(2, device="cuda", dtype=torch.uint8)
    out = torch.empty(2, 32, device="cuda", dtype=torch.float32)
    launch = build_mxfp4_dequant_module()
    compiled = flyc.compile(launch, q, scale, out, 2, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "mxfp4_dequant_gfx120x")


def compile_and_export_awq_dequant(out_dir: Path) -> Path:
    """Compile AWQ weight dequant and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_awq_w4a16 import build_awq_dequant_w4a16_module

    _need_cuda("awq_dequant")
    import torch

    q = torch.empty(2, 64, device="cuda", dtype=torch.uint8)
    scales = torch.empty(2, 2, device="cuda", dtype=torch.bfloat16)
    zeros = torch.empty(2, 2, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(2, 128, device="cuda", dtype=torch.bfloat16)
    launch = build_awq_dequant_w4a16_module(k=128, group_size=64, out_dtype="bfloat16")
    compiled = flyc.compile(launch, q, scales, zeros, out, 2, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "awq_dequant_gfx120x")


def compile_and_export_int8_convrot_dequant(out_dir: Path) -> Path:
    """Compile int8 ConvRot dequant and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_int8_convrot import build_int8_convrot_dequant_module

    _need_cuda("int8_convrot_dequant")
    import torch

    q = torch.empty(2, 64, device="cuda", dtype=torch.int8)
    scale = torch.empty(2, 1, device="cuda", dtype=torch.float32)
    out = torch.empty(2, 64, device="cuda", dtype=torch.bfloat16)
    launch = build_int8_convrot_dequant_module(out_dtype="bfloat16", group_size=64, k=64)
    compiled = flyc.compile(launch, q, scale, out, 2, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "int8_convrot_dequant_gfx120x")


def compile_and_export_tensorwise(out_dir: Path) -> Path:
    """Compile int8 tensorwise quant (partial kernel) and export C object."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.quant.rdna4_quantize_int8_tensorwise import build_quantize_int8_tensorwise_module

    launch_partial, _launch_reduce, _launch_quant = build_quantize_int8_tensorwise_module("bfloat16")
    compiled = flyc.compile(launch_partial, _ptr(), _ptr(), 64, 1, fx.Stream(None))
    if compiled is None:
        raise RuntimeError("flyc.compile returned None")
    return _export(compiled, out_dir, "quantize_int8_tensorwise_partial_gfx120x")


def _try_export(label: str, fn: object, out_dir: Path, written: list, skips: list) -> None:
    """Export one family; isolate abort-prone Tensor compiles in a subprocess."""
    import subprocess
    import textwrap

    isolate = label in {"adaln", "rms_rope"}
    fmap = {
        "adaln": "compile_and_export_adaln",
        "rms_rope": "compile_and_export_rms_rope",
    }

    def _record_skip(reason: str) -> None:
        skips.append(f"SKIP {label}: {reason}")
        print(f"  {label:12s} SKIP — {reason}")

    if not isolate:
        try:
            obj = fn(out_dir)
            written.append(obj)
            print(f"  {label:12s} {obj} ({obj.stat().st_size} bytes)")
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            _record_skip(msg[5:] if msg.startswith("SKIP ") else f"compile/export failed: {msg}")
        return

    import tests.python.examples.aot_gfx120x_example as _aotmod

    root = str(Path(_aotmod.__file__).resolve().parents[3])
    fn_name = fmap[label]
    script = textwrap.dedent(f"""
        import os, sys
        from pathlib import Path  # noqa: E402
        os.environ.setdefault("ARCH", "gfx1201")
        os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
        root = {root!r}
        sys.path.insert(0, root)
        bp = root + "/build-fly/python_packages"
        if bp not in sys.path:
            sys.path.insert(0, bp)
        from tests.python.examples import aot_gfx120x_example as aot
        obj = getattr(aot, {fn_name!r})(Path({str(out_dir)!r}))
        print("OK", obj)
        """)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if r.returncode == 0 and "OK " in (r.stdout or ""):
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("OK ")][-1]
        obj = Path(line[3:].strip())
        written.append(obj)
        print(f"  {label:12s} {obj} ({obj.stat().st_size} bytes)")
        return
    err = r.stderr or ""
    if "profile mismatch" in err:
        reason = (
            "fx.Tensor launch compile profile mismatch with placeholder shapes " "(export via live host tests instead)"
        )
    elif "Mismatched ranks" in err or r.returncode < 0:
        reason = "compiler assert on fx.Tensor placeholder ranks " "(export via live host tests instead)"
    else:
        tail = (err or r.stdout or f"rc={r.returncode}").strip().splitlines()
        reason = (tail[-1] if tail else f"rc={r.returncode}")[:200]
    _record_skip(reason)


def run_pack(
    out_dir: Path,
    *,
    include_fa: bool = True,
    include_w8a16: bool = True,
    include_fa_fp8: bool = False,
    include_fa_int8: bool = False,
    include_iu4_gemm: bool = False,
    include_awq: bool = False,
    include_svd: bool = False,
    include_convrot: bool = False,
    include_extra: bool = False,
) -> list[Path]:
    saved_arch = os.environ.get("ARCH")
    saved_gpu = os.environ.get("FLYDSL_GPU_ARCH")
    arch = _ensure_gfx120x_arch()
    try:
        return _run_pack_body(
            out_dir,
            arch,
            include_fa=include_fa,
            include_w8a16=include_w8a16,
            include_fa_fp8=include_fa_fp8,
            include_fa_int8=include_fa_int8,
            include_iu4_gemm=include_iu4_gemm,
            include_awq=include_awq,
            include_svd=include_svd,
            include_convrot=include_convrot,
            include_extra=include_extra,
        )
    finally:
        if saved_arch is None:
            os.environ.pop("ARCH", None)
        else:
            os.environ["ARCH"] = saved_arch
        if saved_gpu is None:
            os.environ.pop("FLYDSL_GPU_ARCH", None)
        else:
            os.environ["FLYDSL_GPU_ARCH"] = saved_gpu


def _run_pack_body(
    out_dir: Path,
    arch: str,
    *,
    include_fa: bool,
    include_w8a16: bool,
    include_fa_fp8: bool,
    include_fa_int8: bool,
    include_iu4_gemm: bool,
    include_awq: bool,
    include_svd: bool,
    include_convrot: bool,
    include_extra: bool,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"gfx120x AOT pack → {out_dir} (ARCH={arch})")
    written: list[Path] = []
    skips: list[str] = []
    written.append(compile_and_export_rope(out_dir))
    print(f"  rope:       {written[-1]} ({written[-1].stat().st_size} bytes)")
    if include_w8a16:
        written.append(compile_and_export_w8a16(out_dir))
        print(f"  w8a16:      {written[-1]} ({written[-1].stat().st_size} bytes)")
    if include_fa:
        written.append(compile_and_export_flash_attn(out_dir))
        print(f"  flash_attn: {written[-1]} ({written[-1].stat().st_size} bytes)")
    if include_fa_fp8:
        written.append(compile_and_export_flash_attn_fp8(out_dir))
        print(f"  fa_fp8:     {written[-1]} ({written[-1].stat().st_size} bytes)")
    if include_fa_int8:
        written.append(compile_and_export_flash_attn_int8(out_dir))
        print(f"  fa_int8:    {written[-1]} ({written[-1].stat().st_size} bytes)")
    if include_iu4_gemm:
        written.append(compile_and_export_iu4_gemm(out_dir))
        print(f"  iu4_gemm:   {written[-1]} ({written[-1].stat().st_size} bytes)")
    if include_awq:
        written.append(compile_and_export_awq_gemv(out_dir))
        print(f"  awq_gemv:   {written[-1]} ({written[-1].stat().st_size} bytes)")
    if include_svd:
        written.append(compile_and_export_svdquant(out_dir))
        print(f"  svdquant:   {written[-1]} ({written[-1].stat().st_size} bytes)")
    if include_convrot:
        written.append(compile_and_export_convrot_quant(out_dir))
        print(f"  convrot:    {written[-1]} ({written[-1].stat().st_size} bytes)")
    if include_extra:
        extras = [
            ("adaln", compile_and_export_adaln),
            ("rms_rope", compile_and_export_rms_rope),
            ("swiglu", compile_and_export_swiglu),
            ("stoch_fp8", compile_and_export_stoch_fp8),
            ("fp8_quant", compile_and_export_fp8_quant),
            ("int8_linear", compile_and_export_int8_linear),
            ("scaled_mm_fp8", compile_and_export_scaled_mm_fp8),
            ("fused_mlp", compile_and_export_fused_mlp),
            ("asym_w4a8", compile_and_export_asym_w4a8),
            ("int8_fused", compile_and_export_int8_fused),
            ("fp8_fused", compile_and_export_fp8_fused),
            ("rowwise", compile_and_export_rowwise),
            ("tensorwise", compile_and_export_tensorwise),
            ("mxfp8_gemm", compile_and_export_mxfp8_gemm),
            ("mxfp4_gemm", compile_and_export_mxfp4_gemm),
            ("silu", compile_and_export_silu),
            ("fp8_dequant", compile_and_export_fp8_dequant),
            ("splitk", compile_and_export_splitk),
            ("alibi", compile_and_export_alibi),
            ("attn_mask", compile_and_export_attn_mask),
            ("mxfp8_quant", compile_and_export_mxfp8_quant),
            ("mxfp8_dequant", compile_and_export_mxfp8_dequant),
            ("mxfp4_quant", compile_and_export_mxfp4_quant),
            ("mxfp4_dequant", compile_and_export_mxfp4_dequant),
            ("awq_dequant", compile_and_export_awq_dequant),
            ("int8_convrot_dequant", compile_and_export_int8_convrot_dequant),
        ]
        for label, fn in extras:
            _try_export(label, fn, out_dir, written, skips)
        if skips:
            print("SKIP notes:")
            for s in skips:
                print(f"  - {s}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "AOT export gfx120x ship kernels (RoPE / W8A16 / FlashAttention + --all families). "
            "Tensor launches need a live CUDA device. Pointer launches do not. "
            "Tile variants that are not a separate product (extra FP8 shapes, "
            "SwiGLU block sizes, non-16 MLP tiles) stay on the one exported specialization."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for .o/.h (default: temp dir under /tmp)",
    )
    parser.add_argument("--skip-fa", action="store_true", help="Skip bf16 FlashAttention export")
    parser.add_argument("--skip-w8a16", action="store_true", help="Skip W8A16 export")
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Also export fp8/int8 FA, iu4 GEMM, AWQ, SVD, ConvRot, plus adaln, rms_rope, "
            "swiglu, silu, stoch_fp8, fp8 quant/dequant, int8_linear, scaled_mm_fp8, "
            "mxfp8/mxfp4 GEMM and quant, split-K combine, ALiBi, bool mask, fused_mlp, "
            "asym_w4a8, fused int8/fp8, rowwise, tensorwise, AWQ dequant, and int8 ConvRot "
            "dequant (export what compiles; print SKIP+reason)"
        ),
    )
    args = parser.parse_args(argv)

    out = args.out
    tmp = None
    if out is None:
        tmp = tempfile.TemporaryDirectory(prefix="flydsl_aot_gfx120x_")
        out = Path(tmp.name)

    try:
        run_pack(
            out,
            include_fa=not args.skip_fa,
            include_w8a16=not args.skip_w8a16,
            include_fa_fp8=args.all,
            include_fa_int8=args.all,
            include_iu4_gemm=args.all,
            include_awq=args.all,
            include_svd=args.all,
            include_convrot=args.all,
            include_extra=args.all,
        )
    finally:
        if tmp is not None:
            print(f"(temp artifacts kept until process exit: {out})")
            # Keep temp dir for inspection when run as script; pytest uses explicit tmp_path.
            tmp.cleanup = lambda: None  # type: ignore[method-assign]

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ---------------------------------------------------------------------------
# pytest interface
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_aot_gfx120x_rope_and_w8a16_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile+export gfx120x RoPE and W8A16 without requiring a live GPU."""
    monkeypatch.setenv("ARCH", "gfx1201")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    objs = run_pack(tmp_path, include_fa=False, include_w8a16=True)
    assert len(objs) == 2
    for obj in objs:
        assert obj.is_file() and obj.stat().st_size > 0
        assert obj.with_suffix(".h").is_file()


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_aot_gfx120x_flash_attn_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile+export gfx120x bf16 FlashAttention (placeholder args)."""
    monkeypatch.setenv("ARCH", "gfx1201")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    objs = run_pack(tmp_path, include_fa=True, include_w8a16=False)
    assert len(objs) == 2  # rope + fa
    fa = tmp_path / "flash_attn_gfx120x_bf16.o"
    assert fa.is_file() and fa.stat().st_size > 0
    assert fa.read_bytes()[:4] == b"\x7fELF"


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
def test_aot_gfx120x_broaden_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile+export --all families (core FA/quant + extra shipped hosts)."""
    monkeypatch.setenv("ARCH", "gfx1201")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    objs = run_pack(
        tmp_path,
        include_fa=False,
        include_w8a16=False,
        include_fa_fp8=True,
        include_fa_int8=True,
        include_iu4_gemm=True,
        include_awq=True,
        include_svd=True,
        include_convrot=True,
        include_extra=True,
    )
    # rope + prior broaden set; extras may skip without CUDA
    assert len(objs) >= 7
    names = {obj.name for obj in objs}
    for required in (
        "mxfp8_block_gemm_gfx120x_bf16.o",
        "mxfp4_block_gemm_gfx120x_bf16.o",
        "silu_mul_gfx120x_bf16.o",
        "fp8_dequant_gfx120x_bf16.o",
        "splitk_combine_gfx120x_bf16.o",
    ):
        assert required in names, f"missing AOT object {required}; skips={names}"
    if torch.cuda.is_available():
        for required in (
            "adaln_gfx120x_bf16.o",
            "rms_rope_gfx120x_bf16.o",
            "alibi_bias_gfx120x.o",
            "attn_mask_gfx120x.o",
            "mxfp8_quant_gfx120x.o",
            "mxfp8_dequant_gfx120x.o",
            "mxfp4_quant_gfx120x.o",
            "mxfp4_dequant_gfx120x.o",
            "awq_dequant_gfx120x.o",
            "int8_convrot_dequant_gfx120x.o",
        ):
            assert required in names, f"missing AOT object {required}"
    for obj in objs:
        assert obj.is_file() and obj.stat().st_size > 0
        assert obj.read_bytes()[:4] == b"\x7fELF"
