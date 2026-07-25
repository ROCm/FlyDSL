#!/usr/bin/env python3
"""Benchmark the FlyDSL fp8 (e4m3fn) forward flash-attention kernel on gfx950.

Mirrors ``run_fp8_config`` in ``tests/kernels/test_flash_attn_fwd.py``: Q/K/V are
per-tensor quantized to ``torch.float8_e4m3fn`` with shape-[1] fp32 descales, the
output is bf16, and the kernel is dispatched through
``kernels.attention.flash_attn_interface.flydsl_flash_attn_func``.

For fp8 inputs that function routes to ``_build_dense_fp8`` ->
``kernels.attention.flash_attn_fp8_gfx950.build_flash_attn_dualwave_swp_fp8_module``,
i.e. the gfx950 DUALWAVE_SWP fp8 kernel
``flash_attn_dualwave_swp_fp8_gfx950_kernel`` in
``kernels/attention/flash_attn_fp8_gfx950.py``.

fp8 forward is gfx950-only and requires head_dim == 128.
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

# aiter's per_tensor_quant is used (when importable) so the FlyDSL and aiter
# benches quantize identically; the torch fallback below is numerically the same.
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import torch

# Make sure the FlyDSL repo root is importable.
_repo = Path(__file__).resolve().parent
_EXPECTED_ROOT = _repo
sys.path.insert(0, str(_repo))
# Drop any other FlyDSL repo roots that might shadow kernels.
sys.path[:] = [p for p in sys.path if not (p.endswith("/FlyDSL") and Path(p).resolve() != _EXPECTED_ROOT)]

logging.basicConfig(level=logging.INFO)

from kernels.attention.flash_attn_interface import flydsl_flash_attn_func  # noqa: E402
import flydsl  # noqa: E402
import importlib

_kernel_mod = importlib.import_module("kernels.attention.flash_attn_interface")
_kernel_path = Path(_kernel_mod.__file__).resolve()
_fp8_kernel_path = (_EXPECTED_ROOT / "kernels" / "attention" / "flash_attn_fp8_gfx950.py").resolve()
_flydsl_root = Path(flydsl.__file__).resolve().parent
if not str(_kernel_path).startswith(str(_EXPECTED_ROOT)):
    raise RuntimeError(f"flash_attn_interface must be under {_EXPECTED_ROOT}, got {_kernel_path}")
if not str(_flydsl_root).startswith(str(_EXPECTED_ROOT / "python" / "flydsl")):
    raise RuntimeError(f"flydsl package must be under {_EXPECTED_ROOT}, got {_flydsl_root}")

# OCP e4m3fn (NOT the fnuz variant) end-to-end on gfx950.
FP8_DTYPE = torch.float8_e4m3fn
UNIFORM_RANGE = (-1, 1)
# fp8 correctness gate (fixed; fp8 is lossy) — used only with --check.
FP8_MAX_ERR = 5e-2
FP8_MIN_COS = 0.98

# Kernel config (mirrors run_fp8_config / test_flash_attn_fwd.py defaults).
FLASH_ATTN_FUNC_KERNEL_CONFIG = {
    "waves_per_eu": int(os.getenv("FLYDSL_WAVES_PER_EU", "2")),
    "daz": True,
    "dualwave_swp_lazy_rescale": os.getenv("FLYDSL_DUALWAVE_SWP_LAZY_RESCALE", "1") == "1",
    "dualwave_swp_setprio": os.getenv("FLYDSL_DUALWAVE_SWP_SETPRIO", "0") == "1",
    "dualwave_swp_enable_stagger": os.getenv("FLYDSL_DUALWAVE_SWP_STAGGER", "1") == "1",
}


def quantize_per_tensor_fp8(x):
    """Per-tensor quantize a float tensor to e4m3fn + a shape-[1] fp32 descale.

    Mirrors aiter.per_tensor_quant: descale = amax / fp8_max, stored value is
    round(x / descale), dequant is fp8_value * descale. Uses aiter's helper when
    available so this bench and the aiter bench quantize identically; falls back
    to a numerically identical torch implementation otherwise.
    """
    try:
        from aiter.utility import dtypes as _adtypes
        from aiter.ops.quant import per_tensor_quant as _ptq

        x_fp8, descale = _ptq(x, quant_dtype=_adtypes.fp8)
        if x_fp8.dtype != FP8_DTYPE:
            raise ValueError(f"aiter per_tensor_quant produced {x_fp8.dtype}, expected {FP8_DTYPE}")
        return x_fp8.contiguous(), descale.to(torch.float32).view(1).contiguous()
    except Exception:
        fp8_max = torch.finfo(FP8_DTYPE).max
        amax = x.abs().max().to(torch.float32)
        descale = (amax / fp8_max).clamp(min=1e-12).view(1)
        x_fp8 = (x.to(torch.float32) / descale).to(FP8_DTYPE)
        return x_fp8.contiguous(), descale.to(torch.float32).contiguous()


def _dequant_fp8(x_fp8, descale):
    return x_fp8.to(torch.float32) * descale.to(torch.float32)


def _fp8_kwargs():
    return dict(
        waves_per_eu=FLASH_ATTN_FUNC_KERNEL_CONFIG["waves_per_eu"],
        daz=FLASH_ATTN_FUNC_KERNEL_CONFIG.get("daz", False),
        dualwave_swp_lazy_rescale=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_lazy_rescale"],
        dualwave_swp_setprio=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_setprio"],
        dualwave_swp_enable_stagger=FLASH_ATTN_FUNC_KERNEL_CONFIG["dualwave_swp_enable_stagger"],
    )


def _reference(q_fp8, q_descale, k_fp8, k_descale, v_fp8, v_descale, causal):
    """Dequantize the SAME e4m3fn Q/K/V and run a high-precision SDPA reference."""
    import torch.nn.functional as F

    q = _dequant_fp8(q_fp8, q_descale).transpose(1, 2)  # [B,H,S,D]
    k = _dequant_fp8(k_fp8, k_descale).transpose(1, 2)
    v = _dequant_fp8(v_fp8, v_descale).transpose(1, 2)
    nh_q, nh_kv = q.shape[1], k.shape[1]
    if nh_q != nh_kv:
        rep = nh_q // nh_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)  # [B,S,H,D]


def time_kernel(fn, warmup, iters):
    """Return mean seconds/iter measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3  # ms -> s per iter


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlyDSL fp8 forward flash_attn (gfx950)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension (fp8 requires 128)")
    parser.add_argument("--num_kv_heads", type=int, default=None, help="Number of KV heads (default == num_heads)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="Validate vs dequantized-input SDPA reference")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/ROCm not available")
        sys.exit(1)

    head = args.num_heads
    batch = args.batch_size
    headdim = args.head_dim
    num_kv_heads = args.num_kv_heads if args.num_kv_heads is not None else head
    device = "cuda"

    try:
        gpu_arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:
        gpu_arch = ""
    if not gpu_arch.startswith("gfx950"):
        print(f"FlyDSL fp8 forward requires gfx950 (got '{gpu_arch or 'unknown'}')")
        sys.exit(1)
    if headdim != 128:
        print(f"FlyDSL fp8 forward requires head_dim == 128 (got {headdim})")
        sys.exit(1)

    print("FlyDSL fp8 (e4m3fn) forward flash_attn Benchmark")
    print(f"FlyDSL root: {_EXPECTED_ROOT}")
    print(f"Interface:   {_kernel_path}")
    print(f"fp8 kernel:  {_fp8_kernel_path}")
    print(f"flydsl pkg:  {_flydsl_root}")
    print(f"batch: {batch}, head: {head}, headdim: {headdim}, num_kv_heads: {num_kv_heads}")
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_arch})")

    seq_lens = sorted({1024, 2048, 4096, 8192, 16384, 32768})

    def run_mode(is_causal):
        print(f"is_causal: {is_causal}")
        for seq_len in seq_lens:
            torch.manual_seed(123)
            q_bf16 = torch.empty(batch, seq_len, head, headdim, dtype=torch.bfloat16, device=device).uniform_(*UNIFORM_RANGE)
            k_bf16 = torch.empty(batch, seq_len, num_kv_heads, headdim, dtype=torch.bfloat16, device=device).uniform_(*UNIFORM_RANGE)
            v_bf16 = torch.empty(batch, seq_len, num_kv_heads, headdim, dtype=torch.bfloat16, device=device).uniform_(*UNIFORM_RANGE)
            q_fp8, q_descale = quantize_per_tensor_fp8(q_bf16)
            k_fp8, k_descale = quantize_per_tensor_fp8(k_bf16)
            v_fp8, v_descale = quantize_per_tensor_fp8(v_bf16)
            o_bf16 = torch.zeros(batch, seq_len, head, headdim, dtype=torch.bfloat16, device=device)

            def fp8_forward():
                flydsl_flash_attn_func(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    causal=is_causal,
                    num_kv_heads=num_kv_heads,
                    out=o_bf16,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **_fp8_kwargs(),
                )

            try:
                fp8_forward()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"{seq_len} ERROR: {e}")
                import traceback

                traceback.print_exc()
                continue

            if args.check and batch * head * seq_len * seq_len > 128 * 1024 * 1024:
                print(f"  [check] skipped (score matrix too large for a dense fp32 reference)")
            elif args.check:
                ref = _reference(q_fp8, q_descale, k_fp8, k_descale, v_fp8, v_descale, is_causal)
                import torch.nn.functional as F

                o_f32 = o_bf16.float().reshape(-1)
                ref_f32 = ref.float().reshape(-1)
                max_err = (o_f32 - ref_f32).abs().max().item()
                min_cos = F.cosine_similarity(o_f32.reshape(-1, headdim), ref_f32.reshape(-1, headdim), dim=1).min().item()
                passed = max_err < FP8_MAX_ERR and min_cos > FP8_MIN_COS
                print(f"  [check] max_err={max_err:.4e} min_cos={min_cos:.4f} -> {'PASS' if passed else 'FAIL'}")

            try:
                t = time_kernel(fp8_forward, warmup=args.warmup, iters=args.iters)
                s_eff = seq_len / 2.0 if is_causal else float(seq_len)
                flops = 4.0 * seq_len * s_eff * headdim * head * batch
                print(f"{seq_len} flops:{flops / t * 1e-12}")
            except Exception as e:
                print(f"{seq_len} ERROR: {e}")
                import traceback

                traceback.print_exc()

    for causal_mode in (False, True):
        run_mode(causal_mode)


if __name__ == "__main__":
    main()
