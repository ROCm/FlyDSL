# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X iu8 int8 linear GEMM (gfx120x / RDNA4).

Computes ``C[M, N] = A[M, K] @ B_T[N, K].T`` with a per-row activation scale
and either a per-column or scalar weight scale in the epilogue. Int8
activations × int8 weights → bf16/fp16 out via the gfx120x **iu8** WMMA atom.

Requires that atom on ``MmaOpGFX120X_WMMAType``. Stock FlyDSL without integer
GFX120X WMMA cannot compile this module. For bf16 activations with 8-bit
weights (no iu8), use ``rdna4_w8a16_linear``. Call each product API explicitly;
there is no size/K auto router between them.

The WMMA atom is constructed with ``clamp=False`` (default): the INT32
accumulator wraps on overflow. ``clamp=True`` would saturate to the input
element type range per AMD IU8 semantics; this kernel keeps wrap for exact
int32 accumulation before the scale epilogue.

The device kernel is ``build_scaled_mm_fp8_module(..., kind="int8")``.
Odd K is handled there. This module keeps the R9700 int8 tile table.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Optional

import torch

from kernels.common.gfx120x_arch import require_gfx120x
from kernels.gemm.rdna4_tile import TileConfig

KERNEL_NAME = "rdna4_int8_linear"

_DEFAULT_WGPS = 32

# Launch tiles: include deeper K on fat shapes where that tile beat the HIP baseline.
_CFG_128_128_64 = TileConfig(128, 128, 64, 4, 2, 2, 4)
_CFG_128_128_128 = TileConfig(128, 128, 128, 4, 2, 2, 4)
_CFG_256_128_64 = TileConfig(256, 128, 64, 4, 2, 4, 4)
_CFG_256_128_128 = TileConfig(256, 128, 128, 4, 2, 4, 4)
_CFG_64_64_64 = TileConfig(64, 64, 64, 2, 2, 2, 2)


_WGP_COUNT_CACHE: dict[int, int] = {}


def _wgp_count(device=None) -> int:
    """Cached CU/WGP count for ``device`` (defaults to current / 0).

    Multi-GPU: query the tensor's device index, not always ``cuda:0``.
    """
    try:
        import torch

        if device is None:
            idx = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        else:
            idx = int(device.index) if getattr(device, "index", None) is not None else int(torch.cuda.current_device())
        if idx not in _WGP_COUNT_CACHE:
            _WGP_COUNT_CACHE[idx] = int(torch.cuda.get_device_properties(idx).multi_processor_count) or _DEFAULT_WGPS
        return _WGP_COUNT_CACHE[idx]
    except Exception:  # noqa: BLE001
        return _DEFAULT_WGPS


def pick_tile_config(M: int, N: int, K: int, wgps: int | None = None, device=None) -> TileConfig:
    """Size-based tile pick; prefer IU8 128×128×128 for deep K.

    Design notes (measured on gfx1201 / R9700, 2026-09-30):
      - FlyDSL ``docs/kernel_tuning_guide.md`` (deeper ``tile_k`` for reuse; LDS budget)
      - FlyDSL ``docs/testing_benchmarking_guide.md`` (CUDA-event median timing)
      - IU8 BK=128 on 128×128 when K≥2048 (LDS about 34 KiB)
      - HIP ``launch_gemm_wmma`` tile thresholds (WGP-aware)
      - Large production-like shapes (e.g. M=N=K around 4k, or 1024×5120×5120)
        prefer 128×128×128 over tall-M 256×128×128

    Re-bench: tall-M 256×128×128 is slower than or ties the HIP baseline on those large shapes;
    128×128×128 is faster on large (~1.60×) and on 1024×5120×5120 (~1.11×) vs the HIP baseline.
    Keep 256 tiles for tall-M mid-K only.
    """
    if wgps is None:
        wgps = _wgp_count(device)
    blocks_128 = ((M + 127) // 128) * ((N + 127) // 128)
    skinny = M <= 64 or N <= 64
    # Deep-K first: prefer 128×128×128 over tall-M 256 on large/deep-K shapes.
    if not skinny and M >= 128 and N >= 128 and K >= 2048:
        return _CFG_128_128_128
    if not skinny and M >= 512 and blocks_128 >= wgps:
        if K >= 512:
            return _CFG_256_128_128
        return _CFG_256_128_64
    if not skinny and M >= 128 and N >= 128:
        if K >= 512:
            return _CFG_128_128_128
        return _CFG_128_128_64
    return _CFG_64_64_64


@lru_cache(maxsize=64)
def build_int8_linear_module(
    out_name: str,
    cfg: TileConfig,
    skip_bounds: bool = False,
    w_scale_per_n: bool = True,
    k_tail: int = 0,
    sign_a: bool = True,
    sign_b: bool = True,
) -> Callable[..., None]:
    """Compile iu8 int8 linear through the shared quant GEMM builder.

    ``k_tail`` is ``K % 16``. Zero keeps the 16-byte loads. ``sign_a`` /
    ``sign_b`` are the iu8 WMMA NEG modifiers (default signed).
    """
    from kernels.gemm.rdna4_scaled_mm_fp8 import build_scaled_mm_fp8_module

    return build_scaled_mm_fp8_module(
        out_name,
        cfg,
        skip_bounds,
        False,
        k_tail,
        kind="int8",
        scale_b_per_n=w_scale_per_n,
        sign_a=sign_a,
        sign_b=sign_b,
        kernel_name=KERNEL_NAME,
    )


def create_wmma_int8_linear_module(
    out_dtype: str = "bfloat16",
    cfg: TileConfig | tuple | None = None,
    *,
    skip_bounds: bool = False,
    w_scale_per_n: bool = True,
    k_tail: int = 0,
    sign_a: bool = True,
    sign_b: bool = True,
) -> Callable[..., None]:
    """Create a gfx120x int8-linear launcher for one tile configuration.

    ``k_tail`` must be ``K % 16``. The product host passes it. A zero tail
    still requires ``K % 16 == 0`` because that specialization uses 16-byte loads.
    """
    if cfg is None:
        cfg = _CFG_128_128_64
    if isinstance(cfg, tuple):
        cfg = TileConfig(*cfg)
    return build_int8_linear_module(out_dtype, cfg, skip_bounds, w_scale_per_n, k_tail, sign_a, sign_b)


def int8_linear(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    stream: Optional[torch.cuda.Stream] = None,
    sign_a: bool = True,
    sign_b: bool = True,
) -> torch.Tensor:
    """int8 activations × int8 weights through the iu8 WMMA.

    ``x_scale`` is ``[M]``. ``w_scale`` is one value or ``[N]``. A K that is
    not a multiple of 16 is zero-filled in the kernel.
    """
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.compiler.jit_argument import PointerJitArg
    from kernels.common.gfx120x_pad import ensure_contiguous

    require_gfx120x(a_int8.device, what="int8_linear (gfx120x)")
    if a_int8.dtype != torch.int8 or b_int8.dtype != torch.int8:
        raise ValueError("iu8 int8 linear requires int8 A/B")
    a = ensure_contiguous(a_int8, stream=stream)
    b = ensure_contiguous(b_int8, stream=stream)
    m, k = a.shape
    n = b.shape[0]
    if b.shape[1] != k:
        raise ValueError(f"inner dim mismatch: a K={k}, b K={b.shape[1]}")
    k_tail = int(k) % 16
    if k_tail == 0:
        if a.data_ptr() % 16:
            a = a.clone()
        if b.data_ptr() % 16:
            b = b.clone()

    if x_scale.dtype == torch.float32 and x_scale.device == a.device and x_scale.is_contiguous():
        x_scale = x_scale.reshape(-1)
    else:
        x_scale = ensure_contiguous(x_scale.to(device=a.device, dtype=torch.float32).reshape(-1), stream=stream)
    if w_scale.dtype == torch.float32 and w_scale.device == a.device and w_scale.is_contiguous():
        w_scale = w_scale.reshape(-1)
    else:
        w_scale = ensure_contiguous(w_scale.to(device=a.device, dtype=torch.float32).reshape(-1), stream=stream)
    if x_scale.numel() != m:
        raise ValueError(f"x_scale must be [M]={m}, got {x_scale.numel()}")
    w_per_n = w_scale.numel() != 1
    if w_per_n and w_scale.numel() != n:
        raise ValueError(f"w_scale must be scalar or [N]={n}, got {w_scale.numel()}")
    if int(k) == 0:
        store_dtype = torch.float32 if bias is not None else out_dtype
        out = torch.zeros((m, n), device=a.device, dtype=store_dtype)
        if bias is not None:
            from kernels.common.gfx120x_row_bias import add_row_bias

            return add_row_bias(out, bias, out_dtype=out_dtype, stream=stream)
        if store_dtype != out_dtype:
            return out.to(out_dtype)
        return out

    store_dtype = torch.float32 if bias is not None else out_dtype
    store_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[store_dtype]
    out = torch.empty((m, n), device=a.device, dtype=store_dtype)
    cfg = pick_tile_config(m, n, k, device=a.device)
    launch = create_wmma_int8_linear_module(
        store_name,
        cfg,
        skip_bounds=(m % cfg.bm == 0 and n % cfg.bn == 0),
        w_scale_per_n=w_per_n,
        k_tail=k_tail,
        sign_a=sign_a,
        sign_b=sign_b,
    )

    def _ptr(t: torch.Tensor) -> PointerJitArg:
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    args = (_ptr(a), _ptr(b), _ptr(out), _ptr(x_scale), _ptr(w_scale), int(m), int(n), int(k))
    if stream is None:
        launch(*args)
    else:
        launch(*args, stream)
    if bias is not None:
        from kernels.common.gfx120x_row_bias import add_row_bias

        out = add_row_bias(out, bias, out_dtype=out_dtype, stream=stream)
    elif store_dtype != out_dtype:
        out = out.to(out_dtype)
    return out
