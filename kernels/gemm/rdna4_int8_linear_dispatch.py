# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Host-side size dispatcher for gfx120x int8 linear (DEFAULT entry rule).

MEASURED (2026-09-30, gfx1201 / R9700): default size rule from a FlyDSL
Device microbenchmark grid (``do_bench``: warm=10, rep=50, backlog timer)
comparing W8A16 vs iu8 GEMM alone. Prefer **iu8 when K >= 256**, else W8A16.
Optional M and M×K caps can still force W8A16 when set positive; shipped
defaults leave those caps at 0 so only the K gate decides. Thresholds live in
``kernels/common/gfx120x_autotune_tables.py``.

W8A16 linear (``rdna4_w8a16_linear``)
  Activations stay bf16/fp16. Int8 / FP8 e4m3fn / e5m2 weights cast to that
  activation dtype in VGPRs, then float WMMA. Faster when K < 256; slower than HIP on large K.

iu8 int8 linear (``rdna4_int8_linear``)
  Activations quantized to int8. Int8×int8 WMMA via the gfx120x iu8 atom.
  Faster when K >= 256, where W8A16 is slower than the HIP baseline.

Rule: if ``force_kernel`` is set → that kernel; else if
``FLYDSL_DISPATCH_MODE=force_flydsl`` → iu8; else if K < K_iu8_min (256) →
W8A16; else if M ≤ m_a_max (optional, default 0) → W8A16; else if M×K ≤
mk_a_max (optional, default 0) → W8A16; else → iu8.

Call ``int8_linear_auto`` / ``int8_linear_dispatched`` for the default path.
"""

from collections.abc import Callable
from typing import Literal, Optional

import torch

from flydsl.compiler.jit_argument import PointerJitArg
from kernels.common.dispatch_mode import get_dispatch_mode
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_autotune_tables import DEFAULT_K_IU8_MIN as _TABLE_K_IU8_MIN

KernelChoice = Literal["w8a16", "iu8"]

# Measured 2026-09-30: K gate dominates (see gfx120x_autotune_tables).
# Legacy M/MK caps default to 0 so they do not override the K rule unless set.
DEFAULT_K_IU8_MIN = _TABLE_K_IU8_MIN  # measured table
DEFAULT_M_W8A16_MAX = 0
DEFAULT_MK_W8A16_MAX = 0
DEFAULT_M_A_MAX = DEFAULT_M_W8A16_MAX  # alias
DEFAULT_MK_A_MAX = DEFAULT_MK_W8A16_MAX


def select_int8_kernel(
    M: int,
    N: int,
    K: int,
    *,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    k_iu8_min: int = DEFAULT_K_IU8_MIN,
    force_kernel: Optional[KernelChoice] = None,
) -> KernelChoice:
    """Cheap static rule: ``"w8a16" | "iu8"``. ``N`` unused by size selection.

    Measured default (gfx1201, 2026-09-30): prefer iu8 when ``K >= k_iu8_min``
    (256). ``FLYDSL_DISPATCH_MODE=force_flydsl`` returns iu8 before that test.
    Optional ``m_a_max`` / ``mk_a_max`` still force W8A16 when positive.
    """
    del N  # API symmetry with MNK docs
    if force_kernel is not None:
        if force_kernel not in ("w8a16", "iu8"):
            raise ValueError(f"force_kernel must be 'w8a16' or 'iu8', got {force_kernel!r}")
        return force_kernel
    # FLYDSL_DISPATCH_MODE: force_flydsl → always iu8 WMMA path.
    # force_hip is handled in int8_linear_auto (PyTorch reference path); keep size rule here.
    # Large MNK (e.g. 1024x4096x4096): auto must not pick W8A16 — forced W8A16 is slower than HIP; iu8 is faster.
    if get_dispatch_mode() == "force_flydsl":
        return "iu8"
    m = int(M)
    k = int(K)
    if k < int(k_iu8_min):
        return "w8a16"
    if int(m_a_max) > 0 and m <= int(m_a_max):
        return "w8a16"
    if int(mk_a_max) > 0 and m * k <= int(mk_a_max):
        return "w8a16"
    return "iu8"


def select_int8_kernel_for_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    k_iu8_min: int = DEFAULT_K_IU8_MIN,
    force_kernel: Optional[KernelChoice] = None,
) -> KernelChoice:
    """M from flattened x rows; N/K from weight [N,K]."""
    x2d = x.reshape(-1, x.shape[-1])
    return select_int8_kernel(
        int(x2d.shape[0]),
        int(weight.shape[0]),
        int(x2d.shape[1]),
        m_a_max=m_a_max,
        mk_a_max=mk_a_max,
        k_iu8_min=k_iu8_min,
        force_kernel=force_kernel,
    )


def _load_iu8_int8_linear() -> Callable[..., None]:
    """Import ``kernels.gemm.rdna4_int8_linear`` (iu8 WMMA int8 linear)."""
    from kernels.gemm import rdna4_int8_linear as mod

    return mod


def _run_iu8_int8_linear(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """iu8 int8 linear via ``rdna4_int8_linear`` raw-pointer launcher."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from kernels.common.tensor_shim import _run_compiled

    _pb = _load_iu8_int8_linear()
    create_wmma_int8_linear_module = _pb.create_wmma_int8_linear_module
    pick_tile_config = _pb.pick_tile_config

    if a_int8.dtype != torch.int8 or b_int8.dtype != torch.int8:
        raise ValueError("iu8 int8 linear requires int8 A/B")
    a = a_int8 if a_int8.is_contiguous() else a_int8.contiguous()
    b = b_int8 if b_int8.is_contiguous() else b_int8.contiguous()
    if a.data_ptr() % 16:
        a = a.clone()
    if b.data_ptr() % 16:
        b = b.clone()
    m, k = a.shape
    n = b.shape[0]
    if b.shape[1] != k:
        raise ValueError(f"inner dim mismatch: a K={k}, b K={b.shape[1]}")
    if k % 16 != 0:
        raise ValueError(f"K must be divisible by 16, got {k}")

    # Hot-path lean: skip .to/contig when scales are already f32 contiguous on device.
    if x_scale.dtype == torch.float32 and x_scale.device == a.device and x_scale.is_contiguous():
        x_scale = x_scale.reshape(-1)
    else:
        x_scale = x_scale.to(device=a.device, dtype=torch.float32).reshape(-1).contiguous()
    if w_scale.dtype == torch.float32 and w_scale.device == a.device and w_scale.is_contiguous():
        w_scale = w_scale.reshape(-1)
    else:
        w_scale = w_scale.to(device=a.device, dtype=torch.float32).reshape(-1).contiguous()
    if x_scale.numel() != m:
        raise ValueError(f"x_scale must be [M]={m}, got {x_scale.numel()}")
    w_per_n = w_scale.numel() != 1
    if w_per_n and w_scale.numel() != n:
        raise ValueError(f"w_scale must be scalar or [N]={n}, got {w_scale.numel()}")

    store_dtype = torch.float32 if bias is not None else out_dtype
    store_name = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }[store_dtype]
    out = torch.empty((m, n), device=a.device, dtype=store_dtype)
    cfg = pick_tile_config(m, n, k)
    launch = create_wmma_int8_linear_module(
        store_name,
        cfg,
        skip_bounds=(m % cfg.bm == 0 and n % cfg.bn == 0),
        w_scale_per_n=w_per_n,
    )

    def _ptr(t: torch.Tensor) -> PointerJitArg:
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    _run_compiled(
        launch,
        _ptr(a),
        _ptr(b),
        _ptr(out),
        _ptr(x_scale),
        _ptr(w_scale),
        int(m),
        int(n),
        int(k),
        torch.cuda.current_stream(device=a.device),
    )
    if bias is not None:
        bias = bias.to(device=out.device, dtype=torch.float32).reshape(-1)
        if bias.numel() != n:
            raise ValueError(f"bias must be [N]={n}")
        out = (out + bias).to(out_dtype)
    elif store_dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def dispatch_rule_doc(
    *,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    m_w8a16_max: int | None = None,
    mk_w8a16_max: int | None = None,
) -> dict:
    """Structured description of the active size dispatcher."""
    if m_w8a16_max is not None:
        m_a_max = m_w8a16_max
    if mk_w8a16_max is not None:
        mk_a_max = mk_w8a16_max
    return {
        "k_iu8_min": DEFAULT_K_IU8_MIN,
        "m_a_max": m_a_max,
        "mk_a_max": mk_a_max,
        "m_w8a16_max": m_a_max,
        "mk_w8a16_max": mk_a_max,
        "default_k_iu8_min": DEFAULT_K_IU8_MIN,
        "default_m_a_max": DEFAULT_M_A_MAX,
        "default_mk_a_max": DEFAULT_MK_A_MAX,
        "rule": (
            f"force_kernel if set; else iu8 if FLYDSL_DISPATCH_MODE=force_flydsl; "
            f"else w8a16 if K < {DEFAULT_K_IU8_MIN}; "
            f"else w8a16 if m_a_max>0 and M <= {m_a_max}; "
            f"else w8a16 if mk_a_max>0 and M*K <= {mk_a_max}; else iu8"
        ),
        "w8a16": (
            "kernels.gemm.rdna4_w8a16_linear — int8 or fp8 weights cast to the act dtype; "
            "bf16/fp16 acts; out defaults to the act dtype"
        ),
        "iu8": (
            "kernels.gemm.rdna4_int8_linear — int8 acts × int8 weights; "
            "out defaults to x.dtype (bf16, fp16, or fp32)"
        ),
        "smoke_anchor": {
            "tiny": {"M": 32, "N": 64, "K": 128, "expect": "w8a16"},
            "mid": {"M": 256, "N": 512, "K": 512, "expect": "iu8"},
            "large": {"M": 1024, "N": 4096, "K": 4096, "expect": "iu8"},
            "large_square": {"M": 4096, "N": 3072, "K": 3072, "expect": "iu8"},
        },
        "disclaimer": (
            "Measured K-gate size rule on gfx120x/R9700 2026-09-30 (do_bench). Not a general FlyDSL autotune API."
        ),
        "measured_date": "2026-09-30",
        "measured_method": "FlyDSL do_bench warm=10 rep=50 vs HIP baseline in a separate process (R9700 2026-09-30)",
    }


gate_rule_doc = dispatch_rule_doc


def int8_linear_dispatched(
    path: KernelChoice,
    *,
    a_int8: Optional[torch.Tensor] = None,
    b_int8: Optional[torch.Tensor] = None,
    x_scale: Optional[torch.Tensor] = None,
    w_scale: Optional[torch.Tensor] = None,
    a_act: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dispatch to W8A16 linear or B by ``path``. Caller supplies matching tensors.

    Path ``"iu8"``: ``a_int8``, ``b_int8``, ``x_scale``, ``w_scale`` (needs iu8 WMMA).
    Path ``"w8a16"``: ``a_act`` (bf16/fp16), ``b_int8``, ``w_scale``.
    """
    _dev = None
    for _t in (a_int8, b_int8, a_act):
        if _t is not None:
            _dev = _t.device
            break
    require_gfx120x(_dev, what="int8_linear_dispatched (gfx120x)")
    if path == "iu8":
        if a_int8 is None or b_int8 is None or x_scale is None or w_scale is None:
            raise ValueError("iu8 int8 linear requires a_int8, b_int8, x_scale, w_scale")
        try:
            return _run_iu8_int8_linear(a_int8, b_int8, x_scale, w_scale, bias, out_dtype)
        except ImportError as e:
            raise ImportError(
                "iu8 int8 linear requires kernels.gemm.rdna4_int8_linear "
                "(gfx120x-iu8-int8 / iu8 WMMA atom). " + str(e)
            ) from e
    if path == "w8a16":
        if a_act is None or b_int8 is None or w_scale is None:
            raise ValueError("W8A16 linear requires a_act, b_int8, w_scale")
        from kernels.gemm.rdna4_w8a16_linear import w8a16_gemm

        return w8a16_gemm(a_act, b_int8, w_scale, bias=bias, out_dtype=out_dtype)
    raise ValueError(f"unknown path {path!r}")


def int8_linear_auto(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    x_scale: Optional[torch.Tensor] = None,
    a_int8: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    m_a_max: int = DEFAULT_M_A_MAX,
    mk_a_max: int = DEFAULT_MK_A_MAX,
    force_kernel: Optional[KernelChoice] = None,
) -> torch.Tensor:
    """Select path from MNK then dispatch.

    W8A16 linear uses ``x`` (bf16/fp16) directly. The iu8 path uses ``a_int8``
    and ``x_scale``. If either is omitted, this calls ``quantize_int8_rowwise``.
    It does not raise for that omission.
    """
    require_gfx120x(x.device, what="int8_linear_auto (gfx120x size-dispatch)")
    if out_dtype is None:
        out_dtype = x.dtype
    # Env bypass: force_hip → pure PyTorch W8A16-style reference (no size gate).
    if force_kernel is None and get_dispatch_mode() == "force_hip":
        from kernels.common.gfx120x_torch_ref import int8_linear_torch

        return int8_linear_torch(
            x,
            weight,
            weight_scale,
            bias=bias,
            out_dtype=out_dtype,
        )
    path = select_int8_kernel_for_tensors(x, weight, m_a_max=m_a_max, mk_a_max=mk_a_max, force_kernel=force_kernel)
    if path == "w8a16":
        return int8_linear_dispatched(
            "w8a16",
            a_act=x if x.ndim == 2 else x.reshape(-1, x.shape[-1]),
            b_int8=weight,
            w_scale=weight_scale,
            bias=bias,
            out_dtype=out_dtype,
        )
    if a_int8 is None or x_scale is None:
        from kernels.quant.rdna4_quantize_int8_rowwise import quantize_int8_rowwise

        x2d = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
        a_int8, x_scale = quantize_int8_rowwise(x2d)
    return int8_linear_dispatched(
        "iu8",
        a_int8=a_int8,
        b_int8=weight,
        x_scale=x_scale,
        w_scale=weight_scale,
        bias=bias,
        out_dtype=out_dtype,
    )


__all__ = [
    "DEFAULT_K_IU8_MIN",
    "DEFAULT_M_A_MAX",
    "DEFAULT_MK_A_MAX",
    "KernelChoice",
    "select_int8_kernel",
    "select_int8_kernel_for_tensors",
    "gate_rule_doc",
    "int8_linear_dispatched",
    "int8_linear_auto",
]
