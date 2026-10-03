# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X Asym W4A8 (grouped INT4 weights + INT8 acts via ConvRot).

Wire format (``AsymW4A8Int8Layout`` / HIP ``w4a8_int8_*``):

* ``qdata[N, K//2]`` int8 -- **unsigned** nibble pack: low = even col, high = odd
  (``(u0 & 0xF) | ((u1 & 0xF) << 4)``); indices in ``[0, 15]``.
* ``s_rel[N, K//group_size]`` -- per-group relative scale (often fp8 e4m3fn).
* ``s_channel[N]`` float32 -- per-row channel scale (``amax/127`` of shifted W).
* Optional ``codebook[16]`` float32 Lloyd-Max levels; when present, decode
  ``values = codebook[idx]``, else ``values = idx - 8``.
* Optional ``correction[groups, N]`` for asymmetric (``symmetric=False``); when
  set, linear falls back to full dequant + fp linear (same as reference).
* Weights are ConvRot-rotated offline (``convrot_groupsize`` in {16,64,256});
  activations use online ConvRot + INT8 before iu8 GEMM.

Quant (codebook ALS / kurtosis gate) stays a **torch host** path matching
eager reference. Hot inference path: FlyDSL ``dequant_int4_grouped_to_int8`` ->
``int8_linear_convrot``.

"""

import math
from collections.abc import Callable
from functools import lru_cache

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_argument import PointerJitArg
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec
from kernels.common.gfx120x_arch import require_gfx120x
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, ptr_buf_tensor

# Default Lloyd-Max table for group-normalized Gaussian (eager w4a8).
_FIXED_LUT = (
    -0.980602,
    -0.794529,
    -0.638165,
    -0.500986,
    -0.377321,
    -0.263187,
    -0.155210,
    -0.050720,
    0.052541,
    0.156985,
    0.265284,
    0.379533,
    0.502636,
    0.638953,
    0.794876,
    0.980671,
)
_W4A8_GATE_KURTOSIS = -0.1
_KURTOSIS_SAMPLE = 1 << 19
_ALS_ITERS = 2
_SUPPORTED_CONVROT = (16, 64, 256)
_MAX_K = 8192
KERNEL_NAME = "w4a8_dequant_int4_to_int8_gfx120x"
WARP = 32


def _kernel_signature(**params: object) -> str:
    return "_".join(
        f"{name}{int(value) if isinstance(value, bool) else value}" for name, value in params.items()
    ).replace("-", "_")


def _block_threads(k: int) -> int:
    from kernels.common.gfx120x_autotune_tables import pick_asym_w4a8_block_threads

    return pick_asym_w4a8_block_threads(k)


def _ptr(t: torch.Tensor) -> PointerJitArg:
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


@lru_cache(maxsize=64)
def build_w4a8_dequant_int4_to_int8_module(
    k: int,
    group_size: int = 16,
    use_codebook: bool = True,
    block_threads: int | None = None,
) -> Callable[..., None]:
    """Decode packed unsigned INT4 (+ optional codebook) × s_rel → INT8 grid."""
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"K={k} must be positive even")
    if group_size < 4 or k % group_size != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size} (>=4)")
    if k > _MAX_K:
        raise ValueError(f"K={k} exceeds budget {_MAX_K}")

    if block_threads is None:
        block_threads = _block_threads(k)
    if block_threads > WARP and block_threads % WARP != 0:
        raise ValueError(f"block_threads={block_threads} must be multiple of {WARP}")

    packed_k = k // 2
    groups = k // group_size
    pack_steps = (packed_k + block_threads - 1) // block_threads
    sig = _kernel_signature(
        block=block_threads,
        group_size=group_size,
        k=k,
        codebook=use_codebook,
        op="w4a8_dequant",
    )

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def dequant_kernel(
        Qdata: fx.Pointer,
        SRel: fx.Pointer,
        Codebook: fx.Pointer,
        Out: fx.Pointer,
        n_rows: fx.Int32,
        K: fx.Int32,
    ) -> None:
        bid = fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        active_row = bid < n_rows

        # Optional codebook[16] into registers via LDS.
        alloc = fx.SharedAllocator()
        cb_bytes = 16 * 4
        cb_base = alloc.allocate(cb_bytes)._ptr
        cb_f = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, cb_base.address_space), cb_base)

        if const_expr(use_codebook):
            if tid < fx.Int32(16):
                cb_buf = ptr_buf_tensor(
                    Codebook,
                    elem=fx.Float32,
                    n=0x3FFFFFFF,
                    unit_elems=1,
                    num_records_bytes=fx.Int64(16 * 4),
                )
                v = buf_copy_load(cb_buf, fx.Int64(tid), elem=fx.Float32, unit_elems=1)
                fx.make_view(fx.add_offset(cb_f, fx.Int64(tid)), fx.make_layout(1, 1)).store(
                    Vec.from_elements([fx.Float32(v)], fx.Float32)
                )
            gpu.barrier()

        q_buf = ptr_buf_tensor(
            Qdata,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(packed_k),
        )
        s_buf = ptr_buf_tensor(
            SRel,
            elem=fx.Float32,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(groups) * fx.Int64(4),
        )
        o_buf = ptr_buf_tensor(
            Out,
            elem=fx.Int8,
            n=0x3FFFFFFF,
            unit_elems=1,
            num_records_bytes=fx.Int64(n_rows) * fx.Int64(K),
        )

        for step in range_constexpr(pack_steps):
            pc = tid + fx.Int32(step * block_threads)
            inb = active_row & (pc < fx.Int32(packed_k))
            gidx = fx.Int64(bid) * fx.Int64(packed_k) + fx.Int64(pc)
            safe_g = inb.select(gidx, fx.Int64(0))
            raw = buf_copy_load(q_buf, safe_g, elem=fx.Int8, unit_elems=1)
            packed = fx.Int32(raw) & fx.Int32(0xFF)
            lo = packed & fx.Int32(0xF)
            hi = (packed >> fx.Int32(4)) & fx.Int32(0xF)

            if const_expr(use_codebook):
                lo_v = fx.Float32(fx.make_view(fx.add_offset(cb_f, fx.Int64(lo)), fx.make_layout(1, 1)).load()[0])
                hi_v = fx.Float32(fx.make_view(fx.add_offset(cb_f, fx.Int64(hi)), fx.make_layout(1, 1)).load()[0])
            else:
                lo_v = fx.Float32(lo) - fx.Float32(8.0)
                hi_v = fx.Float32(hi) - fx.Float32(8.0)

            # Both nibbles share a group when group_size >= 2 (layout constraint).
            col0 = pc * fx.Int32(2)
            g = col0 // fx.Int32(group_size)
            sidx = fx.Int64(bid) * fx.Int64(groups) + fx.Int64(g)
            safe_s = inb.select(sidx, fx.Int64(0))
            s = fx.Float32(buf_copy_load(s_buf, safe_s, elem=fx.Float32, unit_elems=1))

            q0f = fmath.roundeven(lo_v * s)
            q1f = fmath.roundeven(hi_v * s)
            q0f = fx.max(fx.min(q0f, fx.Float32(127.0)), fx.Float32(-127.0))
            q1f = fx.max(fx.min(q1f, fx.Float32(127.0)), fx.Float32(-127.0))
            q0 = q0f.to(fx.Int8)
            q1 = q1f.to(fx.Int8)

            o0 = fx.Int64(bid) * fx.Int64(K) + fx.Int64(col0)
            o1 = o0 + fx.Int64(1)
            if inb:
                buf_copy_store(o_buf, o0, q0, elem=fx.Int8, unit_elems=1)
                buf_copy_store(o_buf, o1, q1, elem=fx.Int8, unit_elems=1)

    dequant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        Qdata: fx.Pointer,
        SRel: fx.Pointer,
        Codebook: fx.Pointer,
        Out: fx.Pointer,
        n_rows: fx.Int32,
        K: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        dequant_kernel(Qdata, SRel, Codebook, Out, n_rows, K).launch(
            grid=(n_rows, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


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


def rotate_convrot_weight(weight: torch.Tensor, convrot_groupsize: int = 256) -> torch.Tensor:
    """``W @ H`` per ConvRot group (H symmetric)."""
    import torch

    n, k = weight.shape
    if k % convrot_groupsize != 0:
        raise ValueError(f"K={k} not divisible by convrot_groupsize={convrot_groupsize}")
    h = _build_hadamard(convrot_groupsize, weight.device, weight.dtype)
    n_groups = k // convrot_groupsize
    wg = weight.reshape(n, n_groups, convrot_groupsize)
    return torch.matmul(wg, h.T).reshape(n, k)


def validate_w4a8_weight_shape(weight: torch.Tensor, group_size: int, convrot_groupsize: int) -> None:
    """Raise unless the packed W4A8 weight matches N, K, and group size."""
    if weight.dim() != 2:
        raise ValueError(f"W4A8 weight must be 2D, got shape {tuple(weight.shape)}")
    k = weight.shape[1]
    if (
        k % 16 != 0
        or k % group_size != 0
        or k % convrot_groupsize != 0
        or group_size < 4
        or (16 % group_size != 0 and group_size % 16 != 0)
    ):
        raise ValueError(
            f"K={k} must be divisible by 16, group_size={group_size}, and "
            f"convrot_groupsize={convrot_groupsize}; group_size must be >=4 "
            f"and divide 16 or be a multiple of 16"
        )


def _codebook_for(normalized: object) -> torch.Tensor:
    """Frozen LUT unless heavy-tailed (eager reference kurtosis gate)."""
    import torch

    x = normalized.detach().flatten()
    if x.numel() > _KURTOSIS_SAMPLE:
        gen = torch.Generator(device=x.device).manual_seed(0)
        x = x[torch.randint(0, x.numel(), (_KURTOSIS_SAMPLE,), device=x.device, generator=gen)]
    x = x.float()
    excess = ((x - x.mean()) / (x.std() + 1e-9)).pow(4).mean() - 3.0
    if excess.item() <= _W4A8_GATE_KURTOSIS:
        return torch.tensor(_FIXED_LUT, device=normalized.device, dtype=torch.float32)
    # Rare path: quantile init + Lloyd-Max (eager reference).
    samples = x
    sample_size = 300000
    if samples.numel() > sample_size:
        gen = torch.Generator(device=samples.device).manual_seed(0)
        idx = torch.randint(0, samples.numel(), (sample_size,), device=samples.device, generator=gen)
        samples = samples[idx]
    codebook = torch.quantile(samples, torch.linspace(0, 1, 16, device=samples.device))
    for _ in range(25):
        assignments = (samples.unsqueeze(-1) - codebook).abs().argmin(-1)
        updated = codebook.clone()
        for index in range(16):
            selected = assignments == index
            if selected.any():
                updated[index] = samples[selected].mean()
        codebook = updated
    return codebook.contiguous()


def _assign_codes(normalized: object, codebook: object) -> torch.Tensor:
    import torch

    last = codebook.numel() - 1
    pos = torch.searchsorted(codebook, normalized.contiguous())
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = (normalized - codebook[lo]).abs()
    dhi = (normalized - codebook[hi]).abs()
    return torch.where(dhi < dlo, hi, lo).to(torch.int32)


def _assign_grid(weight: torch.Tensor, levels: object, s_channel: object, stochastic_rounding: int = 0) -> torch.Tensor:
    import torch

    n, groups, gsize = weight.shape
    last = levels.shape[-1] - 1
    lv = levels.reshape(n * groups, last + 1).contiguous()
    tg = (weight / s_channel.view(-1, 1, 1)).reshape(n * groups, gsize).contiguous()
    pos = torch.searchsorted(lv, tg)
    if stochastic_rounding > 0:
        raise ValueError("FlyDSL host quant path: stochastic_rounding not supported")
    lo = (pos - 1).clamp(0, last)
    hi = pos.clamp(0, last)
    dlo = tg.sub(torch.gather(lv, 1, lo)).abs()
    dhi = tg.sub(torch.gather(lv, 1, hi)).abs()
    return torch.where(dhi < dlo, hi, lo).to(torch.int32).reshape(n, groups, gsize)


def _quantize_rotated_w4a8(
    weight: torch.Tensor,
    group_size: int = 16,
    symmetric: bool = True,
    scale_dtype: object = None,
    codebook: bool = True,
    codebook_override: object = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eager-compatible rotated W4A8 pack (host)."""
    import torch

    if scale_dtype is None:
        scale_dtype = torch.float8_e4m3fn
    if scale_dtype not in (torch.float32, torch.float8_e4m3fn):
        raise ValueError(f"scale_dtype must be float32 or float8_e4m3fn, got {scale_dtype}")

    n, k = weight.shape
    groups = k // group_size
    grouped = weight.float().view(n, groups, group_size)

    codebook_tensor = None
    if symmetric and codebook:
        group_scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = grouped / group_scale
        codebook_tensor = codebook_override if codebook_override is not None else _codebook_for(normalized)
        quantized = _assign_codes(normalized, codebook_tensor)
        for _ in range(_ALS_ITERS):
            qc = codebook_tensor[quantized]
            group_scale = (
                (grouped * qc).sum(-1, keepdim=True) / (qc * qc).sum(-1, keepdim=True).clamp(min=1e-8)
            ).clamp(min=1e-8)
            quantized = _assign_codes(grouped / group_scale, codebook_tensor)
        unsigned = quantized.to(torch.int32).view(n, k)
        shifted = codebook_tensor[quantized] * group_scale
        correction = None
    elif symmetric:
        group_scale = (grouped.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)
        signed = torch.round(grouped / group_scale).clamp(-8, 7).to(torch.int32)
        unsigned = (signed + 8).view(n, k)
        shifted = signed * group_scale
        correction = None
    else:
        minimum = grouped.amin(dim=-1, keepdim=True)
        group_scale = ((grouped.amax(dim=-1, keepdim=True) - minimum) / 15.0).clamp(min=1e-8)
        unsigned = torch.round((grouped - minimum) / group_scale).clamp(0, 15).to(torch.int32).view(n, k)
        shifted = (unsigned.view(n, groups, group_size) - 8) * group_scale
        correction = (8.0 * group_scale + minimum).squeeze(-1).t().contiguous().to(weight.dtype)

    s_channel = (shifted.abs().amax(dim=(1, 2)) / 127.0).clamp(min=1e-8)
    s_rel = (group_scale.squeeze(-1) / s_channel.unsqueeze(1)).float().contiguous()
    if scale_dtype != torch.float32:
        s_rel = s_rel.to(scale_dtype).contiguous()
    if codebook_tensor is not None:
        levels = (codebook_tensor.view(1, 1, 16) * s_rel.float().unsqueeze(-1)).round_().clamp_(-127, 127)
        unsigned = _assign_grid(grouped, levels, s_channel, 0).view(n, k)

    packed = ((unsigned[:, 0::2] & 0xF) | ((unsigned[:, 1::2] & 0xF) << 4)).to(torch.int8).contiguous()
    return packed, s_rel, s_channel.float().contiguous(), correction, codebook_tensor


def quantize_w4a8_int8_weight(
    weight: torch.Tensor,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    symmetric: bool = True,
    scale_dtype: object = None,
    codebook: bool = True,
    codebook_tensor: object = None,
    stochastic_rounding: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate + pack W4A8 (torch host; reference-eager wire-compatible).

    Codebook ALS / kurtosis gate matches eager reference. Stochastic rounding is
    rejected on this FlyDSL host path (use HIP for LoRA SR requant).
    """
    import torch

    require_gfx120x(weight.device, what="quantize_w4a8_int8_weight (gfx120x)")
    if scale_dtype is None:
        scale_dtype = torch.float8_e4m3fn
    if stochastic_rounding:
        raise ValueError("quantize_w4a8_int8_weight FlyDSL host path: no stochastic_rounding")
    if convrot_groupsize not in _SUPPORTED_CONVROT:
        raise ValueError(f"convrot_groupsize must be one of {_SUPPORTED_CONVROT}")
    validate_w4a8_weight_shape(weight, group_size, convrot_groupsize)
    rotated = rotate_convrot_weight(weight.contiguous(), convrot_groupsize)
    return _quantize_rotated_w4a8(
        rotated,
        group_size=group_size,
        symmetric=symmetric,
        scale_dtype=scale_dtype,
        codebook=codebook,
        codebook_override=codebook_tensor,
    )


def reference_dequant_int4_grouped_to_int8(
    qdata: object, s_rel: object, codebook: object, group_size: int = 16
) -> torch.Tensor:
    """Torch reference for packed INT4 → INT8 grid (eager reference)."""
    import torch

    n, k_half = qdata.shape
    k = k_half * 2
    groups = k // group_size
    packed = qdata.to(torch.int32) & 0xFF
    quantized = torch.empty(n, k, dtype=torch.int32, device=qdata.device)
    quantized[:, 0::2] = packed & 0xF
    quantized[:, 1::2] = (packed >> 4) & 0xF
    if codebook is not None:
        values = codebook.to(device=qdata.device, dtype=torch.float32)[quantized]
    else:
        values = quantized.float() - 8.0
    values = values.view(n, groups, group_size) * s_rel.float().unsqueeze(-1)
    return values.view(n, k).round().clamp(-127, 127).to(torch.int8)


def dequant_int4_grouped_to_int8(
    qdata: object,
    s_rel: object,
    codebook: object = None,
    group_size: int = 16,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """FlyDSL decode of packed W4A8 → INT8 GEMM grid."""
    require_gfx120x(qdata.device, what="dequant_int4_grouped_to_int8 (gfx120x)")
    import torch

    from kernels.common.tensor_shim import _run_compiled

    if qdata.dim() != 2 or qdata.dtype != torch.int8:
        raise ValueError("qdata must be 2D int8")
    if qdata.device.type != "cuda":
        raise ValueError("qdata must be on CUDA/ROCm")
    n, k_half = qdata.shape
    k = k_half * 2
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    groups = k // group_size
    s = s_rel
    if s.dtype == torch.float8_e4m3fn:
        s = s.float()
    elif s.dtype != torch.float32:
        s = s.float()
    if tuple(s.shape) != (n, groups):
        raise ValueError(f"s_rel must have shape {(n, groups)}, got {tuple(s.shape)}")
    s = s.contiguous()
    use_cb = codebook is not None
    if use_cb:
        cb = codebook.to(device=qdata.device, dtype=torch.float32).reshape(16).contiguous()
        if cb.numel() != 16:
            raise ValueError(f"codebook must have 16 entries, got {cb.numel()}")
    else:
        # Dummy buffer; kernel ignores when use_codebook=False.
        cb = torch.zeros(16, device=qdata.device, dtype=torch.float32)

    out = torch.empty((n, k), device=qdata.device, dtype=torch.int8)
    launch = build_w4a8_dequant_int4_to_int8_module(k=k, group_size=group_size, use_codebook=use_cb)
    if stream is None:
        stream = torch.cuda.current_stream()
    _run_compiled(
        launch,
        _ptr(qdata.contiguous()),
        _ptr(s),
        _ptr(cb),
        _ptr(out),
        n,
        k,
        stream,
    )
    return out


def dequantize_w4a8_int8_weight(
    qdata: object,
    s_rel: object,
    s_channel: object,
    codebook: object = None,
    correction: object = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    output_dtype: object = None,
) -> torch.Tensor:
    """Full W4A8 → floating weight in original basis (host un-rotate)."""
    import torch

    if output_dtype is None:
        output_dtype = torch.bfloat16
    int8_w = dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size=group_size)
    n, k = int8_w.shape
    groups = k // group_size
    wr = int8_w.float().view(n, groups, group_size)
    wr = wr * s_channel.float().view(n, 1, 1)
    if correction is not None:
        wr = wr + correction.t().unsqueeze(-1).float()
    wr = wr.view(n, k)
    return rotate_convrot_weight(wr.to(output_dtype), convrot_groupsize).to(output_dtype)


def w4a8_int8_linear(
    x: object,
    qdata: object,
    s_rel: object,
    s_channel: object,
    codebook: object = None,
    correction: object = None,
    bias: torch.Tensor | None = None,
    group_size: int = 16,
    convrot_groupsize: int = 256,
    out_dtype: torch.dtype | None = None,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """``x @ W.T + bias`` via INT4→INT8 decode + ConvRot INT8 linear.

    Requires gfx120x iu8 WMMA (``rdna4_int8_linear``). Asymmetric ``correction``
    falls back to full dequant + ``F.linear`` (same as reference).
    """
    require_gfx120x(x.device, what="w4a8_int8_linear (gfx120x)")
    import torch

    from kernels.quant.rdna4_int8_convrot import int8_linear_convrot

    if out_dtype is None:
        out_dtype = x.dtype
    if x.shape[-1] != qdata.shape[-1] * 2:
        raise ValueError(f"Input K={x.shape[-1]} does not match qdata K={qdata.shape[-1] * 2}")
    if correction is not None:
        weight = dequantize_w4a8_int8_weight(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook,
            correction=correction,
            group_size=group_size,
            convrot_groupsize=convrot_groupsize,
            output_dtype=x.dtype,
        )
        return torch.nn.functional.linear(x, weight, bias).to(out_dtype)

    int8_w = dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size=group_size, stream=stream)
    return int8_linear_convrot(
        x,
        int8_w,
        s_channel,
        group_size=convrot_groupsize,
        bias=bias,
        out_dtype=out_dtype,
        stream=stream,
    )
