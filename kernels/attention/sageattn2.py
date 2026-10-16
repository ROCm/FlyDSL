# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""SageAttention2 host entry point.

``sageattn2`` is the public call. Quantization and the gfx120x prefill kernel
are internal. Long sequences may split into several launches over Q-head
groups so the KV working set fits in the GPU's last-level cache. The output
is the same as one launch.
"""

from __future__ import annotations

import math

import torch

from flydsl.runtime.device import get_llc_cache_bytes, get_rocm_arch
from kernels.attention.sageattn2_quant import _quant_qk_int4, _quant_v_fp8
from kernels.attention.sageattn2_rdna4 import CTA_Q, _sageattn2_rdna4_prefill

_ALLOWED_D = (64, 128)
_ALLOWED_DTYPES = (torch.float16, torch.bfloat16)
_LAYOUTS = ("HND", "NHD")
_KV_ALIGN = 16
_LLC_WORKING_SET_RATIO = 1.35
# Group only when KV is well above the LLC. Around 3x LLC, grouping was a
# net loss end to end; around 10x it won. 5x sits between those.
_ENABLE_LLC_RATIO = 5.0
# A grouped launch must cover at least this many multiples of the CU count,
# or a short query grid under-fills the GPU.
_MIN_BLOCKS_PER_LAUNCH_CUS = 4
_FORCE_GROUP_SIZE: int | None = None


def _require_arch(arch: str) -> None:
    """gfx120x only. Host gate uses the arch string, not an MLIR type."""
    if arch.startswith("gfx120"):
        return
    raise RuntimeError(f"sageattn2 requires gfx120x, got {arch!r}")


def _normalize_layout(t: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    """Return a contiguous ``[B, H, S, D]`` view of ``t``.

    The explicit ``contiguous()`` is not redundant for ``NHD``: after
    ``transpose(1, 2)`` the last stride is still 1, so G2's
    ``_maybe_contiguous`` (``sageattn2_quant.py:66-70``) will not fire and the
    copy would happen implicitly inside ``reshape`` instead. Doing it here
    makes the cost visible at the layer that accepted ``NHD``, and guarantees
    it happens once. ``HND`` input is already contiguous, so this is a no-op.
    """
    if tensor_layout == "NHD":
        t = t.transpose(1, 2)
    return t.contiguous()


def _validate_inputs(q, k, v, tensor_layout, out_dtype, sm_scale) -> None:
    """Layout-independent checks. Runs before ``_normalize_layout``."""
    if tensor_layout not in _LAYOUTS:
        raise ValueError(f"tensor_layout must be one of {_LAYOUTS}, got {tensor_layout!r}")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if t.dim() != 4:
            raise ValueError(f"{name} expected rank 4, got shape {tuple(t.shape)}")
        if t.dtype not in _ALLOWED_DTYPES:
            raise ValueError(f"{name} dtype must be fp16 or bf16, got {t.dtype}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q, k, v must share dtype; got {q.dtype}, {k.dtype}, {v.dtype}")
    if out_dtype is not None and out_dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"out_dtype must be float16 or bfloat16, got {out_dtype}")
    if sm_scale is not None and not math.isfinite(float(sm_scale)):
        raise ValueError(f"sm_scale must be finite, got {sm_scale}")


def _validate_shapes(q, k, v) -> None:
    """Shape agreement on already-normalized ``[B, H, S, D]`` tensors."""
    b, hq, _, d = q.shape
    bk, hkv, sk, dk = k.shape
    if d not in _ALLOWED_D:
        raise ValueError(f"head_dim must be 64 or 128, got {d}")
    if (bk, dk) != (b, d):
        raise ValueError(f"k must match q on (B, D); got k {tuple(k.shape)} vs q {tuple(q.shape)}")
    if tuple(v.shape) != (bk, hkv, sk, dk):
        raise ValueError(f"v shape {tuple(v.shape)} must equal k shape {tuple(k.shape)}")
    if hq % hkv != 0:
        raise ValueError(f"Hq ({hq}) must be a multiple of Hkv ({hkv})")


def _pad_kv(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Zero-pad ``k`` and ``v`` on the S axis to a multiple of ``_KV_ALIGN``.

    Quantization and prefill both require that alignment. A zero row cannot
    raise an amax, so the real rows quantize the same as they would unpadded.
    """
    sk = int(k.shape[2])
    skp = -(-sk // _KV_ALIGN) * _KV_ALIGN
    if skp == sk:
        return k, v, skp
    pad = skp - sk
    return (
        torch.nn.functional.pad(k, (0, 0, 0, pad)),
        torch.nn.functional.pad(v, (0, 0, 0, pad)),
        skp,
    )


def _kv_bytes_per_head(*, skp: int, d: int) -> int:
    """Quantized K/V bytes (K_i4 + k_scale + V_e4m3 + v_scale) for one KV head.

    Matches the tensor shapes in ``sageattn2_rdna4_prefill``'s docstring
    (`sageattn2_rdna4.py:327-332`): K_i4 [Skp, D/8] int32, k_scale [Skp] f32,
    V_e4m3 [D/16, Skp/16, 2, 16, 8] uint8, v_scale [D] f32.
    """
    k_i4_bytes = skp * (d // 8) * 4
    k_scale_bytes = skp * 4
    v_e4m3_bytes = (d // 16) * (skp // 16) * 2 * 16 * 8
    v_scale_bytes = d * 4
    return k_i4_bytes + k_scale_bytes + v_e4m3_bytes + v_scale_bytes


def _head_group_size(
    *,
    hq: int,
    hkv: int,
    skp: int,
    d: int,
    b: int,
    is_causal: bool,
    llc_bytes: int,
    grid_x: int,
    multi_processor_count: int,
) -> int:
    """Q heads per prefill launch; ``hq`` means one launch, i.e. no grouping.

    Grouping is only for the measured case: one batch, MHA, non-causal, known
    LLC size. GQA slicing would mis-pair heads with KV, so that path stays one
    launch.

    ``grid_x`` and ``multi_processor_count`` feed an occupancy guard (final
    review fix wave, "occupancy guard"): the byte-based sizing above never
    looked at ``Sq`` or the GPU's CU count, so a short-Sq/long-Sk shape (e.g.
    chunked prefill) could still group and block-starve the GPU. Pure
    function still — ints in, int out, no tensors, no device calls.
    """
    if _FORCE_GROUP_SIZE is not None:
        return max(1, min(_FORCE_GROUP_SIZE, hq))
    if llc_bytes <= 0 or b != 1 or hq != hkv or is_causal:
        return hq

    kv_bytes = _kv_bytes_per_head(skp=skp, d=d)
    total_kv = hkv * kv_bytes
    enable_target = _ENABLE_LLC_RATIO * llc_bytes
    if total_kv <= enable_target:
        return hq
    target = _LLC_WORKING_SET_RATIO * llc_bytes

    group_size = round(target / kv_bytes)
    group_size = max(1, min(group_size, hq))
    if grid_x * group_size < _MIN_BLOCKS_PER_LAUNCH_CUS * multi_processor_count:
        return hq
    if group_size >= hq:
        return hq
    return group_size


def sageattn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: float | None = None,
    smooth_k: bool = True,
    out_dtype: torch.dtype | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """SageAttention2 dense prefill on gfx120x.

    INT4 QK + FP8 (e4m3) PV on gfx120x. The result is approximate. Cosine
    versus a 16-key online-softmax reference stays at or above 0.999. On a
    short sequence (about one 64-key stage) relative L2 is about 1–3%, because
    the running max updates once per stage before FP8 quantization. At long
    sequences (thousands of stages) relative L2 is about 0.1%.

    q, k, v  fp16 or bf16, all three the same dtype:
        tensor_layout="HND"  q [B, Hq, Sq, D], k/v [B, Hkv, Sk, D]
        tensor_layout="NHD"  q [B, Sq, Hq, D], k/v [B, Sk, Hkv, D]
    D must be 64 or 128; Hq must be a multiple of Hkv; Sq and Sk are free.

    sm_scale   defaults to 1/sqrt(D).
    smooth_k   subtract the per-(b, h, d) mean of K before quantizing. This
               leaves the output mathematically unchanged (it shifts every
               score in a query row by the same constant) and only reduces
               INT4 quantization error. Default True, matching upstream
               SageAttention and ``sageattn2_ref``.
    out_dtype  defaults to ``q.dtype``; fp16 or bf16.

    Returns O in the same ``tensor_layout`` as the input. Under "NHD" the
    result is a ``transpose(1, 2)`` view and is therefore **not contiguous**.

    Raises on any arch other than gfx120x — there is no SDPA fallback.
    """
    _require_arch(str(get_rocm_arch() or ""))
    _validate_inputs(q, k, v, tensor_layout, out_dtype, sm_scale)

    qn = _normalize_layout(q, tensor_layout)
    kn = _normalize_layout(k, tensor_layout)
    vn = _normalize_layout(v, tensor_layout)
    _validate_shapes(qn, kn, vn)

    sk = int(kn.shape[2])
    d = int(qn.shape[3])
    scale = (1.0 / math.sqrt(d)) if sm_scale is None else float(sm_scale)
    out_dt = qn.dtype if out_dtype is None else out_dtype
    stream = torch.cuda.current_stream() if stream is None else stream

    with torch.cuda.stream(stream):
        # Mean comes from the unpadded K. Padding first would pull the mean
        # toward zero.
        km = kn.float().mean(dim=2, keepdim=True) if smooth_k else None
        k_pad, v_pad, skp = _pad_kv(kn, vn)
        q_i4, k_i4, q_scale, k_scale = _quant_qk_int4(qn, k_pad, km=km, stream=stream)
        v_e4m3, v_scale = _quant_v_fp8(v_pad, stream=stream)

        b, hq, sq, d = (int(x) for x in qn.shape)
        hkv = int(kn.shape[1])
        device_props = torch.cuda.get_device_properties(qn.device)
        llc_bytes = get_llc_cache_bytes(pci_bus_id=device_props.pci_bus_id)
        grid_x = -(-sq // CTA_Q)
        group_size = _head_group_size(
            hq=hq,
            hkv=hkv,
            skp=skp,
            d=d,
            b=b,
            is_causal=is_causal,
            llc_bytes=llc_bytes,
            grid_x=grid_x,
            multi_processor_count=device_props.multi_processor_count,
        )

        if group_size >= hq:
            # One launch. Return that tensor directly, without an extra copy.
            out = _sageattn2_rdna4_prefill(
                q_i4,
                q_scale,
                k_i4,
                k_scale,
                v_e4m3,
                v_scale,
                seq_len_k=sk,
                is_causal=is_causal,
                sm_scale=scale,
                out_dtype=out_dt,
                stream=stream,
            )
        else:
            # Several launches over head groups. Per-head arithmetic is unchanged.
            n_groups = -(-hq // group_size)
            out = torch.empty(b, hq, sq, d, dtype=out_dt, device=q_i4.device)
            for g in range(n_groups):
                h0 = g * group_size
                h1 = min(h0 + group_size, hq)
                out[:, h0:h1].copy_(
                    _sageattn2_rdna4_prefill(
                        q_i4[:, h0:h1],
                        q_scale[:, h0:h1],
                        k_i4[:, h0:h1],
                        k_scale[:, h0:h1],
                        v_e4m3[:, h0:h1],
                        v_scale[:, h0:h1],
                        seq_len_k=sk,
                        is_causal=is_causal,
                        sm_scale=scale,
                        out_dtype=out_dt,
                        stream=stream,
                    )
                )
    return out.transpose(1, 2) if tensor_layout == "NHD" else out
