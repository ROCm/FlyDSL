# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x FA host helpers (not a dense-FA fallback; never gfx950 dualwave).

Routing guards and caller-``out`` handling for the gfx120x host.
Dense, packed varlen, paged KV (linear, linear3d, vectorized), varlen+paged,
and split-K run in-kernel. This module does not gather paged KV, pad varlen,
or run a host online-softmax loop.
"""

import torch


def reject_varlen_with_paged(varlen: bool, paged: bool) -> None:
    """Packed Q + paged KV is in-kernel. Kept as a no-op for callers."""
    return None


def reject_gqa(num_q_heads: int, num_kv_heads_from_tensor: int | None, num_kv_heads: int | None = None) -> None:
    """Allow GQA/MQA when ``Hq % Hkv == 0`` (including MHA, ``Hq == Hkv``).

    The kernel indexes ``kv_head = q_head // (Hq // Hkv)``; this does not repeat
    KV on the host. Raise ``ValueError`` when ``Hq`` is not divisible by ``Hkv``,
    or when an explicit ``num_kv_heads`` disagrees with the K/V tensor head count.
    """
    hq = int(num_q_heads)
    if num_kv_heads_from_tensor is None:
        raise ValueError("gfx120x FA: KV head count is required (got None)")
    hkv = int(num_kv_heads_from_tensor)
    if num_kv_heads is not None and int(num_kv_heads) != hkv:
        raise ValueError(
            "gfx120x FA: num_kv_heads does not match the KV tensor "
            f"(num_kv_heads={int(num_kv_heads)} != kv heads={hkv})"
        )
    if hkv <= 0 or hq <= 0 or hq % hkv != 0:
        raise ValueError("gfx120x FA: num_heads must be divisible by num_kv_heads " f"(q heads={hq}, kv heads={hkv})")


def reject_quant_extras(
    dtype_label: str,
    *,
    bias: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_lse: bool = False,
) -> None:
    """fp8/int8 bias, ALiBi, sink, and LSE run in the quant kernel. This check is a no-op."""
    return None


def reject_paged_layout(kv_cache_layout: str | None) -> None:
    """Raise unless the paged KV layout is linear, linear3d, or vectorized.

    ``None`` / empty defaults to ``"linear"``.
    """
    layout = kv_cache_layout or "linear"
    if layout in ("linear", "linear3d", "vectorized"):
        return
    raise NotImplementedError(
        "gfx120x FA paged KV does not support " f"layout {layout!r} (linear, linear3d, vectorized only)"
    )


def lengths_uniform(lengths: torch.Tensor, max_len: int | None = None) -> bool:
    """True when every batch length is equal (optionally to ``max_len``).

    Uses one small metadata D2H (B ints) rather than a device reduction +
    ``.item()`` sync, so uniform/ragged routing stays host-side without a device reduction sync.
    When ``max_len`` is None, returns True iff all lengths equal each other.
    Missing lengths fail closed (raise). Empty lengths are uniform only when
    ``max_len`` is None or 0. Requires a flat 1-D length vector; negative
    lengths raise (uniform-but-invalid must not gate the dense path).
    """
    if lengths is None:
        raise ValueError("lengths_uniform: lengths is required")
    if lengths.dim() != 1:
        raise ValueError(f"lengths_uniform: lengths must be a flat 1-D vector, got shape {tuple(lengths.shape)}")
    if int(lengths.numel()) == 0:
        return max_len is None or int(max_len) == 0
    host = lengths.detach().to(device="cpu", dtype=torch.int64).tolist()
    if not host:
        return max_len is None or int(max_len) == 0
    if any(int(x) < 0 for x in host):
        raise ValueError(f"lengths_uniform: lengths must be >= 0, got min={min(int(x) for x in host)}")
    if max_len is None:
        first = int(host[0])
        return all(int(x) == first for x in host)
    ml = int(max_len)
    return all(int(x) == ml for x in host)


def cu_seqlens_from_seqlens(seqlens: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
    """Build int32 ``cu_seqlens`` ``[B+1]`` from per-batch lengths (device cumsum).

    Used when gfx120x dense-paged traffic with ragged ``seqlen_k`` auto-routes to
    the varlen-paged host (Q packed from BSHD; KV lengths from ``seqlen_k``).
    """
    if seqlens.dim() != 1:
        raise ValueError(f"cu_seqlens_from_seqlens: seqlens must be 1-D, got shape {tuple(seqlens.shape)}")
    dev = device if device is not None else seqlens.device
    flat = seqlens.to(device=dev, dtype=torch.int32).reshape(-1)
    if int(flat.numel()) == 0:
        return torch.zeros(1, device=dev, dtype=torch.int32)
    out = torch.empty(int(flat.numel()) + 1, device=dev, dtype=torch.int32)
    out[0] = 0
    out[1:] = flat.cumsum(0)
    return out


def reject_sink_with_alibi(sink: torch.Tensor | None, alibi_slopes: torch.Tensor | None) -> None:
    """Sink and ALiBi both apply inside one gfx120x launch (ALiBi via bias, sink in epilogue)."""
    return None


def reject_splitk_extras(*, alibi_slopes: torch.Tensor | None = None, return_lse: bool = False) -> None:
    """Split-K partials include bias/ALiBi; combine writes LSE. No longer rejected."""
    return None


def commit_caller_out(out: torch.Tensor | None, produced: torch.Tensor) -> torch.Tensor:
    """Return ``produced``, or copy it into caller-owned ``out`` and return that.

    ``out is None`` keeps the freshly allocated tensor. A provided buffer must
    match shape, dtype, and device; otherwise this raises instead of dropping
    the caller's storage (varlen pack and the per-head ALiBi loop both allocate).
    """
    if out is None:
        return produced
    if tuple(out.shape) != tuple(produced.shape) or out.dtype != produced.dtype or out.device != produced.device:
        raise ValueError(
            "flydsl_flash_attn_func: out shape/dtype/device mismatch: "
            f"got {tuple(out.shape)}/{out.dtype}/{out.device}, "
            f"want {tuple(produced.shape)}/{produced.dtype}/{produced.device}"
        )
    if out.data_ptr() != produced.data_ptr():
        out.copy_(produced)
    return out
