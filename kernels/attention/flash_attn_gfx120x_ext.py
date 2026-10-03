# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x FA host helpers (not a dense-FA fallback; never gfx950 dualwave).

Routing guards, bias slicing, and LSE/sink folding for the gfx120x host.
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
    """Dense fp8/int8 apply bias, ALiBi, sink, and LSE in-kernel.

    Varlen, paged, and split-K stay rejected by the caller: those ABIs are not
    in the quantized WMMA kernel, and dequant-then-bf16 would drop the contract.
    """
    return None


def reject_paged_layout(kv_cache_layout: str) -> None:
    """Raise unless the paged KV layout is linear, linear3d, or vectorized."""
    layout = kv_cache_layout or "linear"
    if layout in ("linear", "linear3d", "vectorized"):
        return
    raise NotImplementedError(
        "gfx120x FA paged KV does not support " f"layout {layout!r} (linear, linear3d, vectorized only)"
    )


def lengths_uniform(lengths: torch.Tensor, max_len: int) -> bool:
    """True when every batch length already equals ``max_len`` (no ragged pad)."""
    if lengths is None or int(lengths.numel()) == 0:
        return True
    return bool(torch.all(lengths.to(torch.int64) == int(max_len)).item())


def varlen_batch_bias(
    bias: torch.Tensor | None,
    cu_seqlens_q: torch.Tensor,
    batch_index: int,
    sq: int,
    sk: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
) -> torch.Tensor | None:
    """Bias tile for one packed-varlen batch on the gfx120x dense host.

    ``None`` stays ``None``. A shared dense mask ``[max_seqlen_q, max_seqlen_kv]``
    contributes the local prefix ``[:sq, :sk]``. The interface packed layout
    ``[total_q, max_seqlen_kv]`` contributes rows ``cu[b]:cu[b+1]`` and columns
    ``[:sk]``. Any other rank is rejected (not dropped).
    """
    if bias is None:
        return None
    if bias.dim() != 2:
        raise NotImplementedError(
            "gfx120x varlen FA bias must be 2D shared [max_seqlen_q, max_seqlen_kv] "
            f"or packed [total_q, max_seqlen_kv], got shape {tuple(bias.shape)}"
        )
    rows, cols = int(bias.shape[0]), int(bias.shape[1])
    sq, sk = int(sq), int(sk)
    if rows == int(max_seqlen_q) and cols == int(max_seqlen_kv):
        if sk > cols or sq > rows:
            raise ValueError(f"gfx120x varlen shared bias {tuple(bias.shape)} cannot cover sq={sq} sk={sk}")
        return bias[:sq, :sk]
    cq = cu_seqlens_q.to(torch.int64)
    qs = int(cq[batch_index])
    qe = int(cq[batch_index + 1])
    if qe - qs != sq:
        raise ValueError(f"gfx120x varlen cu_seqlens_q batch {batch_index} length {qe - qs} != sq={sq}")
    if rows < qe or cols < sk:
        raise ValueError(
            f"gfx120x varlen bias shape {tuple(bias.shape)} cannot cover "
            f"batch {batch_index} rows [{qs}:{qe}) cols [:{sk})"
        )
    return bias[qs:qe, :sk]


def slice_attn_mask_kv(attn_mask: torch.Tensor | None, ks: int, ke: int) -> torch.Tensor | None:
    """Slice ``attn_mask`` on the KV axis (last dimension)."""
    if attn_mask is None:
        return None
    if attn_mask.dim() < 1:
        raise ValueError(f"gfx120x FA attn_mask rank {attn_mask.dim()} has no KV axis")
    return attn_mask[..., ks:ke]


def mask_as_additive(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Bool masks become 0 / -inf. Numeric masks are unchanged."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        out = torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
        return out.masked_fill(~mask, float("-inf"))
    return mask


def fold_sink_lse(lse: torch.Tensor, sink: torch.Tensor | None) -> torch.Tensor:
    """Add a sink logit into a pre-sink LSE ``[B, H, Sq]``.

    ``lse_new = log(exp(lse) + exp(sink))``. ``sink`` is ``[H]`` or ``[B, H]``
    in the same post-scale logit space as ``lse``.
    """
    if sink is None or lse is None:
        return lse
    _b, h, _sq = lse.shape
    s = sink.detach().float()
    if s.dim() == 1:
        if int(s.numel()) != int(h):
            raise ValueError(f"sink must have H={int(h)} entries, got {int(s.numel())}")
        s = s.view(1, int(h), 1)
    elif s.dim() == 2:
        if tuple(s.shape) != (int(lse.shape[0]), int(h)):
            raise ValueError(f"sink shape {tuple(s.shape)} != {(int(lse.shape[0]), int(h))}")
        s = s.unsqueeze(-1)
    else:
        raise ValueError(f"sink must be [H] or [B, H], got {tuple(sink.shape)}")
    return torch.logaddexp(lse, s.to(device=lse.device, dtype=torch.float32))


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
