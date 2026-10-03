# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlyDSL Project Contributors
"""High-level FlyDSL Flash Attention APIs for gfx120x (RDNA4).

FlyDSL FlashAttention for the gfx120x family:
  - bf16 / fp16 dense self + non-causal cross (unequal Sq/Sk)
  - causal self (equal seqlens) and causal×cross via in-kernel bottom-right masking
  - FP8 e4m3fn (+ e5m2) with per-tensor descales; self + cross (+ causal×cross)
  - KV tile pad + in-kernel ``seq_len_kv_valid`` mask
  - adaptive BLOCK_M / waves_per_eu
  - head_dim in [64, 384], ``% 32 == 0`` (LDS ≤ ~48 KiB bf16 at D=384;
    RDNA4 workgroup LDS is 64 KiB; multi-batch ceil cover)
  - optional dense additive attn bias / general mask (fp32 [Sq, Skv]);
    common multi-rank masks (B,H,Sq,Skv)/(1,1,Sq,Skv)/(B,1,Sq,Skv) reduced when
    leading dims are broadcast-singleton or identical across B/H
  - noop mask detection on host (bool all-True / additive all-zero)
  - optional ``return_lse`` via in-kernel fp32 LSE epilogue ``[B,H,Sq]``
  - uniform ALiBi slopes folded into the additive bias; per-head-varying
    slopes fold to ``[H,Sq,Skv]`` and load by head inside one kernel launch
  - optional attention-sink folded into online softmax in-kernel (dense)
  - packed varlen (cu_seqlens) and paged KV (linear, linear3d, and vectorized) in-kernel (bf16/fp16)
  - split-K is an in-kernel partial (fp32 workspace) plus a wave32 combine

Int8 QKV uses ``flydsl_flash_attn_int8_func`` (iu8 WMMA); callers that pass
int8 into the bf16 entry get a clear ValueError.

Family name is gfx120x; hardware may report gfx1201/gfx1200/…. Native pack is
dense BSHD plus packed-varlen and paged KV (linear, linear3d, and vectorized). gfx950 dualwave
and gfx1250 paths stay on their own arches via ``flash_attn_interface``.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F

from kernels.common.gfx120x_arch import is_gfx120x, require_gfx120x

__all__ = [
    "flydsl_flash_attn_func",
    "flydsl_flash_attn_varlen_func",
    "flydsl_flash_attn_paged_func",
    "flydsl_flash_attn_varlen_paged_func",
    "flydsl_flash_attn_fp8_func",
    "flydsl_flash_attn_int8_func",
    "flydsl_flash_attn_iu4_func",
    "is_gfx120x",
    "normalize_attn_mask",
    "mask_is_noop",
    "bottom_right_causal_bias",
    "fold_alibi_to_bias",
]

_KERNEL_BLOCK_M = 128
_KERNEL_BLOCK_N = 32
_ADAPTIVE_BLOCK_M_CROSS = (16, 32, 64, 128)
_MAX_HEAD_DIM = 384  # LDS: D=384 @ prefetch1 ≈ 48.5 KiB bf16 < RDNA4 64 KiB
_MIN_HEAD_DIM = 64
_DUMMY_BIAS_CACHE: dict = {}
_DUMMY_I32_CACHE: dict = {}


def _as_contig(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.view(-1) if t.is_contiguous() else t.reshape(-1)


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_flash_attn_func only supports bf16/f16 for the dense path, got {dtype!r}")


def _pick_block_m(seq_len_q: int, cross: bool, seq_len_kv: int = 0) -> int:
    """Choose BLOCK_M. Soft-gap sweep 2026-09-24 (median on an otherwise idle GPU)."""
    if not cross:
        if seq_len_q <= 128:
            return 64
        return _KERNEL_BLOCK_M
    if seq_len_q <= 96:
        return 16
    if seq_len_q <= 128:
        return 32
    if seq_len_kv and seq_len_kv <= 128 and seq_len_q >= 512:
        return 64
    return _KERNEL_BLOCK_M


def _pick_waves_per_eu(seq_len_q: int, seq_len_kv: int, cross: bool, requested: int) -> int:
    if requested != 2:
        return requested
    if cross and seq_len_q <= 96 and seq_len_kv >= 512:
        return 4
    return requested


def _dummy_bias(device: torch.device) -> torch.Tensor:
    key = device.index if getattr(device, "index", None) is not None else int(device)
    buf = _DUMMY_BIAS_CACHE.get(key)
    if buf is None or buf.device != device:
        buf = torch.zeros(1, dtype=torch.float32, device=device)
        _DUMMY_BIAS_CACHE[key] = buf
    return buf


def _dummy_i32(device: torch.device) -> torch.Tensor:
    key = device.index if getattr(device, "index", None) is not None else int(device)
    buf = _DUMMY_I32_CACHE.get(key)
    if buf is None or buf.device != device:
        buf = torch.zeros(1, dtype=torch.int32, device=device)
        _DUMMY_I32_CACHE[key] = buf
    return buf


def mask_is_noop(mask: torch.Tensor | None) -> bool:
    """True when mask can be ignored (None / bool all-True / additive all-zero)."""
    if mask is None:
        return True
    try:
        if not hasattr(mask, "dtype"):
            return False
        if mask.numel() == 0:
            return True
        if mask.dtype == torch.bool:
            return bool(mask.all().item())
        return bool((mask == 0).all().item())
    except Exception:  # noqa: BLE001
        return False


def normalize_attn_mask(
    mask: torch.Tensor | None,
    seq_len_q: int,
    seq_len_kv: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize SDPA-style mask to dense fp32 additive bias [Sq, Skv].

    Returns None for noop masks. Bool False → -inf; True → 0. Additive masks
    are cast to fp32.

    Accepted ranks (reduced to ``[Sq, Skv]`` when leading dims broadcast):
      - ``[Sq, Skv]``
      - ``[1, Sq, Skv]`` / ``[1, 1, Sq, Skv]``
      - ``[B, 1, Sq, Skv]`` / ``[1, H, Sq, Skv]`` / ``[B, H, Sq, Skv]`` when
        every leading slice is identical (or singleton); otherwise raises so
        the caller can FALLBACK (per-batch / per-head unique masks need a
        different kernel ABI).

    Raises ValueError for shapes that cannot be reduced.
    """
    if mask is None or mask_is_noop(mask):
        return None
    m = mask
    if m.device != device:
        m = m.to(device)

    # Reduce leading dims → 2D [Sq, Skv].
    if m.dim() == 4:
        # (B, H, Sq, Skv) or broadcast variants.
        bsz, nh, sq, sk = m.shape
        if sq != seq_len_q or sk != seq_len_kv:
            raise ValueError(
                f"gfx120x FA attn_mask trailing dims {sq}x{sk} incompatible with "
                f"Sq={seq_len_q} Skv={seq_len_kv} (full shape={tuple(mask.shape)})"
            )
        flat = m.reshape(bsz * nh, sq, sk)
        if flat.shape[0] == 1:
            m = flat[0]
        else:
            # Require identical slices across B*H (broadcast intent).
            ref = flat[0]
            if not torch.equal(flat, ref.unsqueeze(0).expand_as(flat)):
                raise ValueError(
                    f"gfx120x FA attn_mask rank-4 shape {tuple(mask.shape)} has "
                    "non-uniform B/H slices; dense gfx120x bias is head/batch-shared "
                    "[Sq, Skv] only (FALLBACK or use gfx950 dualwave)"
                )
            m = ref
    elif m.dim() == 3:
        # (1, Sq, Skv) or (B, Sq, Skv) with identical rows, or (H, Sq, Skv).
        lead, sq, sk = m.shape
        if sq != seq_len_q or sk != seq_len_kv:
            # Maybe (B, H, Skv) style — not supported without Sq.
            raise ValueError(
                f"gfx120x FA attn_mask shape {tuple(mask.shape)} incompatible with Sq={seq_len_q} Skv={seq_len_kv}"
            )
        if lead == 1:
            m = m[0]
        else:
            ref = m[0]
            if not torch.equal(m, ref.unsqueeze(0).expand_as(m)):
                raise ValueError(
                    f"gfx120x FA attn_mask rank-3 shape {tuple(mask.shape)} has "
                    "non-uniform leading slices; need shared [Sq, Skv]"
                )
            m = ref
    elif m.dim() == 2:
        pass
    elif m.dim() > 4:
        # Squeeze leading singletons then retry once.
        while m.dim() > 4 and m.shape[0] == 1:
            m = m.squeeze(0)
        if m.dim() != 4 and m.dim() != 2:
            raise ValueError(f"gfx120x FA attn_mask must reduce to 2D [Sq, Skv], got shape={tuple(mask.shape)}")
        return normalize_attn_mask(m, seq_len_q, seq_len_kv, device)
    else:
        raise ValueError(f"gfx120x FA attn_mask must reduce to 2D [Sq, Skv], got shape={tuple(mask.shape)}")

    if m.dim() != 2:
        raise ValueError(f"gfx120x FA attn_mask must reduce to 2D [Sq, Skv], got shape={tuple(mask.shape)}")
    if m.shape[0] != seq_len_q or m.shape[1] != seq_len_kv:
        if m.shape[0] == 1 and m.shape[1] == seq_len_kv:
            m = m.expand(seq_len_q, seq_len_kv)
        elif m.shape[0] == seq_len_q and m.shape[1] == 1:
            m = m.expand(seq_len_q, seq_len_kv)
        else:
            raise ValueError(
                f"gfx120x FA attn_mask shape {tuple(m.shape)} incompatible with Sq={seq_len_q} Skv={seq_len_kv}"
            )
    if m.dtype == torch.bool:
        out = torch.zeros(m.shape, dtype=torch.float32, device=device)
        out = out.masked_fill(~m, float("-inf"))
        return _as_contig(out)
    return _as_contig(m.to(torch.float32))


def bottom_right_causal_bias(
    seq_len_q: int,
    seq_len_kv: int,
    device: torch.device,
) -> torch.Tensor:
    """Bottom-right-aligned causal additive bias ``[Sq, Skv]`` (0 / -inf).

    Matches FlashAttention: query ``i`` attends to keys ``j <= i + Skv - Sq``.
    The gfx120x host does not add this tensor. Causal and causal×cross
    masking run inside the kernel.
    """
    q_idx = torch.arange(seq_len_q, device=device, dtype=torch.int32)[:, None]
    k_idx = torch.arange(seq_len_kv, device=device, dtype=torch.int32)[None, :]
    # j <= i + (Skv - Sq)
    allow = k_idx <= (q_idx + (seq_len_kv - seq_len_q))
    bias = torch.zeros(seq_len_q, seq_len_kv, dtype=torch.float32, device=device)
    bias = bias.masked_fill(~allow, float("-inf"))
    return bias


def fold_alibi_to_bias(
    alibi_slopes: torch.Tensor | None,
    seq_len_q: int,
    seq_len_kv: int,
    device: torch.device,
) -> torch.Tensor:
    """Fold ALiBi slopes into additive bias (bottom-right aligned).

    ``-slope * |i + Skv - Sq - j|``. Scalar/uniform → ``[Sq, Skv]``;
    per-head-varying (1D ``[H]``) → ``[H, Sq, Skv]``.
    """
    if alibi_slopes is None:
        raise ValueError("fold_alibi_to_bias: alibi_slopes is None")
    q_idx = torch.arange(seq_len_q, device=device, dtype=torch.float32)[:, None]
    k_idx = torch.arange(seq_len_kv, device=device, dtype=torch.float32)[None, :]
    dist = (q_idx + (seq_len_kv - seq_len_q) - k_idx).abs()

    if isinstance(alibi_slopes, (int, float)):
        slope = float(alibi_slopes)
        if slope < 0:
            raise ValueError(f"fold_alibi_to_bias: slope must be >= 0, got {slope}")
        return (-slope * dist).contiguous()

    if not isinstance(alibi_slopes, torch.Tensor):
        raise TypeError(f"fold_alibi_to_bias: expected float or Tensor, got {type(alibi_slopes).__name__}")
    s = alibi_slopes.detach().float()
    if s.numel() == 0:
        raise ValueError("fold_alibi_to_bias: empty alibi_slopes")
    if (s < 0).any():
        raise ValueError("fold_alibi_to_bias: slopes must be >= 0")
    if s.dim() == 2:
        if not bool(torch.allclose(s, s[:1].expand_as(s))):
            raise ValueError(f"gfx120x FA: batch-varying alibi_slopes not supported (got {tuple(alibi_slopes.shape)})")
        s = s[0]
    s = s.reshape(-1)
    if s.numel() == 1 or bool(torch.allclose(s, s[:1].expand_as(s))):
        return (-float(s[0].item()) * dist).contiguous()
    return (-s[:, None, None] * dist[None, :, :]).contiguous()


def _merge_bias(*parts) -> Optional[torch.Tensor]:
    """Elementwise-add non-None fp32 bias tensors.

    Same-rank tensors must match shape. A shared 2D ``[Sq, Sk]`` may broadcast
    onto a per-head 3D ``[H, Sq, Sk]`` (added to every head).
    """
    acc = None
    for p in parts:
        if p is None:
            continue
        if acc is None:
            acc = p
            continue
        if acc.shape == p.shape:
            acc = acc + p
        elif acc.dim() == 2 and p.dim() == 3 and tuple(acc.shape) == tuple(p.shape[1:]):
            acc = p + acc.unsqueeze(0)
        elif acc.dim() == 3 and p.dim() == 2 and tuple(p.shape) == tuple(acc.shape[1:]):
            acc = acc + p.unsqueeze(0)
        else:
            raise ValueError(f"gfx120x FA cannot merge bias shapes {tuple(acc.shape)} and {tuple(p.shape)}")
    return acc


def _host_row_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool,
    bias: Optional[torch.Tensor],
    chunk_kv: int = 512,
) -> torch.Tensor:
    """Chunked host logsumexp of ``sm_scale * q@k^T (+ bias)`` → ``[B, H, Sq]``.

    Host reference for the natural-log, scale-folded LSE contract. Dense
    gfx120x launches write LSE from the kernel. Nothing in the tree calls
    this helper.
    """
    import math as _math

    B, Sq, H, D = q.shape
    Skv = k.shape[1]
    scale = 1.0 / _math.sqrt(D)
    qf = q.float().transpose(1, 2)  # B H Sq D
    kf = k.float().transpose(1, 2)  # B H Skv D
    lse = torch.full((B, H, Sq), float("-inf"), device=q.device, dtype=torch.float32)
    for start in range(0, Skv, chunk_kv):
        end = min(start + chunk_kv, Skv)
        scores = torch.matmul(qf, kf[:, :, start:end, :].transpose(-1, -2)) * scale
        if bias is not None:
            # Last axis is KV. 2D [Sq, Sk] and 3D [H, Sq, Sk] both slice that way;
            # slicing dim 0 of a rank-3 bias would cut heads instead of keys.
            scores = scores + bias[..., start:end].to(dtype=torch.float32)
        if causal:
            q_idx = torch.arange(Sq, device=q.device)[None, None, :, None]
            k_idx = torch.arange(start, end, device=q.device)[None, None, None, :]
            allow = k_idx <= (q_idx + (Skv - Sq))
            scores = scores.masked_fill(~allow, float("-inf"))
        lse = torch.logaddexp(lse, torch.logsumexp(scores, dim=-1))
    return lse


@lru_cache(maxsize=64)
def _get_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    waves_per_eu: int,
    daz: bool,
    block_m: int = 128,
    has_attn_bias: bool = False,
    has_per_head_bias: bool = False,
    return_lse: bool = False,
    has_sink: bool = False,
    varlen: bool = False,
    paged: bool = False,
    page_size: int = 16,
    num_kv_heads: int | None = None,
    kv_cache_layout: str = "linear",
) -> Callable[..., None]:
    from kernels.attention.flash_attn_gfx120x import build_flash_attn_func_module

    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        has_attn_bias=has_attn_bias,
        has_per_head_bias=has_per_head_bias,
        return_lse=return_lse,
        has_sink=has_sink,
        varlen=varlen,
        paged=paged,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        kv_cache_layout=kv_cache_layout,
    )


def _nosplit_args(device: torch.device) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    z = _flat(_dummy_bias(device))
    return (1, z, z, z, 0)


@lru_cache(maxsize=32)
def _get_splitk_combine(
    num_heads: int, head_dim: int, dtype_str: str, has_sink: bool, return_lse: bool
) -> Callable[..., None]:
    from kernels.attention.flash_attn_gfx120x_splitk import build_splitk_combine_module

    return build_splitk_combine_module(
        num_heads=num_heads,
        head_dim=head_dim,
        dtype_str=dtype_str,
        has_sink=has_sink,
        return_lse=return_lse,
    )


def flydsl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    out: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_lse: bool = False,
    sink: torch.Tensor | None = None,
    num_kv_splits: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run FlyDSL Flash Attention on RDNA4 (gfx120x family).

    Args:
        q, k, v: BSHD ``[B, S, H, D]`` bf16/fp16. Sq may differ from Sk (cross).
        causal: bottom-right aligned. Equal seqlens use the in-kernel causal
            path; unequal (causal×cross) uses in-kernel bottom-right masking.
        attn_mask / bias: optional general mask. Noop is ignored; otherwise
            normalized to fp32 additive ``[Sq, Skv]`` and applied in-kernel.
        alibi_slopes: optional uniform ALiBi slopes folded into the bias;
            per-head-varying slopes fold to [H,Sq,Skv] and load by head in one launch.
        return_lse: when True, return ``(out, lse)`` with fp32 ``[B, H, Sq]``
            written by the kernel epilogue (natural log, scale folded).
        sink: optional attention-sink logits ``[H]`` or ``[B,H]`` folded into
            online softmax in-kernel. Combinable with ALiBi (bias) in one launch.
        num_kv_splits: >1 runs split-K partials in this kernel plus a wave32 combine.
        head_dim: must be in ``[64, 384]`` and ``% 32 == 0``.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    if q.dtype in (torch.int8, torch.uint8):
        raise ValueError(
            "flydsl_flash_attn_func: int8 QKV belongs on flydsl_flash_attn_int8_func "
            "(iu8 WMMA + descales); bf16 path is bf16/fp16 only. "
            "Packed native int4 QKV → flydsl_flash_attn_iu4_func."
        )
    # Self-contained family gate at this gfx120x-only entry (shared FA soft-routes first).
    require_gfx120x(q.device, what="flydsl_flash_attn_func (gfx120x)")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"expected 4D BSHD tensors, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    if not (
        q.shape[0] == k.shape[0] == v.shape[0] and q.shape[3] == k.shape[3] == v.shape[3] and k.shape[2] == v.shape[2]
    ):
        raise ValueError(
            "flydsl_flash_attn_func: q/k/v must share batch and head_dim; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    num_kv_heads = int(k.shape[2])
    if int(q.shape[2]) % num_kv_heads != 0:
        raise ValueError(f"gfx120x FA: num_heads {int(q.shape[2])} must be divisible by num_kv_heads {num_kv_heads}")
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"flydsl_flash_attn_func: k/v seq_len must match, got k={k.shape[1]} v={v.shape[1]}")

    batch, seq_len_q_real, num_heads, head_dim = q.shape
    seq_len_kv_real = int(k.shape[1])
    cross = seq_len_q_real != seq_len_kv_real
    # Causal self + causal×cross: in-kernel bottom-right (host bias fold retired).
    causal_cross = False
    kernel_causal = bool(causal)
    if head_dim < _MIN_HEAD_DIM or head_dim % 32 != 0:
        raise ValueError(f"kernel requires head_dim >= {_MIN_HEAD_DIM} and head_dim % 32 == 0, got {head_dim}")
    if head_dim > _MAX_HEAD_DIM:
        raise ValueError(
            f"flydsl_flash_attn_func: head_dim={head_dim} > {_MAX_HEAD_DIM} is not "
            "supported on gfx120x FlyDSL FA (LDS / register tile budget; "
            f"RDNA4 WG LDS 64 KiB holds D<=384 @ BLOCK_N=32 prefetch1)."
        )

    # Merge attn_mask / bias aliases + optional causal×cross / ALiBi folds.
    mask_in = attn_mask if attn_mask is not None else bias
    bias_t = normalize_attn_mask(mask_in, seq_len_q_real, seq_len_kv_real, q.device)
    causal_bias = None
    if causal_cross:
        causal_bias = bottom_right_causal_bias(seq_len_q_real, seq_len_kv_real, q.device)
    alibi_bias = None
    if alibi_slopes is not None:
        alibi_bias = fold_alibi_to_bias(alibi_slopes, seq_len_q_real, seq_len_kv_real, q.device)
    bias_t = _merge_bias(bias_t, causal_bias, alibi_bias)
    has_per_head_bias = False
    if bias_t is not None and bias_t.dim() == 3:
        if int(bias_t.shape[0]) != num_heads:
            raise ValueError(f"gfx120x FA per-head bias H={int(bias_t.shape[0])} != num_heads={num_heads}")
        has_per_head_bias = True
    has_bias = bias_t is not None

    # Sink: expand to contiguous [B, H] fp32 for the kernel.
    sink_t = None
    if sink is not None:
        s = sink.detach().float()
        if s.dim() == 1:
            if int(s.numel()) != num_heads:
                raise ValueError(f"gfx120x FA sink must have H={num_heads} entries, got {int(s.numel())}")
            sink_t = s.view(1, num_heads).expand(batch, num_heads).contiguous()
        elif s.dim() == 2:
            if tuple(s.shape) != (batch, num_heads):
                raise ValueError(f"gfx120x FA sink shape {tuple(s.shape)} != {(batch, num_heads)}")
            sink_t = _as_contig(s)
        else:
            raise ValueError(f"gfx120x FA sink must be [H] or [B,H], got {tuple(sink.shape)}")
        if sink_t.device != q.device:
            sink_t = sink_t.to(q.device)
    has_sink = sink_t is not None

    dtype_str = _torch_dtype_to_str(q.dtype)
    block_m = _pick_block_m(seq_len_q_real, cross, seq_len_kv_real)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, seq_len_kv_real, cross, waves_per_eu)

    seq_len_q_launch = seq_len_q_real
    if cross:
        seq_len_kv_pad = ((seq_len_kv_real + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
    else:
        seq_len_kv_pad = ((seq_len_kv_real + block_m - 1) // block_m) * block_m
        seq_len_kv_pad = ((seq_len_kv_pad + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N

    n_pad_kv = seq_len_kv_pad - seq_len_kv_real
    q_p = _as_contig(q)
    if n_pad_kv > 0:
        k_p = F.pad(_as_contig(k), (0, 0, 0, 0, 0, n_pad_kv))
        v_p = F.pad(_as_contig(v), (0, 0, 0, 0, 0, n_pad_kv))
        if has_bias:
            # Pad last dim (KV) for both [Sq,Sk] and [H,Sq,Sk].
            bias_t = F.pad(bias_t, (0, n_pad_kv))
    else:
        k_p = _as_contig(k)
        v_p = _as_contig(v)

    bias_arg = _as_contig(bias_t) if has_bias else _dummy_bias(q.device)

    o_shape = (batch, seq_len_q_launch, num_heads, head_dim)
    if out is not None:
        if tuple(out.shape) != o_shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError(
                f"flydsl_flash_attn_func: out shape/dtype/device mismatch: "
                f"got {tuple(out.shape)}/{out.dtype}/{out.device}, "
                f"want {o_shape}/{q.dtype}/{q.device}"
            )
        o_p = out
    else:
        o_p = torch.empty(o_shape, dtype=q.dtype, device=q.device)

    lse_p = None
    if return_lse:
        lse_p = torch.empty(
            (batch, num_heads, seq_len_q_launch),
            dtype=torch.float32,
            device=q.device,
        )
    lse_arg = _flat(lse_p) if return_lse else _dummy_bias(q.device)
    sink_arg = _flat(sink_t) if has_sink else _dummy_bias(q.device)

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        if launch_stream.device != q.device:
            raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
        exe = _get_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=kernel_causal,
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            daz=daz,
            block_m=block_m,
            has_attn_bias=has_bias and not has_per_head_bias,
            has_per_head_bias=has_per_head_bias,
            return_lse=(return_lse and int(num_kv_splits) <= 1),
            has_sink=(has_sink and int(num_kv_splits) <= 1),
            num_kv_heads=num_kv_heads,
        )
        _di = _dummy_i32(q.device)
        nsplits = int(num_kv_splits)
        if nsplits < 1:
            raise ValueError(f"gfx120x FA: num_kv_splits must be >= 1, got {nsplits}")
        if nsplits > 1:
            ws_rows = int(batch) * int(num_heads) * int(seq_len_q_launch)
            ws_m = torch.empty((nsplits, ws_rows), dtype=torch.float32, device=q.device)
            ws_l = torch.empty((nsplits, ws_rows), dtype=torch.float32, device=q.device)
            ws_o = torch.empty((nsplits, ws_rows, head_dim), dtype=torch.float32, device=q.device)
            split_args = (nsplits, _flat(ws_m), _flat(ws_l), _flat(ws_o), ws_rows)
        else:
            split_args = _nosplit_args(q.device)
        exe(
            _flat(q_p),
            _flat(k_p),
            _flat(v_p),
            _flat(o_p),
            batch,
            seq_len_q_launch,
            seq_len_kv_pad,
            seq_len_kv_real,
            _flat(bias_arg),
            lse_arg,
            sink_arg,
            _flat(_di),  # CuSeqlensQ dummy
            _flat(_di),  # CuSeqlensKV dummy
            _flat(_di),  # BlockTable dummy
            _flat(_di),  # SeqlenK dummy
            0,  # block_table_stride
            *split_args,
            stream=launch_stream,
        )
        if nsplits > 1:
            if return_lse and lse_p is None:
                raise RuntimeError("split-K return_lse missing lse buffer")
            lse_c = lse_p if return_lse else _dummy_bias(q.device)
            sink_c = sink_t if has_sink else _dummy_bias(q.device)
            comb = _get_splitk_combine(num_heads, head_dim, dtype_str, bool(has_sink), bool(return_lse))
            comb(
                ws_m,
                ws_l,
                ws_o,
                o_p,
                lse_c,
                sink_c,
                batch,
                seq_len_q_launch,
                nsplits,
                ws_rows,
                stream=launch_stream,
            )

    if return_lse:
        return o_p, lse_p
    return o_p


def _expand_shared_bias_to_packed(
    bias: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv_pad: int,
) -> torch.Tensor:
    """Expand shared [max_q, max_kv] (or already-packed [total_q, max_kv]) to packed rows.

    Kernel VARLEN bias indexing uses (q_off + q_row) * seq_len_kv + kv_col, so the
    host always passes [total_q, max_kv_pad]. Shared masks are broadcast-copied
    into every batch's row range (truncated to that batch's sq).
    """
    cq = cu_seqlens_q.to(torch.int64)
    total_q = int(cq[-1].item())
    b = bias
    if b.dim() != 2:
        raise ValueError(f"gfx120x varlen FA bias must be 2D [total_q|max_q, max_kv], got {tuple(b.shape)}")
    # Pad KV dim to launch pad.
    if b.shape[1] < max_seqlen_kv_pad:
        b = torch.nn.functional.pad(b, (0, max_seqlen_kv_pad - int(b.shape[1])))
    elif b.shape[1] > max_seqlen_kv_pad:
        b = b[:, :max_seqlen_kv_pad]
    if int(b.shape[0]) == total_q:
        return _as_contig(b)
    if int(b.shape[0]) < max_seqlen_q:
        raise ValueError(f"gfx120x varlen shared bias rows {int(b.shape[0])} < max_seqlen_q={max_seqlen_q}")
    # Shared [max_q, Sk] → expand into packed [total_q, Sk_pad].
    out = torch.zeros(total_q, max_seqlen_kv_pad, device=b.device, dtype=torch.float32)
    B = int(cq.numel() - 1)
    for bi in range(B):
        qs, qe = int(cq[bi]), int(cq[bi + 1])
        sq = qe - qs
        if sq <= 0:
            continue
        out[qs:qe] = b[:sq].to(dtype=torch.float32)
    return out


def flydsl_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    out: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_lse: bool = False,
    sink: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Packed-varlen FA: Q/K/V/O as [total, H, D] with cu_seqlens int32 [B+1]."""
    from kernels.common.gfx120x_arch import require_gfx120x

    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_varlen_func requires CUDA/HIP tensors")
    require_gfx120x(q.device, what="flydsl_flash_attn_varlen_func")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"varlen expects packed 3D [total,H,D], got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2] or k.shape[1] != v.shape[1]:
        raise ValueError(f"varlen D/Hkv mismatch: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if int(q.shape[1]) % int(k.shape[1]) != 0:
        raise ValueError(f"varlen GQA: q heads {int(q.shape[1])} not divisible by kv heads {int(k.shape[1])}")

    num_heads, head_dim = int(q.shape[1]), int(q.shape[2])
    if head_dim < _MIN_HEAD_DIM or head_dim % 32 != 0 or head_dim > _MAX_HEAD_DIM:
        raise ValueError(f"varlen head_dim={head_dim} out of gfx120x FA range")

    cu_q = _as_contig(cu_seqlens_q.to(torch.int32))
    cu_kv = _as_contig(cu_seqlens_kv.to(torch.int32))
    if cu_q.device != q.device:
        cu_q = cu_q.to(q.device)
    if cu_kv.device != q.device:
        cu_kv = cu_kv.to(q.device)
    batch = int(cu_q.numel() - 1)
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_kv = int(max_seqlen_kv)

    # Merge attn_mask into bias. If both are set they are added.
    # Dense keeps only attn_mask when both are set.
    bias_t = None
    if attn_mask is not None and not mask_is_noop(attn_mask):
        bias_t = attn_mask
    if bias is not None and not mask_is_noop(bias):
        bias_t = bias if bias_t is None else (bias_t + bias)

    # ALiBi: fold against max seqlens then expand to packed.
    alibi_bias = None
    if alibi_slopes is not None:
        alibi_bias = fold_alibi_to_bias(alibi_slopes, max_seqlen_q, max_seqlen_kv, q.device)
    bias_t = _merge_bias(bias_t, alibi_bias)

    # Causal uses in-kernel bottom-right with per-batch (sk-sq).
    kernel_causal = bool(causal)

    # Sink → [B, H]
    sink_t = None
    if sink is not None:
        s = sink.detach().float()
        if s.dim() == 1:
            if int(s.numel()) != num_heads:
                raise ValueError(f"varlen sink must have H={num_heads}, got {int(s.numel())}")
            sink_t = s.view(1, num_heads).expand(batch, num_heads).contiguous()
        elif s.dim() == 2:
            if tuple(s.shape) != (batch, num_heads):
                raise ValueError(f"varlen sink shape {tuple(s.shape)} != {(batch, num_heads)}")
            sink_t = _as_contig(s)
        else:
            raise ValueError(f"varlen sink must be [H] or [B,H], got {tuple(sink.shape)}")
        if sink_t.device != q.device:
            sink_t = sink_t.to(q.device)

    cross = max_seqlen_q != max_seqlen_kv
    dtype_str = _torch_dtype_to_str(q.dtype)
    block_m = _pick_block_m(max_seqlen_q, cross, max_seqlen_kv)
    waves_per_eu = _pick_waves_per_eu(max_seqlen_q, max_seqlen_kv, cross, waves_per_eu)

    # Tile-pad max_seqlen_kv like dense (BLOCK_N=32).
    block_n = 32
    n_pad_kv = (block_n - (max_seqlen_kv % block_n)) % block_n
    seq_len_kv_pad = max_seqlen_kv + n_pad_kv
    seq_len_q_launch = max_seqlen_q

    has_bias = bias_t is not None
    if has_bias and bias_t.dim() == 3:
        raise NotImplementedError("gfx120x varlen FA does not support per-head bias yet; pass shared/packed 2D")
    if has_bias:
        bias_t = _expand_shared_bias_to_packed(bias_t.float(), cu_q, max_seqlen_q, seq_len_kv_pad)
    bias_arg = _as_contig(bias_t) if has_bias else _dummy_bias(q.device)

    total_q = int(q.shape[0])
    o_shape = (total_q, num_heads, head_dim)
    if out is not None:
        if tuple(out.shape) != o_shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError(
                "varlen out mismatch: "
                f"got {tuple(out.shape)}/{out.dtype}/{out.device}, "
                f"want {o_shape}/{q.dtype}/{q.device}"
            )
        o_p = out
    else:
        o_p = torch.empty(o_shape, dtype=q.dtype, device=q.device)

    lse_p = None
    if return_lse:
        lse_p = torch.full(
            (batch, num_heads, seq_len_q_launch),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )
    lse_arg = _flat(lse_p) if return_lse else _flat(_dummy_bias(q.device))
    sink_arg = _flat(sink_t) if sink_t is not None else _flat(_dummy_bias(q.device))
    _di = _dummy_i32(q.device)

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        exe = _get_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=kernel_causal,
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            daz=daz,
            block_m=block_m,
            has_attn_bias=has_bias,
            has_per_head_bias=False,
            return_lse=return_lse,
            has_sink=sink_t is not None,
            varlen=True,
            paged=False,
            page_size=16,
            num_kv_heads=int(k.shape[1]),
        )
        exe(
            _flat(_as_contig(q)),
            _flat(_as_contig(k)),
            _flat(_as_contig(v)),
            _flat(o_p),
            batch,
            seq_len_q_launch,
            seq_len_kv_pad,
            max_seqlen_kv,  # ignored by VARLEN kernel (uses sk)
            _flat(bias_arg),
            lse_arg,
            sink_arg,
            _flat(cu_q),
            _flat(cu_kv),
            _flat(_di),
            _flat(_di),
            0,
            *_nosplit_args(q.device),
            stream=launch_stream,
        )

    if return_lse:
        return o_p, lse_p
    return o_p


def flydsl_flash_attn_paged_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen_k: torch.Tensor,
    *,
    page_size: int | None = None,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    out: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_lse: bool = False,
    sink: torch.Tensor | None = None,
    kv_cache_layout: str = "linear",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Paged FA. linear 4D, linear3d page-1 (same bytes), or vectorized 5D."""
    from kernels.common.gfx120x_arch import require_gfx120x

    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise ValueError("flydsl_flash_attn_paged_func requires CUDA/HIP tensors")
    require_gfx120x(q.device, what="flydsl_flash_attn_paged_func")
    if q.dim() != 4:
        raise ValueError(f"paged Q must be 4D BSHD, got {tuple(q.shape)}")
    layout = kv_cache_layout or "linear"
    batch, seq_len_q_real, num_heads, head_dim = q.shape
    if layout == "linear3d":
        if k_cache.dim() != 3 or v_cache.dim() != 3:
            raise ValueError("linear3d paged K/V must be [num_blocks, Hkv, D]")
        if tuple(k_cache.shape) != tuple(v_cache.shape):
            raise ValueError(f"linear3d K/V shape mismatch {tuple(k_cache.shape)} vs {tuple(v_cache.shape)}")
        # Same bytes as linear page_size=1: [Nb, Hkv, D] == [Nb, 1, Hkv, D].
        k_cache = k_cache.view(k_cache.shape[0], 1, k_cache.shape[1], k_cache.shape[2])
        v_cache = v_cache.view(v_cache.shape[0], 1, v_cache.shape[1], v_cache.shape[2])
        layout = "linear"
        page_size = 1
    elif layout == "vectorized":
        if k_cache.dim() != 5 or v_cache.dim() != 5:
            raise ValueError(f"vectorized paged K/V must be 5D, got {tuple(k_cache.shape)} {tuple(v_cache.shape)}")
        kvs = 16 // k_cache.element_size()
        if int(k_cache.shape[4]) != kvs:
            raise ValueError(f"vectorized K last dim {int(k_cache.shape[4])} != kVS={kvs}")
        if int(k_cache.shape[3]) % kvs != 0:
            raise ValueError(f"vectorized page_size {int(k_cache.shape[3])} not divisible by kVS={kvs}")
        hkv = int(k_cache.shape[1])
        page_size = int(k_cache.shape[3])
        k_dim = int(k_cache.shape[2]) * int(k_cache.shape[4])
        if k_dim != head_dim:
            raise ValueError(f"vectorized K logical D {k_dim} != Q D {head_dim}")
        expect_v = (int(k_cache.shape[0]), hkv, page_size // kvs, head_dim, kvs)
        if tuple(v_cache.shape) != expect_v:
            raise ValueError(f"vectorized V shape {tuple(v_cache.shape)} != {expect_v}")
        if num_heads % hkv != 0:
            raise ValueError(f"paged GQA: Hq {num_heads} not divisible by Hkv {hkv}")
        num_kv_heads = hkv
    elif layout == "linear":
        num_kv_heads = None  # filled below
    else:
        raise ValueError(f"gfx120x paged layout {layout!r} is not linear/linear3d/vectorized")

    if layout != "vectorized":
        if k_cache.dim() != 4 or v_cache.dim() != 4:
            raise ValueError(
                "paged K/V must be 4D [num_blocks,page_size,H,D], "
                f"got k={tuple(k_cache.shape)} v={tuple(v_cache.shape)}"
            )
        if tuple(k_cache.shape[1:]) != tuple(v_cache.shape[1:]):
            raise ValueError(f"paged K/V page/H/D mismatch: {tuple(k_cache.shape)} vs {tuple(v_cache.shape)}")
        cache_page_size = int(k_cache.shape[1])
        if page_size is None:
            page_size = cache_page_size
        page_size = int(page_size)
        if page_size != cache_page_size:
            raise ValueError(f"page_size={page_size} != cache dim1={cache_page_size}")
        num_kv_heads = int(k_cache.shape[2])
        if int(k_cache.shape[3]) != head_dim:
            raise ValueError(f"paged cache D {int(k_cache.shape[3])} != Q D {head_dim}")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"paged GQA: Hq {num_heads} not divisible by Hkv {num_kv_heads}")
    if head_dim < _MIN_HEAD_DIM or head_dim % 32 != 0 or head_dim > _MAX_HEAD_DIM:
        raise ValueError(f"paged head_dim={head_dim} out of gfx120x FA range")

    bt = _as_contig(block_table.to(torch.int32))
    sk = _as_contig(seqlen_k.to(torch.int32))
    if bt.device != q.device:
        bt = bt.to(q.device)
    if sk.device != q.device:
        sk = sk.to(q.device)
    if bt.dim() != 2 or int(bt.shape[0]) != batch:
        raise ValueError(f"block_table must be [B, n_pages], got {tuple(bt.shape)} for B={batch}")
    if sk.numel() != batch:
        raise ValueError(f"seqlen_k must have B={batch} entries, got {int(sk.numel())}")
    block_table_stride = int(bt.shape[1])

    max_sk = int(sk.max().item()) if sk.numel() else 0
    cross = seq_len_q_real != max_sk
    dtype_str = _torch_dtype_to_str(q.dtype)
    block_m = _pick_block_m(seq_len_q_real, cross, max_sk)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, max_sk, cross, waves_per_eu)

    block_n = 32
    n_pad_kv = (block_n - (max_sk % block_n)) % block_n if max_sk else 0
    seq_len_kv_pad = max_sk + n_pad_kv
    seq_len_q_launch = seq_len_q_real

    # Bias / ALiBi (shared dense [Sq, Sk] against max_sk; ragged + bias rejected upstream).
    bias_t = None
    if attn_mask is not None and not mask_is_noop(attn_mask):
        bias_t = normalize_attn_mask(attn_mask, seq_len_q_real, max_sk, q.device)
    if bias is not None and not mask_is_noop(bias):
        nb = normalize_attn_mask(bias, seq_len_q_real, max_sk, q.device) if bias.dim() != 2 else bias
        bias_t = nb if bias_t is None else _merge_bias(bias_t, nb)
    alibi_bias = None
    if alibi_slopes is not None:
        alibi_bias = fold_alibi_to_bias(alibi_slopes, seq_len_q_real, max_sk, q.device)
    bias_t = _merge_bias(bias_t, alibi_bias)
    has_per_head_bias = False
    if bias_t is not None and bias_t.dim() == 3:
        if int(bias_t.shape[0]) != num_heads:
            raise ValueError(f"paged per-head bias H={int(bias_t.shape[0])} != {num_heads}")
        has_per_head_bias = True
    has_bias = bias_t is not None
    if has_bias and n_pad_kv > 0:
        bias_t = torch.nn.functional.pad(bias_t, (0, n_pad_kv))

    sink_t = None
    if sink is not None:
        s = sink.detach().float()
        if s.dim() == 1:
            sink_t = s.view(1, num_heads).expand(batch, num_heads).contiguous()
        elif s.dim() == 2:
            sink_t = _as_contig(s)
        else:
            raise ValueError(f"paged sink must be [H] or [B,H], got {tuple(sink.shape)}")
        if sink_t.device != q.device:
            sink_t = sink_t.to(q.device)

    bias_arg = _as_contig(bias_t) if has_bias else _dummy_bias(q.device)
    o_shape = (batch, seq_len_q_launch, num_heads, head_dim)
    if out is not None:
        if tuple(out.shape) != o_shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError(
                "paged out mismatch: "
                f"got {tuple(out.shape)}/{out.dtype}/{out.device}, "
                f"want {o_shape}/{q.dtype}/{q.device}"
            )
        o_p = out
    else:
        o_p = torch.empty(o_shape, dtype=q.dtype, device=q.device)

    lse_p = None
    if return_lse:
        lse_p = torch.empty((batch, num_heads, seq_len_q_launch), dtype=torch.float32, device=q.device)
    lse_arg = _flat(lse_p) if return_lse else _flat(_dummy_bias(q.device))
    sink_arg = _flat(sink_t) if sink_t is not None else _flat(_dummy_bias(q.device))
    _di = _dummy_i32(q.device)

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        exe = _get_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=bool(causal),
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            daz=daz,
            block_m=block_m,
            has_attn_bias=has_bias and not has_per_head_bias,
            has_per_head_bias=has_per_head_bias,
            return_lse=return_lse,
            has_sink=sink_t is not None,
            varlen=False,
            paged=True,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            kv_cache_layout=("vectorized" if (kv_cache_layout or "linear") == "vectorized" else "linear"),
        )
        exe(
            _flat(_as_contig(q)),
            _flat(_as_contig(k_cache)),
            _flat(_as_contig(v_cache)),
            _flat(o_p),
            batch,
            seq_len_q_launch,
            seq_len_kv_pad,
            max_sk,  # overridden per-batch by SeqlenK loads
            _flat(bias_arg),
            lse_arg,
            sink_arg,
            _flat(_di),
            _flat(_di),
            _flat(bt),
            _flat(sk),
            block_table_stride,
            *_nosplit_args(q.device),
            stream=launch_stream,
        )

    if return_lse:
        return o_p, lse_p
    return o_p


def flydsl_flash_attn_varlen_paged_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    block_table: torch.Tensor,
    seqlen_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    *,
    page_size: int = None,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    return_lse: bool = False,
    sink: torch.Tensor | None = None,
    kv_cache_layout: str = "linear",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Packed Q with paged K/V. cu_seqlens_q selects Q rows; block_table/seqlen_k select KV.

    cu_seqlens_kv is required and must describe the same batch as seqlen_k. KV
    addresses come from the block table (no host gather, no dense pad).
    """
    if cu_seqlens_q is None or cu_seqlens_kv is None:
        raise ValueError("varlen paged requires cu_seqlens_q and cu_seqlens_kv")
    if q.dim() != 3:
        raise ValueError(f"varlen paged Q must be packed [total,H,D], got {tuple(q.shape)}")
    # Reuse dense-Q paged prep by viewing each batch through the kernel's own cu_seqlens.
    # Build a 4D Q launch by calling the paged kernel with varlen=True via _get_kernel below.
    from kernels.common.gfx120x_arch import require_gfx120x

    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise ValueError("varlen paged requires CUDA/HIP tensors")
    require_gfx120x(q.device, what="flydsl_flash_attn_varlen_paged_func")
    # Normalize cache through the dense paged entry's layout rules by a tiny Q stand-in
    # only for validation — real launch is below so we duplicate the layout parse.
    layout = kv_cache_layout or "linear"
    num_heads = int(q.shape[1])
    head_dim = int(q.shape[2])
    cu_q = _as_contig(cu_seqlens_q.to(torch.int32))
    cu_kv = _as_contig(cu_seqlens_kv.to(torch.int32))
    if cu_q.device != q.device:
        cu_q = cu_q.to(q.device)
    if cu_kv.device != q.device:
        cu_kv = cu_kv.to(q.device)
    if cu_q.numel() != cu_kv.numel():
        raise ValueError("varlen paged cu_seqlens_q/kv batch mismatch")
    batch = int(cu_q.numel() - 1)
    bt = _as_contig(block_table.to(torch.int32))
    sk = _as_contig(seqlen_k.to(torch.int32).view(-1))
    if bt.device != q.device:
        bt = bt.to(q.device)
    if sk.device != q.device:
        sk = sk.to(q.device)
    if bt.dim() != 2 or int(bt.shape[0]) != batch:
        raise ValueError(f"varlen paged block_table must be [B, n_pages], got {tuple(bt.shape)}")
    if int(sk.numel()) != batch:
        raise ValueError("varlen paged seqlen_k batch mismatch")
    # KV lengths follow seqlen_k (paged). cu_seqlens_kv must match those lengths.
    cu_kv64 = cu_kv.to(torch.int64)
    cu_lens = (cu_kv64[1:] - cu_kv64[:-1]).to(torch.int32)
    if not torch.equal(cu_lens, sk):
        raise ValueError("varlen paged cu_seqlens_kv lengths must equal seqlen_k")

    if layout == "linear3d":
        if k_cache.dim() != 3:
            raise ValueError("varlen paged linear3d cache must be [Nb,Hkv,D]")
        k_cache = k_cache.view(k_cache.shape[0], 1, k_cache.shape[1], k_cache.shape[2])
        v_cache = v_cache.view(v_cache.shape[0], 1, v_cache.shape[1], v_cache.shape[2])
        layout_kernel = "linear"
        page_size = 1
        num_kv_heads = int(k_cache.shape[2])
        if int(k_cache.shape[3]) != head_dim:
            raise ValueError("varlen paged linear3d D mismatch")
    elif layout == "vectorized":
        layout_kernel = "vectorized"
        kvs = 16 // k_cache.element_size()
        num_kv_heads = int(k_cache.shape[1])
        page_size = int(k_cache.shape[3])
        if int(k_cache.shape[2]) * kvs != head_dim:
            raise ValueError("varlen paged vectorized D mismatch")
    else:
        layout_kernel = "linear"
        if k_cache.dim() != 4:
            raise ValueError("varlen paged linear cache must be 4D")
        if page_size is None:
            page_size = int(k_cache.shape[1])
        num_kv_heads = int(k_cache.shape[2])
        if int(k_cache.shape[3]) != head_dim or int(k_cache.shape[1]) != int(page_size):
            raise ValueError("varlen paged linear cache H/D/page mismatch")
    if num_heads % int(num_kv_heads) != 0:
        raise ValueError("varlen paged GQA group invalid")
    page_size = int(page_size)
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_kv = int(max_seqlen_kv)
    dtype_str = _torch_dtype_to_str(q.dtype)
    block_m = _pick_block_m(max_seqlen_q, max_seqlen_q != max_seqlen_kv, max_seqlen_kv)
    waves_per_eu = _pick_waves_per_eu(max_seqlen_q, max_seqlen_kv, max_seqlen_q != max_seqlen_kv, waves_per_eu)
    block_n = 32
    n_pad = (block_n - (max_seqlen_kv % block_n)) % block_n
    seq_len_kv_pad = max_seqlen_kv + n_pad

    bias_t = None
    if bias is not None and not mask_is_noop(bias):
        bias_t = bias if bias.dim() == 2 else normalize_attn_mask(bias, max_seqlen_q, max_seqlen_kv, q.device)
    if alibi_slopes is not None:
        bias_t = _merge_bias(bias_t, fold_alibi_to_bias(alibi_slopes, max_seqlen_q, max_seqlen_kv, q.device))
    has_bias = bias_t is not None
    if has_bias and bias_t.dim() == 3:
        raise NotImplementedError("gfx120x varlen paged FA does not support per-head bias yet")
    if has_bias:
        bias_t = _expand_shared_bias_to_packed(bias_t.float(), cu_q, max_seqlen_q, seq_len_kv_pad)
    sink_t = None
    if sink is not None:
        ss = sink.detach().float()
        if ss.dim() == 1:
            sink_t = ss.view(1, num_heads).expand(batch, num_heads).contiguous()
        elif ss.dim() == 2:
            sink_t = _as_contig(ss)
        else:
            raise ValueError("varlen paged sink must be [H] or [B,H]")
        if sink_t.device != q.device:
            sink_t = sink_t.to(q.device)
    total_q = int(q.shape[0])
    o_shape = (total_q, num_heads, head_dim)
    if out is None:
        o_p = torch.empty(o_shape, dtype=q.dtype, device=q.device)
    else:
        if tuple(out.shape) != o_shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError(f"varlen paged out mismatch {tuple(out.shape)}")
        o_p = out
    lse_p = None
    if return_lse:
        lse_p = torch.full((batch, num_heads, max_seqlen_q), float("-inf"), dtype=torch.float32, device=q.device)
    bias_arg = _as_contig(bias_t) if has_bias else _dummy_bias(q.device)
    lse_arg = _flat(lse_p) if return_lse else _flat(_dummy_bias(q.device))
    sink_arg = _flat(sink_t) if sink_t is not None else _flat(_dummy_bias(q.device))
    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        exe = _get_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=bool(causal),
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            daz=daz,
            block_m=block_m,
            has_attn_bias=has_bias,
            has_per_head_bias=False,
            return_lse=return_lse,
            has_sink=sink_t is not None,
            varlen=True,
            paged=True,
            page_size=page_size,
            num_kv_heads=int(num_kv_heads),
            kv_cache_layout=layout_kernel,
        )
        exe(
            _flat(_as_contig(q)),
            _flat(_as_contig(k_cache)),
            _flat(_as_contig(v_cache)),
            _flat(o_p),
            batch,
            max_seqlen_q,
            seq_len_kv_pad,
            max_seqlen_kv,
            _flat(bias_arg),
            lse_arg,
            sink_arg,
            _flat(cu_q),
            _flat(cu_kv),
            _flat(bt),
            _flat(sk),
            int(bt.shape[1]),
            *_nosplit_args(q.device),
            stream=launch_stream,
        )
    if return_lse:
        return o_p, lse_p
    return o_p


# ---------------------------------------------------------------------------
# FP8 (E4M3FN / E5M2) — self + cross, descales, seq_len_kv_valid
# ---------------------------------------------------------------------------

_FP8_KERNEL_BLOCK_M = 128


def _prepare_quant_extras(
    q: torch.Tensor,
    bias: torch.Tensor | None,
    alibi_slopes: torch.Tensor | None,
    sink: torch.Tensor | None,
    seq_q: int,
    seq_k: int,
    n_pad_kv: int,
    batch: int,
    num_heads: int,
) -> tuple[object, object, object]:
    """Bias / ALiBi / sink for the fp8 and int8 kernels. Returns tensors, not host loops."""
    bias_t = None
    if bias is not None and not mask_is_noop(bias):
        bias_t = normalize_attn_mask(bias, seq_q, seq_k, q.device)
    if alibi_slopes is not None:
        bias_t = _merge_bias(bias_t, fold_alibi_to_bias(alibi_slopes, seq_q, seq_k, q.device))
    has_per_head = False
    if bias_t is not None and bias_t.dim() == 3:
        if int(bias_t.shape[0]) != int(num_heads):
            raise ValueError(f"gfx120x quant FA per-head bias H={int(bias_t.shape[0])} != num_heads={num_heads}")
        has_per_head = True
    if bias_t is not None and n_pad_kv > 0:
        bias_t = F.pad(bias_t, (0, n_pad_kv))
    sink_t = None
    if sink is not None:
        ss = sink.detach().float()
        if ss.dim() == 1:
            if int(ss.numel()) != int(num_heads):
                raise ValueError(f"quant sink must have H={num_heads}, got {int(ss.numel())}")
            sink_t = ss.view(1, int(num_heads)).expand(int(batch), int(num_heads)).contiguous()
        elif ss.dim() == 2:
            if tuple(ss.shape) != (int(batch), int(num_heads)):
                raise ValueError(f"quant sink shape {tuple(ss.shape)} != {(int(batch), int(num_heads))}")
            sink_t = _as_contig(ss)
        else:
            raise ValueError(f"quant sink must be [H] or [B,H], got {tuple(sink.shape)}")
        if sink_t.device != q.device:
            sink_t = sink_t.to(q.device)
    return bias_t, has_per_head, sink_t


@lru_cache(maxsize=32)
def _get_fp8_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    dtype_str: str = "fp8_e4m3fn",
    block_m: int = 128,
    has_attn_bias: bool = False,
    has_per_head_bias: bool = False,
    return_lse: bool = False,
    has_sink: bool = False,
    num_kv_heads: int | None = None,
) -> Callable[..., None]:
    from kernels.attention.flash_attn_fp8_gfx120x import (
        build_flash_attn_func_fp8_module,
    )

    return build_flash_attn_func_fp8_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        has_attn_bias=has_attn_bias,
        has_per_head_bias=has_per_head_bias,
        return_lse=return_lse,
        has_sink=has_sink,
        num_kv_heads=num_kv_heads,
    )


def _fp8_descale_to_float(name: str, scale: torch.Tensor | float | None, device: torch.device) -> float:
    import math as _math

    if scale is None:
        return 1.0
    if isinstance(scale, (int, float)):
        val = float(scale)
    elif isinstance(scale, torch.Tensor):
        if not scale.is_cuda:
            raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be a CUDA tensor, got device={scale.device}")
        if scale.device != device:
            raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be on {device}, got {scale.device}")
        if scale.dtype != torch.float32:
            raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be float32, got {scale.dtype}")
        if scale.numel() != 1:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: {name} must have numel==1 "
                f"(per-tensor), got shape={tuple(scale.shape)}"
            )
        val = float(scale.reshape(-1)[0].item())
    else:
        raise TypeError(
            f"flydsl_flash_attn_fp8_func: {name} must be None, float, or "
            f"float32 CUDA tensor[1], got {type(scale).__name__}"
        )
    if not (_math.isfinite(val) and val > 0.0):
        raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be positive and finite, got {val}")
    return val


def _fp8_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.float8_e4m3fn:
        return "fp8_e4m3fn"
    if dtype == torch.float8_e5m2:
        return "fp8_e5m2"
    raise ValueError(f"flydsl_flash_attn_fp8_func expects float8_e4m3fn or float8_e5m2, got {dtype}")


def flydsl_flash_attn_fp8_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    q_descale: torch.Tensor | float | None = None,
    k_descale: torch.Tensor | float | None = None,
    v_descale: torch.Tensor | float | None = None,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    return_lse: bool = False,
    num_kv_heads: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run FlyDSL FP8 Flash Attention on RDNA4 (gfx120x).

    Q/K/V: ``float8_e4m3fn`` or ``float8_e5m2`` BSHD. Supports self and
    non-causal cross (unequal Sq/Sk) via ``seq_len_kv`` / ``seq_len_kv_valid``.
    Output is bf16. Descales match the gfx950 per-tensor contract.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_fp8_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    require_gfx120x(q.device, what="flydsl_flash_attn_fp8_func (gfx120x)")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"flydsl_flash_attn_fp8_func: q/k/v dtype must match, got {q.dtype}/{k.dtype}/{v.dtype}")
    dtype_str = _fp8_dtype_str(q.dtype)
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"expected 4D BSHD tensors, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if not (
        q.shape[0] == k.shape[0] == v.shape[0] and q.shape[3] == k.shape[3] == v.shape[3] and k.shape[2] == v.shape[2]
    ):
        raise ValueError(
            "flydsl_flash_attn_fp8_func: q/k/v must share batch and head_dim; "
            "k/v must share head count; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )

    from kernels.attention.flash_attn_gfx120x_ext import reject_gqa

    reject_gqa(int(q.shape[2]), int(k.shape[2]), num_kv_heads)
    num_kv_heads = int(k.shape[2])
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"flydsl_flash_attn_fp8_func: k/v seq_len must match, got k={k.shape[1]} v={v.shape[1]}")

    batch, seq_len_q_real, num_heads, head_dim = q.shape
    seq_len_kv_real = int(k.shape[1])
    cross = seq_len_q_real != seq_len_kv_real
    # Causal×cross: in-kernel bottom-right (no dequant→bf16 FALLBACK).
    if head_dim < _MIN_HEAD_DIM or head_dim % 32 != 0:
        raise ValueError(f"kernel requires head_dim >= {_MIN_HEAD_DIM} and head_dim % 32 == 0, got {head_dim}")
    if head_dim > _MAX_HEAD_DIM:
        raise ValueError(
            f"flydsl_flash_attn_fp8_func: head_dim={head_dim} > {_MAX_HEAD_DIM} is not "
            "supported on gfx120x FlyDSL FA (LDS / register tile budget)."
        )

    qd = _fp8_descale_to_float("q_descale", q_descale, q.device)
    kd = _fp8_descale_to_float("k_descale", k_descale, q.device)
    vd = _fp8_descale_to_float("v_descale", v_descale, q.device)

    block_m = _pick_block_m(seq_len_q_real, cross, seq_len_kv_real)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, seq_len_kv_real, cross, waves_per_eu)

    seq_len_q_launch = seq_len_q_real
    if cross:
        seq_len_kv_pad = ((seq_len_kv_real + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
    else:
        seq_len_kv_pad = ((seq_len_kv_real + block_m - 1) // block_m) * block_m
        seq_len_kv_pad = ((seq_len_kv_pad + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N

    n_pad_kv = seq_len_kv_pad - seq_len_kv_real
    q_p = _as_contig(q)
    if n_pad_kv > 0:
        k_p = F.pad(_as_contig(k), (0, 0, 0, 0, 0, n_pad_kv))
        v_p = F.pad(_as_contig(v), (0, 0, 0, 0, 0, n_pad_kv))
    else:
        k_p = _as_contig(k)
        v_p = _as_contig(v)

    o_shape = (batch, seq_len_q_launch, num_heads, head_dim)
    if out is not None:
        if tuple(out.shape) != o_shape or out.dtype != torch.bfloat16 or out.device != q.device:
            raise ValueError(f"flydsl_flash_attn_fp8_func: out must be bf16 {o_shape} on {q.device}")
        o_p = out
    else:
        o_p = torch.empty(o_shape, dtype=torch.bfloat16, device=q.device)

    if sm_scale is not None:
        import math as _math

        default = 1.0 / _math.sqrt(head_dim)
        if abs(sm_scale - default) > 1e-7:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func only supports default sm_scale=1/sqrt(D) ({default}), got {sm_scale}"
            )

    bias_t, has_per_head, sink_t = _prepare_quant_extras(
        q, bias, alibi_slopes, sink, seq_len_q_real, seq_len_kv_real, n_pad_kv, batch, num_heads
    )
    has_bias = bias_t is not None
    has_sink_b = sink_t is not None
    bias_arg = _flat(_as_contig(bias_t)) if has_bias else _flat(_dummy_bias(q.device))
    lse_p = None
    if return_lse:
        lse_p = torch.empty((batch, num_heads, seq_len_q_launch), dtype=torch.float32, device=q.device)
    lse_arg = _flat(lse_p) if return_lse else _flat(_dummy_bias(q.device))
    sink_arg = _flat(sink_t) if has_sink_b else _flat(_dummy_bias(q.device))

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        if launch_stream.device != q.device:
            raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
        exe = _get_fp8_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            waves_per_eu=waves_per_eu,
            daz=daz,
            dtype_str=dtype_str,
            block_m=block_m,
            has_attn_bias=has_bias and not has_per_head,
            has_per_head_bias=has_per_head,
            return_lse=return_lse,
            has_sink=has_sink_b,
            num_kv_heads=num_kv_heads,
        )
        exe(
            _flat(q_p),
            _flat(k_p),
            _flat(v_p),
            _flat(o_p),
            batch,
            seq_len_q_launch,
            seq_len_kv_pad,
            seq_len_kv_real,
            qd,
            kd,
            vd,
            bias_arg,
            lse_arg,
            sink_arg,
            stream=launch_stream,
        )

    if return_lse:
        return o_p, lse_p
    return o_p


# ---------------------------------------------------------------------------
# Int8 (iu8 WMMA) — self + cross, descales, seq_len_kv_valid (FP8-patterned)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_int8_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    block_m: int = 128,
    has_attn_bias: bool = False,
    has_per_head_bias: bool = False,
    return_lse: bool = False,
    has_sink: bool = False,
    num_kv_heads: int | None = None,
) -> Callable[..., None]:
    from kernels.attention.flash_attn_int8_gfx120x import (
        build_flash_attn_func_int8_module,
    )

    return build_flash_attn_func_int8_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str="int8",
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        has_attn_bias=has_attn_bias,
        has_per_head_bias=has_per_head_bias,
        return_lse=return_lse,
        has_sink=has_sink,
        num_kv_heads=num_kv_heads,
    )


def flydsl_flash_attn_int8_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    q_descale: torch.Tensor | float | None = None,
    k_descale: torch.Tensor | float | None = None,
    v_descale: torch.Tensor | float | None = None,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    return_lse: bool = False,
    num_kv_heads: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """FlyDSL int8 Flash Attention on RDNA4 (iu8 WMMA).

    Q/K/V: ``torch.int8`` BSHD. Per-tensor descales match the FP8 contract.
    Output bf16. Self + cross; causal/causal×cross via in-kernel bottom-right
    (no dequant→bf16 FALLBACK). Patterned on ``flydsl_flash_attn_fp8_func``.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_int8_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    require_gfx120x(q.device, what="flydsl_flash_attn_int8_func (gfx120x)")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise ValueError(f"flydsl_flash_attn_int8_func expects torch.int8 QKV, got {q.dtype}/{k.dtype}/{v.dtype}")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"expected 4D BSHD tensors, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if not (
        q.shape[0] == k.shape[0] == v.shape[0] and q.shape[3] == k.shape[3] == v.shape[3] and k.shape[2] == v.shape[2]
    ):
        raise ValueError(
            "flydsl_flash_attn_int8_func: q/k/v must share batch and head_dim; "
            "k/v must share head count; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )

    from kernels.attention.flash_attn_gfx120x_ext import reject_gqa

    reject_gqa(int(q.shape[2]), int(k.shape[2]), num_kv_heads)
    num_kv_heads = int(k.shape[2])
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"flydsl_flash_attn_int8_func: k/v seq_len must match, got k={k.shape[1]} v={v.shape[1]}")

    batch, seq_len_q_real, num_heads, head_dim = q.shape
    seq_len_kv_real = int(k.shape[1])
    cross = seq_len_q_real != seq_len_kv_real
    # Causal×cross: in-kernel bottom-right (no dequant→bf16 FALLBACK).
    if head_dim < _MIN_HEAD_DIM or head_dim % 32 != 0:
        raise ValueError(f"kernel requires head_dim >= {_MIN_HEAD_DIM} and head_dim % 32 == 0, got {head_dim}")
    if head_dim > _MAX_HEAD_DIM:
        raise ValueError(
            f"flydsl_flash_attn_int8_func: head_dim={head_dim} > {_MAX_HEAD_DIM} is not "
            "supported on gfx120x FlyDSL FA."
        )

    qd = _fp8_descale_to_float("q_descale", q_descale, q.device)
    kd = _fp8_descale_to_float("k_descale", k_descale, q.device)
    vd = _fp8_descale_to_float("v_descale", v_descale, q.device)

    block_m = _pick_block_m(seq_len_q_real, cross, seq_len_kv_real)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, seq_len_kv_real, cross, waves_per_eu)

    seq_len_q_launch = seq_len_q_real
    if cross:
        seq_len_kv_pad = ((seq_len_kv_real + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
    else:
        seq_len_kv_pad = ((seq_len_kv_real + block_m - 1) // block_m) * block_m
        seq_len_kv_pad = ((seq_len_kv_pad + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N

    n_pad_kv = seq_len_kv_pad - seq_len_kv_real
    q_p = _as_contig(q)
    if n_pad_kv > 0:
        k_p = F.pad(_as_contig(k), (0, 0, 0, 0, 0, n_pad_kv))
        v_p = F.pad(_as_contig(v), (0, 0, 0, 0, 0, n_pad_kv))
    else:
        k_p = _as_contig(k)
        v_p = _as_contig(v)

    o_shape = (batch, seq_len_q_launch, num_heads, head_dim)
    if out is not None:
        if tuple(out.shape) != o_shape or out.dtype != torch.bfloat16 or out.device != q.device:
            raise ValueError(f"flydsl_flash_attn_int8_func: out must be bf16 {o_shape} on {q.device}")
        o_p = out
    else:
        o_p = torch.empty(o_shape, dtype=torch.bfloat16, device=q.device)

    if sm_scale is not None:
        import math as _math

        default = 1.0 / _math.sqrt(head_dim)
        if abs(sm_scale - default) > 1e-7:
            raise ValueError(
                f"flydsl_flash_attn_int8_func only supports default sm_scale=1/sqrt(D) ({default}), got {sm_scale}"
            )

    bias_t, has_per_head, sink_t = _prepare_quant_extras(
        q, bias, alibi_slopes, sink, seq_len_q_real, seq_len_kv_real, n_pad_kv, batch, num_heads
    )
    has_bias = bias_t is not None
    has_sink_b = sink_t is not None
    bias_arg = _flat(_as_contig(bias_t)) if has_bias else _flat(_dummy_bias(q.device))
    lse_p = None
    if return_lse:
        lse_p = torch.empty((batch, num_heads, seq_len_q_launch), dtype=torch.float32, device=q.device)
    lse_arg = _flat(lse_p) if return_lse else _flat(_dummy_bias(q.device))
    sink_arg = _flat(sink_t) if has_sink_b else _flat(_dummy_bias(q.device))

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        if launch_stream.device != q.device:
            raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
        exe = _get_int8_kernel(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            waves_per_eu=waves_per_eu,
            daz=daz,
            block_m=block_m,
            has_attn_bias=has_bias and not has_per_head,
            has_per_head_bias=has_per_head,
            return_lse=return_lse,
            has_sink=has_sink_b,
            num_kv_heads=num_kv_heads,
        )
        exe(
            _flat(q_p),
            _flat(k_p),
            _flat(v_p),
            _flat(o_p),
            batch,
            seq_len_q_launch,
            seq_len_kv_pad,
            seq_len_kv_real,
            qd,
            kd,
            vd,
            bias_arg,
            lse_arg,
            sink_arg,
            stream=launch_stream,
        )

    if return_lse:
        return o_p, lse_p
    return o_p


# ---------------------------------------------------------------------------
# Native int4 (iu4 WMMA) — nibble-packed QKV, FP8-patterned descales
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _get_iu4_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    block_m: int = 128,
    prefer_native: bool = True,
    num_kv_heads: int | None = None,
) -> Callable[..., None]:
    from kernels.attention.flash_attn_iu4_gfx120x import (
        build_flash_attn_func_iu4_module,
    )

    return build_flash_attn_func_iu4_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str="iu4",
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        prefer_native=prefer_native,
        num_kv_heads=num_kv_heads,
    )


def _unpack_iu4_nibbles_to_i8(packed: torch.Tensor) -> torch.Tensor:
    """Unpack nibble-packed int4 bytes ``[..., D//2]`` → signed int8 ``[..., D]``."""
    lo = (packed.to(torch.int16) & 0xF).to(torch.int8)
    hi = ((packed.to(torch.int16) >> 4) & 0xF).to(torch.int8)
    # sign-extend nibbles
    lo = torch.where(lo >= 8, lo - 16, lo.to(torch.int16)).to(torch.int8)
    hi = torch.where(hi >= 8, hi - 16, hi.to(torch.int16)).to(torch.int8)
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def flydsl_flash_attn_iu4_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    q_descale: torch.Tensor | float | None = None,
    k_descale: torch.Tensor | float | None = None,
    v_descale: torch.Tensor | float | None = None,
    out: torch.Tensor | None = None,
    head_dim: int | None = None,
    prefer_native: bool = True,
    num_kv_heads: int | None = None,
    bias: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """FlyDSL native int4 Flash Attention on RDNA4 (nibble-pack ABI).

    Q/K/V: nibble-packed ``torch.int8`` ``[B, S, H, D//2]`` (two signed
    nibbles per byte, low first — same pack as ``rdna4_iu4_gemm``, not AWQ).
    Logical ``head_dim`` defaults to ``2 * q.shape[-1]``.

    Prefers in-kernel iu4 WMMA FA (i32 nibble-pack load path from
    ``rdna4_iu4_gemm``) for dense GQA when bias, ALiBi, sink, and
    ``return_lse`` are all unset. The native body has no knobs for those,
    so any one of them skips native and unpacks to int8 FA, which applies
    them. On toolchain refusal, or ``prefer_native=False``, the same
    nibble-unpack → iu8 path runs. Arguments are never dropped.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_iu4_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    require_gfx120x(q.device, what="flydsl_flash_attn_iu4_func (gfx120x)")
    if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
        raise ValueError(
            f"flydsl_flash_attn_iu4_func expects nibble-packed int8 QKV, got {q.dtype}/{k.dtype}/{v.dtype}"
        )
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"expected 4D packed BSHD, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if not (
        q.shape[0] == k.shape[0] == v.shape[0] and q.shape[3] == k.shape[3] == v.shape[3] and k.shape[2] == v.shape[2]
    ):
        raise ValueError(
            "flydsl_flash_attn_iu4_func: q/k/v must share batch and packed_dim; "
            "k/v must share head count; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )

    from kernels.attention.flash_attn_gfx120x_ext import reject_gqa

    reject_gqa(int(q.shape[2]), int(k.shape[2]), num_kv_heads)
    num_kv_heads = int(k.shape[2])
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"flydsl_flash_attn_iu4_func: k/v seq_len must match, got k={k.shape[1]} v={v.shape[1]}")

    d_pack = int(q.shape[3])
    logical_d = int(head_dim) if head_dim is not None else d_pack * 2
    if logical_d != d_pack * 2:
        raise ValueError(f"flydsl_flash_attn_iu4_func: head_dim={logical_d} must equal 2*packed_last ({d_pack * 2})")
    if logical_d < _MIN_HEAD_DIM or logical_d % 32 != 0:
        raise ValueError(f"kernel requires head_dim >= {_MIN_HEAD_DIM} and head_dim % 32 == 0, got {logical_d}")
    if logical_d > _MAX_HEAD_DIM:
        raise ValueError(
            f"flydsl_flash_attn_iu4_func: head_dim={logical_d} > {_MAX_HEAD_DIM} is not "
            "supported on gfx120x FlyDSL FA."
        )

    batch, seq_len_q_real, num_heads, _ = q.shape
    seq_len_kv_real = int(k.shape[1])
    cross = seq_len_q_real != seq_len_kv_real

    qd = _fp8_descale_to_float("q_descale", q_descale, q.device)
    kd = _fp8_descale_to_float("k_descale", k_descale, q.device)
    vd = _fp8_descale_to_float("v_descale", v_descale, q.device)

    block_m = _pick_block_m(seq_len_q_real, cross, seq_len_kv_real)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, seq_len_kv_real, cross, waves_per_eu)

    if sm_scale is not None:
        import math as _math

        default = 1.0 / _math.sqrt(logical_d)
        if abs(sm_scale - default) > 1e-7:
            raise ValueError(
                f"flydsl_flash_attn_iu4_func only supports default sm_scale=1/sqrt(D) ({default}), got {sm_scale}"
            )

    # Native iu4 has no bias / ALiBi / sink / LSE knobs. Any of those takes
    # unpack→int8, which applies them. Dense GQA with none set stays native.
    wants_quant_extra = bias is not None or alibi_slopes is not None or sink is not None or bool(return_lse)
    if prefer_native and not wants_quant_extra:
        try:
            exe = _get_iu4_kernel(
                num_heads=num_heads,
                head_dim=logical_d,
                causal=bool(causal),
                waves_per_eu=int(waves_per_eu),
                daz=bool(daz),
                block_m=block_m,
                prefer_native=True,
                num_kv_heads=num_kv_heads,
            )
        except Exception:
            exe = None
        if exe is not None and getattr(exe, "is_native_iu4_fa", False):
            seq_len_q_launch = seq_len_q_real
            if cross:
                seq_len_kv_pad = ((seq_len_kv_real + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
            else:
                seq_len_kv_pad = ((seq_len_kv_real + block_m - 1) // block_m) * block_m
                seq_len_kv_pad = ((seq_len_kv_pad + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
            n_pad_kv = seq_len_kv_pad - seq_len_kv_real
            q_p = _as_contig(q)
            if n_pad_kv > 0:
                k_p = F.pad(_as_contig(k), (0, 0, 0, 0, 0, n_pad_kv))
                v_p = F.pad(_as_contig(v), (0, 0, 0, 0, 0, n_pad_kv))
            else:
                k_p = _as_contig(k)
                v_p = _as_contig(v)
            o_shape = (batch, seq_len_q_launch, num_heads, logical_d)
            if out is not None:
                if tuple(out.shape) != o_shape or out.dtype != torch.bfloat16 or out.device != q.device:
                    raise ValueError(f"flydsl_flash_attn_iu4_func: out must be bf16 {o_shape} on {q.device}")
                o_p = out
            else:
                o_p = torch.empty(o_shape, dtype=torch.bfloat16, device=q.device)
            with torch.cuda.device(q.device.index):
                launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
                if launch_stream.device != q.device:
                    raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
                exe(
                    _flat(q_p),
                    _flat(k_p),
                    _flat(v_p),
                    _flat(o_p),
                    batch,
                    seq_len_q_launch,
                    seq_len_kv_pad,
                    seq_len_kv_real,
                    qd,
                    kd,
                    vd,
                    stream=launch_stream,
                )
            return o_p

    # Fallback: unpack nibble-packed int4 → iu8 FA.
    q8 = _unpack_iu4_nibbles_to_i8(_as_contig(q))
    k8 = _unpack_iu4_nibbles_to_i8(_as_contig(k))
    v8 = _unpack_iu4_nibbles_to_i8(_as_contig(v))
    return flydsl_flash_attn_int8_func(
        q8,
        k8,
        v8,
        causal=causal,
        sm_scale=sm_scale,
        waves_per_eu=waves_per_eu,
        daz=daz,
        stream=stream,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        out=out,
        bias=bias,
        alibi_slopes=alibi_slopes,
        sink=sink,
        return_lse=return_lse,
        num_kv_heads=num_kv_heads,
    )
