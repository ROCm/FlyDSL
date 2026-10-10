# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlyDSL Project Contributors
"""High-level FlyDSL Flash Attention APIs for gfx120x (RDNA4).

FlyDSL FlashAttention for the gfx120x family:
  - bf16 / fp16 dense self + non-causal cross (unequal Sq/Sk)
  - causal self (equal seqlens) and causal×cross via in-kernel bottom-right masking
  - FP8 e4m3fn (+ e5m2) with per-tensor descales; self + cross (+ causal×cross)
  - KV tile pad + in-kernel ``seq_len_kv_valid`` mask
  - adaptive BLOCK_M / waves_per_eu
  - head_dim in [64, 480], ``% 32 == 0`` after pad-to-tile (hosts zero-pad D and
    keep ``sm_scale=1/sqrt(orig_D)`` when safe; D>480 raises — bf16 LDS budget
    at prefetch1; paged keeps a hard gate unless layout-safe)
  - optional dense additive attn bias / general mask (fp32 [Sq, Skv]);
    multi-rank masks reduced only when leading dims are broadcast-singleton (1);
    identical-across-B/H slices are not probed on host
  - noop mask detection on host (None / empty only; vacuous all-True/all-zero
    not short-circuited without a host sync)
  - optional ``return_lse`` via in-kernel fp32 LSE epilogue ``[B,H,Sq]``
  - uniform ALiBi slopes folded into the additive bias; per-head-varying
    slopes fold to ``[H,Sq,Skv]`` and load by head inside one kernel launch
  - optional attention-sink folded into online softmax in-kernel (dense)
  - packed varlen (cu_seqlens) and paged KV (linear, linear3d, and vectorized) in-kernel (bf16/fp16)
  - split-K is an in-kernel partial (fp32 workspace) plus a wave32 combine

Int8 QKV uses ``flydsl_flash_attn_int8_func`` (iu8 WMMA); callers that pass
int8 into the bf16 entry get a clear ValueError.

Family name is gfx120x; hardware may report gfx1201/gfx1200/…. Native pack is
dense BSHD plus packed-varlen and paged KV (linear, linear3d, and vectorized); dense-paged + ragged seqlen_k auto-routes to varlen-paged. gfx950 dualwave
and gfx1250 paths stay on their own arches via ``flash_attn_interface``.
"""

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Optional

import torch

import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec
from kernels.common.gfx120x_arch import is_gfx120x, require_gfx120x
from kernels.common.gfx120x_pad import device_pad

# WMMA score columns for one lane: low K-half, then the +16 half.
_SCORE_COLUMNS = (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23)


def _column_killed(kc, op: str, limit):
    if op == ">":
        return kc > limit
    if op == ">=":
        return kc >= limit
    if op == "<":
        return kc < limit
    raise AssertionError(op)


def kill_score_columns(scores, kv_start_i32, klane_off_i32, preds, c_neg_inf):
    """Replace a score with -inf when any ``(op, limit)`` pred matches.

    The caller keeps the tile-level ``if`` so a fully valid tile emits no
    compares.
    """
    out = []
    for i in range_constexpr(len(_SCORE_COLUMNS)):
        kc = kv_start_i32 + fx.Int32(_SCORE_COLUMNS[i]) + klane_off_i32
        score = scores[i]
        for op, limit in preds:
            score = _column_killed(kc, op, limit).select(c_neg_inf, score)
        out.append(score)
    return out


def apply_sliding_window(
    scores,
    kv_start_i32,
    klane_i32,
    q_row_i32,
    *,
    swa_left: int,
    swa_right: int,
    causal: bool,
    causal_br_off_i32,
    seq_len_kv_valid_i32,
    reduction_peer,
    fmax,
    c_neg_inf,
    c_zero_f,
    c_one_f,
):
    """Drop scores outside the window. Returns the masked scores and row keep-mass.

    The band is measured from the causal bottom-right key when ``causal`` is
    set, and from ``q_row`` otherwise. Unsigned ``(key - origin + left)``
    rejects a negative key.
    """
    keeps = []
    out = []
    left_i32 = fx.Int32(swa_left)
    band_u32 = fx.Uint32(swa_left + swa_right)
    if causal:
        origin_i32 = q_row_i32 + causal_br_off_i32
    else:
        origin_i32 = q_row_i32
    for i in range_constexpr(len(_SCORE_COLUMNS)):
        key = kv_start_i32 + fx.Int32(_SCORE_COLUMNS[i]) + klane_i32 * fx.Int32(8)
        shifted = (key - origin_i32 + left_i32).bitcast(fx.Uint32)
        in_band = shifted <= band_u32
        if causal:
            in_keep = key <= (q_row_i32 + causal_br_off_i32)
        else:
            in_keep = key < seq_len_kv_valid_i32
        band_f = in_band.select(c_one_f, c_zero_f)
        keep_f = in_keep.select(band_f, c_zero_f)
        keeps.append(keep_f)
        out.append((keep_f > c_zero_f).select(scores[i], c_neg_inf))
    live = keeps[0]
    for keep in keeps[1:]:
        live = fmax(live, keep)
    return out, fmax(live, reduction_peer(live))


def add_score_bias(
    scores,
    kv_start_i32,
    klane_off_i32,
    row_base_i64,
    bias_buf,
    *,
    scale,
    clamp: bool,
    seq_len_kv_valid_i32,
    c_zero_f,
):
    """Add one bias row onto the 16 scores. ``scale`` is None for a bf16 logit."""
    out = []
    for i in range_constexpr(len(_SCORE_COLUMNS)):
        kc = kv_start_i32 + fx.Int32(_SCORE_COLUMNS[i]) + klane_off_i32
        kc_load = kc
        if clamp:
            # A partial tile can address past this batch's K. Clamp and drop it.
            in_k = kc < seq_len_kv_valid_i32
            kc_load = in_k.select(kc, fx.Int32(0))
        bv = fx.Float32(fx.ptr_load(bias_buf + fx.Int32(row_base_i64 + fx.Int64(kc_load))))
        if clamp:
            bv = in_k.select(bv, c_zero_f)
        if scale is not None:
            bv = bv * scale
        out.append(scores[i] + bv)
    return out


def add_alibi_scores(
    scores,
    kv_start_i32,
    klane_off_i32,
    q_row_i32,
    sq_i32,
    sk_i32,
    slope_f,
    *,
    scale,
):
    """Add ``-slope * |q_row + sk - sq - key|`` onto the 16 unscaled scores.

    Bottom-right, one slope per head or one shared slope. ``scale`` matches
    ``add_score_bias`` so the softmax sees ``qk * sm_scale + alibi``. A score
    that is already ``-inf`` stays ``-inf``.
    """
    # Same expression as fill_alibi_bias: 0 - slope * abs(float(i + sk - sq - j)).
    pos = q_row_i32 + sk_i32 - sq_i32
    out = []
    for i in range_constexpr(len(_SCORE_COLUMNS)):
        key = kv_start_i32 + fx.Int32(_SCORE_COLUMNS[i]) + klane_off_i32
        dist = fx.absf(fx.Float32(pos - key))
        term = (fx.Float32(0.0) - slope_f * dist) * scale
        # A pad column is already -inf. Adding a finite term under no-nans
        # can poison it. Those columns contribute nothing.
        term = (key < sk_i32).select(term, fx.Float32(0.0))
        out.append(scores[i] + term)
    return out


def attention_lse(m_scaled, l_final, has_mass, c_neg_inf, *, has_sink: bool):
    """``m + log(l)``. An empty row with no sink stores -inf."""
    lse = m_scaled + fx.log(l_final, fastmath=fx.arith.FastMathFlags.fast)
    if has_sink:
        return lse
    return has_mass.select(lse, c_neg_inf)


def attention_inv_l(l_final, has_mass, c_one_f, c_zero_f):
    """``1/l`` when the row has mass, else 0. Callers apply their own output scale."""
    return has_mass.select(c_one_f / l_final, c_zero_f)


def clear_o_if_empty(o_finals, has_mass, zero_vec):
    """Zero O when the row has no mass so a later ``0 * NaN`` cannot leak."""
    for dc in range_constexpr(len(o_finals)):
        o_finals[dc] = has_mass.select(o_finals[dc], zero_vec)
    return o_finals


def online_softmax_tile(
    s_raw,
    m_running,
    l_running,
    o_accs,
    scale,
    *,
    reduction_peer,
    fmax,
    fadd,
    fmul,
    fsub,
    c_neg_inf,
    c_zero_f,
    c_one_f,
    guard_prev_dead: bool,
    window=None,
):
    """One KV tile of the running max, sum, and output rescale.

    ``window`` is ``(seen, row_live)`` for the fp8 sliding-window path.
    ``guard_prev_dead`` is the fp8 non-window correction. bf16 leaves it false.
    """
    n_scores = len(s_raw)
    local_max = s_raw[0]
    for r in range_constexpr(n_scores - 1):
        local_max = fmax(local_max, s_raw[r + 1])
    row_max = fmax(local_max, reduction_peer(local_max))

    if window is not None:
        seen, row_live = window
        prev_alive = seen > c_zero_f
        tile_alive = row_live > c_zero_f
        prev_f = prev_alive.select(c_one_f, c_zero_f)
        tile_f = tile_alive.select(c_one_f, c_zero_f)
        both = (prev_f * tile_f) > c_zero_f
        m_paired = fmax(prev_alive.select(m_running, c_zero_f), tile_alive.select(row_max, c_zero_f))
        m_new_raw = both.select(m_paired, row_max)
        row_alive = tile_alive
        diff_m_raw = fsub(both.select(m_running, c_zero_f), both.select(m_paired, c_zero_f))
        corr = fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(fmul(diff_m_raw, scale)).ir_value()))
        corr = both.select(corr, c_zero_f)
        corr_o = both.select(corr, c_one_f)
        m_new_for_scale = tile_alive.select(m_new_raw, c_zero_f)
        s_raw = [tile_alive.select(s, c_zero_f) for s in s_raw]
    elif guard_prev_dead:
        tile_alive = row_max > c_neg_inf
        m_new_raw = tile_alive.select(fmax(m_running, row_max), m_running)
        row_alive = m_new_raw > c_neg_inf
        prev_alive = m_running > c_neg_inf
        diff_m_raw = fsub(m_running, m_new_raw)
        diff_m_raw = row_alive.select(diff_m_raw, c_zero_f)
        diff_m_raw = prev_alive.select(diff_m_raw, c_zero_f)
        corr = fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(fmul(diff_m_raw, scale)).ir_value()))
        corr = row_alive.select(corr, c_zero_f)
        corr_o = row_alive.select(corr, c_one_f)
        m_new_for_scale = row_alive.select(m_new_raw, c_zero_f)
    else:
        m_new_raw = fmax(m_running, row_max)
        row_alive = m_new_raw > c_neg_inf
        diff_m_raw = row_alive.select(fsub(m_running, m_new_raw), c_zero_f)
        corr = fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(fmul(diff_m_raw, scale)).ir_value()))
        corr = row_alive.select(corr, c_zero_f)
        corr_o = row_alive.select(corr, c_one_f)
        m_new_for_scale = row_alive.select(m_new_raw, c_zero_f)

    neg_scaled_max = fsub(c_zero_f, fmul(scale, m_new_for_scale))
    p_vals = []
    local_sum = c_zero_f
    for r in range_constexpr(n_scores):
        diff = fx.math.fma(s_raw[r], scale, neg_scaled_max)
        p_vals.append(fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(diff).ir_value())))
        local_sum = fadd(local_sum, p_vals[-1])
    if window is not None:
        p_vals = [tile_alive.select(p, c_zero_f) for p in p_vals]
        local_sum = tile_alive.select(local_sum, c_zero_f)
    tile_sum = fadd(local_sum, reduction_peer(local_sum))
    l_new = fadd(fmul(corr, l_running), tile_sum)

    corr_vec = Vec.from_elements([corr_o], fx.Float32).broadcast_to(8)
    for dc in range_constexpr(len(o_accs)):
        o_accs[dc] = fmul(o_accs[dc], corr_vec)
    return p_vals, m_new_raw, l_new, o_accs, row_alive


def fold_attention_sink(
    o_finals,
    l_final,
    m_nat,
    sink_logit,
    has_mass_pre,
    *,
    fmax,
    fadd,
    fmul,
    fsub,
    c_zero_f,
    c_log2e,
):
    """Rescale O and l when a sink logit wins the final max. Returns O, l, m."""
    m_new = fmax(m_nat, sink_logit)
    corr = fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(fmul(fsub(m_nat, m_new), c_log2e)).ir_value()))
    sink_w = fx.Float32(
        fx.rocdl.exp2(fx.Float32.ir_type, fx.Float32(fmul(fsub(sink_logit, m_new), c_log2e)).ir_value())
    )
    zero_vec = Vec.from_elements([c_zero_f], fx.Float32).broadcast_to(8)
    corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8)
    for dc in range_constexpr(len(o_finals)):
        o_finals[dc] = fmul(has_mass_pre.select(o_finals[dc], zero_vec), corr_vec)
    l_final = fadd(fmul(has_mass_pre.select(l_final, c_zero_f), corr), sink_w)
    return o_finals, l_final, m_new


__all__ = [
    "flydsl_flash_attn_func",
    "flydsl_flash_attn_varlen_func",
    "flydsl_flash_attn_paged_func",
    "flydsl_flash_attn_varlen_paged_func",
    "flydsl_flash_attn_fp8_func",
    "flydsl_flash_attn_int8_func",
    "is_gfx120x",
    "normalize_attn_mask",
    "mask_is_noop",
    "fold_alibi_to_bias",
    "add_alibi_scores",
    "add_score_bias",
    "apply_sliding_window",
    "attention_inv_l",
    "attention_lse",
    "clear_o_if_empty",
    "fold_attention_sink",
    "kill_score_columns",
    "online_softmax_tile",
]

_KERNEL_BLOCK_M = 128
_KERNEL_BLOCK_N = 32
_RDNA4_LDS_BYTES = 65536
_FA_BLOCK_N = 32
_FA_PREFETCH_KV = 1  # NUM_PREFETCH_K == NUM_PREFETCH_V == 1 in the kernel
# LDS = 2 * prefetch * BLOCK_N * (D+4) * sizeof(bf16) <= 64 KiB
# => D+4 <= 512 => D <= 508 => align down to %32 == 0 => 480 (≈60.5 KiB)
_MAX_HEAD_DIM = (((_RDNA4_LDS_BYTES // (2 * _FA_PREFETCH_KV * _FA_BLOCK_N * 2)) - 4) // 32) * 32
_MIN_HEAD_DIM = 64
_DUMMY_BIAS_CACHE: dict = {}
_DUMMY_I32_CACHE: dict = {}


def _head_dim_tile_target(head_dim: int, *, what: str) -> int:
    """Next legal FA head_dim tile (>=64, %32==0, <=LDS max), or raise if too large."""
    d = int(head_dim)
    if d > _MAX_HEAD_DIM:
        raise ValueError(
            f"{what}: head_dim={d} > {_MAX_HEAD_DIM} is not "
            "supported on gfx120x FlyDSL FA (LDS / register tile budget)."
        )
    target = max(_MIN_HEAD_DIM, ((d + 31) // 32) * 32)
    if target > _MAX_HEAD_DIM:
        raise ValueError(
            f"{what}: head_dim={d} pads to {target} > {_MAX_HEAD_DIM} (cannot pad-to-tile within LDS budget)."
        )
    return target


def _crop_head_dim(out: torch.Tensor, orig_d: int) -> torch.Tensor:
    """Crop padded FA output back to the caller head_dim."""
    if int(out.shape[-1]) == orig_d:
        return out
    return out[..., :orig_d].contiguous()


def _as_contig(t: torch.Tensor, stream: torch.cuda.Stream | None = None) -> torch.Tensor:
    from kernels.common.gfx120x_pad import ensure_contiguous

    return ensure_contiguous(t, stream=stream)


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.view(-1) if t.is_contiguous() else t.reshape(-1)


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_flash_attn_func only supports bf16/f16 for the dense path, got {dtype!r}")


def _pinned_block_m(requested: int | None, picked: int) -> int:
    """Use a caller tile when ``fp8_block_m`` is set. Otherwise keep the sweep pick."""
    if requested is None:
        return int(picked)
    bm = int(requested)
    if bm <= 0 or bm % 16 != 0 or bm > 256:
        raise ValueError(f"gfx120x FA fp8_block_m must be a positive multiple of 16 and <= 256, got {bm}")
    return bm


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


def _quant_index_tail(device: torch.device) -> tuple:
    """Dummy cu_seqlens / block table / split workspace for the dense quant launch."""
    z = _flat(_dummy_i32(device))
    zb = _flat(_dummy_bias(device))
    return (z, z, z, z, 0, 1, zb, zb, zb, 0)


def _quant_run_exe(
    exe,
    parts: tuple,
    *,
    device: torch.device,
    launch_stream: torch.cuda.Stream,
    o_p: torch.Tensor,
    batch: int,
    seq_q: int,
    num_heads: int,
    tile_d: int,
    logical_d: int,
    num_kv_splits: int,
    has_sink: bool,
    return_lse: bool,
    sink_t: torch.Tensor | None,
    lse_p: torch.Tensor | None,
    daz: bool,
) -> torch.Tensor:
    """Launch a quant FA kernel. Split-K writes partials, then the combine kernel.

    The workspace is the partial m/l/o result. Q, K, and V are not cloned.
    """
    nsplits = int(num_kv_splits)
    if nsplits < 1:
        raise ValueError(f"gfx120x quant FA: num_kv_splits must be >= 1, got {nsplits}")
    comb_out = o_p
    if nsplits == 1:
        tail = _quant_index_tail(device)
        ws = None
    else:
        ws_rows = int(batch) * int(num_heads) * int(seq_q)
        ws_m, ws_l, ws_o = _new_split_workspace(device, nsplits, ws_rows, tile_d)
        if int(logical_d) != int(tile_d):
            comb_out = torch.empty(
                (int(batch), int(seq_q), int(num_heads), int(tile_d)),
                dtype=torch.bfloat16,
                device=device,
            )
        z = _flat(_dummy_i32(device))
        tail = (
            z,
            z,
            z,
            z,
            0,
            nsplits,
            _flat(ws_m),
            _flat(ws_l),
            _flat(ws_o),
            ws_rows,
        )
        ws = (ws_m, ws_l, ws_o, ws_rows)
    exe(*parts, *tail, stream=launch_stream)
    if ws is not None:
        ws_m, ws_l, ws_o, ws_rows = ws
        lse_c = lse_p if return_lse and lse_p is not None else _dummy_bias(device)
        sink_c = sink_t if has_sink and sink_t is not None else _dummy_bias(device)
        comb = _get_splitk_combine(int(num_heads), int(tile_d), "bf16", bool(has_sink), bool(return_lse), bool(daz))
        comb(
            ws_m,
            ws_l,
            ws_o,
            comb_out,
            lse_c,
            sink_c,
            int(batch),
            int(seq_q),
            nsplits,
            ws_rows,
            stream=launch_stream,
        )
        if comb_out.data_ptr() != o_p.data_ptr():
            with torch.cuda.stream(launch_stream):
                o_p.copy_(comb_out[..., : int(logical_d)])
    return o_p


def _dummy_i32(device: torch.device) -> torch.Tensor:
    key = device.index if getattr(device, "index", None) is not None else int(device)
    buf = _DUMMY_I32_CACHE.get(key)
    if buf is None or buf.device != device:
        buf = torch.zeros(1, dtype=torch.int32, device=device)
        _DUMMY_I32_CACHE[key] = buf
    return buf


def mask_is_noop(mask: torch.Tensor | None) -> bool:
    """True when mask can be ignored without reading values.

    Only ``None`` / empty tensors are treated as noop. Detecting all-True bool
    or all-zero additive masks requires a host sync; callers that know the
    mask is vacuous should pass ``None`` instead. Nonempty masks always go
    through :func:`normalize_attn_mask` (bool→device additive).
    """
    if mask is None:
        return True
    if not hasattr(mask, "dtype"):
        return False
    return int(mask.numel()) == 0


def normalize_attn_mask(
    mask: torch.Tensor | None,
    seq_len_q: int,
    seq_len_kv: int,
    device: torch.device,
    *,
    stream: torch.cuda.Stream | None = None,
) -> Optional[torch.Tensor]:
    """Normalize SDPA-style mask to dense fp32 additive bias [Sq, Skv].

    Returns None for noop masks. Bool False → -inf; True → 0. Additive masks
    are cast to fp32.

    Accepted ranks (reduced to ``[Sq, Skv]`` when leading dims broadcast):
      - ``[Sq, Skv]``
      - ``[1, Sq, Skv]`` / ``[1, 1, Sq, Skv]`` (leading dims must be 1 — no
        host ``torch.equal`` probe of multi-slice masks).

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
        if flat.shape[0] != 1:
            raise ValueError(
                f"gfx120x FA attn_mask rank-4 shape {tuple(mask.shape)} must "
                "broadcast to a single [Sq, Skv] slice (leading dims all 1); "
                "pass shared [Sq, Skv] explicitly (no host torch.equal probe)"
            )
        m = flat[0]
    elif m.dim() == 3:
        # (1, Sq, Skv) or (B, Sq, Skv) with identical rows, or (H, Sq, Skv).
        lead, sq, sk = m.shape
        if sq != seq_len_q or sk != seq_len_kv:
            # Maybe (B, H, Skv) style — not supported without Sq.
            raise ValueError(
                f"gfx120x FA attn_mask shape {tuple(mask.shape)} incompatible with Sq={seq_len_q} Skv={seq_len_kv}"
            )
        if lead != 1:
            raise ValueError(
                f"gfx120x FA attn_mask rank-3 shape {tuple(mask.shape)} must be "
                "[1, Sq, Skv] (shared) or pass per-head as [H, Sq, Skv] via bias= "
                "with explicit 3D after bool conversion; no host torch.equal probe"
            )
        m = m[0]
    elif m.dim() == 2:
        pass
    elif m.dim() > 4:
        # Squeeze leading singletons then retry once.
        while m.dim() > 4 and m.shape[0] == 1:
            m = m.squeeze(0)
        if m.dim() != 4 and m.dim() != 2:
            raise ValueError(f"gfx120x FA attn_mask must reduce to 2D [Sq, Skv], got shape={tuple(mask.shape)}")
        return normalize_attn_mask(m, seq_len_q, seq_len_kv, device, stream=stream)
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
        from kernels.attention.gfx120x_attn_mask import bool_mask_to_additive

        return _as_contig(bool_mask_to_additive(m, stream=stream))
    if m.dtype == torch.float32:
        return _as_contig(m, stream=stream)
    return _as_contig(m.to(torch.float32), stream=stream)


def _as_score_bias(
    mask: torch.Tensor | None,
    seq_q: int,
    seq_k: int,
    num_heads: int,
    device: torch.device,
    *,
    stream: torch.cuda.Stream | None = None,
) -> Optional[torch.Tensor]:
    """fp32 score bias. ``[H, Sq, Sk]`` with ``H == num_heads > 1`` stays per-head."""
    if mask is None or mask_is_noop(mask):
        return None
    if mask.dim() == 3 and int(mask.shape[0]) == int(num_heads) and int(num_heads) > 1:
        if int(mask.shape[1]) != int(seq_q) or int(mask.shape[2]) != int(seq_k):
            raise ValueError(f"gfx120x FA per-head bias {tuple(mask.shape)} incompatible with Sq={seq_q} Skv={seq_k}")
        if mask.dtype == torch.bool:
            from kernels.attention.gfx120x_attn_mask import bool_mask_to_additive

            return bool_mask_to_additive(mask, stream=stream)
        if mask.dtype == torch.float32:
            return _as_contig(mask, stream=stream)
        return _as_contig(mask.to(torch.float32), stream=stream)
    return normalize_attn_mask(mask, seq_q, seq_k, device, stream=stream)


def _normalize_alibi_slopes(
    alibi_slopes: torch.Tensor | float | None,
    device: torch.device,
    *,
    stream: torch.cuda.Stream | None = None,
    num_heads: int | None = None,
) -> tuple[torch.Tensor | None, bool]:
    """fp32 slopes, ``[1]`` or ``[H]``. ``None`` when ALiBi is off.

    The kernel adds ``-slope * |i + sk - sq - j|`` itself. This does not
    build an ``[Sq, Sk]`` tensor.
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    if alibi_slopes is None:
        return None, False
    if isinstance(alibi_slopes, (int, float)):
        slope = float(alibi_slopes)
        if slope < 0:
            raise ValueError(f"gfx120x FA: ALiBi slope must be >= 0, got {slope}")
        return torch.tensor([slope], device=device, dtype=torch.float32), False
    if not isinstance(alibi_slopes, torch.Tensor):
        raise TypeError(f"gfx120x FA: alibi_slopes must be float or Tensor, got {type(alibi_slopes).__name__}")
    s = alibi_slopes.detach().to(device=device, dtype=torch.float32)
    if s.numel() == 0:
        raise ValueError("gfx120x FA: empty alibi_slopes")
    if s.dim() == 2:
        if s.shape[0] > 1:
            raise ValueError(f"gfx120x FA: batch-varying alibi_slopes not supported (got {tuple(alibi_slopes.shape)})")
        s = s[0]
    s = ensure_contiguous(s.reshape(-1), stream=stream)
    per_head = int(s.numel()) != 1
    if per_head and num_heads is not None and int(s.numel()) != int(num_heads):
        raise ValueError(f"gfx120x FA: alibi_slopes H={int(s.numel())} != num_heads={int(num_heads)}")
    return s, per_head


def fold_alibi_to_bias(
    alibi_slopes: torch.Tensor | None,
    seq_len_q: int,
    seq_len_kv: int,
    device: torch.device,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Fold ALiBi slopes into additive bias (bottom-right aligned) on device.

    ``-slope * |i + Skv - Sq - j|``. Scalar/uniform → ``[Sq, Skv]``;
    per-head-varying (1D ``[H]``) → ``[H, Sq, Skv]``. Uses
    ``kernels.attention.gfx120x_alibi_bias.fill_alibi_bias`` (no host arange/abs).
    Pass ``stream`` so fill runs on the same stream as the attention launch.
    The attention kernels do not read this tensor. They consume the slopes.
    Tests use the folded bias as the reference mask.
    """
    from kernels.attention.gfx120x_alibi_bias import fill_alibi_bias

    slopes, per_head = _normalize_alibi_slopes(alibi_slopes, device, stream=stream)
    if slopes is None:
        raise ValueError("fold_alibi_to_bias: alibi_slopes is None")
    return fill_alibi_bias(slopes, seq_len_q, seq_len_kv, per_head=per_head, stream=stream)


def _lengths_are_ragged(cu_q, max_q: int, max_k: int, cu_kv=None, kv_lengths=None) -> bool:
    """True when any sequence is shorter than the max lengths the bias was built on."""
    cq = cu_q.detach().to(device="cpu", dtype=torch.int64).tolist()
    lens = _varlen_kv_lengths(cu_kv, kv_lengths) if (cu_kv is not None or kv_lengths is not None) else None
    if lens is not None and len(lens) != len(cq) - 1:
        raise ValueError(f"gfx120x varlen KV lengths {len(lens)} != batches {len(cq) - 1}")
    for i in range(len(cq) - 1):
        sq = int(cq[i + 1]) - int(cq[i])
        sk = int(lens[i]) if lens is not None else int(max_k)
        if sq < int(max_q) or sk < int(max_k):
            return True
    return False


def _varlen_alibi_parts(
    user_bias: torch.Tensor | None,
    alibi_slopes,
    *,
    num_heads: int,
    device: torch.device,
    stream: torch.cuda.Stream | None,
    cu_q: torch.Tensor,
    max_q: int,
    max_k: int,
    cu_kv: torch.Tensor | None = None,
    kv_lengths: list[int] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, bool, bool]:
    """Return ``(folded_bias_or_none, slopes_or_none, has_alibi, alibi_per_head)``.

    Ragged per-head mixes cannot share one ``[H, max_q, max_k]`` tensor: the
    user mask is top-left and ALiBi is bottom-right. Those launches pass the
    slopes into ``add_alibi_scores`` and leave the user mask as the only bias.
    Every other case still folds ALiBi into the bias so equal lengths stay
    one scaled add.
    """
    slopes, per_head = _normalize_alibi_slopes(alibi_slopes, device, stream=stream, num_heads=num_heads)
    user_ph = user_bias is not None and user_bias.dim() == 3 and int(user_bias.shape[0]) == int(num_heads)
    ragged = slopes is not None and user_bias is not None and (user_ph or per_head)
    if ragged and _lengths_are_ragged(cu_q, max_q, max_k, cu_kv, kv_lengths):
        return None, slopes, True, per_head
    if slopes is None:
        return None, None, False, False
    return (
        fold_alibi_to_bias(alibi_slopes, max_q, max_k, device, stream=stream),
        None,
        False,
        False,
    )


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
    sm_scale: float | None = None,
    logical_head_dim: int | None = None,
    kv_oob: bool = False,
    sliding_window: tuple[int, int] | None = None,
    bias_bottom_right: bool = False,
    has_alibi: bool = False,
    alibi_per_head: bool = False,
) -> Callable[..., None]:
    from kernels.attention.flash_attn_gfx120x import build_flash_attn_func_module

    if sliding_window is not None:
        sliding_window = (int(sliding_window[0]), int(sliding_window[1]))
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
        sm_scale=sm_scale,
        logical_head_dim=logical_head_dim,
        kv_oob=kv_oob,
        sliding_window=sliding_window,
        bias_bottom_right=bias_bottom_right,
        has_alibi=has_alibi,
        alibi_per_head=alibi_per_head,
    )


def _new_split_workspace(device: torch.device, nsplits: int, ws_rows: int, head_dim: int):
    """Partials the kernel does not store must look empty, not uninitialized."""
    ws_m = torch.full((int(nsplits), int(ws_rows)), float("-inf"), dtype=torch.float32, device=device)
    ws_l = torch.zeros((int(nsplits), int(ws_rows)), dtype=torch.float32, device=device)
    ws_o = torch.zeros((int(nsplits), int(ws_rows), int(head_dim)), dtype=torch.float32, device=device)
    return ws_m, ws_l, ws_o


def _nosplit_args(device: torch.device) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    z = _flat(_dummy_bias(device))
    return (1, z, z, z, 0)


def _float_split_tail(
    device: torch.device,
    nsplits: int,
    batch: int,
    seq_q: int,
    num_heads: int,
    head_dim: int,
):
    """Split-K workspace for the bf16/fp16 kernel. ``None`` workspace means one split."""
    if int(nsplits) <= 1:
        return _nosplit_args(device), None
    ws_rows = int(batch) * int(num_heads) * int(seq_q)
    ws_m, ws_l, ws_o = _new_split_workspace(device, nsplits, ws_rows, head_dim)
    tail = (int(nsplits), _flat(ws_m), _flat(ws_l), _flat(ws_o), ws_rows)
    return tail, (ws_m, ws_l, ws_o, ws_rows)


def _combine_float_splits(
    ws,
    o_p: torch.Tensor,
    *,
    batch: int,
    seq_q: int,
    num_heads: int,
    head_dim: int,
    dtype_str: str,
    has_sink: bool,
    return_lse: bool,
    sink_t: torch.Tensor | None,
    lse_p: torch.Tensor | None,
    daz: bool,
    stream: torch.cuda.Stream,
) -> None:
    ws_m, ws_l, ws_o, ws_rows = ws
    device = o_p.device
    comb = _get_splitk_combine(int(num_heads), int(head_dim), dtype_str, bool(has_sink), bool(return_lse), bool(daz))
    comb(
        ws_m,
        ws_l,
        ws_o,
        o_p,
        lse_p if return_lse and lse_p is not None else _dummy_bias(device),
        sink_t if has_sink and sink_t is not None else _dummy_bias(device),
        int(batch),
        int(seq_q),
        int(ws_m.shape[0]),
        ws_rows,
        stream=stream,
    )


@lru_cache(maxsize=32)
def _get_splitk_combine(
    num_heads: int, head_dim: int, dtype_str: str, has_sink: bool, return_lse: bool, daz: bool = True
) -> Callable[..., None]:
    from kernels.attention.flash_attn_gfx120x_splitk import build_splitk_combine_module

    return build_splitk_combine_module(
        num_heads=num_heads,
        head_dim=head_dim,
        dtype_str=dtype_str,
        has_sink=has_sink,
        return_lse=return_lse,
        daz=daz,
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
    sliding_window: tuple[int, int] | None = None,
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
        head_dim: padded to ``[64, 480]`` with ``% 32 == 0`` when safe; ``>480`` raises (LDS).
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    if q.dtype in (torch.int8, torch.uint8):
        raise ValueError(
            "flydsl_flash_attn_func: int8 QKV belongs on flydsl_flash_attn_int8_func "
            "(iu8 WMMA + descales); bf16 path is bf16/fp16 only."
        )
    # Self-contained family gate at this gfx120x-only entry (shared FA soft-routes first).
    require_gfx120x(q.device, what="flydsl_flash_attn_func (gfx120x)")
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    if launch_stream.device != q.device:
        raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
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
    # Causal self + causal×cross: in-kernel bottom-right masking.
    kernel_causal = bool(causal)
    orig_head_dim = int(head_dim)
    launch_head_dim = _head_dim_tile_target(orig_head_dim, what="flydsl_flash_attn_func")
    sm_scale_pad = None
    logical_head_dim = None
    if launch_head_dim != orig_head_dim:
        # Tail lanes are zeroed in the kernel. Scale stays on the caller D.
        sm_scale_pad = 1.0 / math.sqrt(float(orig_head_dim))
        logical_head_dim = orig_head_dim
        head_dim = launch_head_dim

    # Merge attn_mask / bias (normalize each, then add) + optional ALiBi fold.
    # Causal / causal×cross masking runs in-kernel (no host bottom_right bias).
    # [H, Sq, Sk] with H>1 is per-head and must not be squeezed to one row.
    # The router stores attn_mask in bias. Adding both would double the logits.
    norm_mask = (
        None
        if attn_mask is bias
        else _as_score_bias(attn_mask, seq_len_q_real, seq_len_kv_real, num_heads, q.device, stream=stream)
    )
    norm_bias = _as_score_bias(bias, seq_len_q_real, seq_len_kv_real, num_heads, q.device, stream=stream)
    bias_t = _merge_bias(norm_mask, norm_bias)
    if alibi_slopes is not None:
        bias_t = _merge_bias(
            bias_t, fold_alibi_to_bias(alibi_slopes, seq_len_q_real, seq_len_kv_real, q.device, stream=stream)
        )
    has_alibi = False
    alibi_per_head = False
    slopes_t = None
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
            sink_t = ensure_contiguous(s.view(1, num_heads).expand(batch, num_heads), stream=stream)
        elif s.dim() == 2:
            if tuple(s.shape) != (batch, num_heads):
                raise ValueError(f"gfx120x FA sink shape {tuple(s.shape)} != {(batch, num_heads)}")
            sink_t = _as_contig(s, stream=launch_stream)
        else:
            raise ValueError(f"gfx120x FA sink must be [H] or [B,H], got {tuple(sink.shape)}")
        if sink_t.device != q.device:
            sink_t = sink_t.to(q.device)
    has_sink = sink_t is not None

    dtype_str = _torch_dtype_to_str(q.dtype)
    block_m = _pick_block_m(seq_len_q_real, cross, seq_len_kv_real)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, seq_len_kv_real, cross, waves_per_eu)

    seq_len_q_launch = seq_len_q_real
    # Real K. A length the old host would have padded uses the bounds-checked
    # K kernel. An already-aligned length keeps the historical load.
    if seq_len_kv_real == 0:
        kv_oob = True
    elif cross:
        kv_oob = seq_len_kv_real % _KERNEL_BLOCK_N != 0
    else:
        _rounded = ((seq_len_kv_real + block_m - 1) // block_m) * block_m
        _rounded = ((_rounded + _KERNEL_BLOCK_N - 1) // _KERNEL_BLOCK_N) * _KERNEL_BLOCK_N
        kv_oob = _rounded != seq_len_kv_real
    seq_len_kv_pad = seq_len_kv_real
    q_p = _as_contig(q, stream=launch_stream)
    k_p = _as_contig(k, stream=launch_stream)
    v_p = _as_contig(v, stream=launch_stream)

    bias_arg = _as_contig(bias_t, stream=launch_stream) if has_bias else _dummy_bias(q.device)

    o_shape = (batch, seq_len_q_launch, num_heads, orig_head_dim)
    caller_out_shape = o_shape
    if out is not None:
        if tuple(out.shape) != caller_out_shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError(
                f"flydsl_flash_attn_func: out shape/dtype/device mismatch: "
                f"got {tuple(out.shape)}/{out.dtype}/{out.device}, "
                f"want {caller_out_shape}/{q.dtype}/{q.device}"
            )
        if not out.is_contiguous():
            raise ValueError("flydsl_flash_attn_func: out must be contiguous")
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
            sm_scale=sm_scale_pad,
            logical_head_dim=logical_head_dim,
            kv_oob=kv_oob,
            sliding_window=sliding_window,
            has_alibi=has_alibi,
            alibi_per_head=alibi_per_head,
        )
        _di = _dummy_i32(q.device)
        nsplits = int(num_kv_splits)
        if nsplits < 1:
            raise ValueError(f"gfx120x FA: num_kv_splits must be >= 1, got {nsplits}")
        if nsplits > 1:
            ws_rows = int(batch) * int(num_heads) * int(seq_len_q_launch)
            ws_m, ws_l, ws_o = _new_split_workspace(q.device, nsplits, ws_rows, head_dim)
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
            _flat(_dummy_bias(q.device) if not has_alibi else slopes_t),
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
            comb = _get_splitk_combine(num_heads, head_dim, dtype_str, bool(has_sink), bool(return_lse), bool(daz))
            # The combine strides by the WMMA tile. A shorter head needs its own buffer.
            comb_o = o_p
            if int(head_dim) != int(orig_head_dim):
                comb_o = torch.empty(
                    (batch, seq_len_q_launch, num_heads, head_dim),
                    dtype=o_p.dtype,
                    device=o_p.device,
                )
            comb(
                ws_m,
                ws_l,
                ws_o,
                comb_o,
                lse_c,
                sink_c,
                batch,
                seq_len_q_launch,
                nsplits,
                ws_rows,
                stream=launch_stream,
            )
            if comb_o.data_ptr() != o_p.data_ptr():
                with torch.cuda.stream(launch_stream):
                    o_p.copy_(comb_o[..., : int(orig_head_dim)])

    o_p = _crop_head_dim(o_p, orig_head_dim)
    if out is not None and o_p.data_ptr() != out.data_ptr():
        if launch_stream is None:
            out.copy_(o_p)
        else:
            with torch.cuda.stream(launch_stream):
                out.copy_(o_p)
        o_p = out
    if return_lse:
        return o_p, lse_p
    return o_p


def _varlen_kv_lengths(
    cu_seqlens_kv: torch.Tensor | None,
    kv_lengths: list[int] | None,
) -> list[int]:
    """Per-sequence key lengths. ``kv_lengths`` wins when the cache is paged."""
    if kv_lengths is not None:
        return [int(x) for x in kv_lengths]
    if cu_seqlens_kv is None:
        raise ValueError("gfx120x varlen ALiBi needs cu_seqlens_kv or kv_lengths")
    ck = cu_seqlens_kv.detach().to(device="cpu", dtype=torch.int64).tolist()
    return [int(ck[i + 1]) - int(ck[i]) for i in range(len(ck) - 1)]


def _expand_shared_bias_to_packed(
    bias: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv_pad: int,
    *,
    total_q: int,
    stream: torch.cuda.Stream | None = None,
    align: str = "top_left",
    cu_seqlens_kv: torch.Tensor | None = None,
    kv_lengths: list[int] | None = None,
    max_seqlen_kv: int | None = None,
) -> torch.Tensor:
    """Expand shared [max_q, max_kv] (or already-packed [total_q, max_kv]) to packed rows.

    Kernel VARLEN bias indexing uses (q_off + q_row) * seq_len_kv + kv_col, so the
    host always passes [total_q, max_kv_pad]. A user mask is copied from the
    top-left of the shared tensor. ALiBi is bottom-right: local (i, j) of a
    shorter sequence is ``|i + sk - sq - j|``, which is the slice starting at
    ``(max_q - sq, max_k - sk)``.

    ``total_q`` comes from the packed Q shape (no ``cu_seqlens[-1].item()`` sync).
    cu_seqlens metadata is copied to CPU once (small B+1 ints) for the expand loop.
    """
    if align not in ("top_left", "bottom_right"):
        raise ValueError(f"gfx120x varlen bias align must be top_left or bottom_right, got {align}")
    total_q = int(total_q)
    b = bias
    if b.dim() != 2:
        raise ValueError(f"gfx120x varlen FA bias must be 2D [total_q|max_q, max_kv], got {tuple(b.shape)}")
    # Pad KV dim to launch pad. Content stays left-aligned, so a bottom-right
    # slice still indexes the unpadded max_k columns.
    if b.shape[1] < max_seqlen_kv_pad:
        b = device_pad(b, (0, max_seqlen_kv_pad - int(b.shape[1])), stream=stream)
    elif b.shape[1] > max_seqlen_kv_pad:
        b = b[:, :max_seqlen_kv_pad]
    if align == "top_left" and int(b.shape[0]) == total_q:
        return _as_contig(b, stream=stream)
    if int(b.shape[0]) < max_seqlen_q:
        raise ValueError(f"gfx120x varlen shared bias rows {int(b.shape[0])} < max_seqlen_q={max_seqlen_q}")
    # One small metadata D2H for cu_seqlens (B+1 ints), then host slice copies.
    cq_host = cu_seqlens_q.detach().to(device="cpu", dtype=torch.int64).tolist()
    lengths = None
    if align == "bottom_right":
        if max_seqlen_kv is None:
            raise ValueError("gfx120x bottom-right varlen bias needs max_seqlen_kv")
        lengths = _varlen_kv_lengths(cu_seqlens_kv, kv_lengths)
        if len(lengths) != len(cq_host) - 1:
            raise ValueError(f"gfx120x varlen KV lengths {len(lengths)} != batches {len(cq_host) - 1}")
    out = torch.zeros(total_q, max_seqlen_kv_pad, device=b.device, dtype=torch.float32)
    b32 = b if b.dtype == torch.float32 else b.to(dtype=torch.float32)
    B = len(cq_host) - 1
    for bi in range(B):
        qs, qe = int(cq_host[bi]), int(cq_host[bi + 1])
        sq = qe - qs
        if sq <= 0:
            continue
        if align == "bottom_right":
            sk = int(lengths[bi])
            if sk <= 0:
                continue
            row0 = int(max_seqlen_q) - sq
            col0 = int(max_seqlen_kv) - sk
            if row0 < 0 or col0 < 0:
                raise ValueError(
                    f"gfx120x ALiBi slice row0={row0} col0={col0} for sq={sq} sk={sk} "
                    f"max=({max_seqlen_q}, {max_seqlen_kv})"
                )
            out[qs:qe, :sk] = b32[row0 : row0 + sq, col0 : col0 + sk]
        else:
            out[qs:qe] = b32[:sq]
    return out


def _assemble_varlen_bias(
    user_bias: torch.Tensor | None,
    alibi_bias: torch.Tensor | None,
    *,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    seq_len_kv_pad: int,
    total_q: int,
    num_heads: int,
    stream: torch.cuda.Stream | None = None,
    cu_seqlens_kv: torch.Tensor | None = None,
    kv_lengths: list[int] | None = None,
) -> tuple[torch.Tensor | None, bool, bool]:
    """Pack varlen bias. Returns ``(bias, per_head, bottom_right)``.

    A 2D user mask is top-left and a 2D ALiBi tensor is bottom-right. They are
    expanded separately, then added. A 3D per-head tensor stays shared. The
    bottom-right flag is set only when that tensor is ALiBi with no user mask,
    so a per-head user mask keeps local row 0 at bias row 0.
    """

    def _per_head(t: torch.Tensor | None) -> bool:
        return t is not None and t.dim() == 3 and int(t.shape[0]) == int(num_heads)

    if user_bias is None and alibi_bias is None:
        return None, False, False
    user_ph = _per_head(user_bias)
    alibi_ph = _per_head(alibi_bias)
    if not user_ph and not alibi_ph:
        parts: list[torch.Tensor] = []
        if user_bias is not None:
            src = user_bias if user_bias.dtype == torch.float32 else user_bias.to(torch.float32)
            parts.append(
                _expand_shared_bias_to_packed(
                    src,
                    cu_seqlens_q,
                    max_seqlen_q,
                    seq_len_kv_pad,
                    total_q=total_q,
                    stream=stream,
                    align="top_left",
                )
            )
        if alibi_bias is not None:
            src = alibi_bias if alibi_bias.dtype == torch.float32 else alibi_bias.to(torch.float32)
            parts.append(
                _expand_shared_bias_to_packed(
                    src,
                    cu_seqlens_q,
                    max_seqlen_q,
                    seq_len_kv_pad,
                    total_q=total_q,
                    stream=stream,
                    align="bottom_right",
                    cu_seqlens_kv=cu_seqlens_kv,
                    kv_lengths=kv_lengths,
                    max_seqlen_kv=max_seqlen_kv,
                )
            )
        acc = parts[0]
        for part in parts[1:]:
            acc = acc + part
        return acc, False, False

    acc = user_bias if alibi_bias is None else alibi_bias if user_bias is None else _merge_bias(user_bias, alibi_bias)
    if acc is None:
        return None, False, False
    if acc.dtype != torch.float32:
        acc = acc.to(torch.float32)
    if int(acc.shape[-1]) < seq_len_kv_pad:
        acc = device_pad(acc, (0, seq_len_kv_pad - int(acc.shape[-1])), stream=stream)
    elif int(acc.shape[-1]) > seq_len_kv_pad:
        acc = acc[..., :seq_len_kv_pad]
    per_head = acc.dim() == 3
    # One shared 3D tensor cannot hold both a top-left user mask and a
    # bottom-right ALiBi when sequence lengths differ. Shift only pure ALiBi.
    bottom_right = bool(per_head and alibi_ph and user_bias is None)
    return acc, per_head, bottom_right


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
    sliding_window: tuple[int, int] | None = None,
    num_kv_splits: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Packed-varlen FA: Q/K/V/O as [total, H, D] with cu_seqlens int32 [B+1]."""
    from kernels.common.gfx120x_arch import require_gfx120x
    from kernels.common.gfx120x_pad import ensure_contiguous

    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_varlen_func requires CUDA/HIP tensors")
    require_gfx120x(q.device, what="flydsl_flash_attn_varlen_func")
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    if launch_stream.device != q.device:
        raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
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
    orig_head_dim = int(head_dim)
    launch_head_dim = _head_dim_tile_target(orig_head_dim, what="flydsl_flash_attn_varlen_func")
    sm_scale_pad = None
    logical_head_dim = None
    if launch_head_dim != orig_head_dim:
        sm_scale_pad = 1.0 / math.sqrt(float(orig_head_dim))
        logical_head_dim = orig_head_dim
        head_dim = launch_head_dim

    cu_q = _as_contig(cu_seqlens_q.to(torch.int32), stream=launch_stream)
    cu_kv = _as_contig(cu_seqlens_kv.to(torch.int32), stream=launch_stream)
    if cu_q.device != q.device:
        cu_q = cu_q.to(q.device)
    if cu_kv.device != q.device:
        cu_kv = cu_kv.to(q.device)
    batch = int(cu_q.numel() - 1)
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_kv = int(max_seqlen_kv)

    # Merge attn_mask into bias (normalize each like dense, then add).
    # A 3D [H, max_q, max_kv] bias is per-head and is not squeezed.
    bias_t = None
    if attn_mask is not None and not mask_is_noop(attn_mask):
        bias_t = normalize_attn_mask(attn_mask, max_seqlen_q, max_seqlen_kv, q.device, stream=stream)
    if bias is not None and not mask_is_noop(bias):
        if bias.dim() == 3 and int(bias.shape[0]) == num_heads:
            nb = bias
        else:
            nb = normalize_attn_mask(bias, max_seqlen_q, max_seqlen_kv, q.device, stream=stream)
        bias_t = nb if bias_t is None else _merge_bias(bias_t, nb)

    # ALiBi stays separate until the packed expand. A shared user mask is
    # top-left; ALiBi is the bottom-right slice of the max-length tensor.
    user_bias = bias_t
    alibi_bias, slopes_t, has_alibi, alibi_per_head = _varlen_alibi_parts(
        user_bias,
        alibi_slopes,
        num_heads=num_heads,
        device=q.device,
        stream=stream,
        cu_q=cu_q,
        max_q=max_seqlen_q,
        max_k=max_seqlen_kv,
        cu_kv=cu_kv,
    )

    # Causal uses in-kernel bottom-right with per-batch (sk-sq).
    kernel_causal = bool(causal)

    # Sink → [B, H]
    sink_t = None
    if sink is not None:
        s = sink.detach().float()
        if s.dim() == 1:
            if int(s.numel()) != num_heads:
                raise ValueError(f"varlen sink must have H={num_heads}, got {int(s.numel())}")
            sink_t = ensure_contiguous(s.view(1, num_heads).expand(batch, num_heads), stream=stream)
        elif s.dim() == 2:
            if tuple(s.shape) != (batch, num_heads):
                raise ValueError(f"varlen sink shape {tuple(s.shape)} != {(batch, num_heads)}")
            sink_t = _as_contig(s, stream=launch_stream)
        else:
            raise ValueError(f"varlen sink must be [H] or [B,H], got {tuple(sink.shape)}")
        if sink_t.device != q.device:
            sink_t = sink_t.to(q.device)

    cross = max_seqlen_q != max_seqlen_kv
    dtype_str = _torch_dtype_to_str(q.dtype)
    block_m = _pick_block_m(max_seqlen_q, cross, max_seqlen_kv)
    waves_per_eu = _pick_waves_per_eu(max_seqlen_q, max_seqlen_kv, cross, waves_per_eu)

    block_n = 32
    # Empty max KV has no legal bias column. One zero tile keeps that launch
    # in range. A positive max stays at the caller's length.
    n_pad_kv = block_n if max_seqlen_kv == 0 else 0
    seq_len_kv_pad = max_seqlen_kv + n_pad_kv
    seq_len_q_launch = max_seqlen_q

    bias_t, has_per_head_bias, bias_bottom_right = _assemble_varlen_bias(
        user_bias,
        alibi_bias,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        seq_len_kv_pad=seq_len_kv_pad,
        total_q=int(q.shape[0]),
        num_heads=num_heads,
        stream=stream,
    )
    has_bias = bias_t is not None
    bias_arg = _as_contig(bias_t, stream=launch_stream) if has_bias else _dummy_bias(q.device)

    total_q = int(q.shape[0])
    o_shape = (total_q, num_heads, head_dim)
    caller_out_shape = (total_q, num_heads, orig_head_dim)
    if out is not None:
        if tuple(out.shape) != caller_out_shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError(
                "varlen out mismatch: "
                f"got {tuple(out.shape)}/{out.dtype}/{out.device}, "
                f"want {caller_out_shape}/{q.dtype}/{q.device}"
            )
        o_p = out if orig_head_dim == head_dim else torch.empty(o_shape, dtype=q.dtype, device=q.device)
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
            has_attn_bias=has_bias and not has_per_head_bias,
            has_per_head_bias=has_per_head_bias,
            return_lse=return_lse,
            has_sink=sink_t is not None,
            varlen=True,
            paged=False,
            page_size=16,
            num_kv_heads=int(k.shape[1]),
            sm_scale=sm_scale_pad,
            logical_head_dim=logical_head_dim,
            sliding_window=sliding_window,
            bias_bottom_right=bias_bottom_right,
            has_alibi=has_alibi,
            alibi_per_head=alibi_per_head,
        )
        nsplits = int(num_kv_splits)
        comb_o = o_p
        if nsplits > 1:
            comb_o = torch.empty(
                (batch, seq_len_q_launch, num_heads, head_dim),
                dtype=q.dtype,
                device=q.device,
            )
        split_tail, split_ws = _float_split_tail(q.device, nsplits, batch, seq_len_q_launch, num_heads, head_dim)
        exe(
            _flat(_as_contig(q, stream=launch_stream)),
            _flat(_as_contig(k, stream=launch_stream)),
            _flat(_as_contig(v, stream=launch_stream)),
            _flat(comb_o),
            batch,
            seq_len_q_launch,
            seq_len_kv_pad,
            max_seqlen_kv,  # ignored by VARLEN kernel (uses sk)
            _flat(bias_arg),
            _flat(_dummy_bias(q.device) if not has_alibi else slopes_t),
            lse_arg,
            sink_arg,
            _flat(cu_q),
            _flat(cu_kv),
            _flat(_di),
            _flat(_di),
            0,
            *split_tail,
            stream=launch_stream,
        )
        if split_ws is not None:
            _combine_float_splits(
                split_ws,
                comb_o,
                batch=batch,
                seq_q=seq_len_q_launch,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype_str=dtype_str,
                has_sink=sink_t is not None,
                return_lse=return_lse,
                sink_t=sink_t,
                lse_p=lse_p,
                daz=daz,
                stream=launch_stream,
            )
            cq_host = cu_q.detach().to(device="cpu", dtype=torch.int64).tolist()
            for bi in range(batch):
                qs, qe = int(cq_host[bi]), int(cq_host[bi + 1])
                n = qe - qs
                if n > 0:
                    o_p[qs:qe].copy_(comb_o[bi, :n], non_blocking=True)

    o_p = _crop_head_dim(o_p, orig_head_dim)
    if out is not None and o_p.data_ptr() != out.data_ptr():
        if launch_stream is None:
            out.copy_(o_p)
        else:
            with torch.cuda.stream(launch_stream):
                out.copy_(o_p)
        o_p = out
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
    sliding_window: tuple[int, int] | None = None,
    num_kv_splits: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Paged FA. linear 4D, linear3d page-1 (same bytes), or vectorized 5D."""
    from kernels.common.gfx120x_arch import require_gfx120x
    from kernels.common.gfx120x_pad import ensure_contiguous

    if not (q.is_cuda and k_cache.is_cuda and v_cache.is_cuda):
        raise ValueError("flydsl_flash_attn_paged_func requires CUDA/HIP tensors")
    require_gfx120x(q.device, what="flydsl_flash_attn_paged_func")
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    if launch_stream.device != q.device:
        raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
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
    # Paged KV cache is caller-owned (page layout / shared blocks). Padding D
    # would require reallocating the cache; vectorized also needs D % kVS.
    # Leave hard-gated — not layout-safe to soft-pad in place.
    if head_dim < _MIN_HEAD_DIM or head_dim % 32 != 0 or head_dim > _MAX_HEAD_DIM:
        raise ValueError(f"paged head_dim={head_dim} out of gfx120x FA range")

    bt = _as_contig(block_table.to(torch.int32), stream=launch_stream)
    sk = _as_contig(seqlen_k.to(torch.int32), stream=launch_stream)
    if bt.device != q.device:
        bt = bt.to(q.device)
    if sk.device != q.device:
        sk = sk.to(q.device)
    if bt.dim() != 2 or int(bt.shape[0]) != batch:
        raise ValueError(f"block_table must be [B, n_pages], got {tuple(bt.shape)} for B={batch}")
    if sk.numel() != batch:
        raise ValueError(f"seqlen_k must have B={batch} entries, got {int(sk.numel())}")
    block_table_stride = int(bt.shape[1])

    # One small metadata D2H for seqlen_k (B ints) — needed for pad/ALiBi bounds
    # and to detect ragged lengths for auto-route to varlen-paged.
    if sk.numel() == 0:
        max_sk = 0
        sk_host = []
    else:
        sk_host = sk.detach().to(device="cpu", dtype=torch.int64).tolist()
        max_sk = int(max(sk_host)) if sk_host else 0
    # Dense BSHD + paged KV with non-uniform seqlen_k: pack Q and hand off to
    # varlen-paged (cu_seqlens from Sq / seqlen_k). Uniform stays on this dense path.
    if sk_host and any(int(x) != int(sk_host[0]) for x in sk_host):
        from kernels.attention.flash_attn_gfx120x_ext import cu_seqlens_from_seqlens

        total_q = int(batch * seq_len_q_real)
        q_pack = q.reshape(total_q, num_heads, head_dim)
        # Multiply form stays valid when Sq==0 (arange step cannot be 0).
        cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * int(seq_len_q_real)
        cu_kv = cu_seqlens_from_seqlens(sk, device=q.device)
        # Fold attn_mask alias into bias (same contract as the dense path below).
        route_bias = bias
        if attn_mask is not None and not mask_is_noop(attn_mask) and attn_mask is not bias:
            if route_bias is None or mask_is_noop(route_bias):
                route_bias = attn_mask
            else:
                route_bias = _merge_bias(
                    _as_score_bias(route_bias, seq_len_q_real, max_sk, num_heads, q.device, stream=stream),
                    _as_score_bias(attn_mask, seq_len_q_real, max_sk, num_heads, q.device, stream=stream),
                )
        out_pack = None if out is None else out.reshape(total_q, num_heads, head_dim)
        got = flydsl_flash_attn_varlen_paged_func(
            q_pack,
            k_cache,
            v_cache,
            cu_q,
            cu_kv,
            block_table,
            seqlen_k,
            int(seq_len_q_real),
            int(max_sk),
            page_size=page_size,
            causal=causal,
            waves_per_eu=waves_per_eu,
            daz=daz,
            stream=stream,
            out=out_pack,
            bias=route_bias,
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            sink=sink,
            kv_cache_layout=layout,
            sliding_window=sliding_window,
            num_kv_splits=num_kv_splits,
        )
        if return_lse:
            o_p, lse_p = got
            return o_p.reshape(batch, seq_len_q_real, num_heads, head_dim), lse_p
        return got.reshape(batch, seq_len_q_real, num_heads, head_dim)
    cross = seq_len_q_real != max_sk
    dtype_str = _torch_dtype_to_str(q.dtype)
    block_m = _pick_block_m(seq_len_q_real, cross, max_sk)
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, max_sk, cross, waves_per_eu)

    block_n = 32
    # Empty max KV has no legal bias column. One zero tile keeps that launch
    # in range. A positive max stays at the caller's length.
    n_pad_kv = block_n if max_sk == 0 else 0
    seq_len_kv_pad = max_sk + n_pad_kv
    seq_len_q_launch = seq_len_q_real

    # Bias / ALiBi (shared dense [Sq, Sk] against max_sk; ragged already routed above).
    bias_t = (
        None
        if attn_mask is bias
        else _as_score_bias(attn_mask, seq_len_q_real, max_sk, num_heads, q.device, stream=stream)
    )
    nb = _as_score_bias(bias, seq_len_q_real, max_sk, num_heads, q.device, stream=stream)
    bias_t = nb if bias_t is None else _merge_bias(bias_t, nb)
    if alibi_slopes is not None:
        bias_t = _merge_bias(bias_t, fold_alibi_to_bias(alibi_slopes, seq_len_q_real, max_sk, q.device, stream=stream))
    has_alibi = False
    alibi_per_head = False
    slopes_t = None
    has_per_head_bias = False
    if bias_t is not None and bias_t.dim() == 3:
        if int(bias_t.shape[0]) != num_heads:
            raise ValueError(f"paged per-head bias H={int(bias_t.shape[0])} != {num_heads}")
        has_per_head_bias = True
    has_bias = bias_t is not None
    if has_bias and n_pad_kv > 0:
        bias_t = device_pad(bias_t, (0, n_pad_kv), stream=stream)

    sink_t = None
    if sink is not None:
        s = sink.detach().float()
        if s.dim() == 1:
            if int(s.numel()) != num_heads:
                raise ValueError(f"paged sink must have H={num_heads} entries, got {int(s.numel())}")
            sink_t = ensure_contiguous(s.view(1, num_heads).expand(batch, num_heads), stream=stream)
        elif s.dim() == 2:
            if tuple(s.shape) != (batch, num_heads):
                raise ValueError(f"paged sink shape {tuple(s.shape)} != {(batch, num_heads)}")
            sink_t = _as_contig(s, stream=launch_stream)
        else:
            raise ValueError(f"paged sink must be [H] or [B,H], got {tuple(sink.shape)}")
        if sink_t.device != q.device:
            sink_t = sink_t.to(q.device)

    bias_arg = _as_contig(bias_t, stream=launch_stream) if has_bias else _dummy_bias(q.device)
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
            sliding_window=sliding_window,
            has_alibi=has_alibi,
            alibi_per_head=alibi_per_head,
        )
        nsplits = int(num_kv_splits)
        split_tail, split_ws = _float_split_tail(q.device, nsplits, batch, seq_len_q_launch, num_heads, head_dim)
        exe(
            _flat(_as_contig(q, stream=launch_stream)),
            _flat(_as_contig(k_cache, stream=launch_stream)),
            _flat(_as_contig(v_cache, stream=launch_stream)),
            _flat(o_p),
            batch,
            seq_len_q_launch,
            seq_len_kv_pad,
            max_sk,  # overridden per-batch by SeqlenK loads
            _flat(bias_arg),
            _flat(_dummy_bias(q.device) if not has_alibi else slopes_t),
            lse_arg,
            sink_arg,
            _flat(_di),
            _flat(_di),
            _flat(bt),
            _flat(sk),
            block_table_stride,
            *split_tail,
            stream=launch_stream,
        )
        if split_ws is not None:
            _combine_float_splits(
                split_ws,
                o_p,
                batch=batch,
                seq_q=seq_len_q_launch,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype_str=dtype_str,
                has_sink=sink_t is not None,
                return_lse=return_lse,
                sink_t=sink_t,
                lse_p=lse_p,
                daz=daz,
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
    page_size: int | None = None,
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
    sliding_window: tuple[int, int] | None = None,
    num_kv_splits: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Packed Q with paged K/V. cu_seqlens_q selects Q rows; block_table/seqlen_k select KV.

    cu_seqlens_kv is required and must describe the same batch as seqlen_k. KV
    addresses come from the block table (no host gather, no dense pad).
    """
    from kernels.common.gfx120x_pad import ensure_contiguous

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
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    if launch_stream.device != q.device:
        raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
    # Normalize cache through the dense paged entry's layout rules by a tiny Q stand-in
    # only for validation — real launch is below so we duplicate the layout parse.
    layout = kv_cache_layout or "linear"
    num_heads = int(q.shape[1])
    head_dim = int(q.shape[2])
    # Paged KV cache is caller-owned — same hard gate as dense-Q paged (no soft-pad).
    if head_dim < _MIN_HEAD_DIM or head_dim % 32 != 0 or head_dim > _MAX_HEAD_DIM:
        raise ValueError(f"varlen paged head_dim={head_dim} out of gfx120x FA range")
    cu_q = _as_contig(cu_seqlens_q.to(torch.int32), stream=launch_stream)
    cu_kv = _as_contig(cu_seqlens_kv.to(torch.int32), stream=launch_stream)
    if cu_q.device != q.device:
        cu_q = cu_q.to(q.device)
    if cu_kv.device != q.device:
        cu_kv = cu_kv.to(q.device)
    if cu_q.numel() != cu_kv.numel():
        raise ValueError("varlen paged cu_seqlens_q/kv batch mismatch")
    batch = int(cu_q.numel() - 1)
    bt = _as_contig(block_table.to(torch.int32), stream=launch_stream)
    sk = _as_contig(seqlen_k.to(torch.int32).view(-1), stream=launch_stream)
    if bt.device != q.device:
        bt = bt.to(q.device)
    if sk.device != q.device:
        sk = sk.to(q.device)
    if bt.dim() != 2 or int(bt.shape[0]) != batch:
        raise ValueError(f"varlen paged block_table must be [B, n_pages], got {tuple(bt.shape)}")
    if int(sk.numel()) != batch:
        raise ValueError("varlen paged seqlen_k batch mismatch")
    # KV lengths follow seqlen_k (paged). cu_seqlens_kv must match those lengths.
    # One small metadata D2H for both length vectors (validation only).
    cu_host = cu_kv.detach().to(device="cpu", dtype=torch.int64).tolist()
    sk_host = sk.detach().to(device="cpu", dtype=torch.int64).tolist()
    cu_lens_host = [int(cu_host[i + 1] - cu_host[i]) for i in range(len(cu_host) - 1)]
    if cu_lens_host != [int(x) for x in sk_host]:
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
    # Empty max KV has no legal bias column. One zero tile keeps that launch
    # in range. A positive max stays at the caller's length.
    n_pad = block_n if max_seqlen_kv == 0 else 0
    seq_len_kv_pad = max_seqlen_kv + n_pad

    user_bias = None
    if bias is not None and not mask_is_noop(bias):
        if bias.dim() == 3 and int(bias.shape[0]) == num_heads:
            user_bias = bias if bias.dtype == torch.float32 else bias.to(torch.float32)
        else:
            user_bias = normalize_attn_mask(bias, max_seqlen_q, max_seqlen_kv, q.device, stream=stream)
    alibi_bias, slopes_t, has_alibi, alibi_per_head = _varlen_alibi_parts(
        user_bias,
        alibi_slopes,
        num_heads=num_heads,
        device=q.device,
        stream=stream,
        cu_q=cu_q,
        max_q=max_seqlen_q,
        max_k=max_seqlen_kv,
        cu_kv=cu_kv,
    )
    bias_t, has_per_head_bias, bias_bottom_right = _assemble_varlen_bias(
        user_bias,
        alibi_bias,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        seq_len_kv_pad=seq_len_kv_pad,
        total_q=int(q.shape[0]),
        num_heads=num_heads,
        stream=stream,
    )
    has_bias = bias_t is not None
    sink_t = None
    if sink is not None:
        ss = sink.detach().float()
        if ss.dim() == 1:
            if int(ss.numel()) != num_heads:
                raise ValueError(f"varlen paged sink must have H={num_heads}, got {int(ss.numel())}")
            sink_t = ensure_contiguous(ss.view(1, num_heads).expand(batch, num_heads), stream=stream)
        elif ss.dim() == 2:
            if tuple(ss.shape) != (batch, num_heads):
                raise ValueError(f"varlen paged sink shape {tuple(ss.shape)} != {(batch, num_heads)}")
            sink_t = _as_contig(ss, stream=launch_stream)
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
    bias_arg = _as_contig(bias_t, stream=launch_stream) if has_bias else _dummy_bias(q.device)
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
            has_attn_bias=has_bias and not has_per_head_bias,
            has_per_head_bias=has_per_head_bias,
            return_lse=return_lse,
            has_sink=sink_t is not None,
            varlen=True,
            paged=True,
            page_size=page_size,
            num_kv_heads=int(num_kv_heads),
            kv_cache_layout=layout_kernel,
            sliding_window=sliding_window,
            bias_bottom_right=bias_bottom_right,
            has_alibi=has_alibi,
            alibi_per_head=alibi_per_head,
        )
        nsplits = int(num_kv_splits)
        comb_o = o_p
        if nsplits > 1:
            comb_o = torch.empty((batch, max_seqlen_q, num_heads, head_dim), dtype=q.dtype, device=q.device)
        split_tail, split_ws = _float_split_tail(q.device, nsplits, batch, max_seqlen_q, num_heads, head_dim)
        exe(
            _flat(_as_contig(q, stream=launch_stream)),
            _flat(_as_contig(k_cache, stream=launch_stream)),
            _flat(_as_contig(v_cache, stream=launch_stream)),
            _flat(comb_o),
            batch,
            max_seqlen_q,
            seq_len_kv_pad,
            max_seqlen_kv,
            _flat(bias_arg),
            _flat(_dummy_bias(q.device) if not has_alibi else slopes_t),
            lse_arg,
            sink_arg,
            _flat(cu_q),
            _flat(cu_kv),
            _flat(bt),
            _flat(sk),
            int(bt.shape[1]),
            *split_tail,
            stream=launch_stream,
        )
        if split_ws is not None:
            _combine_float_splits(
                split_ws,
                comb_o,
                batch=batch,
                seq_q=max_seqlen_q,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype_str=dtype_str,
                has_sink=sink_t is not None,
                return_lse=return_lse,
                sink_t=sink_t,
                lse_p=lse_p,
                daz=daz,
                stream=launch_stream,
            )
            cq_host = cu_q.detach().to(device="cpu", dtype=torch.int64).tolist()
            for bi in range(batch):
                qs, qe = int(cq_host[bi]), int(cq_host[bi + 1])
                n = qe - qs
                if n > 0:
                    o_p[qs:qe].copy_(comb_o[bi, :n], non_blocking=True)
    if return_lse:
        return o_p, lse_p
    return o_p


# ---------------------------------------------------------------------------
# FP8 (E4M3FN / E5M2) — self + cross, descales, seq_len_kv_valid
# ---------------------------------------------------------------------------


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
    stream: torch.cuda.Stream | None = None,
) -> tuple[object, object, object]:
    """Bias / ALiBi / sink for the fp8 and int8 kernels. Returns tensors, not host loops."""
    from kernels.common.gfx120x_pad import ensure_contiguous

    bias_t = None
    if bias is not None and not mask_is_noop(bias):
        if bias.dim() == 3 and int(bias.shape[0]) == int(num_heads):
            bias_t = bias if bias.dtype == torch.float32 else bias.float()
        else:
            bias_t = normalize_attn_mask(bias, seq_q, seq_k, q.device, stream=stream)
    if alibi_slopes is not None:
        bias_t = _merge_bias(bias_t, fold_alibi_to_bias(alibi_slopes, seq_q, seq_k, q.device, stream=stream))
    has_per_head = False
    if bias_t is not None and bias_t.dim() == 3:
        if int(bias_t.shape[0]) != int(num_heads):
            raise ValueError(f"gfx120x quant FA per-head bias H={int(bias_t.shape[0])} != num_heads={num_heads}")
        has_per_head = True
    if bias_t is not None and n_pad_kv > 0:
        bias_t = device_pad(bias_t, (0, n_pad_kv), stream=stream)
    sink_t = None
    if sink is not None:
        ss = sink.detach().float()
        if ss.dim() == 1:
            if int(ss.numel()) != int(num_heads):
                raise ValueError(f"quant sink must have H={num_heads}, got {int(ss.numel())}")
            sink_t = ensure_contiguous(ss.view(1, int(num_heads)).expand(int(batch), int(num_heads)), stream=stream)
        elif ss.dim() == 2:
            if tuple(ss.shape) != (int(batch), int(num_heads)):
                raise ValueError(f"quant sink shape {tuple(ss.shape)} != {(int(batch), int(num_heads))}")
            sink_t = _as_contig(ss, stream=stream)
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
    sm_scale: float | None = None,
    logical_head_dim: int | None = None,
    varlen: bool = False,
    paged: bool = False,
    page_size: int = 16,
    kv_cache_layout: str = "linear",
    split_k: bool = False,
    sliding_window: tuple[int, int] | None = None,
    bias_bottom_right: bool = False,
    has_alibi: bool = False,
    alibi_per_head: bool = False,
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
        sm_scale=sm_scale,
        logical_head_dim=logical_head_dim,
        varlen=varlen,
        paged=paged,
        page_size=page_size,
        kv_cache_layout=kv_cache_layout,
        split_k=split_k,
        sliding_window=sliding_window,
        bias_bottom_right=bias_bottom_right,
        has_alibi=has_alibi,
        alibi_per_head=alibi_per_head,
    )


_DESCALE_ONE_CACHE: dict[object, torch.Tensor] = {}


def _fp8_descale_to_device(name: str, scale: torch.Tensor | float | None, device: torch.device) -> torch.Tensor:
    """Return a 1-element fp32 CUDA descale buffer (gfx950-style; no host sync)."""
    import math as _math

    if scale is None:
        key = device.index if getattr(device, "index", None) is not None else int(device)
        buf = _DESCALE_ONE_CACHE.get(key)
        if buf is None or buf.device != device:
            buf = torch.ones(1, dtype=torch.float32, device=device)
            _DESCALE_ONE_CACHE[key] = buf
        return buf
    if isinstance(scale, (int, float)):
        val = float(scale)
        if not (_math.isfinite(val) and val > 0.0):
            raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be positive and finite, got {val}")
        return torch.tensor([val], device=device, dtype=torch.float32)
    if isinstance(scale, torch.Tensor):
        if not scale.is_cuda:
            raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be a CUDA tensor, got device={scale.device}")
        if scale.device != device:
            raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be on {device}, got {scale.device}")
        if scale.dtype != torch.float32:
            raise ValueError(f"flydsl_flash_attn_fp8_func: {name} must be float32, got {scale.dtype}")
        if scale.numel() != 1:
            raise ValueError(
                f"flydsl_flash_attn_fp8_func: {name} must have numel==1 (per-tensor), got shape={tuple(scale.shape)}"
            )
        # Keep on device — kernel loads the scalar (no .item() sync).
        if scale.is_contiguous() and scale.dim() == 1:
            return scale
        return scale.detach().reshape(1).contiguous()
    raise TypeError(
        f"flydsl_flash_attn_fp8_func: {name} must be None, float, or float32 CUDA tensor[1], got {type(scale).__name__}"
    )


def _fp8_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.float8_e4m3fn:
        return "fp8_e4m3fn"
    if dtype == torch.float8_e5m2:
        return "fp8_e5m2"
    raise ValueError(f"flydsl_flash_attn_fp8_func expects float8_e4m3fn or float8_e5m2, got {dtype}")


def _quant_dense_launch(
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
    sliding_window: tuple[int, int] | None = None,
    num_kv_splits: int = 1,
    block_m: int | None = None,
    *,
    kind: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Dense fp8 or int8 flash attention. One host for both dtypes."""
    what = f"flydsl_flash_attn_{kind}_func"
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError(f"{what} requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    require_gfx120x(q.device, what=f"{what} (gfx120x)")
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    if launch_stream.device != q.device:
        raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
    if kind == "int8":
        if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
            raise ValueError(f"{what} expects torch.int8 QKV, got {q.dtype}/{k.dtype}/{v.dtype}")
        dtype_str = "int8"
    else:
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError(f"{what}: q/k/v dtype must match, got {q.dtype}/{k.dtype}/{v.dtype}")
        dtype_str = _fp8_dtype_str(q.dtype)
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"{what}: expected 4D BSHD tensors, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if not (
        q.shape[0] == k.shape[0] == v.shape[0] and q.shape[3] == k.shape[3] == v.shape[3] and k.shape[2] == v.shape[2]
    ):
        raise ValueError(
            f"{what}: q/k/v must share batch and head_dim; "
            "k/v must share head count; "
            f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )

    from kernels.attention.flash_attn_gfx120x_ext import reject_gqa

    reject_gqa(int(q.shape[2]), int(k.shape[2]), num_kv_heads)
    num_kv_heads = int(k.shape[2])
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"{what}: k/v seq_len must match, got k={k.shape[1]} v={v.shape[1]}")

    batch, seq_len_q_real, num_heads, head_dim = q.shape
    seq_len_kv_real = int(k.shape[1])
    cross = seq_len_q_real != seq_len_kv_real
    # Causal×cross: in-kernel bottom-right (no dequant→bf16 FALLBACK).
    orig_head_dim = int(head_dim)
    launch_head_dim = _head_dim_tile_target(orig_head_dim, what=what)
    sm_scale_pad = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(float(orig_head_dim))
    # Head-dim tail is masked in the kernel, including a partial last dword.
    kernel_logical_d = orig_head_dim
    head_dim = launch_head_dim

    qd = _fp8_descale_to_device("q_descale", q_descale, q.device)
    kd = _fp8_descale_to_device("k_descale", k_descale, q.device)
    vd = _fp8_descale_to_device("v_descale", v_descale, q.device)

    block_m = _pinned_block_m(block_m, _pick_block_m(seq_len_q_real, cross, seq_len_kv_real))
    waves_per_eu = _pick_waves_per_eu(seq_len_q_real, seq_len_kv_real, cross, waves_per_eu)

    seq_len_q_launch = seq_len_q_real
    # Real K length. The buffer record zero-fills a short last tile. Empty K
    # has a 0-byte record, so the load returns zeros and the epilogue clears O.
    seq_len_kv_pad = seq_len_kv_real
    n_pad_kv = 0
    q_p = _as_contig(q, stream=launch_stream)
    k_p = _as_contig(k, stream=launch_stream)
    v_p = _as_contig(v, stream=launch_stream)

    caller_out_shape = (batch, seq_len_q_launch, num_heads, orig_head_dim)
    store_d = kernel_logical_d
    if out is not None and store_d == orig_head_dim:
        if tuple(out.shape) != caller_out_shape or out.dtype != torch.bfloat16 or out.device != q.device:
            raise ValueError(f"{what}: out must be bf16 {caller_out_shape} on {q.device}")
        o_p = out
    else:
        if out is not None and (
            tuple(out.shape) != caller_out_shape or out.dtype != torch.bfloat16 or out.device != q.device
        ):
            raise ValueError(f"{what}: out must be bf16 {caller_out_shape} on {q.device}")
        o_p = torch.empty((batch, seq_len_q_launch, num_heads, store_d), dtype=torch.bfloat16, device=q.device)

    bias_t, has_per_head, sink_t = _prepare_quant_extras(
        q, bias, alibi_slopes, sink, seq_len_q_real, seq_len_kv_real, n_pad_kv, batch, num_heads, stream=stream
    )
    has_bias = bias_t is not None
    has_sink_b = sink_t is not None
    has_alibi = False
    alibi_per_head = False
    slopes_t = None
    bias_arg = _flat(_as_contig(bias_t, stream=launch_stream)) if has_bias else _flat(_dummy_bias(q.device))
    lse_p = None
    if return_lse:
        lse_p = torch.empty((batch, num_heads, seq_len_q_launch), dtype=torch.float32, device=q.device)
    lse_arg = _flat(lse_p) if return_lse else _flat(_dummy_bias(q.device))
    sink_arg = _flat(sink_t) if has_sink_b else _flat(_dummy_bias(q.device))

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        if launch_stream.device != q.device:
            raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
        _ns = int(num_kv_splits)
        getter = _get_int8_kernel if kind == "int8" else _get_fp8_kernel
        extra = {} if kind == "int8" else {"dtype_str": dtype_str}
        exe = getter(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            waves_per_eu=waves_per_eu,
            daz=daz,
            block_m=block_m,
            has_attn_bias=has_bias and not has_per_head,
            has_per_head_bias=has_per_head,
            return_lse=return_lse and _ns <= 1,
            has_sink=has_sink_b and _ns <= 1,
            num_kv_heads=num_kv_heads,
            sm_scale=sm_scale_pad,
            logical_head_dim=kernel_logical_d,
            sliding_window=sliding_window,
            split_k=_ns > 1,
            has_alibi=has_alibi,
            alibi_per_head=alibi_per_head,
            **extra,
        )
        _quant_run_exe(
            exe,
            (
                _flat(q_p),
                _flat(k_p),
                _flat(v_p),
                _flat(o_p),
                batch,
                seq_len_q_launch,
                seq_len_kv_pad,
                seq_len_kv_real,
                _flat(qd),
                _flat(kd),
                _flat(vd),
                bias_arg,
                _flat(_dummy_bias(q.device) if not has_alibi else slopes_t),
                lse_arg,
                sink_arg,
            ),
            device=q.device,
            launch_stream=launch_stream,
            o_p=o_p,
            batch=batch,
            seq_q=seq_len_q_launch,
            num_heads=num_heads,
            tile_d=head_dim,
            logical_d=kernel_logical_d,
            num_kv_splits=_ns,
            has_sink=has_sink_b,
            return_lse=return_lse,
            sink_t=sink_t if has_sink_b else None,
            lse_p=lse_p,
            daz=daz,
        )

    o_p = _crop_head_dim(o_p, orig_head_dim)
    if out is not None and o_p.data_ptr() != out.data_ptr():
        if launch_stream is None:
            out.copy_(o_p)
        else:
            with torch.cuda.stream(launch_stream):
                out.copy_(o_p)
        o_p = out
    if return_lse:
        return o_p, lse_p
    return o_p


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
    sliding_window: tuple[int, int] | None = None,
    num_kv_splits: int = 1,
    block_m: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """fp8 e4m3 or e5m2 flash attention. Output is bf16."""
    return _quant_dense_launch(
        q,
        k,
        v,
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
        sliding_window=sliding_window,
        num_kv_splits=num_kv_splits,
        block_m=block_m,
        kind="fp8",
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
    sliding_window: tuple[int, int] | None = None,
    num_kv_splits: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """int8 flash attention. Same launch as fp8, iu8 WMMA. Output is bf16."""
    return _quant_dense_launch(
        q,
        k,
        v,
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
        sliding_window=sliding_window,
        num_kv_splits=num_kv_splits,
        kind="int8",
    )


# ---------------------------------------------------------------------------
# Int8 kernel cache. The dense host is _quant_dense_launch.
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
    sm_scale: float | None = None,
    logical_head_dim: int | None = None,
    varlen: bool = False,
    paged: bool = False,
    page_size: int = 16,
    kv_cache_layout: str = "linear",
    split_k: bool = False,
    sliding_window: tuple[int, int] | None = None,
    bias_bottom_right: bool = False,
    has_alibi: bool = False,
    alibi_per_head: bool = False,
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
        sm_scale=sm_scale,
        logical_head_dim=logical_head_dim,
        varlen=varlen,
        paged=paged,
        page_size=page_size,
        kv_cache_layout=kv_cache_layout,
        split_k=split_k,
        sliding_window=sliding_window,
        bias_bottom_right=bias_bottom_right,
        has_alibi=has_alibi,
        alibi_per_head=alibi_per_head,
    )


def _quant_varlen_launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    *,
    kind: str,
    causal: bool,
    sm_scale: float | None,
    waves_per_eu: int,
    daz: bool,
    stream: torch.cuda.Stream | None,
    q_descale: torch.Tensor | float | None,
    k_descale: torch.Tensor | float | None,
    v_descale: torch.Tensor | float | None,
    out: torch.Tensor | None,
    bias: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    return_lse: bool = False,
    sliding_window: tuple[int, int] | None = None,
    num_kv_splits: int = 1,
    block_m: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Packed quant FA. Q/K/V are [total, H, D]. Output is bf16 [total, H, D]."""
    from kernels.common.gfx120x_arch import require_gfx120x

    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"quant varlen expects packed [total,H,D], got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if q.shape[2] != k.shape[2] or k.shape[1] != v.shape[1] or q.shape[2] != v.shape[2]:
        raise ValueError(f"quant varlen shape mismatch q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    require_gfx120x(q.device, what=f"flydsl_flash_attn_{kind}_varlen_func")
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    num_heads = int(q.shape[1])
    num_kv_heads = int(k.shape[1])
    orig_head_dim = int(q.shape[2])
    what = f"flydsl_flash_attn_{kind}_varlen_func"
    launch_head_dim = _head_dim_tile_target(orig_head_dim, what=what)
    sm_scale_pad = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(float(orig_head_dim))
    cu_q = _as_contig(cu_seqlens_q.to(torch.int32), stream=launch_stream)
    cu_kv = _as_contig(cu_seqlens_kv.to(torch.int32), stream=launch_stream)
    batch = int(cu_q.numel() - 1)
    max_seqlen_q = int(max_seqlen_q)
    max_seqlen_kv = int(max_seqlen_kv)
    cross = max_seqlen_q != max_seqlen_kv
    block_m = _pinned_block_m(block_m, _pick_block_m(max_seqlen_q, cross, max_seqlen_kv))
    waves_per_eu = _pick_waves_per_eu(max_seqlen_q, max_seqlen_kv, cross, waves_per_eu)
    block_n = _KERNEL_BLOCK_N
    # Empty max KV keeps one tile so the clamped prefetch has an address.
    # A positive max is the caller's length; the kernel clamps past it.
    n_pad_kv = block_n if max_seqlen_kv == 0 else 0
    seq_len_kv_pad = max_seqlen_kv + n_pad_kv
    if kind == "fp8":
        dtype_str = _fp8_dtype_str(q.dtype)
        qd = _fp8_descale_to_device("q_descale", q_descale, q.device)
        kd = _fp8_descale_to_device("k_descale", k_descale, q.device)
        vd = _fp8_descale_to_device("v_descale", v_descale, q.device)
        getter = _get_fp8_kernel
    else:
        if q.dtype != torch.int8 or k.dtype != torch.int8 or v.dtype != torch.int8:
            raise ValueError(f"int8 varlen expects torch.int8 QKV, got {q.dtype}/{k.dtype}/{v.dtype}")
        dtype_str = "int8"
        qd = _fp8_descale_to_device("q_descale", q_descale, q.device)
        kd = _fp8_descale_to_device("k_descale", k_descale, q.device)
        vd = _fp8_descale_to_device("v_descale", v_descale, q.device)
        getter = _get_int8_kernel
    total_q = int(q.shape[0])
    o_p = torch.empty((total_q, num_heads, orig_head_dim), dtype=torch.bfloat16, device=q.device)
    if out is not None:
        if tuple(out.shape) != tuple(o_p.shape) or out.dtype != torch.bfloat16 or out.device != q.device:
            raise ValueError(f"{what}: out must be bf16 {tuple(o_p.shape)} on {q.device}")
        o_p = out
    z = _flat(_dummy_i32(q.device))
    # The router already stored attn_mask in bias. Adding it again doubles the logits.
    if attn_mask is not None and not mask_is_noop(attn_mask) and attn_mask is not bias:
        mask_b = normalize_attn_mask(attn_mask, max_seqlen_q, seq_len_kv_pad, q.device, stream=stream)
        bias = mask_b if bias is None else _merge_bias(bias, mask_b)
    user_bias, _, sink_t = _prepare_quant_extras(
        q,
        bias,
        None,
        sink,
        max_seqlen_q,
        max_seqlen_kv,
        0,
        batch,
        num_heads,
        stream=stream,
    )
    alibi_bias, slopes_t, has_alibi, alibi_per_head = _varlen_alibi_parts(
        user_bias,
        alibi_slopes,
        num_heads=num_heads,
        device=q.device,
        stream=stream,
        cu_q=cu_q,
        max_q=max_seqlen_q,
        max_k=max_seqlen_kv,
        cu_kv=cu_kv,
    )
    bias_t, has_per_head, bias_bottom_right = _assemble_varlen_bias(
        user_bias,
        alibi_bias,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        seq_len_kv_pad=seq_len_kv_pad,
        total_q=total_q,
        num_heads=num_heads,
        stream=stream,
    )
    has_bias = bias_t is not None
    bias_arg = _flat(_as_contig(bias_t, stream=launch_stream)) if has_bias else _flat(_dummy_bias(q.device))
    lse_p = None
    if return_lse:
        lse_p = torch.empty((batch, num_heads, max_seqlen_q), dtype=torch.float32, device=q.device)
    lse_arg = _flat(lse_p) if return_lse else _flat(_dummy_bias(q.device))
    sink_arg = _flat(sink_t) if sink_t is not None else _flat(_dummy_bias(q.device))
    nsplits = int(num_kv_splits)
    with torch.cuda.device(q.device.index):
        exe = getter(
            num_heads=num_heads,
            head_dim=launch_head_dim,
            causal=causal,
            waves_per_eu=waves_per_eu,
            daz=daz,
            block_m=block_m,
            has_attn_bias=has_bias and not has_per_head,
            has_per_head_bias=has_per_head,
            return_lse=return_lse,
            has_sink=sink_t is not None,
            num_kv_heads=num_kv_heads,
            sm_scale=sm_scale_pad,
            logical_head_dim=orig_head_dim,
            varlen=True,
            sliding_window=sliding_window,
            split_k=nsplits > 1,
            bias_bottom_right=bias_bottom_right,
            has_alibi=has_alibi,
            alibi_per_head=alibi_per_head,
            **({} if kind == "int8" else {"dtype_str": dtype_str}),
        )
        comb_o = o_p
        if nsplits > 1:
            comb_o = torch.empty(
                (batch, max_seqlen_q, num_heads, launch_head_dim),
                dtype=torch.bfloat16,
                device=q.device,
            )
            ws_rows = batch * num_heads * max_seqlen_q
            ws_m, ws_l, ws_o = _new_split_workspace(q.device, nsplits, ws_rows, launch_head_dim)
            split_tail = (nsplits, _flat(ws_m), _flat(ws_l), _flat(ws_o), ws_rows)
        else:
            ws_m = ws_l = ws_o = None
            ws_rows = 0
            split_tail = (
                1,
                _flat(_dummy_bias(q.device)),
                _flat(_dummy_bias(q.device)),
                _flat(_dummy_bias(q.device)),
                0,
            )
        exe(
            _flat(_as_contig(q, stream=launch_stream)),
            _flat(_as_contig(k, stream=launch_stream)),
            _flat(_as_contig(v, stream=launch_stream)),
            _flat(comb_o),
            batch,
            max_seqlen_q,
            seq_len_kv_pad,
            max_seqlen_kv,
            _flat(qd),
            _flat(kd),
            _flat(vd),
            bias_arg,
            _flat(_dummy_bias(q.device) if not has_alibi else slopes_t),
            lse_arg,
            sink_arg,
            _flat(cu_q),
            _flat(cu_kv),
            z,
            z,
            0,
            *split_tail,
            stream=launch_stream,
        )
        if nsplits > 1:
            _combine_float_splits(
                (ws_m, ws_l, ws_o, ws_rows),
                comb_o,
                batch=batch,
                seq_q=max_seqlen_q,
                num_heads=num_heads,
                head_dim=launch_head_dim,
                dtype_str="bf16",
                has_sink=sink_t is not None,
                return_lse=return_lse,
                sink_t=sink_t,
                lse_p=lse_p,
                daz=daz,
                stream=launch_stream,
            )
            cq_host = cu_q.detach().to(device="cpu", dtype=torch.int64).tolist()
            for bi in range(batch):
                qs, qe = int(cq_host[bi]), int(cq_host[bi + 1])
                n = qe - qs
                if n > 0:
                    o_p[qs:qe].copy_(comb_o[bi, :n, :, :orig_head_dim], non_blocking=True)
    if return_lse:
        return o_p, lse_p
    return o_p


def flydsl_flash_attn_fp8_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
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
    attn_mask: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    return_lse: bool = False,
    sliding_window: tuple[int, int] | None = None,
    num_kv_splits: int = 1,
    block_m: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _quant_varlen_launch(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        kind="fp8",
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
        attn_mask=attn_mask,
        sink=sink,
        return_lse=return_lse,
        sliding_window=sliding_window,
        num_kv_splits=num_kv_splits,
        block_m=block_m,
    )


def flydsl_flash_attn_quant_paged_func(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqlen_k: torch.Tensor,
    *,
    kind: str,
    causal: bool = False,
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    q_descale: torch.Tensor | float | None = None,
    k_descale: torch.Tensor | float | None = None,
    v_descale: torch.Tensor | float | None = None,
    out: torch.Tensor | None = None,
    kv_cache_layout: str | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    sliding_window: tuple[int, int] | None = None,
    bias: torch.Tensor | None = None,
    alibi_slopes: torch.Tensor | None = None,
    attn_mask: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    return_lse: bool = False,
    num_kv_splits: int = 1,
    block_m: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Quant paged KV. The cache and block table are passed through. No gather copy."""
    from kernels.common.gfx120x_arch import require_gfx120x

    layout = kv_cache_layout or "linear"
    if layout not in ("linear", "linear3d", "vectorized"):
        raise NotImplementedError(f"gfx120x quant FA paged layout {layout!r} is not supported")
    require_gfx120x(q.device, what=f"flydsl_flash_attn_{kind}_paged")
    launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
    varlen = cu_seqlens_q is not None
    if layout == "linear3d":
        if k_cache.dim() != 3 or v_cache.dim() != 3:
            raise ValueError(
                f"linear3d paged cache must be [blocks, Hkv, D], got k={tuple(k_cache.shape)} v={tuple(v_cache.shape)}"
            )
        page_size = 1
        num_kv_heads = int(k_cache.shape[1])
        cache_d = int(k_cache.shape[2])
    elif layout == "vectorized":
        if k_cache.dim() != 5 or v_cache.dim() != 5:
            raise ValueError(
                "vectorized paged cache must be 5D, " f"got k={tuple(k_cache.shape)} v={tuple(v_cache.shape)}"
            )
        num_kv_heads = int(k_cache.shape[1])
        page_size = int(k_cache.shape[3])
        cache_d = int(v_cache.shape[3])
    else:
        if k_cache.dim() != 4 or v_cache.dim() != 4:
            raise ValueError(
                f"linear paged cache must be [blocks, page, Hkv, D], got k={tuple(k_cache.shape)} v={tuple(v_cache.shape)}"
            )
        page_size = int(k_cache.shape[1])
        num_kv_heads = int(k_cache.shape[2])
        cache_d = int(k_cache.shape[3])
    if varlen:
        if q.dim() != 3 or max_seqlen_q is None:
            raise ValueError("quant varlen-paged expects packed Q [total,H,D] and max_seqlen_q")
        num_heads = int(q.shape[1])
        orig_d = int(q.shape[2])
        batch = int(cu_seqlens_q.numel()) - 1
        seq_q = int(max_seqlen_q)
        out_shape = (int(q.shape[0]), num_heads, orig_d)
    else:
        if q.dim() != 4:
            raise ValueError(f"quant paged Q must be [B,Sq,H,D], got {tuple(q.shape)}")
        batch, seq_q, num_heads, orig_d = (int(x) for x in q.shape)
        out_shape = (batch, seq_q, num_heads, orig_d)
    if orig_d != cache_d:
        raise ValueError(f"quant paged head_dim mismatch q={orig_d} cache={cache_d}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"quant paged GQA invalid H={num_heads} Hkv={num_kv_heads}")
    if block_table.dim() != 2 or int(block_table.shape[0]) != batch:
        raise ValueError(f"block_table must be [B, pages], got {tuple(block_table.shape)} B={batch}")
    # Same handoff as float paged: unequal seqlen_k packs Q and uses varlen-paged,
    # so ALiBi, masks, causal, and the window see each sequence's own length.
    if not varlen:
        sk_host = seqlen_k.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
        if sk_host and any(int(x) != int(sk_host[0]) for x in sk_host):
            total_q = int(batch) * int(seq_q)
            q_pack = q.reshape(total_q, num_heads, orig_d)
            cu_pack = torch.arange(int(batch) + 1, device=q.device, dtype=torch.int32) * int(seq_q)
            out_pack = None if out is None else out.reshape(total_q, num_heads, orig_d)
            got = flydsl_flash_attn_quant_paged_func(
                q_pack,
                k_cache,
                v_cache,
                block_table,
                seqlen_k,
                kind=kind,
                causal=causal,
                sm_scale=sm_scale,
                waves_per_eu=waves_per_eu,
                daz=daz,
                stream=stream,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                out=out_pack,
                kv_cache_layout=kv_cache_layout,
                cu_seqlens_q=cu_pack,
                max_seqlen_q=int(seq_q),
                sliding_window=sliding_window,
                bias=bias,
                alibi_slopes=alibi_slopes,
                attn_mask=attn_mask,
                sink=sink,
                return_lse=return_lse,
                num_kv_splits=num_kv_splits,
                block_m=block_m,
            )
            if return_lse:
                o_got, lse_got = got
                return o_got.reshape(int(batch), int(seq_q), num_heads, orig_d), lse_got
            return got.reshape(int(batch), int(seq_q), num_heads, orig_d)
    n_pages = int(block_table.shape[1])
    seq_kv = n_pages * page_size
    what = f"flydsl_flash_attn_{kind}_paged"
    tile = _head_dim_tile_target(orig_d, what=what)
    sm = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(float(orig_d))
    if kind == "fp8":
        dtype_str = _fp8_dtype_str(q.dtype)
        qd = _fp8_descale_to_device("q_descale", q_descale, q.device)
        kd = _fp8_descale_to_device("k_descale", k_descale, q.device)
        vd = _fp8_descale_to_device("v_descale", v_descale, q.device)
        getter = _get_fp8_kernel
    else:
        dtype_str = "int8"
        qd = _fp8_descale_to_device("q_descale", q_descale, q.device)
        kd = _fp8_descale_to_device("k_descale", k_descale, q.device)
        vd = _fp8_descale_to_device("v_descale", v_descale, q.device)
        getter = _get_int8_kernel
    block_m = _pinned_block_m(block_m, _pick_block_m(seq_q, seq_q != seq_kv, seq_kv))
    waves = _pick_waves_per_eu(seq_q, seq_kv, seq_q != seq_kv, waves_per_eu)
    if out is None:
        o_p = torch.empty(out_shape, dtype=torch.bfloat16, device=q.device)
    else:
        if tuple(out.shape) != out_shape or out.dtype != torch.bfloat16:
            raise ValueError(f"quant paged out must be bf16 {out_shape}")
        o_p = out
    if attn_mask is not None and not mask_is_noop(attn_mask) and attn_mask is not bias:
        mask_b = normalize_attn_mask(attn_mask, seq_q, seq_kv, q.device, stream=stream)
        bias = mask_b if bias is None else _merge_bias(bias, mask_b)
    # The kernel loads these as int32. An int64 table is read at a 4-byte stride.
    if varlen:
        cu_q = _as_contig(cu_seqlens_q.to(torch.int32), stream=launch_stream)
    else:
        cu_q = _dummy_i32(q.device)
    bt = _as_contig(block_table.to(torch.int32), stream=launch_stream)
    sk = _as_contig(seqlen_k.to(torch.int32).reshape(-1), stream=launch_stream)
    if varlen:
        user_bias, _, sink_t = _prepare_quant_extras(
            q, bias, None, sink, seq_q, seq_kv, 0, batch, num_heads, stream=stream
        )
        sk_lens = [int(x) for x in sk.detach().to(device="cpu", dtype=torch.int64).tolist()]
        alibi_bias, slopes_t, has_alibi, alibi_per_head = _varlen_alibi_parts(
            user_bias,
            alibi_slopes,
            num_heads=num_heads,
            device=q.device,
            stream=stream,
            cu_q=cu_q,
            max_q=seq_q,
            max_k=seq_kv,
            kv_lengths=sk_lens,
        )
        bias_t, has_per_head, bias_bottom_right = _assemble_varlen_bias(
            user_bias,
            alibi_bias,
            cu_seqlens_q=cu_q,
            max_seqlen_q=seq_q,
            max_seqlen_kv=seq_kv,
            seq_len_kv_pad=seq_kv,
            total_q=int(q.shape[0]),
            num_heads=num_heads,
            stream=stream,
            kv_lengths=sk_lens,
        )
    else:
        bias_t, has_per_head, sink_t = _prepare_quant_extras(
            q, bias, alibi_slopes, sink, seq_q, seq_kv, 0, batch, num_heads, stream=stream
        )
        bias_bottom_right = False
        has_alibi = False
        alibi_per_head = False
        slopes_t = None
    has_bias = bias_t is not None
    bias_arg = _flat(_as_contig(bias_t, stream=launch_stream)) if has_bias else _flat(_dummy_bias(q.device))
    lse_p = None
    if return_lse:
        lse_p = torch.empty((batch, num_heads, seq_q), dtype=torch.float32, device=q.device)
    lse_arg = _flat(lse_p) if return_lse else _flat(_dummy_bias(q.device))
    sink_arg = _flat(sink_t) if sink_t is not None else _flat(_dummy_bias(q.device))
    z = _flat(_dummy_i32(q.device))
    zb = _flat(_dummy_bias(q.device))
    nsplits = int(num_kv_splits)
    exe = getter(
        num_heads=num_heads,
        head_dim=tile,
        causal=causal,
        waves_per_eu=waves,
        daz=daz,
        block_m=block_m,
        num_kv_heads=num_kv_heads,
        sm_scale=sm,
        logical_head_dim=orig_d,
        varlen=varlen,
        paged=True,
        page_size=page_size,
        kv_cache_layout=layout,
        sliding_window=sliding_window,
        bias_bottom_right=bias_bottom_right,
        has_alibi=has_alibi,
        alibi_per_head=alibi_per_head,
        has_attn_bias=has_bias and not has_per_head,
        has_per_head_bias=has_per_head,
        return_lse=return_lse,
        has_sink=sink_t is not None,
        split_k=nsplits > 1,
        **({} if kind == "int8" else {"dtype_str": dtype_str}),
    )
    comb_o = o_p
    if nsplits > 1:
        if varlen or tile != orig_d:
            comb_o = torch.empty((batch, seq_q, num_heads, tile), dtype=torch.bfloat16, device=q.device)
        ws_rows = batch * num_heads * seq_q
        ws_m, ws_l, ws_o = _new_split_workspace(q.device, nsplits, ws_rows, tile)
        split_tail = (nsplits, _flat(ws_m), _flat(ws_l), _flat(ws_o), ws_rows)
    else:
        ws_m = ws_l = ws_o = None
        ws_rows = 0
        split_tail = (1, zb, zb, zb, 0)
    exe(
        _flat(_as_contig(q, stream=launch_stream)),
        _flat(_as_contig(k_cache, stream=launch_stream)),
        _flat(_as_contig(v_cache, stream=launch_stream)),
        _flat(comb_o),
        batch,
        seq_q,
        seq_kv,
        seq_kv,
        _flat(qd),
        _flat(kd),
        _flat(vd),
        bias_arg,
        _flat(_dummy_bias(q.device) if not has_alibi else slopes_t),
        lse_arg,
        sink_arg,
        _flat(cu_q),
        z,
        _flat(bt),
        _flat(sk),
        n_pages,
        *split_tail,
        stream=launch_stream,
    )
    if nsplits > 1:
        _combine_float_splits(
            (ws_m, ws_l, ws_o, ws_rows),
            comb_o,
            batch=batch,
            seq_q=seq_q,
            num_heads=num_heads,
            head_dim=tile,
            dtype_str="bf16",
            has_sink=sink_t is not None,
            return_lse=return_lse,
            sink_t=sink_t,
            lse_p=lse_p,
            daz=daz,
            stream=launch_stream,
        )
        if varlen:
            cq_host = cu_seqlens_q.detach().to(device="cpu", dtype=torch.int64).tolist()
            for bi in range(batch):
                qs, qe = int(cq_host[bi]), int(cq_host[bi + 1])
                n = qe - qs
                if n > 0:
                    o_p[qs:qe].copy_(comb_o[bi, :n, :, :orig_d], non_blocking=True)
        elif comb_o.data_ptr() != o_p.data_ptr():
            o_p.copy_(comb_o[..., :orig_d])
    o_p = _crop_head_dim(o_p, orig_d) if not varlen else o_p
    if return_lse:
        return o_p, lse_p
    return o_p


def flydsl_flash_attn_int8_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
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
    attn_mask: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    return_lse: bool = False,
    sliding_window: tuple[int, int] | None = None,
    num_kv_splits: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    return _quant_varlen_launch(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        kind="int8",
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
        attn_mask=attn_mask,
        sink=sink,
        return_lse=return_lse,
        sliding_window=sliding_window,
        num_kv_splits=num_kv_splits,
    )
