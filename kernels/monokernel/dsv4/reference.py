# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Weights, layouts and the torch golden for one rank's DeepSeek-V4 attention+MoE layer.

:func:`golden_layer` covers sliding-window, HCA and CSA layers and hyper-connections. Quantization
mirrors the kernel (FP8 block-scaled matrices, bf16 GEMV activations), not the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from kernels.monokernel.dsv4.config import (
    BLOCK_TOKENS,
    COMPRESS_CSA,
    COMPRESS_ROPE_THETA,
    COMPRESS_SWA,
    EPS,
    FP8_MAX,
    HC_EPS,
    HC_MULT,
    HC_SINKHORN_ITERS,
    HEAD_DIM,
    HIDDEN,
    INDEX_HEAD_DIM,
    INDEX_HEADS,
    INDEX_TOPK,
    INTER,
    KEY_BLOCK,
    N_EXPERTS,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    ROUTE_SCALE,
    SWIGLU_LIMIT,
    TOP_K,
    WINDOW,
    ExpertActivation,
    ExpertWeight,
    MoeMode,
    as_moe_mode,
    moe_format,
)
from kernels.monokernel.formats import quantize_mxfp4
from kernels.monokernel.reference import (  # noqa: F401  (re-exported for the dsv4 tests)
    _rand_fp8,
    bf,
    dequant,
    dequant_expert,
    quant_dequant,
    scale_shape,
)

# rope_table() spans the RoPE tail of a head.
PE_DIM = ROPE_DIM


def rope_table(max_seq: int, theta: float = 8.0e6, device="cuda"):
    inv = 1.0 / theta ** (torch.arange(0, PE_DIM, 2, device=device, dtype=torch.float64) / PE_DIM)
    ang = torch.arange(max_seq, device=device, dtype=torch.float64)[:, None] * inv[None]
    return torch.cos(ang).float().contiguous(), torch.sin(ang).float().contiguous()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Interleaved pairs (2i, 2i+1); ``inverse`` de-rotates (V4's output, since V is the RoPE'd K)."""
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    s = -sin if inverse else sin
    out[..., 0::2] = x0 * cos - x1 * s
    out[..., 1::2] = x0 * s + x1 * cos
    return out


@dataclass
class V4Config:
    """One rank's shard of a DeepSeek-V4 layer. Defaults are V4-Pro at TP8."""

    heads: int = 16  # local; 128 global / 8 ranks
    hidden: int = HIDDEN
    q_lora: int = Q_LORA
    head_dim: int = HEAD_DIM  # K and V share this; rope lives in its tail
    rope_dim: int = ROPE_DIM
    o_groups: int = O_GROUPS  # local; 16 global / 8 ranks
    o_lora: int = O_LORA
    n_experts: int = N_EXPERTS
    top_k: int = TOP_K
    inter: int = INTER  # local; 3072 global / 8 ranks
    window: int = WINDOW
    route_scale: float = ROUTE_SCALE
    swiglu_limit: float = SWIGLU_LIMIT
    eps: float = EPS
    rope_theta: float = 1.0e4
    compress_ratio: int = COMPRESS_SWA  # 0 SWA, 4 CSA, 128 HCA
    index_heads: int = INDEX_HEADS  # all of them on every rank (the indexer is replicated)
    index_head_dim: int = INDEX_HEAD_DIM
    index_topk: int = INDEX_TOPK
    compress_rope_theta: float = COMPRESS_ROPE_THETA
    max_seq: int = 4096  # sizes the compressed half of the KV cache
    hc_mult: int = HC_MULT  # 1 = plain residual
    hc_sinkhorn_iters: int = HC_SINKHORN_ITERS
    hc_eps: float = HC_EPS
    # KV rows in ATOM's fp8 layout (NoPE + E8M0 plane, bf16 RoPE plane) instead of one bf16 plane
    kv_fp8: bool = False
    # Hadamard-rotate indexer q/k as the model does; ATOM's cache is unrotated, so serving it needs False
    indexer_hadamard: bool = True

    @property
    def hc_mix(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    def for_layer(self, layer_id: int, ratios=None) -> "V4Config":
        """This shard with layer ``layer_id``'s attention variant (``ratios`` defaults to V4-Pro's)."""
        from kernels.monokernel.dsv4.config import compress_ratios

        ratios = compress_ratios() if ratios is None else ratios
        if not 0 <= layer_id < len(ratios):
            raise ValueError(f"layer {layer_id} is outside a schedule of {len(ratios)}")
        return replace(self, compress_ratio=ratios[layer_id])

    @property
    def rope_base(self) -> float:
        """Per layer, not per consumer: a compressing layer rotates everything on ``compress_rope_theta``."""
        return self.compress_rope_theta if self.compress_ratio else self.rope_theta

    @property
    def overlap(self) -> bool:
        """CSA (ratio 4) pools 2 * ratio tokens per entry at a stride of ratio."""
        return self.compress_ratio == COMPRESS_CSA

    @property
    def indexed(self) -> bool:
        """CSA selects compressed entries with the lightning indexer; HCA takes them all."""
        return self.compress_ratio == COMPRESS_CSA

    @property
    def n_index(self) -> int:
        """Compressed entries the attention can reach in one step."""
        if not self.compress_ratio:
            return 0
        return min(self.index_topk, self.n_compressed) if self.indexed else self.n_compressed

    @property
    def c_coff(self) -> int:
        """Compressor channel multiplier: overlapping windows carry previous and current halves."""
        return 2 if self.overlap else 1

    @property
    def c_rows(self) -> int:
        """Rows of compressor state: two windows' worth when overlapping."""
        return self.compress_ratio * self.c_coff

    @property
    def n_compressed(self) -> int:
        """Compressed cache slots, stored after the ``window`` rows of the same cache."""
        return 0 if self.compress_ratio == 0 else self.max_seq // self.compress_ratio

    @property
    def cache_rows(self) -> int:
        return self.window + self.n_compressed

    @property
    def n_keys(self) -> int:
        """Length of the attention's index list (window + compressed), padded to the key tile."""
        rows = self.window + self.n_index
        return (rows + KEY_BLOCK - 1) // KEY_BLOCK * KEY_BLOCK

    @property
    def hc_rows(self) -> int:
        """hc_mix padded to the MFMA row group (24 -> 32 at hc_mult 4)."""
        return (self.hc_mix + 15) // 16 * 16

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.rope_dim

    @property
    def group_dim(self) -> int:
        """Slice of the concatenated heads that one ``o_a`` group consumes."""
        return self.heads * self.head_dim // self.o_groups

    @property
    def shared_expert(self) -> int:
        """Last in an FP8 bank; an MXFP4 bank keeps it as FP8 ``w_sug`` / ``w_sdn`` beside it."""
        return self.n_experts

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    def validate(self) -> None:
        assert self.compress_ratio == 0 or self.window % self.compress_ratio == 0
        assert self.nope_dim % 64 == 0, "act_quant blocks the nope part by 64"
        assert self.heads % self.o_groups == 0
        assert self.head_dim % 2 == 0 and self.rope_dim % 2 == 0


def qkv_a_tail(cfg: V4Config) -> int:
    """Rows of the fused qkv_a GEMV past q_a and kv: the compressors' wkv / wgate pairs."""
    if not cfg.compress_ratio:
        return 0
    rows = 2 * cfg.c_coff * cfg.head_dim
    if cfg.indexed:
        rows += 2 * cfg.c_coff * cfg.index_head_dim
    return rows


def qkv_a_split(cfg: V4Config):
    """Column ranges of the fused qkv_a output, in the order it is laid out."""
    hd, cw = cfg.head_dim, cfg.c_coff * cfg.head_dim
    iw = cfg.c_coff * cfg.index_head_dim
    o = {"q_a": (0, cfg.q_lora), "kv": (cfg.q_lora, cfg.q_lora + hd)}
    p = cfg.q_lora + hd
    if cfg.compress_ratio:
        o["c_kv"], o["c_gate"] = (p, p + cw), (p + cw, p + 2 * cw)
        p += 2 * cw
        if cfg.indexed:
            o["i_kv"], o["i_gate"] = (p, p + iw), (p + iw, p + 2 * iw)
    return o


def qkv_a_matrix(t: dict) -> torch.Tensor:
    """The fused qkv_a matrix, dequantized: FP8 q_a | kv rows, then BF16 ``w_qkv_c`` rows."""
    w = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
    return torch.cat([w, t["w_qkv_c"].float()]) if "w_qkv_c" in t else w


def fp8_mats(cfg: V4Config):
    """(rows, K, BK) of every FP8 attention matrix in one rank's shard."""
    return {
        # q_a and kv only: the compressors' rows are BF16 (w_qkv_c), as ATOM applies them
        "qkv_a": (cfg.q_lora + cfg.head_dim, cfg.hidden, 128),
        "q_b": (cfg.heads * cfg.head_dim, cfg.q_lora, 128),
        **({"i_q_b": (cfg.index_heads * cfg.index_head_dim, cfg.q_lora, 128)} if cfg.indexed else {}),
        "o_a": (cfg.o_groups * cfg.o_lora, cfg.group_dim, 128),
        "o_b": (cfg.hidden, cfg.o_groups * cfg.o_lora, 128),
    }


@dataclass
class LayerWeights:
    cfg: V4Config
    t: dict  # name -> tensor


def quant_dequant_mxfp8(x: torch.Tensor) -> torch.Tensor:
    """Per-1x32 MXFP8 E4M3 quantization, dequantized; the E8M0 scale rounds up so no block clips."""
    shape = x.shape
    blocks = x.float().reshape(*shape[:-1], shape[-1] // 32, 32)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0))).clamp_min(torch.finfo(torch.float32).tiny)
    q = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(shape)


def expert_matrix(t: dict, name: str, e: int, cfg: V4Config, weight: ExpertWeight) -> torch.Tensor:
    """Expert ``e``'s dequantized ``ug`` or ``dn`` matrix (the shared one may be FP8 ``w_s<name>``)."""
    if e == cfg.shared_expert and f"w_s{name}" in t:
        return dequant(t[f"w_s{name}"], t[f"s_s{name}"], 128)
    return dequant_expert(t[f"w_{name}"][e], t[f"s_{name}"][e], weight)


def make_weights(
    rank: int,
    cfg: V4Config | None = None,
    device="cuda",
    seed: int = 1234,
    moe_mode: MoeMode | str = MoeMode.A8W4,
) -> LayerWeights:
    """Replicated tensors share ``seed``; TP shards add ``rank`` to it."""
    cfg = cfg or V4Config()
    cfg.validate()
    expert_weight = moe_format(moe_mode).weight
    rep = torch.Generator(device=device).manual_seed(seed)
    shd = torch.Generator(device=device).manual_seed(seed + 1 + rank)
    t = {}
    bfl = torch.bfloat16

    t["g_in"] = (1 + 0.1 * torch.randn(cfg.hidden, generator=rep, device=device)).to(bfl)
    t["g_q"] = (1 + 0.1 * torch.randn(cfg.q_lora, generator=rep, device=device)).to(bfl)
    t["g_kv"] = (1 + 0.1 * torch.randn(cfg.head_dim, generator=rep, device=device)).to(bfl)
    t["g_post"] = (1 + 0.1 * torch.randn(cfg.hidden, generator=rep, device=device)).to(bfl)
    t["attn_sink"] = 0.5 * torch.randn(cfg.heads, generator=shd, device=device)

    for name, (rows, k, bk) in fp8_mats(cfg).items():
        # qkv_a and the indexer's query are replicated; the rest are shards
        gen = rep if name in ("qkv_a", "i_q_b") else shd
        t[f"w_{name}"], t[f"s_{name}"] = _rand_fp8(rows, k, bk, gen, device)
    if qkv_a_tail(cfg):
        t["w_qkv_c"] = (torch.randn(qkv_a_tail(cfg), cfg.hidden, generator=rep, device=device) / cfg.hidden**0.5).to(
            bfl
        )

    if cfg.compress_ratio:
        # compressors run in fp32, replicated
        r = cfg.compress_ratio
        t["ape"] = 0.5 * torch.randn(r, cfg.c_coff * cfg.head_dim, generator=rep, device=device)
        t["g_ckv"] = (1 + 0.1 * torch.randn(cfg.head_dim, generator=rep, device=device)).to(bfl)
        if cfg.indexed:
            ihd = cfg.index_head_dim
            t["i_ape"] = 0.5 * torch.randn(r, cfg.c_coff * ihd, generator=rep, device=device)
            t["g_ickv"] = (1 + 0.1 * torch.randn(ihd, generator=rep, device=device)).to(bfl)
            t["i_w"] = (torch.randn(cfg.index_heads, cfg.hidden, generator=rep, device=device) / cfg.hidden**0.5).to(
                bfl
            )

    if cfg.hc_mult > 1:
        # hyper-connection mixers: fp32, row-padded, replicated
        for side in ("attn", "ffn"):
            fn = torch.zeros(cfg.hc_rows, cfg.hc_mult * cfg.hidden, device=device)
            fn[: cfg.hc_mix] = (
                torch.randn(cfg.hc_mix, cfg.hc_mult * cfg.hidden, generator=rep, device=device)
                / (cfg.hc_mult * cfg.hidden) ** 0.5
            )
            t[f"hc_{side}_fn"] = fn
            t[f"hc_{side}_base"] = torch.randn(cfg.hc_mix, generator=rep, device=device) * 0.5
            t[f"hc_{side}_scale"] = torch.rand(3, generator=rep, device=device) + 0.5

    t["w_r"] = (torch.randn(cfg.n_experts, cfg.hidden, generator=rep, device=device) / cfg.hidden**0.5 * 4).to(bfl)
    t["bias"] = torch.randn(cfg.n_experts, generator=rep, device=device) * 0.1

    n_bank = cfg.n_experts + 1
    if expert_weight is ExpertWeight.FP8_BLOCK128:
        ug_q = torch.empty(n_bank, 2 * cfg.inter, cfg.hidden, dtype=torch.float8_e4m3fn, device=device)
        ug_s = torch.empty(n_bank, *scale_shape(2 * cfg.inter, cfg.hidden, 128), device=device)
        dn_q = torch.empty(n_bank, cfg.hidden, cfg.inter, dtype=torch.float8_e4m3fn, device=device)
        dn_s = torch.empty(n_bank, *scale_shape(cfg.hidden, cfg.inter, 128), device=device)
        for e in range(n_bank):
            ug_q[e], ug_s[e] = _rand_fp8(2 * cfg.inter, cfg.hidden, 128, shd, device)
            dn_q[e], dn_s[e] = _rand_fp8(cfg.hidden, cfg.inter, 128, shd, device)
    else:
        n_bank = cfg.n_experts  # the shared expert is FP8, beside the bank
        ug_q = torch.empty(n_bank, 2 * cfg.inter, cfg.hidden // 2, dtype=torch.uint8, device=device)
        ug_s = torch.empty(n_bank, 2 * cfg.inter, cfg.hidden // 32, dtype=torch.uint8, device=device)
        dn_q = torch.empty(n_bank, cfg.hidden, cfg.inter // 2, dtype=torch.uint8, device=device)
        dn_s = torch.empty(n_bank, cfg.hidden, cfg.inter // 32, dtype=torch.uint8, device=device)
        for e in range(n_bank):
            ug = torch.randn(2 * cfg.inter, cfg.hidden, generator=shd, device=device) / cfg.hidden**0.5
            dn = torch.randn(cfg.hidden, cfg.inter, generator=shd, device=device) / cfg.inter**0.5
            ug_q[e], ug_s[e] = quantize_mxfp4(ug)
            dn_q[e], dn_s[e] = quantize_mxfp4(dn)
        t["w_sug"], t["s_sug"] = _rand_fp8(2 * cfg.inter, cfg.hidden, 128, shd, device)
        t["w_sdn"], t["s_sdn"] = _rand_fp8(cfg.hidden, cfg.inter, 128, shd, device)
    t["w_ug"], t["s_ug"], t["w_dn"], t["s_dn"] = ug_q, ug_s, dn_q, dn_s
    return LayerWeights(cfg, t)


def rmsnorm(x: torch.Tensor, g: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm; ``g=None`` is the weightless per-head scale applied to the query."""
    x = x.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y if g is None else y * g.float()


FP4_MAX = 6.0
# e2m1 representable magnitudes in code order; the code IS the index
FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_BLOCK = 32


# KV NoPE is FP8 in 64-wide groups with E8M0 scales 2**ceil(log2(amax / 448)), as ATOM writes it.
# ATOM's fp8 row: 448 FP8 bytes, each group's scale byte twice, pad to 512; RoPE is a bf16 plane.
KV_GROUP = 64
KV_ROW_BYTES = 512


def _kv_group_exp(xb: torch.Tensor) -> torch.Tensor:
    """Per-group biased E8M0 exponent of ``2**ceil(log2(amax / 448))`` (int32)."""
    amax = xb.abs().amax(-1).clamp(min=FP8_MAX * 2.0**-126)
    bits = (amax / FP8_MAX).view(torch.int32)
    return ((bits >> 23) & 0xFF) + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)


def kv_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """The KV NoPE part's FP8 round trip: 64-wide groups, power-of-two scales."""
    xb = x.float().reshape(*x.shape[:-1], -1, KV_GROUP)
    s = torch.ldexp(torch.ones_like(xb[..., 0]), _kv_group_exp(xb) - 127)[..., None]
    return ((xb / s).to(torch.float8_e4m3fn).float() * s).reshape(x.shape)


def encode_kv_fp8(rows: torch.Tensor, rope_dim: int = ROPE_DIM) -> tuple[torch.Tensor, torch.Tensor]:
    """KV rows [..., head_dim] -> ATOM's fp8 layout: (NoPE plane uint8 [..., 512], RoPE bf16 [..., rope_dim])."""
    nope = rows[..., :-rope_dim].float()
    xb = nope.reshape(*nope.shape[:-1], -1, KV_GROUP)
    e = _kv_group_exp(xb)
    q = (xb / torch.ldexp(torch.ones_like(xb[..., 0]), e - 127)[..., None]).to(torch.float8_e4m3fn)
    out = torch.zeros(*rows.shape[:-1], KV_ROW_BYTES, dtype=torch.uint8, device=rows.device)
    n = nope.shape[-1]
    out[..., :n] = q.reshape(nope.shape).view(torch.uint8)
    out[..., n : n + 2 * e.shape[-1]] = e.to(torch.uint8).repeat_interleave(2, dim=-1)
    return out, rows[..., -rope_dim:].to(torch.bfloat16).contiguous()


def decode_kv_fp8(nope: torch.Tensor, rope: torch.Tensor, head_dim: int = HEAD_DIM) -> torch.Tensor:
    """ATOM's fp8 KV layout -> float rows [..., head_dim] (first copy of each scale byte)."""
    n = head_dim - rope.shape[-1]
    ng = n // KV_GROUP
    q = nope[..., :n].contiguous().view(torch.float8_e4m3fn).float().reshape(*nope.shape[:-1], ng, KV_GROUP)
    e = nope[..., n : n + 2 * ng : 2].long()
    v = (q * torch.ldexp(torch.ones_like(q[..., 0]), e - 127)[..., None]).reshape(*nope.shape[:-1], n)
    return torch.cat([v, rope.float()], dim=-1)


def fp4_pool_shapes(cfg: V4Config, samples: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(codes, scales) shapes of the indexer's paged FP4 pool, one run of blocks per sample."""
    k_pb = BLOCK_TOKENS // cfg.compress_ratio
    nb = -(-cfg.n_compressed // k_pb)
    ng = cfg.index_head_dim // FP4_BLOCK
    return (samples, nb, ng * k_pb * 16), (samples, nb, ng * k_pb)


def _fp4_pool_index(n, bt_row, k_pb, device):
    e = torch.arange(n, device=device)
    blk = bt_row.long()[e // k_pb]
    sl = e % k_pb
    # the scale pool's entry axis is interleaved in runs of 16 (ATOM / aiter's writer)
    return blk, sl, (sl % 16) * (k_pb // 16) + sl // 16


def fp4_pool_rows(codes, scales, bt_row, n, index_head_dim=INDEX_HEAD_DIM, k_pb=BLOCK_TOKENS // COMPRESS_CSA):
    """The first ``n`` entries of one sequence's paged FP4 pool (ATOM's gfx950 layout) as ``pack_fp4`` rows."""
    ng = index_head_dim // FP4_BLOCK
    blk, sl, sfl = _fp4_pool_index(n, bt_row, k_pb, codes.device)
    c = codes.reshape(-1, ng, k_pb, 16)[blk, :, sl, :].reshape(n, ng * 16)
    e = scales.reshape(-1, ng, k_pb)[blk, :, sfl]
    return torch.cat([c, e], dim=-1)


def fp4_pool_store(codes, scales, bt_row, rows, index_head_dim=INDEX_HEAD_DIM, k_pb=BLOCK_TOKENS // COMPRESS_CSA):
    """Write ``pack_fp4`` rows [n, fp4_row_bytes] into one sequence's paged FP4 pool."""
    ng = index_head_dim // FP4_BLOCK
    n = rows.shape[0]
    blk, sl, sfl = _fp4_pool_index(n, bt_row, k_pb, codes.device)
    cv = codes.view(-1, ng, k_pb, 16)
    sv = scales.view(-1, ng, k_pb)
    cv[blk, :, sl, :] = rows[:, : ng * 16].reshape(n, ng, 16)
    sv[blk, :, sfl] = rows[:, ng * 16 :]


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard transform over the last dim."""
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"hadamard size must be a power of two, got {n}"
    y = x.float().reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :].clone(), y[:, :, 1, :].clone()
        y[:, :, 0, :], y[:, :, 1, :] = a + b, a - b
        y = y.reshape(-1, n)
        h *= 2
    return (y * n**-0.5).reshape(x.shape)


def _fp4_quant(x: torch.Tensor, block: int):
    """Blocked e2m1 quantization: value = ``sign * FP4_LEVELS[index] * 2**e``."""
    n = x.shape[-1]
    xb = x.float().reshape(*x.shape[:-1], n // block, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=FP4_MAX * 2.0**-126)
    # ceil(log2(amax / FP4_MAX)) by exponent arithmetic, as the model does
    bits = (amax / FP4_MAX).view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    q = (xb / torch.ldexp(torch.ones_like(amax), e)).clamp(-FP4_MAX, FP4_MAX)
    lv = torch.tensor(FP4_LEVELS, device=x.device, dtype=torch.float32)
    mid = (lv[1:] + lv[:-1]) / 2
    mag = q.abs()
    down, up = torch.bucketize(mag, mid, right=False), torch.bucketize(mag, mid, right=True)
    idx = torch.where(up != down, torch.where(down % 2 == 0, down, up), down)
    return torch.sign(q), idx, e


def quant_dequant_fp4(x: torch.Tensor, block: int = FP4_BLOCK) -> torch.Tensor:
    """FP4 (e2m1) round trip with power-of-2 scales rounded up; ties round to the even code."""
    sign, idx, e = _fp4_quant(x, block)
    lv = torch.tensor(FP4_LEVELS, device=x.device, dtype=torch.float32)
    return (sign * torch.ldexp(lv[idx], e)).reshape(x.shape).to(x.dtype)


def fp4_row_bytes(n: int) -> int:
    """Bytes of one ``pack_fp4`` row of ``n`` elements: codes, then scales."""
    return n // 2 + n // FP4_BLOCK


def pack_fp4(x: torch.Tensor) -> torch.Tensor:
    """``quant_dequant_fp4``'s result as uint8 rows: nibble codes (even ``i`` low), then E8M0 bytes."""
    sign, idx, e = _fp4_quant(x, FP4_BLOCK)
    code = (idx | ((sign < 0).to(idx.dtype) << 3)).reshape(*x.shape[:-1], -1, 2)
    codes = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    return torch.cat([codes, (e[..., 0] + 127).to(torch.uint8)], dim=-1)


def unpack_fp4(p: torch.Tensor) -> torch.Tensor:
    """``pack_fp4``'s rows [..., fp4_row_bytes(n)] back to float32 [..., n]."""
    n = p.shape[-1] * 2 * FP4_BLOCK // (FP4_BLOCK + 2)
    b = p[..., : n // 2].long()
    code = torch.stack([b & 0xF, b >> 4], dim=-1).reshape(*p.shape[:-1], n // FP4_BLOCK, FP4_BLOCK)
    lv = torch.tensor(FP4_LEVELS, device=p.device, dtype=torch.float32)
    v = torch.where(code >= 8, -1.0, 1.0) * lv[code & 7]
    e = p[..., n // 2 :].long() - 127
    return torch.ldexp(v, e[..., None]).reshape(*p.shape[:-1], n)


def hc_split_sinkhorn(mixes: torch.Tensor, scale, base, cfg: V4Config):
    """Split pre | post | comb mixes into coefficients; ``comb`` is Sinkhorn-normalised."""
    hc, eps = cfg.hc_mult, cfg.hc_eps
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x: torch.Tensor, fn: torch.Tensor, scale, base, cfg: V4Config):
    """[S, hc, hidden] -> ([S, hidden], post [S, hc], comb [S, hc, hc])."""
    shape, dtype = x.size(), x.dtype
    xf = x.flatten(-2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.eps)
    mixes = torch.nn.functional.linear(xf, fn[: cfg.hc_mix].float()) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, cfg)
    y = (pre.unsqueeze(-1) * xf.view(shape)).sum(dim=-2)
    return y.to(dtype), post, comb


def hc_post(x: torch.Tensor, residual: torch.Tensor, post, comb):
    """out[k] = post[k] * x + sum_j comb[j, k] * residual[j]."""
    mixed = (comb.unsqueeze(-1) * residual.unsqueeze(-2).float()).sum(dim=-3)
    return (post.unsqueeze(-1) * x.unsqueeze(-2).float() + mixed).type_as(x)


def compress_step(
    kv,
    score,
    cur_pos,
    cfg,
    t,
    kv_state,
    score_state,
    cache,
    cos_c,
    sin_c,
    head_dim=None,
    ape=None,
    gamma=None,
    dest_row=None,  # default cur_pos // ratio
    rotate=False,
):
    """One decode step of the KV compressor; mutates the states and ``cache``.

    The overrides serve the indexer's compressor; ``rotate`` gives it the Hadamard + FP4 tail.
    """
    r, rd = cfg.compress_ratio, cfg.rope_dim
    d = cfg.head_dim if head_dim is None else head_dim
    ape = t["ape"] if ape is None else ape
    gamma = t["g_ckv"] if gamma is None else gamma
    kv = kv.float()
    score = score.float() + ape[cur_pos % r]
    if cfg.overlap:
        # ring of 2r rows (ATOM's layout): previous window's first half, current window's second
        kv_state[cur_pos % (2 * r)] = kv
        score_state[cur_pos % (2 * r)] = score
        if (cur_pos + 1) % r:
            return None
        rows = [(cur_pos + 1 + i) % (2 * r) for i in range(2 * r)]  # oldest first
        ks = torch.cat([kv_state[rows[:r], :d], kv_state[rows[r:], d:]], dim=0)
        ss = torch.cat([score_state[rows[:r], :d], score_state[rows[r:], d:]], dim=0)
        pooled = (ks * ss.softmax(dim=0)).sum(dim=0)
    else:
        kv_state[cur_pos % r] = kv
        score_state[cur_pos % r] = score
        if (cur_pos + 1) % r:
            return None
        pooled = (kv_state * score_state.softmax(dim=0)).sum(dim=0)
    # bf16 as in the model: RoPE and the quantization below see bf16
    v = bf(rmsnorm(pooled.to(torch.bfloat16), gamma, cfg.eps))
    anchor = cur_pos + 1 - r  # the window's FIRST position carries the rotation
    v = torch.cat([v[:-rd], bf(rope(v[-rd:], cos_c[anchor], sin_c[anchor]))])
    row = cur_pos // r if dest_row is None else dest_row
    if rotate:
        if cfg.indexer_hadamard:  # bf(): FP4 sees bf16, as in the model
            v = bf(hadamard(v))
        cache[row] = pack_fp4(v)
        return quant_dequant_fp4(v)
    v = torch.cat([kv_quant_dequant(v[:-rd]), v[-rd:]])
    cache[row] = v.to(torch.bfloat16)
    return v


def indexer_step(x, q_a_n, i_kv, i_gate, cur_pos, cfg, t, i_state, i_score_state, i_cache, cos, sin):
    """One lightning-indexer step: the cache slots CSA attends to, padded with -1 to ``cfg.n_index``.

    Every rank holds all index heads, so the score needs no all-reduce.
    """
    ih, ihd, rd, r = cfg.index_heads, cfg.index_head_dim, cfg.rope_dim, cfg.compress_ratio
    compress_step(
        i_kv,
        i_gate,
        cur_pos,
        cfg,
        t,
        i_state,
        i_score_state,
        i_cache,
        cos,
        sin,
        head_dim=ihd,
        ape=t["i_ape"],
        gamma=t["g_ickv"],
        rotate=True,
    )
    dq = dequant(t["w_i_q_b"], t["s_i_q_b"], 128)
    q = (q_a_n.float() @ dq.T).view(ih, ihd)
    q = torch.stack([torch.cat([q[h, :-rd], rope(q[h, -rd:], cos[cur_pos], sin[cur_pos])]) for h in range(ih)])
    q = quant_dequant_fp4(bf(hadamard(q)) if cfg.indexer_hadamard else bf(q))
    w = bf(x.float() @ t["i_w"].float().T) * (ihd**-0.5 * ih**-0.5)

    n = (cur_pos + 1) // r  # compressed entries written so far
    out = torch.full((cfg.n_index,), -1, dtype=torch.int32, device=x.device)
    if n:
        score = torch.einsum("hd,td->ht", q, unpack_fp4(i_cache[:n]))
        score = (score.relu() * w.view(ih, 1)).sum(dim=0)
        k = min(cfg.index_topk, n)
        out[:k] = score.topk(k)[1].to(torch.int32)
    return out


def layer_idxs(positions, cfg: V4Config, device) -> torch.Tensor:
    """Window ring slots, then compressed entries so far (-1 unwritten), one row per position.

    On a CSA layer the compressed half is empty: the indexer fills it inside the layer.
    """
    win = window_idxs(positions, cfg.window, device)
    if not cfg.compress_ratio:
        return win
    rows = []
    for p in positions:
        n = 0 if cfg.indexed else (p + 1) // cfg.compress_ratio
        rows.append([cfg.window + i for i in range(n)] + [-1] * (cfg.n_index - n))
    comp = torch.tensor(rows, dtype=torch.int32, device=device)
    idx = torch.cat([win, comp], dim=1)
    pad = cfg.n_keys - idx.shape[1]
    if pad:
        idx = torch.cat([idx, torch.full((len(positions), pad), -1, dtype=torch.int32, device=device)], dim=1)
    return idx


def contiguous_pool(positions, cfg: V4Config, device):
    """``(indices, dest_rows)`` of absolute plane rows for a trivial pool: sample ``s`` owns
    ``cache_rows`` rows from ``s * cache_rows``. ``dest_rows`` [2, S]: the window row, then
    compressed entry 0's row."""
    base = torch.tensor([s * cfg.cache_rows for s in range(len(positions))], dtype=torch.int32, device=device)
    idx = layer_idxs(positions, cfg, device)
    idx = torch.where(idx >= 0, idx + base[:, None], idx)
    win = torch.tensor([p % cfg.window for p in positions], dtype=torch.int32, device=device)
    comp = torch.full_like(base, cfg.window)
    return idx, torch.stack([win + base, comp + base])


def window_idxs(positions, window: int, device) -> torch.Tensor:
    """Ring slots oldest first, -1 where unfilled (DeepSeek's ``get_window_topk_idxs``)."""
    rows = []
    for p in positions:
        if p + 1 >= window:
            start = (p + 1) % window
            rows.append([(start + i) % window for i in range(window)])
        else:
            rows.append(list(range(p + 1)) + [-1] * (window - p - 1))
    return torch.tensor(rows, dtype=torch.int32, device=device)


def route(scores: torch.Tensor, bias: torch.Tensor, cfg: V4Config):
    """Flat top-k: (indices, probs) in score order; weights from the unbiased score (``noaux_tc``).

    As in the kernel's packed-key argmax, the key's low bits are the inverted expert id.
    """
    n = cfg.n_experts
    bits = (scores.float() + bias.float()).view(torch.int32).long()
    okey = torch.where(bits >= 0, bits ^ (1 << 31), ~bits & 0xFFFFFFFF) & 0xFFFFFFFF
    id_bits = max(8, (n - 1).bit_length())
    id_mask = (1 << id_bits) - 1
    key = (okey & (0xFFFFFFFF ^ id_mask)) | (id_mask - torch.arange(n, device=scores.device))
    idx = torch.argsort(key, descending=True)[: cfg.top_k]
    p = scores[idx]
    return idx, p / p.sum() * cfg.route_scale


def sparse_attention(q, kv_cache, keys, sink, scale, split=64):
    """One sample's shared-KV attention over plane rows ``keys`` (-1 masked); ``sink`` is denominator only.

    Merged flash-style in ``split``-key chunks with bf16 P, the kernel's rounding order.
    """
    valid = keys >= 0
    k = kv_cache.float()[keys.clamp(min=0).long()]
    sc = (bf(q) @ k.T) * scale
    sc = sc.masked_fill(~valid.unsqueeze(0), float("-inf"))
    ms, ls, accs = [], [], []
    for k0 in range(0, keys.numel(), split):
        scs = sc[:, k0 : k0 + split]
        m = scs.amax(-1, keepdim=True)
        m = torch.where(torch.isneginf(m), torch.zeros_like(m), m)
        p = torch.exp(scs - m)
        ms.append(m)
        ls.append(p.sum(-1, keepdim=True))
        accs.append(bf(p) @ k[k0 : k0 + split])
    mx = torch.stack(ms).amax(0)
    w = [torch.exp(m - mx) for m in ms]
    denom = sum(li * wi for li, wi in zip(ls, w)) + torch.exp(sink.unsqueeze(-1) - mx)
    return sum(a * wi for a, wi in zip(accs, w)) / denom


def golden_layer(
    W: LayerWeights,
    h,
    positions,
    kv_cache,
    dest_rows,
    indices,
    cos,
    sin,
    allreduce,
    moe_mode: MoeMode | str = MoeMode.A8W4,
    kv_state=None,
    score_state=None,
    cos_c=None,
    sin_c=None,
    i_state=None,
    i_score_state=None,
    i_cache=None,
    tokens=None,
):
    """One rank's view of a V4 layer; mutates ``kv_cache`` and the compressor state.

    ``indices`` / ``dest_rows`` are absolute rows of one ``kv_cache`` plane (see ``contiguous_pool``).
    Returns intermediates keyed like the kernel's debug scratch.
    """
    cfg, t = W.cfg, W.t
    H, S = cfg.heads, h.shape[0]
    rd, hd = cfg.rope_dim, cfg.head_dim
    dq = {n: dequant(t[f"w_{n}"], t[f"s_{n}"], bk) for n, (_, _, bk) in fp8_mats(cfg).items()}

    if cfg.hc_mult > 1:
        xin, post_a, comb_a = hc_pre(h, t["hc_attn_fn"], t["hc_attn_scale"], t["hc_attn_base"], cfg)
    else:
        xin, post_a, comb_a = h, None, None
    x = bf(rmsnorm(xin, t["g_in"], cfg.eps))
    qkv = x @ qkv_a_matrix(t).T
    hd = cfg.head_dim
    cut = qkv_a_split(cfg)
    part = lambda n: qkv[:, slice(*cut[n])] if n in cut else None  # noqa: E731
    q_a, kv = part("q_a"), part("kv")
    c_kv, c_gate = part("c_kv"), part("c_gate")
    i_kv, i_gate = part("i_kv"), part("i_gate")

    q_an = bf(rmsnorm(q_a, t["g_q"], cfg.eps))
    q = (q_an @ dq["q_b"].T).view(S, H, hd)
    q = rmsnorm(q, None, cfg.eps)
    q = torch.stack(
        [
            torch.cat([q[s, :, :-rd], rope(q[s, :, -rd:], cos[positions[s]], sin[positions[s]])], dim=-1)
            for s in range(S)
        ]
    )
    if cfg.kv_fp8:
        # ATOM's fp8 attention quantizes the query's NoPE part like the KV's
        q = torch.cat([kv_quant_dequant(q[..., :-rd]), q[..., -rd:]], dim=-1)

    for s in range(S):
        p = positions[s]
        v = rmsnorm(kv[s], t["g_kv"], cfg.eps)
        v = torch.cat([kv_quant_dequant(v[:-rd]), rope(v[-rd:], cos[p], sin[p])])
        kv_cache[int(dest_rows[0, s])] = v.to(torch.bfloat16)
        if cfg.compress_ratio:
            compress_step(
                c_kv[s],
                c_gate[s],
                p,
                cfg,
                t,
                kv_state[s],
                score_state[s],
                kv_cache,
                cos_c,
                sin_c,
                dest_row=int(dest_rows[1, s]) + p // cfg.compress_ratio,
            )

    if cfg.indexed:
        # CSA: keep the window half of the index list, replace the rest with the indexer's picks
        picks = torch.stack(
            [
                indexer_step(
                    x[s],
                    q_an[s],
                    i_kv[s],
                    i_gate[s],
                    positions[s],
                    cfg,
                    t,
                    i_state[s],
                    i_score_state[s],
                    i_cache[s],
                    cos_c,
                    sin_c,
                )
                for s in range(S)
            ]
        )
        indices = indices.clone()
        # picks are entry ids: rebase onto the caller's entry-0 row
        indices[:, cfg.window : cfg.window + cfg.n_index] = torch.where(
            picks >= 0, picks + dest_rows[1][:, None], picks
        )
        indices[:, cfg.window + cfg.n_index :] = -1

    sink = t["attn_sink"].float()
    o = torch.stack([sparse_attention(q[s], kv_cache, indices[s], sink, cfg.softmax_scale) for s in range(S)])

    # V is the RoPE'd K: de-rotate the output
    o = torch.stack(
        [
            torch.cat([o[s, :, :-rd], rope(o[s, :, -rd:], cos[positions[s]], sin[positions[s]], True)], dim=-1)
            for s in range(S)
        ]
    )

    og = bf(o).reshape(S, cfg.o_groups, cfg.group_dim)
    wa = dq["o_a"].view(cfg.o_groups, cfg.o_lora, cfg.group_dim)
    o_lora = torch.einsum("sgd,grd->sgr", og, wa).reshape(S, cfg.o_groups * cfg.o_lora)
    attn_out = allreduce(bf(o_lora) @ dq["o_b"].T)
    if cfg.hc_mult > 1:
        a = hc_post(attn_out.to(torch.bfloat16), h, post_a, comb_a)
    else:
        a = (h.float() + attn_out).to(torch.bfloat16)

    hash_ids = None
    if "tid2eid" in t:
        assert tokens is not None, "a hash-routed layer needs the token ids"
        hash_ids = t["tid2eid"][tokens.long()]
    moe = golden_moe(W, a, allreduce, moe_mode=moe_mode, hash_ids=hash_ids)
    # attn_out is before the residual / hc mix, as ATOM's DeepseekV4Attention.forward_impl returns it
    res = dict(q_a=q_a, kv=kv, q=q, o=o, o_lora=o_lora, attn_out=attn_out, a=a, xin=xin)
    if cfg.indexed:
        res["picks"] = picks
    res.update(moe)
    return res


def golden_moe(
    W: LayerWeights,
    a,
    allreduce,
    mid=None,
    sel=None,
    prob=None,
    xq=None,
    hash_ids=None,
    moe_mode: MoeMode | str = MoeMode.A8W4,
):
    """MoE half from the post-attention state ``a``; the overrides feed a stage the kernel's own inputs.

    ``hash_ids`` [S, top_k] replaces scored routing (V4's hash layers).
    """
    cfg, t = W.cfg, W.t
    mode = as_moe_mode(moe_mode)
    fmt = moe_format(mode)
    S = a.shape[0]
    out = {k: [] for k in ("sel", "prob", "mid")}

    if cfg.hc_mult > 1:
        ain, post_f, comb_f = hc_pre(a, t["hc_ffn_fn"], t["hc_ffn_scale"], t["hc_ffn_base"], cfg)
    else:
        ain, post_f, comb_f = a, None, None
    x2 = rmsnorm(ain, t["g_post"], cfg.eps)
    # bf16 logits, as ATOM's gate GEMM writes them
    scores = torch.nn.functional.softplus(bf(bf(x2) @ t["w_r"].float().T)).sqrt()

    if fmt.activation is ExpertActivation.FP8_BLOCK128:
        xq_ref = quant_dequant(x2)
    elif fmt.activation is ExpertActivation.MXFP8_BLOCK32:
        xq_ref = quant_dequant_mxfp8(x2)
    else:
        xq_ref = bf(x2)
    xq = xq_ref if xq is None else xq.float()

    lim = cfg.swiglu_limit
    y = torch.zeros(S, cfg.hidden, device=a.device)
    for s in range(S):
        if hash_ids is None:
            idx, p = route(scores[s], t["bias"], cfg)
        else:
            idx = hash_ids[s].long()
            raw = scores[s][idx]
            p = raw / raw.sum() * cfg.route_scale
        # given routing (the kernel's own) drives the up/gate as well as the down
        experts = [cfg.shared_expert] + idx.tolist() if sel is None else sel[s].tolist()
        weights = [1.0] + p.tolist() if prob is None else prob[s].tolist()
        mids = []
        for e in experts:
            ug = expert_matrix(t, "ug", e, cfg, fmt.weight) @ xq[s]
            gate, up = ug[: cfg.inter], ug[cfg.inter :]
            if lim > 0:
                # up is clamped both sides, gate only above
                gate = gate.clamp(max=lim)
                up = up.clamp(min=-lim, max=lim)
            value = torch.nn.functional.silu(gate) * up
            mids.append(bf(value) if fmt.activation is ExpertActivation.BF16 else value)
        out["sel"].append(torch.tensor(experts, device=a.device, dtype=torch.int32))
        out["prob"].append(torch.tensor(weights, device=a.device))
        out["mid"].append(torch.stack(mids))

    for s in range(S):
        experts, weights = out["sel"][s].tolist(), out["prob"][s].tolist()
        for j, (e, wgt) in enumerate(zip(experts, weights)):
            m = out["mid"][s][j] if mid is None else mid[s, j].float()
            if fmt.activation is ExpertActivation.FP8_BLOCK128:
                activation = quant_dequant(m)
            elif fmt.activation is ExpertActivation.MXFP8_BLOCK32:
                activation = quant_dequant_mxfp8(m)
            else:
                activation = bf(m)
            y[s] += wgt * (expert_matrix(t, "dn", e, cfg, fmt.weight) @ activation)

    ffn_out = allreduce(y)
    if cfg.hc_mult > 1:
        x_out = hc_post(ffn_out.to(torch.bfloat16), a, post_f, comb_f)
    else:
        x_out = (a.float() + ffn_out).to(torch.bfloat16)
    return dict(
        scores=scores,
        xq=xq_ref,
        x_out=x_out,
        sel=torch.stack(out["sel"]),
        prob=torch.stack(out["prob"]),
        mid=torch.stack(out["mid"]),
    )
