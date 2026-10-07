# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""One DeepSeek-V4 layer's TP-sharded weights from the HF checkpoint, as ``LayerWeights``.

Formats are kept as ATOM runs them: FP8 128x128 (shared expert included), MXFP4 routed
experts, BF16 compressor projections and FP32 hyper-connection mixers.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import torch
from safetensors import safe_open

from kernels.monokernel.dsv4.config import ExpertWeight, MoeMode, moe_format
from kernels.monokernel.dsv4.reference import LayerWeights, V4Config, fp8_mats


def config_for_layer(path: str, layer: int, tp: int) -> V4Config:
    """This rank's ``V4Config`` for ``layer``, from the checkpoint's own config.json."""
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    cfg = V4Config(
        heads=c["num_attention_heads"] // tp,
        hidden=c["hidden_size"],
        q_lora=c["q_lora_rank"],
        head_dim=c["head_dim"],
        rope_dim=c["qk_rope_head_dim"],
        o_groups=c["o_groups"] // tp,
        o_lora=c["o_lora_rank"],
        n_experts=c["n_routed_experts"],
        top_k=c["num_experts_per_tok"],
        inter=c["moe_intermediate_size"] // tp,
        window=c["sliding_window"],
        route_scale=c["routed_scaling_factor"],
        swiglu_limit=c["swiglu_limit"],
        eps=c["rms_norm_eps"],
        rope_theta=c["rope_theta"],
        index_heads=c["index_n_heads"],  # replicated, not sharded
        index_head_dim=c["index_head_dim"],
        index_topk=c["index_topk"],
        compress_rope_theta=c["compress_rope_theta"],
        hc_mult=c["hc_mult"],
        hc_sinkhorn_iters=c["hc_sinkhorn_iters"],
        hc_eps=c["hc_eps"],
    )
    return replace(cfg, compress_ratio=c["compress_ratios"][layer])


def is_hash_layer(path: str, layer: int) -> bool:
    with open(os.path.join(path, "config.json")) as f:
        return layer < json.load(f)["num_hash_layers"]


class Checkpoint:
    """Lazy, sliced reads of a sharded safetensors checkpoint."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.index = json.load(f)["weight_map"]
        self._files = {}

    def has(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str, rows=None, cols=None) -> torch.Tensor:
        fname = self.index[name]
        if fname not in self._files:
            self._files[fname] = safe_open(os.path.join(self.path, fname), "pt")
        sl = self._files[fname].get_slice(name)
        r = slice(*rows) if rows else slice(None)
        if len(sl.get_shape()) == 1:
            return sl[r]
        return sl[r, slice(*cols) if cols else slice(None)]


def e8m0_float(s: torch.Tensor) -> torch.Tensor:
    """E8M0 scale bytes -> their float32 values, 2**(byte - 127)."""
    return torch.exp2(s.view(torch.uint8).float() - 127.0)


def layer_prefix(path: str, layer: int) -> str:
    """``layers.N.`` for the main stack, ``mtp.M.`` for layer num_hidden_layers + M (ATOM's MTP layer_id)."""
    with open(os.path.join(path, "config.json")) as f:
        n = json.load(f)["num_hidden_layers"]
    return f"layers.{layer}." if layer < n else f"mtp.{layer - n}."


def load_expert(ck: Checkpoint, layer: int, e: int, rank: int, tp: int, device="cuda"):
    """Routed expert ``e``'s MXFP4 [gate; up] and down matrices and E8M0 scales (uint8, row-major)."""
    cfg = config_for_layer(ck.path, layer, tp)
    inter = cfg.inter

    def shard(n):
        return (n * rank, n * (rank + 1))

    x = layer_prefix(ck.path, layer) + f"ffn.experts.{e}."
    rows = shard(inter)
    ug = [ck.get(x + w + ".weight", rows).view(torch.uint8) for w in ("w1", "w3")]
    ugs = [ck.get(x + w + ".scale", rows).view(torch.uint8) for w in ("w1", "w3")]
    dn = ck.get(x + "w2.weight", cols=shard(inter // 2)).view(torch.uint8)
    dns = ck.get(x + "w2.scale", cols=shard(inter // 32)).view(torch.uint8)
    return tuple(v.to(device).contiguous() for v in (torch.cat(ug), torch.cat(ugs), dn, dns))


def load_layer(
    ck: Checkpoint,
    layer: int,
    rank: int,
    tp: int,
    device="cuda",
    moe_mode: MoeMode | str = MoeMode.A8W4,
    experts: bool = True,
) -> LayerWeights:
    """``layer``'s weights for TP rank ``rank`` of ``tp``, in ``make_weights``' layout.

    ``experts=False`` skips the routed bank, for callers passing ``Dsv4MonoKernel(experts=)``.
    """
    cfg = config_for_layer(ck.path, layer, tp)
    cfg.validate()
    p = layer_prefix(ck.path, layer)
    t = {}

    def fp8(name, rows=None, cols=None):
        """An FP8 matrix and its block scales, sliced to this rank's shard."""
        q = ck.get(name + ".weight", rows, cols).view(torch.float8_e4m3fn)
        sr = (rows[0] // 128, rows[1] // 128) if rows else None
        sc = (cols[0] // 128, cols[1] // 128) if cols else None
        return q.to(device), e8m0_float(ck.get(name + ".scale", sr, sc)).to(device)

    def shard(n):
        return (n * rank, n * (rank + 1))

    t["g_in"] = ck.get(p + "attn_norm.weight").to(device)
    t["g_q"] = ck.get(p + "attn.q_norm.weight").to(device)
    t["g_kv"] = ck.get(p + "attn.kv_norm.weight").to(device)
    t["g_post"] = ck.get(p + "ffn_norm.weight").to(device)
    t["attn_sink"] = ck.get(p + "attn.attn_sink", shard(cfg.heads)).float().to(device)

    # qkv_a, in qkv_a_split's order: wq_a | wkv | compressor wkv | wgate | indexer's pair
    parts = [fp8(p + "attn.wq_a"), fp8(p + "attn.wkv")]
    bf16_rows = []
    if cfg.compress_ratio:
        bf16_rows += [p + "attn.compressor.wkv.weight", p + "attn.compressor.wgate.weight"]
        if cfg.indexed:
            bf16_rows += [p + "attn.indexer.compressor.wkv.weight", p + "attn.indexer.compressor.wgate.weight"]
    t["w_qkv_a"] = torch.cat([q for q, _ in parts])
    t["s_qkv_a"] = torch.cat([s for _, s in parts])
    if bf16_rows:
        t["w_qkv_c"] = torch.cat([ck.get(name).to(device).to(torch.bfloat16) for name in bf16_rows])

    t["w_q_b"], t["s_q_b"] = fp8(p + "attn.wq_b", rows=shard(cfg.heads * cfg.head_dim))
    t["w_o_a"], t["s_o_a"] = fp8(p + "attn.wo_a", rows=shard(cfg.o_groups * cfg.o_lora))
    t["w_o_b"], t["s_o_b"] = fp8(p + "attn.wo_b", cols=shard(cfg.o_groups * cfg.o_lora))

    if cfg.compress_ratio:
        t["ape"] = ck.get(p + "attn.compressor.ape").float().to(device)
        t["g_ckv"] = ck.get(p + "attn.compressor.norm.weight").to(device)
        if cfg.indexed:
            t["w_i_q_b"], t["s_i_q_b"] = fp8(p + "attn.indexer.wq_b")
            t["i_ape"] = ck.get(p + "attn.indexer.compressor.ape").float().to(device)
            t["g_ickv"] = ck.get(p + "attn.indexer.compressor.norm.weight").to(device)
            t["i_w"] = ck.get(p + "attn.indexer.weights_proj.weight").to(device)

    if cfg.hc_mult > 1:
        for side in ("attn", "ffn"):
            fn = torch.zeros(cfg.hc_rows, cfg.hc_mult * cfg.hidden, dtype=torch.float32, device=device)
            fn[: cfg.hc_mix] = ck.get(p + f"hc_{side}_fn").float().to(device)
            t[f"hc_{side}_fn"] = fn  # fp32: packing splits it into bf16 hi / lo
            t[f"hc_{side}_base"] = ck.get(p + f"hc_{side}_base").float().to(device)
            t[f"hc_{side}_scale"] = ck.get(p + f"hc_{side}_scale").float().to(device)

    t["w_r"] = ck.get(p + "ffn.gate.weight").to(device)
    bias = p + "ffn.gate.bias"
    # explicit dtypes: ATOM sets torch's default dtype to bf16
    t["bias"] = (
        ck.get(bias).float().to(device)
        if ck.has(bias)
        else torch.zeros(cfg.n_experts, dtype=torch.float32, device=device)
    )
    if ck.has(p + "ffn.gate.tid2eid"):
        t["tid2eid"] = ck.get(p + "ffn.gate.tid2eid").to(torch.int32).to(device)

    if moe_format(moe_mode).weight is not ExpertWeight.MXFP4_BLOCK32:
        raise NotImplementedError("the checkpoint's routed experts are MXFP4: load them with MoeMode.A8W4")
    inter = cfg.inter
    rows = shard(inter)
    if experts:
        bank = [load_expert(ck, layer, e, rank, tp, device) for e in range(cfg.n_experts)]
        t["w_ug"], t["s_ug"], t["w_dn"], t["s_dn"] = (torch.stack(v) for v in zip(*bank))
    sh = p + "ffn.shared_experts."
    (gq, gs), (uq, us) = fp8(sh + "w1", rows=rows), fp8(sh + "w3", rows=rows)
    t["w_sug"], t["s_sug"] = torch.cat([gq, uq]), torch.cat([gs, us])
    # contiguous: a column slice, read as dense row-major
    t["w_sdn"], t["s_sdn"] = (v.contiguous() for v in fp8(sh + "w2", cols=rows))

    for name, (r, k, _bk) in fp8_mats(cfg).items():
        assert t[f"w_{name}"].shape == (r, k), f"{name}: {tuple(t[f'w_{name}'].shape)} != {(r, k)}"
    return LayerWeights(cfg, t)
