# SPDX-License-Identifier: Apache-2.0
"""Load one native Pro MoE layer, following ATOM's config and TP slicing."""

import json
from pathlib import Path
from types import SimpleNamespace

from .config import Dsv4Config


def load_moe(path, layer, tp, rank, device):
    import torch
    from atom.models.deepseek_v4 import DeepseekV4Args
    from safetensors import safe_open

    path = Path(path)
    raw = json.loads((path / "config.json").read_text())
    args = DeepseekV4Args.from_hf_config(SimpleNamespace(**raw))
    cfg = Dsv4Config.from_atom_args(args)
    cfg.validate_pro()
    cfg.validate_moe(1, tp)
    if raw.get("expert_dtype") != "fp4" or not 0 <= layer < args.n_layers:
        raise ValueError("expected a native FP4 Pro checkpoint and a backbone layer index")
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{layer}.ffn."
    handles = {}

    def read(name, selection=None, dtype=None):
        key = prefix + name
        shard = index[key]
        if shard not in handles:
            handles[shard] = safe_open(path / shard, framework="pt", device="cpu")
        value = handles[shard].get_slice(key)
        value = value[:] if selection is None else value[selection]
        # FP4 and E8M0 are raw bytes; numeric casts would corrupt their bits.
        if dtype is not None:
            value = value.view(dtype)
        return value.to(device=device).contiguous()

    h, i, e = cfg.hidden, cfg.intermediate // tp, cfg.experts
    rows = slice(rank * i, (rank + 1) * i)
    cols = slice(rank * i // 2, (rank + 1) * i // 2)
    scale_cols = slice(rank * i // 32, (rank + 1) * i // 32)
    router = read("gate.weight")
    bias = torch.zeros(e, dtype=torch.float32, device=device) if layer < cfg.hash_layers else read("gate.bias")
    tid2eid = read("gate.tid2eid").to(torch.int32) if layer < cfg.hash_layers else None
    if tid2eid is not None and ((tid2eid < 0).any() or (tid2eid >= e).any()):
        raise ValueError("checkpoint hash routing contains an out-of-range expert id")
    up = torch.empty(e, 2 * i, h // 2, dtype=torch.uint8, device=device)
    us = torch.empty(e, 2 * i, h // 32, dtype=torch.uint8, device=device)
    down = torch.empty(e, h, i // 2, dtype=torch.uint8, device=device)
    ds = torch.empty(e, h, i // 32, dtype=torch.uint8, device=device)
    for expert in range(e):
        for w, dest, sel in (
            ("w1.weight", up[expert, :i], (rows, slice(None))),
            ("w3.weight", up[expert, i:], (rows, slice(None))),
            ("w1.scale", us[expert, :i], (rows, slice(None))),
            ("w3.scale", us[expert, i:], (rows, slice(None))),
            ("w2.weight", down[expert], (slice(None), cols)),
            ("w2.scale", ds[expert], (slice(None), scale_cols)),
        ):
            dest.copy_(read(f"experts.{expert}.{w}", sel, torch.uint8))
    shared_rows = slice(rank * i // 128, (rank + 1) * i // 128)
    su = torch.cat([read(f"shared_experts.w{w}.weight", (rows, slice(None))) for w in (1, 3)])
    sus = torch.cat([read(f"shared_experts.w{w}.scale", (shared_rows, slice(None)), torch.uint8) for w in (1, 3)])
    sd = read("shared_experts.w2.weight", (slice(None), rows))
    sds = read("shared_experts.w2.scale", (slice(None), shared_rows), torch.uint8)
    return cfg, (router, bias, up, us, down, ds), (su, sus, sd, sds), tid2eid
