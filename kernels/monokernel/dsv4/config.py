# SPDX-License-Identifier: Apache-2.0
"""Geometry and semantics taken from ATOM's DeepseekV4Args.

The defaults describe V4 Pro. Integration must call ``from_atom_args`` so
checkpoint overrides (including Flash) are never replaced by these defaults.
"""

from dataclasses import dataclass


def validate_shape(batch_size, seq_len):
    if batch_size != 1 or seq_len not in (1, 2, 3, 4):
        raise ValueError(
            f"DSV4-Pro initial validation supports batch_size=1 and seq_len=1..4 "
            f"(decode through MTP3), got batch_size={batch_size}, seq_len={seq_len}"
        )


@dataclass(frozen=True)
class Dsv4Config:
    hidden: int = 7168
    intermediate: int = 3072
    experts: int = 384
    top_k: int = 6
    shared_experts: int = 1
    route_scale: float = 2.5
    swiglu_limit: float = 10.0
    hash_layers: int = 3
    heads: int = 128
    head_dim: int = 512
    rope_dim: int = 64
    q_lora: int = 1536
    o_lora: int = 1024
    o_groups: int = 16
    window: int = 128
    index_heads: int = 64
    index_dim: int = 128
    index_topk: int = 1024
    hc_mult: int = 4
    sinkhorn_iters: int = 20
    norm_eps: float = 1e-6
    hc_eps: float = 1e-6
    compress_ratios: tuple[int, ...] = ()
    scoring_func: str = "sqrtsoftplus"

    @classmethod
    def from_atom_args(cls, args):
        names = {
            "hidden": "dim",
            "intermediate": "moe_inter_dim",
            "experts": "n_routed_experts",
            "top_k": "n_activated_experts",
            "shared_experts": "n_shared_experts",
            "hash_layers": "n_hash_layers",
            "heads": "n_heads",
            "rope_dim": "rope_head_dim",
            "q_lora": "q_lora_rank",
            "o_lora": "o_lora_rank",
            "window": "window_size",
            "index_heads": "index_n_heads",
            "index_dim": "index_head_dim",
            "sinkhorn_iters": "hc_sinkhorn_iters",
            "scoring_func": "score_func",
        }
        values = {field: getattr(args, names.get(field, field)) for field in cls.__dataclass_fields__}
        values["compress_ratios"] = tuple(values["compress_ratios"])
        return cls(**values)

    def validate_moe(self, samples, tp):
        if not 1 <= samples <= 8:
            raise ValueError("DSV4 decode supports every batch size from 1 through 8")
        if tp not in (1, 2, 4, 8):
            raise ValueError("DSV4 resident TP supports 1, 2, 4 or 8 local GPUs")
        if self.hidden % 128 or self.intermediate % (128 * tp):
            raise ValueError("hidden and the unpadded local intermediate must be multiples of 128")
        if self.experts % 16 or not 1 <= self.top_k <= 7:
            raise ValueError("experts must be a multiple of 16 and top_k must fit seven routed waves")
        if self.shared_experts != 1 or self.scoring_func != "sqrtsoftplus":
            raise ValueError("DSV4 requires one shared expert and sqrtsoftplus routing")
        if self.route_scale <= 0 or self.swiglu_limit < 0:
            raise ValueError("invalid route scale or SwiGLU clamp")

    def validate_pro(self):
        """Reject other model geometries at the public Pro-only entry point."""
        baseline = Dsv4Config()
        for name in self.__dataclass_fields__:
            if name != "compress_ratios" and getattr(self, name) != getattr(baseline, name):
                raise ValueError(f"DSV4-Pro requires {name}={getattr(baseline, name)}, got {getattr(self, name)}")

    def attention_kind(self, layer):
        ratio = self.compress_ratios[layer % len(self.compress_ratios)] if self.compress_ratios else 0
        return "swa" if ratio == 0 else "csa" if ratio == 4 else "hca"
