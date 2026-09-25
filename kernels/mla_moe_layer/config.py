# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Model shard dimensions and supported MoE arithmetic modes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MoeMode(str, Enum):
    """Public arithmetic modes for the expert up/gate and down projections."""

    W8A8 = "w8a8"
    W8A16 = "w8a16"
    A16W4 = "a16w4"
    A8W4 = "a8w4"


class ExpertActivation(str, Enum):
    """Activation representation consumed by both expert projections."""

    FP8_BLOCK128 = "fp8_block128"
    MXFP8_BLOCK32 = "mxfp8_block32"
    BF16 = "bf16"


class ExpertWeight(str, Enum):
    """Packed expert-weight representation."""

    FP8_BLOCK128 = "fp8_block128"
    MXFP4_BLOCK32 = "mxfp4_block32"


class AttentionWeight(str, Enum):
    """Packed attention-weight representation."""

    FP8_BLOCK128 = "fp8_block128"
    BF16 = "bf16"


@dataclass(frozen=True)
class MoeFormat:
    activation: ExpertActivation
    weight: ExpertWeight

    @property
    def activation_group(self) -> int | None:
        if self.activation is ExpertActivation.FP8_BLOCK128:
            return 128
        if self.activation is ExpertActivation.MXFP8_BLOCK32:
            return 32
        return None


MOE_FORMATS = {
    MoeMode.W8A8: MoeFormat(ExpertActivation.FP8_BLOCK128, ExpertWeight.FP8_BLOCK128),
    MoeMode.W8A16: MoeFormat(ExpertActivation.BF16, ExpertWeight.FP8_BLOCK128),
    MoeMode.A16W4: MoeFormat(ExpertActivation.BF16, ExpertWeight.MXFP4_BLOCK32),
    MoeMode.A8W4: MoeFormat(ExpertActivation.MXFP8_BLOCK32, ExpertWeight.MXFP4_BLOCK32),
}


def as_moe_mode(value: MoeMode | str) -> MoeMode:
    """Normalize a public mode argument and report supported values clearly."""

    if isinstance(value, MoeMode):
        return value
    try:
        return MoeMode(value)
    except ValueError as error:
        choices = ", ".join(mode.value for mode in MoeMode)
        raise ValueError(f"unsupported MoE mode {value!r}; expected one of: {choices}") from error


def moe_format(value: MoeMode | str) -> MoeFormat:
    """Return the independent activation and weight formats for a public mode."""

    return MOE_FORMATS[as_moe_mode(value)]


@dataclass(frozen=True)
class LayerConfig:
    """Compile-time geometry for one tensor-parallel decode shard."""

    name: str
    hidden: int
    q_lora: int
    kv_lora: int
    pe_dim: int
    nope_dim: int
    v_dim: int
    n_experts: int
    top_k: int
    inter: int
    route_scale: float
    local_heads: int
    attention_weight: AttentionWeight = AttentionWeight.FP8_BLOCK128
    attention_output_gate: bool = False
    attention_input_norm: bool = True
    attention_residual: bool = True

    @property
    def qkv_a_rows(self) -> int:
        rows = self.q_lora + self.kv_lora + self.pe_dim
        if self.attention_output_gate:
            rows += self.local_heads * self.v_dim
        return rows

    @property
    def moe_slots(self) -> int:
        return 1 + self.top_k

    @property
    def shared_expert(self) -> int:
        return self.n_experts

    @property
    def softmax_scale(self) -> float:
        return (self.nope_dim + self.pe_dim) ** -0.5


GLM5_CONFIG = LayerConfig(
    name="glm5",
    hidden=6144,
    q_lora=2048,
    kv_lora=512,
    pe_dim=64,
    nope_dim=192,
    v_dim=256,
    n_experts=256,
    top_k=8,
    inter=256,
    route_scale=2.5,
    local_heads=8,
)

KIMI_K3_CONFIG = LayerConfig(
    name="kimi_k3",
    hidden=7168,
    q_lora=1536,
    kv_lora=512,
    pe_dim=64,
    nope_dim=128,
    v_dim=128,
    n_experts=896,
    top_k=16,
    inter=384,
    route_scale=1.0,
    local_heads=12,
    attention_weight=AttentionWeight.BF16,
    attention_output_gate=True,
    attention_input_norm=False,
    attention_residual=False,
)

MODEL_CONFIGS = {config.name: config for config in (GLM5_CONFIG, KIMI_K3_CONFIG)}


def as_layer_config(value: LayerConfig | str) -> LayerConfig:
    """Normalize a public model-profile argument."""

    if isinstance(value, LayerConfig):
        return value
    try:
        return MODEL_CONFIGS[value]
    except KeyError as error:
        choices = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"unsupported model profile {value!r}; expected one of: {choices}") from error


HIDDEN = GLM5_CONFIG.hidden
Q_LORA = GLM5_CONFIG.q_lora
KV_LORA = GLM5_CONFIG.kv_lora
PE_DIM = GLM5_CONFIG.pe_dim
NOPE_DIM = GLM5_CONFIG.nope_dim
V_DIM = GLM5_CONFIG.v_dim
QKV_A_ROWS = GLM5_CONFIG.qkv_a_rows
N_EXPERTS = GLM5_CONFIG.n_experts
TOP_K = GLM5_CONFIG.top_k
MOE_SLOTS = GLM5_CONFIG.moe_slots
SHARED_EXPERT = GLM5_CONFIG.shared_expert
INTER = GLM5_CONFIG.inter
ROUTE_SCALE = GLM5_CONFIG.route_scale
EPS = 1e-5
SCALE_BM = 128
FP8_MAX = 448.0
SOFTMAX_SCALE = GLM5_CONFIG.softmax_scale

SUPPORTED_SAMPLES = (1, 2, 4, 8)
SUPPORTED_PEERS = (1, 2, 4, 8)
LOCAL_HEADS = GLM5_CONFIG.local_heads
MAX_LAYERS_PER_STEP = 128


def validate_shard(
    samples: int,
    heads: int,
    rank: int,
    npes: int,
    topk: int,
    model_config: LayerConfig | str = GLM5_CONFIG,
) -> None:
    """Validate a fixed model shard contract before allocating GPU buffers."""

    config = as_layer_config(model_config)

    if samples not in SUPPORTED_SAMPLES:
        raise ValueError(f"samples must be one of {SUPPORTED_SAMPLES}, got {samples}")
    if heads != config.local_heads:
        raise ValueError(f"{config.name} requires {config.local_heads} local heads, got {heads}")
    if npes not in SUPPORTED_PEERS:
        raise ValueError(f"npes must be one of {SUPPORTED_PEERS}, got {npes}")
    if not 0 <= rank < npes:
        raise ValueError(f"rank must be in [0, {npes}), got {rank}")
    if topk <= 0 or topk % 64:
        raise ValueError(f"topk must be a positive multiple of 64, got {topk}")
