# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Static GLM-5 shard dimensions and supported MoE arithmetic modes."""

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


HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
PE_DIM = 64
NOPE_DIM = 192
V_DIM = 256
QKV_A_ROWS = Q_LORA + KV_LORA + PE_DIM
N_EXPERTS = 256
EXPERT_TOP_K = 8
MOE_SLOTS = 1 + EXPERT_TOP_K
SHARED_EXPERT = N_EXPERTS
INTER = 256
ROUTE_SCALE = 2.5
EPS = 1e-5
SCALE_BM = 128
FP8_MAX = 448.0
SOFTMAX_SCALE = (NOPE_DIM + PE_DIM) ** -0.5

SUPPORTED_SAMPLES = (1, 2, 4, 8)
SUPPORTED_PEERS = (1, 2, 4, 8)
LOCAL_HEADS = 8
MAX_LAYERS_PER_STEP = 128


def validate_shard(samples: int, heads: int, rank: int, npes: int, sparse_attention_topk: int) -> None:
    """Validate the fixed GLM-5 shard contract before allocating GPU buffers."""

    if samples not in SUPPORTED_SAMPLES:
        raise ValueError(f"samples must be one of {SUPPORTED_SAMPLES}, got {samples}")
    if heads != LOCAL_HEADS:
        raise ValueError(f"this kernel requires {LOCAL_HEADS} local heads, got {heads}")
    if npes not in SUPPORTED_PEERS:
        raise ValueError(f"npes must be one of {SUPPORTED_PEERS}, got {npes}")
    if not 0 <= rank < npes:
        raise ValueError(f"rank must be in [0, {npes}), got {rank}")
    if sparse_attention_topk <= 0 or sparse_attention_topk % 64:
        raise ValueError("sparse_attention_topk must be a positive multiple of 64, " f"got {sparse_attention_topk}")
