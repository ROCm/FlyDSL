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


class Mxfp4WeightLayout(str, Enum):
    """Physical layout of packed MXFP4 expert values."""

    NATIVE = "native"
    ATOM = "atom"


class Mxfp4ScaleLayout(str, Enum):
    """Physical layout of per-row MXFP4 E8M0 scales."""

    NATIVE = "native"
    ATOM = "atom"


class RouterWeightLayout(str, Enum):
    """Physical layout of the BF16 router matrix."""

    NATIVE = "native"
    ATOM = "atom"


class KvCacheLayout(str, Enum):
    """Physical layout of the BF16 MLA KV cache."""

    SPLIT = "split"
    ATOM = "atom"


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


def _as_layout(value, enum_type, name):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(layout.value for layout in enum_type)
        raise ValueError(f"unsupported {name} {value!r}; expected one of: {choices}") from error


def as_mxfp4_weight_layout(value: Mxfp4WeightLayout | str) -> Mxfp4WeightLayout:
    return _as_layout(value, Mxfp4WeightLayout, "MXFP4 weight layout")


def as_mxfp4_scale_layout(value: Mxfp4ScaleLayout | str) -> Mxfp4ScaleLayout:
    return _as_layout(value, Mxfp4ScaleLayout, "MXFP4 scale layout")


def as_router_weight_layout(value: RouterWeightLayout | str) -> RouterWeightLayout:
    return _as_layout(value, RouterWeightLayout, "router weight layout")


def as_kv_cache_layout(value: KvCacheLayout | str) -> KvCacheLayout:
    return _as_layout(value, KvCacheLayout, "KV-cache layout")


def resolve_storage_layouts(
    moe_mode: MoeMode | str,
    mxfp4_weight_layout: Mxfp4WeightLayout | str | None = None,
    mxfp4_scale_layout: Mxfp4ScaleLayout | str | None = None,
    router_weight_layout: RouterWeightLayout | str | None = None,
    kv_cache_layout: KvCacheLayout | str | None = None,
) -> tuple[Mxfp4WeightLayout, Mxfp4ScaleLayout, RouterWeightLayout, KvCacheLayout]:
    """Resolve storage defaults for each MoE arithmetic mode."""

    is_mxfp4 = moe_format(moe_mode).weight is ExpertWeight.MXFP4_BLOCK32
    weight_default = Mxfp4WeightLayout.ATOM if is_mxfp4 else Mxfp4WeightLayout.NATIVE
    scale_default = Mxfp4ScaleLayout.ATOM if is_mxfp4 else Mxfp4ScaleLayout.NATIVE
    # ATOM keeps the unquantized router row-major, but direct row-major loads
    # regress the A16W4 S=8 mono-kernel. Keep the MFMA-native router packing by
    # default while allowing zero-copy ATOM router experiments explicitly.
    router_default = RouterWeightLayout.NATIVE
    cache_default = KvCacheLayout.ATOM if is_mxfp4 else KvCacheLayout.SPLIT
    return (
        weight_default if mxfp4_weight_layout is None else as_mxfp4_weight_layout(mxfp4_weight_layout),
        scale_default if mxfp4_scale_layout is None else as_mxfp4_scale_layout(mxfp4_scale_layout),
        router_default if router_weight_layout is None else as_router_weight_layout(router_weight_layout),
        cache_default if kv_cache_layout is None else as_kv_cache_layout(kv_cache_layout),
    )


HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
PE_DIM = 64
NOPE_DIM = 192
V_DIM = 256
QKV_A_ROWS = Q_LORA + KV_LORA + PE_DIM
N_EXPERTS = 256
TOP_K = 8
MOE_SLOTS = 1 + TOP_K
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


def validate_shard(samples: int, heads: int, rank: int, npes: int, topk: int) -> None:
    """Validate the fixed GLM-5 shard contract before allocating GPU buffers."""

    if samples not in SUPPORTED_SAMPLES:
        raise ValueError(f"samples must be one of {SUPPORTED_SAMPLES}, got {samples}")
    if heads != LOCAL_HEADS:
        raise ValueError(f"this kernel requires {LOCAL_HEADS} local heads, got {heads}")
    if npes not in SUPPORTED_PEERS:
        raise ValueError(f"npes must be one of {SUPPORTED_PEERS}, got {npes}")
    if not 0 <= rank < npes:
        raise ValueError(f"rank must be in [0, {npes}), got {rank}")
    if topk <= 0 or topk % 64:
        raise ValueError(f"topk must be a positive multiple of 64, got {topk}")
