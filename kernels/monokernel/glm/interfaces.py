# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tensor contracts for the inlined GLM ops.

Names follow AITER's normalization, MLA and fused-MoE contracts. Mailboxes are
lazy views of existing tagged scratch tensors, not allocations or device ops.
Inputs can alias outputs across phases; completion is enforced by existing
mailbox epochs and barriers. These are compile-time descriptors, not a torch ABI.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Mailbox:
    name: str


@dataclass(frozen=True)
class LinearWeight:
    weight: object
    scale: object = None


@dataclass(frozen=True)
class Rope:
    cos: object
    sin: object


@dataclass(frozen=True)
class KvCache:
    kv_buffer: object
    rope_buffer: object


@dataclass(frozen=True)
class MlaQuery:
    latent: Mailbox
    rope: Mailbox


@dataclass(frozen=True)
class IndexQuery:
    kv_buffer: object
    q: Mailbox
    q_quant: Mailbox
    weights: Mailbox


@dataclass(frozen=True)
class IndexWeights:
    params: object


@dataclass(frozen=True)
class PreattnWeights:
    input_norm: object
    query_norm: object
    kv_norm: object
    qkv_proj: LinearWeight
    q_proj: LinearWeight
    q_absorb: LinearWeight
    index: IndexWeights


@dataclass(frozen=True)
class PreattnInput:
    hidden_states: object
    weights: PreattnWeights
    rope: Rope


@dataclass(frozen=True)
class PreattnOutput:
    q: MlaQuery
    index: IndexQuery
    cache: KvCache


@dataclass(frozen=True)
class IndexerInput:
    query: IndexQuery
    rope: Rope
    weights: IndexWeights


@dataclass(frozen=True)
class IndexerOutput:
    kv_indices: Mailbox
    ready: Mailbox


@dataclass(frozen=True)
class AttentionWeights:
    v_proj: LinearWeight
    out_proj: LinearWeight


@dataclass(frozen=True)
class AttentionInput:
    q: MlaQuery
    cache: KvCache
    kv_indices: object
    indices_ready: Mailbox
    residual: object
    weights: AttentionWeights


@dataclass(frozen=True)
class AttentionOutput:
    o: Mailbox
    hidden_states: Mailbox


@dataclass(frozen=True)
class RouterInput:
    hidden_states: Mailbox
    norm_weight: object
    gate: object


@dataclass(frozen=True)
class QuantizedActivation:
    data: Mailbox
    scale: Mailbox
    dequant: Mailbox


@dataclass(frozen=True)
class RouterOutput:
    gating_output: Mailbox
    hidden_states: QuantizedActivation


@dataclass(frozen=True)
class MoeStage1Input:
    hidden_states: Mailbox
    gating_output: Mailbox
    activation: QuantizedActivation
    e_score_correction_bias: object
    norm_weight: object
    w1: LinearWeight


@dataclass(frozen=True)
class MoeStage1Output:
    intermediate: Mailbox
    topk_ids: Mailbox
    topk_weights: Mailbox


@dataclass(frozen=True)
class MoeStage2Input:
    hidden_states: Mailbox
    topk_ids: Mailbox
    topk_weights: Mailbox
    gating_output: Mailbox
    e_score_correction_bias: object
    residual: Mailbox
    w2: LinearWeight


@dataclass(frozen=True)
class MoeOutput:
    out: object
    all_out: object
