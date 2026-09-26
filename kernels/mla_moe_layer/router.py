# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Reusable low-token sigmoid router selection kernels."""

from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.mla_moe_layer.kernel_common import exp, rcp, rsrc

WAVE_SIZE = 64


@functools.cache
def build_sigmoid_topk_router(num_experts: int, topk: int, samples: int):
    """Build BF16-logit sigmoid + correction-bias top-k for a fixed small batch."""

    if num_experts <= 0 or num_experts % WAVE_SIZE:
        raise ValueError(f"num_experts must be a positive multiple of {WAVE_SIZE}, got {num_experts}")
    if not 0 < topk <= 32:
        raise ValueError(f"topk must be in [1, 32], got {topk}")
    if samples not in {1, 2, 4, 8}:
        raise ValueError(f"samples must be one of {{1, 2, 4, 8}}, got {samples}")

    values_per_lane = num_experts // WAVE_SIZE
    threads = samples * WAVE_SIZE

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def sigmoid_topk_router(
        logits: Int64,
        correction_bias: Int64,
        scores_out: Int64,
        ids_out: Int64,
        weights_out: Int64,
    ):
        tid = fx.thread_idx.x
        lane = tid % WAVE_SIZE
        sample = tid // WAVE_SIZE
        logits_rsrc = rsrc(logits)
        bias_rsrc = rsrc(correction_bias)
        scores_rsrc = rsrc(scores_out)
        ids_rsrc = rsrc(ids_out)
        weights_rsrc = rsrc(weights_out)

        raw_scores = []
        corrected_scores = []
        expert_ids = []
        for i in range_constexpr(values_per_lane):
            expert = lane + i * WAVE_SIZE
            offset = sample * num_experts + expert
            logit = fx.Float32(fx.BFloat16(bo.buffer_load(logits_rsrc, offset, vec_width=1, dtype=T.bf16)))
            score = rcp(fx.Float32(1.0) + exp(-logit))
            bias = fx.Float32(fx.BFloat16(bo.buffer_load(bias_rsrc, expert, vec_width=1, dtype=T.bf16)))
            bo.buffer_store(score, scores_rsrc, offset)
            raw_scores.append(score)
            corrected_scores.append(score + bias)
            expert_ids.append(fx.Int32(expert))

        selected_scores = []
        selected_ids = []
        selected_sum = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))
        for _ in range_constexpr(topk):
            best_score = corrected_scores[0]
            best_raw = raw_scores[0]
            best_id = expert_ids[0]
            for i in range_constexpr(1, values_per_lane):
                candidate_score = corrected_scores[i]
                candidate_id = expert_ids[i]
                take = (candidate_score > best_score) | (
                    (ArithValue(candidate_score) == ArithValue(best_score)) & (candidate_id < best_id)
                )
                best_score = take.select(candidate_score, best_score)
                best_raw = take.select(raw_scores[i], best_raw)
                best_id = take.select(candidate_id, best_id)

            for offset in (32, 16, 8, 4, 2, 1):
                peer_score = best_score.shuffle_xor(fx.Int32(offset), WAVE_SIZE)
                peer_raw = best_raw.shuffle_xor(fx.Int32(offset), WAVE_SIZE)
                peer_id = best_id.shuffle_xor(fx.Int32(offset), WAVE_SIZE)
                take = (peer_score > best_score) | (
                    (ArithValue(peer_score) == ArithValue(best_score)) & (peer_id < best_id)
                )
                best_score = take.select(peer_score, best_score)
                best_raw = take.select(peer_raw, best_raw)
                best_id = take.select(peer_id, best_id)

            selected_scores.append(best_raw)
            selected_ids.append(best_id)
            selected_sum = selected_sum + best_raw
            for i in range_constexpr(values_per_lane):
                is_winner = expert_ids[i] == best_id
                corrected_scores[i] = is_winner.select(neg_inf, corrected_scores[i])

        if lane == 0:
            inv_sum = rcp(selected_sum)
            for k in range_constexpr(topk):
                out_offset = sample * topk + k
                bo.buffer_store(selected_ids[k], ids_rsrc, out_offset)
                bo.buffer_store(selected_scores[k] * inv_sum, weights_rsrc, out_offset)

    @flyc.jit
    def launch(
        logits: Int64,
        correction_bias: Int64,
        scores_out: Int64,
        ids_out: Int64,
        weights_out: Int64,
        stream: Stream = Stream(None),
    ):
        sigmoid_topk_router(
            logits,
            correction_bias,
            scores_out,
            ids_out,
            weights_out,
            value_attrs={"rocdl.flat_work_group_size": f"{threads},{threads}"},
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)

    launch.func.__name__ = f"sigmoid_topk_router_e{num_experts}_k{topk}_s{samples}"
    return launch


class SigmoidTopkRouter:
    """Torch tensor adapter for the fixed-shape low-token router kernel."""

    def __init__(self, num_experts: int, topk: int, samples: int) -> None:
        self.launch = build_sigmoid_topk_router(num_experts, topk, samples)
        self.num_experts = num_experts
        self.topk = topk
        self.samples = samples

    def __call__(
        self,
        logits: torch.Tensor,
        correction_bias: torch.Tensor,
        scores_out: torch.Tensor,
        ids_out: torch.Tensor,
        weights_out: torch.Tensor,
    ) -> None:
        if logits.shape != (self.samples, self.num_experts) or logits.dtype != torch.bfloat16:
            raise ValueError("router logits must be contiguous BF16 [samples, experts]")
        if correction_bias.shape != (self.num_experts,) or correction_bias.dtype != torch.bfloat16:
            raise ValueError("router correction bias must be BF16 [experts]")
        if scores_out.shape != logits.shape or scores_out.dtype != torch.float32:
            raise ValueError("router scores output must be FP32 [samples, experts]")
        if ids_out.shape != (self.samples, self.topk) or ids_out.dtype != torch.int32:
            raise ValueError("router ids output must be int32 [samples, topk]")
        if weights_out.shape != ids_out.shape or weights_out.dtype != torch.float32:
            raise ValueError("router weights output must be FP32 [samples, topk]")
        if any(not tensor.is_contiguous() for tensor in (logits, correction_bias, scores_out, ids_out, weights_out)):
            raise ValueError("router inputs and outputs must be contiguous")
        self.launch(
            logits.data_ptr(),
            correction_bias.data_ptr(),
            scores_out.data_ptr(),
            ids_out.data_ptr(),
            weights_out.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
