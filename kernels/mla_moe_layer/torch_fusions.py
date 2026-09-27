# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Reusable Torch-side fusions for model-composed MLA + MoE layers."""

from __future__ import annotations

import statistics
from contextlib import contextmanager

import torch

from kernels.mla_moe_layer.config import EPS


def rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply RMSNorm in FP32 and return BF16 activations."""

    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS) * weight.float()).to(torch.bfloat16)


def situ(x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    """Apply the bounded SiTU gate/up activation and return BF16."""

    gate, up = x.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(torch.bfloat16)


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def compiled_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Graph-capturable compiled form of :func:`rmsnorm`."""

    return rmsnorm(x, weight)


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def compiled_rmsnorm_out(x: torch.Tensor, weight: torch.Tensor, output: torch.Tensor) -> None:
    """Write RMSNorm directly to a caller-owned graph-stable buffer."""

    output.copy_(rmsnorm(x, weight))


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def compiled_attn_res_no_delta(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Fuse AttnRes source mixing and its output RMSNorm."""

    sources = torch.cat((blocks, prefix[:, None]), dim=1)
    sf = sources.float()
    normalized = sf * torch.rsqrt(sf.square().mean(-1, keepdim=True) + EPS)
    logits = (normalized * norm_weight.float() * qk_weight.float()).sum(-1)
    mixed = (torch.softmax(logits, dim=-1)[..., None] * sf).sum(1)
    return compiled_rmsnorm(mixed, output_norm_weight)


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def compiled_attn_res_with_delta(
    prefix: torch.Tensor,
    delta: torch.Tensor,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse prefix update, AttnRes source mixing, and output RMSNorm."""

    updated = (prefix.float() + delta.float()).to(torch.bfloat16)
    mixed = compiled_attn_res_no_delta(updated, blocks, norm_weight, qk_weight, output_norm_weight)
    return mixed, updated


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def compiled_shared_experts(
    hidden_states: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_gate_out: torch.Tensor,
    mid_out: torch.Tensor,
    partial_out: torch.Tensor,
    beta: float,
    linear_beta: float,
) -> None:
    """Fuse a BF16 shared-expert up/gate, SiTU, and down path."""

    up_gate = hidden_states @ up_gate_weight.t()
    mid = situ(up_gate, beta, linear_beta)
    partial = mid @ down_weight.t()
    up_gate_out.copy_(up_gate)
    mid_out.copy_(mid)
    partial_out.copy_(partial)


class CudaStageProfiler:
    """Optional median CUDA-event profiler for composed layer stages."""

    def __init__(self) -> None:
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] | None = None

    @contextmanager
    def stage(self, name: str):
        if self._events is None:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self._events.setdefault(name, []).append((start, end))

    def start(self) -> None:
        if self._events is not None:
            raise RuntimeError("stage profiling is already active")
        self._events = {}

    def finish(self) -> dict[str, float]:
        if self._events is None:
            raise RuntimeError("stage profiling is not active")
        torch.cuda.synchronize()
        result = {
            name: statistics.median(start.elapsed_time(end) * 1000.0 for start, end in events)
            for name, events in self._events.items()
        }
        self._events = None
        return result
