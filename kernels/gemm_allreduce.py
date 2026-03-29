#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused GEMM + all-reduce wrapper.

This module provides a practical fused operator entrypoint based on the current
FlyDSL GEMM and custom all-reduce implementations:

1) GEMM: `kernels.preshuffle_gemm.compile_preshuffle_gemm_a8`
2) All-reduce: a user-supplied callback (or `FlyDSLAllreduce.custom_all_reduce`)

The execution is stream-consistent (single stream) and keeps GEMM output on
device to avoid extra host round-trips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8


AllReduceFn = Callable[[torch.Tensor, Optional[torch.Tensor], Optional[int]], torch.Tensor]


@dataclass(frozen=True)
class GemmAllReduceConfig:
    """Compile-time config for GEMM + all-reduce fused op."""

    K: int
    tile_m: int
    tile_n: int
    tile_k: int
    in_dtype: str = "bf16"
    out_dtype: str = "bf16"
    lds_stage: int = 2
    use_cshuffle_epilog: bool = False
    waves_per_eu: Optional[int] = None
    use_async_copy: bool = False


def _identity_allreduce(inp: torch.Tensor, out: Optional[torch.Tensor], stream_ptr: Optional[int]) -> torch.Tensor:
    _ = stream_ptr
    if out is None:
        return inp
    out.copy_(inp)
    return out


def make_flydsl_allreduce_fn(
    ar_obj,
    *,
    open_fp8_quant: bool = False,
    validate: bool = True,
) -> AllReduceFn:
    """Adapt `FlyDSLAllreduce.custom_all_reduce` to `AllReduceFn` signature."""

    def _fn(inp: torch.Tensor, out: Optional[torch.Tensor], stream_ptr: Optional[int]) -> torch.Tensor:
        inp_flat = inp.contiguous().view(-1)
        out_flat = None if out is None else out.contiguous().view(-1)
        out_ret = ar_obj.custom_all_reduce(
            inp_flat,
            out=out_flat,
            open_fp8_quant=open_fp8_quant,
            validate=validate,
            stream_ptr=stream_ptr,
        )
        if out is not None:
            return out
        return out_ret.view_as(inp)

    return _fn


class GemmAllReduceOp:
    """Callable fused operator for GEMM followed by all-reduce."""

    def __init__(self, config: GemmAllReduceConfig, *, allreduce_fn: Optional[AllReduceFn] = None):
        self.config = config
        self._allreduce_fn = allreduce_fn or _identity_allreduce
        # M/N are runtime values in this launcher, keep compile-time placeholders as 0.
        self._gemm_launch = compile_preshuffle_gemm_a8(
            M=0,
            N=0,
            K=int(config.K),
            tile_m=int(config.tile_m),
            tile_n=int(config.tile_n),
            tile_k=int(config.tile_k),
            in_dtype=str(config.in_dtype),
            out_dtype=str(config.out_dtype),
            lds_stage=int(config.lds_stage),
            use_cshuffle_epilog=bool(config.use_cshuffle_epilog),
            waves_per_eu=config.waves_per_eu,
            use_async_copy=bool(config.use_async_copy),
        )

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Run fused GEMM + all-reduce.

        Args:
            a: [M, K]
            b: preshuffled-B tensor expected by `compile_preshuffle_gemm_a8`
               (logical shape corresponds to [N, K], kernel computes A @ B^T)
            scale_a/scale_b: scaling tensors for quantized input types; can be None
                for bf16/fp16 paths.
            out: optional final output buffer [M, N]
            stream: optional CUDA stream; defaults to current stream.
        """

        if a.dim() != 2 or b.dim() != 2:
            raise ValueError("a and b must be rank-2 tensors")
        m, k_a = int(a.shape[0]), int(a.shape[1])
        n, k_b = int(b.shape[0]), int(b.shape[1])
        if k_a != int(self.config.K) or k_b != int(self.config.K):
            raise ValueError(f"K mismatch: expected {self.config.K}, got a.K={k_a}, b.K={k_b}")
        if a.device.type != "cuda" or b.device.type != "cuda":
            raise ValueError("a and b must be CUDA tensors")
        if a.device != b.device:
            raise ValueError("a and b must be on the same device")

        if stream is None:
            stream = torch.cuda.current_stream(device=a.device)

        gemm_out_dtype = torch.bfloat16 if self.config.out_dtype == "bf16" else torch.float16
        gemm_out = torch.empty((m, n), dtype=gemm_out_dtype, device=a.device)

        if scale_a is None:
            scale_a = torch.empty((0,), dtype=torch.float32, device=a.device)
        if scale_b is None:
            scale_b = torch.empty((0,), dtype=torch.float32, device=a.device)

        # Keep compatibility with fp8 path in preshuffle tests.
        a_in = a.view(torch.int8) if "float8" in str(a.dtype) else a
        b_in = b.view(torch.int8) if "float8" in str(b.dtype) else b

        self._gemm_launch(
            gemm_out.contiguous().view(-1),
            a_in.contiguous().view(-1),
            b_in.contiguous().view(-1),
            scale_a.contiguous().view(-1),
            scale_b.contiguous().view(-1),
            m,
            n,
            stream,
        )

        stream_ptr = int(stream.cuda_stream)
        return self._allreduce_fn(gemm_out, out, stream_ptr)


def build_gemm_allreduce_operator(
    *,
    config: GemmAllReduceConfig,
    allreduce_fn: Optional[AllReduceFn] = None,
) -> GemmAllReduceOp:
    """Build a fused GEMM + all-reduce operator."""

    return GemmAllReduceOp(config=config, allreduce_fn=allreduce_fn)


__all__ = [
    "AllReduceFn",
    "GemmAllReduceConfig",
    "GemmAllReduceOp",
    "build_gemm_allreduce_operator",
    "make_flydsl_allreduce_fn",
]

