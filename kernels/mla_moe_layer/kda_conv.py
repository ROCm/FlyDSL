# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Slot-indexed Kimi-K3 causal-convolution decode update."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.typing import Int32, Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.common.act import sigmoid_batch
from kernels.mla_moe_layer.kernel_common import rsrc

_HEADS = 12
_HEAD_DIM = 128
_PROJECTION = _HEADS * _HEAD_DIM
_CHANNELS = 3 * _PROJECTION
_STATE_LENGTH = 3
_KERNEL_WIDTH = 4
_THREADS = 256


@functools.cache
def build_kimi_k3_kda_causal_conv(samples: int):
    """Build the fixed-width Kimi-K3 decode convolution."""

    if samples not in {1, 2, 4, 8}:
        raise ValueError(f"samples must be one of {{1, 2, 4, 8}}, got {samples}")
    blocks = (samples * _CHANNELS + _THREADS - 1) // _THREADS

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def kimi_k3_kda_causal_conv_kernel(
        mixed_qkv: Int64,
        conv_weight: Int64,
        state_indices: Int64,
        conv_state: Int64,
        query: Int64,
        key: Int64,
        value: Int64,
        input_stride: Int32,
    ):
        linear = gpu.block_idx.x * _THREADS + gpu.thread_idx.x
        if linear < samples * _CHANNELS:
            sample = linear // _CHANNELS
            channel = linear % _CHANNELS
            indices_rsrc = rsrc(state_indices)
            slot = fx.Int32(bo.buffer_load(indices_rsrc, sample, vec_width=1, dtype=T.i32))

            def update():
                state_rsrc = rsrc(conv_state + fx.Int64(slot) * fx.Int64(_CHANNELS * _STATE_LENGTH * 2))
                input_rsrc = rsrc(mixed_qkv)
                weight_rsrc = rsrc(conv_weight)
                q_rsrc = rsrc(query)
                k_rsrc = rsrc(key)
                v_rsrc = rsrc(value)

                state_base = channel * _STATE_LENGTH
                state0 = fx.BFloat16(bo.buffer_load(state_rsrc, state_base, vec_width=1, dtype=T.bf16))
                state1 = fx.BFloat16(bo.buffer_load(state_rsrc, state_base + 1, vec_width=1, dtype=T.bf16))
                state2 = fx.BFloat16(bo.buffer_load(state_rsrc, state_base + 2, vec_width=1, dtype=T.bf16))
                current = fx.BFloat16(
                    bo.buffer_load(
                        input_rsrc,
                        sample * input_stride + channel,
                        vec_width=1,
                        dtype=T.bf16,
                    )
                )
                weights = fx.Vector(
                    bo.buffer_load(
                        weight_rsrc,
                        channel * _KERNEL_WIDTH,
                        vec_width=_KERNEL_WIDTH,
                        dtype=T.bf16,
                    )
                ).to(fx.Float32)
                inputs = fx.Vector.from_elements(
                    [
                        fx.Float32(state0),
                        fx.Float32(state1),
                        fx.Float32(state2),
                        fx.Float32(current),
                    ],
                    fx.Float32,
                )
                convolution = (inputs * weights).reduce(fx.ReductionOp.ADD)
                activated = convolution * sigmoid_batch([convolution])[0]

                bo.buffer_store(state1, state_rsrc, state_base)
                bo.buffer_store(state2, state_rsrc, state_base + 1)
                bo.buffer_store(current, state_rsrc, state_base + 2)

                output_channel = channel % _PROJECTION
                output_offset = sample * _PROJECTION + output_channel
                output_value = activated.to(fx.BFloat16)
                if channel < _PROJECTION:
                    bo.buffer_store(output_value, q_rsrc, output_offset)
                elif channel < 2 * _PROJECTION:
                    bo.buffer_store(output_value, k_rsrc, output_offset)
                else:
                    bo.buffer_store(output_value, v_rsrc, output_offset)

            def zero_output():
                q_rsrc = rsrc(query)
                k_rsrc = rsrc(key)
                v_rsrc = rsrc(value)
                output_channel = channel % _PROJECTION
                output_offset = sample * _PROJECTION + output_channel
                zero = fx.Float32(0.0).to(fx.BFloat16)
                if channel < _PROJECTION:
                    bo.buffer_store(zero, q_rsrc, output_offset)
                elif channel < 2 * _PROJECTION:
                    bo.buffer_store(zero, k_rsrc, output_offset)
                else:
                    bo.buffer_store(zero, v_rsrc, output_offset)

            if slot >= 0:
                update()
            else:
                zero_output()

    @flyc.jit
    def launch(
        mixed_qkv: Int64,
        conv_weight: Int64,
        state_indices: Int64,
        conv_state: Int64,
        query: Int64,
        key: Int64,
        value: Int64,
        input_stride: Int32,
        stream: Stream,
    ):
        kimi_k3_kda_causal_conv_kernel(
            mixed_qkv,
            conv_weight,
            state_indices,
            conv_state,
            query,
            key,
            value,
            input_stride,
        ).launch(grid=(blocks, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    return launch


class KimiK3KdaCausalConv:
    """Graph-safe tensor adapter for Kimi-K3's q/k/v short convolution."""

    def __init__(self, samples: int) -> None:
        if samples not in {1, 2, 4, 8}:
            raise ValueError(f"samples must be one of {{1, 2, 4, 8}}, got {samples}")
        self.samples = samples
        self.launch = build_kimi_k3_kda_causal_conv(samples)

    def __call__(
        self,
        mixed_qkv: torch.Tensor,
        conv_weight: torch.Tensor,
        state_indices: torch.Tensor,
        conv_state: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            mixed_qkv.shape != (self.samples, _CHANNELS)
            or mixed_qkv.dtype != torch.bfloat16
            or mixed_qkv.stride(1) != 1
        ):
            raise ValueError(f"mixed_qkv must be a feature-contiguous BF16 [{self.samples}, {_CHANNELS}] view")
        if (
            conv_weight.shape != (_CHANNELS, _KERNEL_WIDTH)
            or conv_weight.dtype != torch.bfloat16
            or not conv_weight.is_contiguous()
        ):
            raise ValueError(f"conv_weight must be contiguous BF16 [{_CHANNELS}, {_KERNEL_WIDTH}]")
        if state_indices.shape != (self.samples,) or state_indices.dtype != torch.int32:
            raise ValueError(f"state_indices must be int32 [{self.samples}]")
        if (
            conv_state.ndim != 3
            or conv_state.shape[1:] != (_CHANNELS, _STATE_LENGTH)
            or conv_state.dtype != torch.bfloat16
            or not conv_state.is_contiguous()
        ):
            raise ValueError(f"conv_state must be contiguous BF16 [slots, {_CHANNELS}, {_STATE_LENGTH}]")
        expected_output = (self.samples, 1, _HEADS, _HEAD_DIM)
        outputs = (query, key, value)
        if any(
            tensor.shape != expected_output or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous()
            for tensor in outputs
        ):
            raise ValueError(f"query, key, and value must be contiguous BF16 {list(expected_output)}")
        tensors = (mixed_qkv, conv_weight, state_indices, conv_state, *outputs)
        if any(tensor.device != mixed_qkv.device for tensor in tensors):
            raise ValueError("all KDA causal-convolution tensors must be on the same device")

        self.launch(
            mixed_qkv.data_ptr(),
            conv_weight.data_ptr(),
            state_indices.data_ptr(),
            conv_state.data_ptr(),
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            mixed_qkv.stride(0),
            stream=torch.cuda.current_stream(),
        )
        return query, key, value
