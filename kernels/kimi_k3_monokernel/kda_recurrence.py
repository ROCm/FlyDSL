# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Kimi-K3 decode recurrence over slot-indexed FP32 state.

The kernel deliberately uses raw device pointers rather than tensor descriptors.
Besides matching the rest of the fused Kimi path, this keeps the generated kernel
ABI small enough for the gfx950 linker.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.common.act import sigmoid_batch
from kernels.monokernel.config import EPS
from kernels.monokernel.ops import exp, rsq, rsrc, xshfl

_HEADS = 12
_HEAD_DIM = 128
_THREADS = 256
_WAVE_SIZE = 64
_WAVES = _THREADS // _WAVE_SIZE
_K_LANES = 8
_V_LANES = _WAVE_SIZE // _K_LANES
_VALUES_PER_THREAD = 4
_K_TILE = _K_LANES * _VALUES_PER_THREAD
_K_ITERS = _HEAD_DIM // _K_TILE
_V_TILE = _WAVES * _V_LANES
_V_BLOCKS = 1
_V_PER_BLOCK = _HEAD_DIM // _V_BLOCKS
_V_ITERS = _V_PER_BLOCK // _V_TILE
_STATE_SLOT_BYTES = _HEADS * _HEAD_DIM * _HEAD_DIM * 4
_CONV_CHANNELS = 3 * _HEADS * _HEAD_DIM
_CONV_STATE_LENGTH = 3
_CONV_KERNEL_WIDTH = 4
_Q_SCALE = _HEAD_DIM**-0.5
_GATE_LOWER_BOUND = -5.0


def _load_bf16x4(resource, offset):
    return fx.Vector(bo.buffer_load(resource, offset, vec_width=_VALUES_PER_THREAD, dtype=T.bf16)).to(fx.Float32)


def _zeros4():
    return fx.Vector.filled(_VALUES_PER_THREAD, fx.Float32(0.0), fx.Float32)


def _subgroup_sum(value):
    """Reduce within one eight-lane K subgroup of an AMD wave."""

    for offset in (4, 2, 1):
        value = value + xshfl(value, offset)
    return value


@functools.cache
def build_kimi_k3_kda_recurrence(
    samples: int,
):
    """Build the fused Kimi-K3 convolution, recurrence, and norm kernel."""

    if not 1 <= samples <= 64:
        raise ValueError(f"samples must be in [1, 64], got {samples}")
    fuse_output_norm = True
    fuse_conv = True
    fuse_gate_projection = True

    @fx.struct
    class SharedStorage:
        query: fx.Array[fx.BFloat16, _HEAD_DIM, 16]
        key: fx.Array[fx.BFloat16, _HEAD_DIM, 16]
        value: fx.Array[fx.BFloat16, _HEAD_DIM, 16]
        gate: fx.Array[fx.BFloat16, _HEAD_DIM, 16]
        norm_sums: fx.Array[fx.Float32, _WAVES, 16]

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def kimi_k3_kda_recurrence_kernel(
        query: Int64,
        key: Int64,
        value: Int64,
        gate: Int64,
        f_a: Int64,
        f_b_weight: Int64,
        beta: Int64,
        dt_bias: Int64,
        a_log: Int64,
        mixed_qkv: Int64,
        conv_weight: Int64,
        conv_state: Int64,
        state_indices: Int64,
        state: Int64,
        output_gate: Int64,
        norm_weight: Int64,
        output: Int64,
        beta_stride: Int32,
        input_stride: Int32,
        output_gate_stride: Int32,
    ):
        block = gpu.block_idx.x
        sample_head = block // _V_BLOCKS
        sample = sample_head // _HEADS
        head = sample_head % _HEADS
        v_block = block % _V_BLOCKS
        tid = gpu.thread_idx.x
        wave = tid // _WAVE_SIZE
        lane = tid % _WAVE_SIZE
        k_lane = lane % _K_LANES
        v_lane = lane // _K_LANES

        query_rsrc = rsrc(query)
        key_rsrc = rsrc(key)
        value_rsrc = rsrc(value)
        gate_rsrc = rsrc(gate)
        f_a_rsrc = rsrc(f_a)
        f_b_weight_rsrc = rsrc(f_b_weight)
        beta_rsrc = rsrc(beta)
        dt_bias_rsrc = rsrc(dt_bias)
        a_log_rsrc = rsrc(a_log)
        mixed_qkv_rsrc = rsrc(mixed_qkv)
        conv_weight_rsrc = rsrc(conv_weight)
        indices_rsrc = rsrc(state_indices)
        output_gate_rsrc = rsrc(output_gate)
        norm_weight_rsrc = rsrc(norm_weight)
        output_rsrc = rsrc(output)
        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        shared_query = shared.query.ptr
        shared_key = shared.key.ptr
        shared_value = shared.value.ptr
        shared_gate = shared.gate.ptr
        norm_sums = shared.norm_sums.ptr

        slot = fx.Int32(bo.buffer_load(indices_rsrc, sample, vec_width=1, dtype=T.i32))

        def decode():
            # Shift the raw address in 64-bit space before making a buffer
            # resource. Slot pools can exceed the descriptor's i32 offset range.
            state_rsrc = rsrc(state + fx.Int64(slot) * fx.Int64(_STATE_SLOT_BYTES))
            vector_base = (sample * _HEADS + head) * _HEAD_DIM

            if const_expr(fuse_conv):
                conv_state_rsrc = rsrc(conv_state + fx.Int64(slot) * fx.Int64(_CONV_CHANNELS * _CONV_STATE_LENGTH * 2))

                def convolve(channel):
                    state_base = channel * _CONV_STATE_LENGTH
                    state0 = fx.BFloat16(
                        bo.buffer_load(
                            conv_state_rsrc,
                            state_base,
                            vec_width=1,
                            dtype=T.bf16,
                        )
                    )
                    state1 = fx.BFloat16(
                        bo.buffer_load(
                            conv_state_rsrc,
                            state_base + 1,
                            vec_width=1,
                            dtype=T.bf16,
                        )
                    )
                    state2 = fx.BFloat16(
                        bo.buffer_load(
                            conv_state_rsrc,
                            state_base + 2,
                            vec_width=1,
                            dtype=T.bf16,
                        )
                    )
                    current = fx.BFloat16(
                        bo.buffer_load(
                            mixed_qkv_rsrc,
                            sample * input_stride + channel,
                            vec_width=1,
                            dtype=T.bf16,
                        )
                    )
                    weights = fx.Vector(
                        bo.buffer_load(
                            conv_weight_rsrc,
                            channel * _CONV_KERNEL_WIDTH,
                            vec_width=_CONV_KERNEL_WIDTH,
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
                    bo.buffer_store(state1, conv_state_rsrc, state_base)
                    bo.buffer_store(state2, conv_state_rsrc, state_base + 1)
                    bo.buffer_store(current, conv_state_rsrc, state_base + 2)
                    return activated.to(fx.BFloat16)

                if tid < _HEAD_DIM:
                    channel_in_head = tid
                    channel = head * _HEAD_DIM + channel_in_head
                    fx.ptr_store(convolve(channel), shared_query + channel_in_head)
                    fx.ptr_store(
                        convolve(2 * _HEADS * _HEAD_DIM + channel),
                        shared_value + channel_in_head,
                    )
                else:
                    channel_in_head = tid - _HEAD_DIM
                    channel = _HEADS * _HEAD_DIM + head * _HEAD_DIM + channel_in_head
                    fx.ptr_store(convolve(channel), shared_key + channel_in_head)
                gpu.barrier()

            if const_expr(fuse_gate_projection):
                if tid < _HEAD_DIM:
                    gate_parts = _zeros4()
                    for feature_base in range_constexpr(0, _HEAD_DIM, _VALUES_PER_THREAD):
                        features = fx.Vector(
                            bo.buffer_load(
                                f_a_rsrc,
                                sample * input_stride + feature_base,
                                vec_width=_VALUES_PER_THREAD,
                                dtype=T.bf16,
                            )
                        ).to(fx.Float32)
                        weights = fx.Vector(
                            bo.buffer_load(
                                f_b_weight_rsrc,
                                (head * _HEAD_DIM + tid) * _HEAD_DIM + feature_base,
                                vec_width=_VALUES_PER_THREAD,
                                dtype=T.bf16,
                            )
                        ).to(fx.Float32)
                        gate_parts = fx.math.fma(features, weights, gate_parts)
                    gate_acc = gate_parts.reduce(fx.ReductionOp.ADD)
                    fx.ptr_store(gate_acc.to(fx.BFloat16), shared_gate + tid)
                gpu.barrier()

            exp_a_log = exp(fx.Float32(bo.buffer_load(a_log_rsrc, head, vec_width=1, dtype=T.f32)))
            beta_logit = fx.Float32(
                fx.BFloat16(
                    bo.buffer_load(
                        beta_rsrc,
                        sample * beta_stride + head,
                        vec_width=1,
                        dtype=T.bf16,
                    )
                )
            )
            beta_value = sigmoid_batch([beta_logit])[0]

            q_vecs = [None] * _K_ITERS
            k_vecs = [None] * _K_ITERS
            decay_vecs = [None] * _K_ITERS
            q_square = fx.Float32(0.0)
            k_square = fx.Float32(0.0)
            for k_iter in range_constexpr(_K_ITERS):
                k_base = k_lane * _VALUES_PER_THREAD + k_iter * _K_TILE
                if const_expr(fuse_conv):
                    q_vec = fx.Vector(
                        fx.ptr_load(
                            shared_query + k_base,
                            result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                        )
                    ).to(fx.Float32)
                    k_vec = fx.Vector(
                        fx.ptr_load(
                            shared_key + k_base,
                            result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                        )
                    ).to(fx.Float32)
                else:
                    q_vec = _load_bf16x4(query_rsrc, vector_base + k_base)
                    k_vec = _load_bf16x4(key_rsrc, vector_base + k_base)
                if const_expr(fuse_gate_projection):
                    gate_vec = fx.Vector(
                        fx.ptr_load(
                            shared_gate + k_base,
                            result_type=fx.Vector.make_type(_VALUES_PER_THREAD, fx.BFloat16),
                        )
                    ).to(fx.Float32)
                else:
                    gate_vec = _load_bf16x4(gate_rsrc, vector_base + k_base)
                dt_vec = _load_bf16x4(dt_bias_rsrc, head * _HEAD_DIM + k_base)

                q_vecs[k_iter] = q_vec
                k_vecs[k_iter] = k_vec
                q_square = q_square + (q_vec * q_vec).reduce(fx.ReductionOp.ADD)
                k_square = k_square + (k_vec * k_vec).reduce(fx.ReductionOp.ADD)

                gate_sigmoid = sigmoid_batch(
                    [
                        exp_a_log * (gate_vec[element] + dt_vec[element])
                        for element in range_constexpr(_VALUES_PER_THREAD)
                    ]
                )
                decay_vecs[k_iter] = fx.Vector.from_elements(
                    [
                        exp(fx.Float32(_GATE_LOWER_BOUND) * gate_sigmoid[element])
                        for element in range_constexpr(_VALUES_PER_THREAD)
                    ],
                    fx.Float32,
                )

            q_inverse_norm = rsq(_subgroup_sum(q_square) + fx.Float32(1.0e-6))
            k_inverse_norm = rsq(_subgroup_sum(k_square) + fx.Float32(1.0e-6))
            q_scale = fx.Vector.filled(
                _VALUES_PER_THREAD,
                q_inverse_norm * fx.Float32(_Q_SCALE),
                fx.Float32,
            )
            k_scale = fx.Vector.filled(_VALUES_PER_THREAD, k_inverse_norm, fx.Float32)
            for k_iter in range_constexpr(_K_ITERS):
                q_vecs[k_iter] = q_vecs[k_iter] * q_scale
                k_vecs[k_iter] = k_vecs[k_iter] * k_scale

            dot_kq_parts = _zeros4()
            for k_iter in range_constexpr(_K_ITERS):
                dot_kq_parts = fx.math.fma(k_vecs[k_iter], q_vecs[k_iter], dot_kq_parts)
            dot_kq = _subgroup_sum(dot_kq_parts.reduce(fx.ReductionOp.ADD))

            state_vecs = [None] * (_V_ITERS * _K_ITERS)
            results = [None] * _V_ITERS
            for v_iter in range_constexpr(_V_ITERS):
                v_index = v_block * _V_PER_BLOCK + wave * _V_LANES + v_lane + v_iter * _V_TILE
                for k_iter in range_constexpr(_K_ITERS):
                    k_base = k_lane * _VALUES_PER_THREAD + k_iter * _K_TILE
                    state_offset = (head * _HEAD_DIM + v_index) * _HEAD_DIM + k_base
                    state_vecs[v_iter * _K_ITERS + k_iter] = fx.Vector(
                        bo.buffer_load(
                            state_rsrc,
                            state_offset,
                            vec_width=_VALUES_PER_THREAD,
                            dtype=T.f32,
                        )
                    )

            for v_iter in range_constexpr(_V_ITERS):
                v_index = v_block * _V_PER_BLOCK + wave * _V_LANES + v_lane + v_iter * _V_TILE
                state_k_parts = _zeros4()
                state_q_parts = _zeros4()
                for k_iter in range_constexpr(_K_ITERS):
                    index = v_iter * _K_ITERS + k_iter
                    decayed = state_vecs[index] * decay_vecs[k_iter]
                    state_vecs[index] = decayed
                    state_k_parts = fx.math.fma(decayed, k_vecs[k_iter], state_k_parts)
                    state_q_parts = fx.math.fma(decayed, q_vecs[k_iter], state_q_parts)

                state_k = _subgroup_sum(state_k_parts.reduce(fx.ReductionOp.ADD))
                state_q = _subgroup_sum(state_q_parts.reduce(fx.ReductionOp.ADD))
                if const_expr(fuse_conv):
                    v_input = fx.Float32(fx.ptr_load(shared_value + v_index))
                else:
                    v_input = fx.Float32(
                        fx.BFloat16(
                            bo.buffer_load(
                                value_rsrc,
                                vector_base + v_index,
                                vec_width=1,
                                dtype=T.bf16,
                            )
                        )
                    )
                v_new = (v_input - state_k) * beta_value
                v_new_vec = fx.Vector.filled(_VALUES_PER_THREAD, v_new, fx.Float32)

                for k_iter in range_constexpr(_K_ITERS):
                    index = v_iter * _K_ITERS + k_iter
                    state_vecs[index] = fx.math.fma(k_vecs[k_iter], v_new_vec, state_vecs[index])
                result = state_q + v_new * dot_kq
                results[v_iter] = result

            for v_iter in range_constexpr(_V_ITERS):
                v_index = v_block * _V_PER_BLOCK + wave * _V_LANES + v_lane + v_iter * _V_TILE
                for k_iter in range_constexpr(_K_ITERS):
                    k_base = k_lane * _VALUES_PER_THREAD + k_iter * _K_TILE
                    state_offset = (head * _HEAD_DIM + v_index) * _HEAD_DIM + k_base
                    bo.buffer_store(
                        state_vecs[v_iter * _K_ITERS + k_iter],
                        state_rsrc,
                        state_offset,
                    )

            if const_expr(fuse_output_norm):
                square_sum = fx.Float32(0.0)
                if k_lane == 0:
                    for v_iter in range_constexpr(_V_ITERS):
                        square_sum = square_sum + results[v_iter] * results[v_iter]
                for offset in (32, 16, 8, 4, 2, 1):
                    square_sum = square_sum + xshfl(square_sum, offset)
                if lane == 0:
                    fx.ptr_store(square_sum, norm_sums + wave)
                gpu.barrier()
                total_square = fx.ptr_load(norm_sums)
                for source_wave in range_constexpr(1, _WAVES):
                    total_square = total_square + fx.ptr_load(norm_sums + source_wave)
                inverse_rms = rsq(total_square * fx.Float32(1.0 / _HEAD_DIM) + fx.Float32(EPS))
                if k_lane == 0:
                    for v_iter in range_constexpr(_V_ITERS):
                        v_index = v_block * _V_PER_BLOCK + wave * _V_LANES + v_lane + v_iter * _V_TILE
                        gate_value = fx.Float32(
                            fx.BFloat16(
                                bo.buffer_load(
                                    output_gate_rsrc,
                                    sample * output_gate_stride + head * _HEAD_DIM + v_index,
                                    vec_width=1,
                                    dtype=T.bf16,
                                )
                            )
                        )
                        weight = fx.Float32(
                            fx.BFloat16(
                                bo.buffer_load(
                                    norm_weight_rsrc,
                                    v_index,
                                    vec_width=1,
                                    dtype=T.bf16,
                                )
                            )
                        )
                        gated = results[v_iter] * inverse_rms * weight * sigmoid_batch([gate_value])[0]
                        bo.buffer_store(
                            gated.to(fx.BFloat16),
                            output_rsrc,
                            vector_base + v_index,
                        )
            else:
                if k_lane == 0:
                    for v_iter in range_constexpr(_V_ITERS):
                        v_index = v_block * _V_PER_BLOCK + wave * _V_LANES + v_lane + v_iter * _V_TILE
                        bo.buffer_store(
                            results[v_iter].to(fx.BFloat16),
                            output_rsrc,
                            vector_base + v_index,
                        )

        def zero_output():
            if tid < _V_PER_BLOCK:
                bo.buffer_store(
                    fx.Float32(0.0).to(fx.BFloat16),
                    output_rsrc,
                    (sample * _HEADS + head) * _HEAD_DIM + v_block * _V_PER_BLOCK + tid,
                )

        if slot >= 0:
            decode()
        else:
            zero_output()

    @flyc.jit
    def launch(
        query: Int64,
        key: Int64,
        value: Int64,
        gate: Int64,
        f_a: Int64,
        f_b_weight: Int64,
        beta: Int64,
        dt_bias: Int64,
        a_log: Int64,
        mixed_qkv: Int64,
        conv_weight: Int64,
        conv_state: Int64,
        state_indices: Int64,
        state: Int64,
        output_gate: Int64,
        norm_weight: Int64,
        output: Int64,
        beta_stride: Int32,
        input_stride: Int32,
        output_gate_stride: Int32,
        stream: Stream,
    ):
        kimi_k3_kda_recurrence_kernel(
            query,
            key,
            value,
            gate,
            f_a,
            f_b_weight,
            beta,
            dt_bias,
            a_log,
            mixed_qkv,
            conv_weight,
            conv_state,
            state_indices,
            state,
            output_gate,
            norm_weight,
            output,
            beta_stride,
            input_stride,
            output_gate_stride,
        ).launch(
            grid=(samples * _HEADS * _V_BLOCKS, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    return launch


class KimiK3KdaRecurrence:
    """Fused q/k/v causal convolution, recurrence, and gated RMSNorm."""

    def __init__(self, samples: int) -> None:
        if not 1 <= samples <= 64:
            raise ValueError(f"samples must be in [1, 64], got {samples}")
        self.samples = samples
        self.launch = build_kimi_k3_kda_recurrence(samples)

    def __call__(
        self,
        mixed_qkv: torch.Tensor,
        beta: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_state: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log: torch.Tensor,
        state_indices: torch.Tensor,
        state: torch.Tensor,
        output_gate: torch.Tensor,
        norm_weight: torch.Tensor,
        output: torch.Tensor,
        *,
        f_a: torch.Tensor,
        f_b_weight: torch.Tensor,
    ) -> torch.Tensor:
        expected_vector = (self.samples, 1, _HEADS, _HEAD_DIM)
        if (
            mixed_qkv.shape != (self.samples, _CONV_CHANNELS)
            or mixed_qkv.dtype != torch.bfloat16
            or mixed_qkv.stride(1) != 1
        ):
            raise ValueError(f"mixed_qkv must be a feature-contiguous BF16 [{self.samples}, {_CONV_CHANNELS}] view")
        if f_a.shape != (self.samples, _HEAD_DIM) or f_a.dtype != torch.bfloat16 or f_a.stride(1) != 1:
            raise ValueError(f"f_a must be a feature-contiguous BF16 [{self.samples}, {_HEAD_DIM}] view")
        if (
            f_b_weight.shape != (_HEADS * _HEAD_DIM, _HEAD_DIM)
            or f_b_weight.dtype != torch.bfloat16
            or not f_b_weight.is_contiguous()
        ):
            raise ValueError("f_b_weight must be contiguous BF16 " f"[{_HEADS * _HEAD_DIM}, {_HEAD_DIM}]")
        if beta.shape != (self.samples, 1, _HEADS) or beta.dtype != torch.bfloat16:
            raise ValueError(f"beta must be BF16 [{self.samples}, 1, {_HEADS}]")
        if (
            conv_weight.shape != (_CONV_CHANNELS, _CONV_KERNEL_WIDTH)
            or conv_weight.dtype != torch.bfloat16
            or not conv_weight.is_contiguous()
        ):
            raise ValueError(f"conv_weight must be contiguous BF16 [{_CONV_CHANNELS}, {_CONV_KERNEL_WIDTH}]")
        if (
            conv_state.ndim != 3
            or conv_state.shape[1:] != (_CONV_CHANNELS, _CONV_STATE_LENGTH)
            or conv_state.dtype != torch.bfloat16
            or not conv_state.is_contiguous()
        ):
            raise ValueError(f"conv_state must be contiguous BF16 [slots, {_CONV_CHANNELS}, {_CONV_STATE_LENGTH}]")
        if dt_bias.shape != (_HEADS, _HEAD_DIM) or dt_bias.dtype != torch.bfloat16:
            raise ValueError(f"dt_bias must be BF16 [{_HEADS}, {_HEAD_DIM}]")
        if a_log.shape != (_HEADS,) or a_log.dtype != torch.float32:
            raise ValueError(f"a_log must be FP32 [{_HEADS}]")
        if state_indices.shape != (self.samples,) or state_indices.dtype != torch.int32:
            raise ValueError(f"state_indices must be int32 [{self.samples}]")
        if (
            state.ndim != 4
            or state.shape[1:] != (_HEADS, _HEAD_DIM, _HEAD_DIM)
            or state.dtype != torch.float32
            or not state.is_contiguous()
        ):
            raise ValueError(f"state must be contiguous FP32 [slots, {_HEADS}, {_HEAD_DIM}, {_HEAD_DIM}]")
        if (
            output_gate.shape
            not in {
                expected_vector,
                (self.samples, _HEADS, _HEAD_DIM),
            }
            or output_gate.dtype != torch.bfloat16
        ):
            raise ValueError("output_gate must be a BF16 [samples, (1,) heads, head_dim] tensor")
        if output_gate.stride(-1) != 1 or output_gate.stride(-2) != _HEAD_DIM:
            raise ValueError("output_gate heads must be contiguous")
        if norm_weight.shape != (_HEAD_DIM,) or norm_weight.dtype != torch.bfloat16:
            raise ValueError(f"norm_weight must be BF16 [{_HEAD_DIM}]")
        if output.shape != expected_vector or output.dtype != torch.bfloat16 or not output.is_contiguous():
            raise ValueError(f"output must be contiguous BF16 {list(expected_vector)}")

        tensors = (
            mixed_qkv,
            f_a,
            f_b_weight,
            beta,
            conv_weight,
            conv_state,
            dt_bias,
            a_log,
            state_indices,
            state,
            output_gate,
            norm_weight,
            output,
        )
        if any(tensor.device != mixed_qkv.device for tensor in tensors):
            raise ValueError("all fused KDA tensors must be on the same device")

        self.launch(
            output.data_ptr(),
            output.data_ptr(),
            output.data_ptr(),
            output.data_ptr(),
            f_a.data_ptr(),
            f_b_weight.data_ptr(),
            beta.data_ptr(),
            dt_bias.data_ptr(),
            a_log.data_ptr(),
            mixed_qkv.data_ptr(),
            conv_weight.data_ptr(),
            conv_state.data_ptr(),
            state_indices.data_ptr(),
            state.data_ptr(),
            output_gate.data_ptr(),
            norm_weight.data_ptr(),
            output.data_ptr(),
            beta.stride(0),
            mixed_qkv.stride(0),
            output_gate.stride(0),
            stream=torch.cuda.current_stream(),
        )
        return output
