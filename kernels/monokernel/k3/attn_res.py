# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Low-token Kimi-K3 attention-residual mixer and output RMSNorm."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T
from kernels.common import buffer_ops as bo
from kernels.monokernel.config import EPS, FP8_MAX
from kernels.monokernel.ops import exp, rcp, rsq, rsrc, xred, xshfl

_THREADS = 896
_WAVE_SIZE = 64
_WAVES = _THREADS // _WAVE_SIZE


@functools.cache
def build_kimi_k3_attn_res(
    samples: int,
    hidden: int,
    num_blocks: int,
    has_delta: bool,
    block_write_idx: int,
    quantize_output: bool = False,
    source_override_idx: int = -1,
):
    """Build one CTA per sample for AttnRes mixing followed by RMSNorm."""

    if samples not in {1, 2, 4, 8}:
        raise ValueError(f"samples must be one of {{1, 2, 4, 8}}, got {samples}")
    if hidden <= 0 or hidden % (2 * _THREADS):
        raise ValueError(f"hidden must be divisible by {2 * _THREADS}, got {hidden}")
    if num_blocks < 0:
        raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
    if source_override_idx < -1 or source_override_idx >= num_blocks:
        raise ValueError(f"source_override_idx must be in [-1, {num_blocks}), got {source_override_idx}")
    num_sources = num_blocks + 1
    pair_rounds = hidden // (2 * _THREADS)

    @fx.struct
    class SharedStorage:
        reduction: fx.Array[fx.Float32, 2 * _WAVES, 16]

    @flyc.kernel(known_block_size=[_THREADS, 1, 1])
    def kimi_k3_attn_res_kernel(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        norm_weight: Int64,
        qk_weight: Int64,
        output_norm_weight: Int64,
        updated_out: Int64,
        output: Int64,
        quantized_output: Int64,
        quantized_scale: Int64,
        block_stride: Int32,
    ):
        sample = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % _WAVE_SIZE
        wave = tid // _WAVE_SIZE

        prefix_rsrc = rsrc(prefix)
        delta_rsrc = rsrc(delta)
        blocks_rsrc = rsrc(blocks)
        norm_weight_rsrc = rsrc(norm_weight)
        qk_weight_rsrc = rsrc(qk_weight)
        output_norm_weight_rsrc = rsrc(output_norm_weight)
        updated_out_rsrc = rsrc(updated_out)
        output_rsrc = rsrc(output)
        quantized_output_rsrc = rsrc(quantized_output)
        quantized_scale_rsrc = rsrc(quantized_scale)
        reduction = fx.SharedAllocator().allocate(SharedStorage).peek().reduction.ptr

        def wave_sum(value):
            for offset in (32, 16, 8, 4, 2, 1):
                value = xred(value, offset, lambda lhs, rhs: lhs + rhs)
            return value

        def block_sums(lhs, rhs):
            lhs_wave = wave_sum(lhs)
            rhs_wave = wave_sum(rhs)
            if lane == 0:
                fx.ptr_store(lhs_wave, reduction + wave)
                fx.ptr_store(rhs_wave, reduction + _WAVES + wave)
            gpu.barrier()
            lhs_total = fx.ptr_load(reduction)
            rhs_total = fx.ptr_load(reduction + _WAVES)
            for source_wave in range_constexpr(1, _WAVES):
                lhs_total = lhs_total + fx.ptr_load(reduction + source_wave)
                rhs_total = rhs_total + fx.ptr_load(reduction + _WAVES + source_wave)
            gpu.barrier()
            return lhs_total, rhs_total

        def load_updated(pair):
            prefix_word = fx.Int32(bo.buffer_load(prefix_rsrc, pair, vec_width=1, dtype=T.i32))
            if has_delta:
                delta_word = fx.Int32(bo.buffer_load(delta_rsrc, pair, vec_width=1, dtype=T.i32))
                prefix_lo = (prefix_word << 16).bitcast(fx.Float32)
                prefix_hi = (prefix_word & fx.Int32(-65536)).bitcast(fx.Float32)
                delta_lo = (delta_word << 16).bitcast(fx.Float32)
                delta_hi = (delta_word & fx.Int32(-65536)).bitcast(fx.Float32)
                return (prefix_lo + delta_lo).to(fx.BFloat16), (prefix_hi + delta_hi).to(fx.BFloat16)
            return (prefix_word << 16).bitcast(fx.Float32).to(fx.BFloat16), (prefix_word & fx.Int32(-65536)).bitcast(
                fx.Float32
            ).to(fx.BFloat16)

        def store_output(pair_in_row, value_lo, value_hi):
            output_values = fx.Vector.from_elements([value_lo, value_hi], fx.Float32).to(fx.BFloat16)
            output_word = output_values.bitcast(fx.Int32)[0]
            bo.buffer_store(output_word, output_rsrc, sample * (hidden // 2) + pair_in_row)
            if const_expr(quantize_output):
                values = output_values.to(fx.Float32)
                amax = fx.max(fx.max(values[0], -values[0]), fx.max(values[1], -values[1]))
                for offset in (8, 4, 2, 1):
                    amax = xred(amax, offset, fx.max)
                raw_scale = amax * fx.Float32(1.0 / FP8_MAX)
                bits = raw_scale.bitcast(fx.Int32)
                exponent = bits.shrui(fx.Int32(23)) & fx.Int32(0xFF)
                round_up = ((bits & fx.Int32(0x400000)) != 0) & (
                    ((bits & fx.Int32(0x200000)) != 0) | ((bits & fx.Int32(0x1FFFFF)) != 0) | (exponent > 0)
                )
                exponent = exponent + round_up.select(fx.Int32(1), fx.Int32(0))
                nonzero = amax > fx.Float32(0.0)
                scale = nonzero.select(
                    (exponent << fx.Int32(23)).bitcast(fx.Float32),
                    fx.Float32(1.0),
                )
                inverse = nonzero.select(rcp(scale), fx.Float32(1.0))
                q0 = fx.min(fx.max(values[0] * inverse, -FP8_MAX), FP8_MAX)
                q1 = fx.min(fx.max(values[1] * inverse, -FP8_MAX), FP8_MAX)
                packed = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & fx.Int32(0xFFFF)
                neighbor = xshfl(packed, 1)
                if lane % 2 == 0:
                    bo.buffer_store(
                        packed | (neighbor << fx.Int32(16)),
                        quantized_output_rsrc,
                        sample * (hidden // 4) + pair_in_row // 2,
                    )
                if lane % 16 == 0:
                    scale_col = pair_in_row // 16
                    scale_offset = (
                        (scale_col // 8) * 256 + (scale_col % 4) * 64 + sample * 4 + ((scale_col // 4) % 2) * 2
                    )
                    bo.buffer_store(
                        exponent.to(fx.Uint8),
                        quantized_scale_rsrc,
                        scale_offset,
                        offset_is_bytes=True,
                    )

        if const_expr(num_sources == 1):
            updated_pairs = []
            square_sum = fx.Float32(0.0)
            for pair_round in range_constexpr(pair_rounds):
                pair_in_row = tid + pair_round * _THREADS
                value_lo_bf16, value_hi_bf16 = load_updated(sample * (hidden // 2) + pair_in_row)
                value_lo = fx.Float32(value_lo_bf16)
                value_hi = fx.Float32(value_hi_bf16)
                updated_pairs.append((value_lo, value_hi))
                square_sum = square_sum + value_lo * value_lo + value_hi * value_hi
                updated_word = (
                    fx.Vector.from_elements([value_lo, value_hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                )
                bo.buffer_store(updated_word, updated_out_rsrc, sample * (hidden // 2) + pair_in_row)
                if const_expr(block_write_idx >= 0):
                    block_pair = (sample * block_stride + block_write_idx) * (hidden // 2) + pair_in_row
                    bo.buffer_store(updated_word, blocks_rsrc, block_pair)

            total_square, _ = block_sums(square_sum, fx.Float32(0.0))
            inverse_rms = rsq(total_square * (1.0 / hidden) + EPS)
            for pair_round in range_constexpr(pair_rounds):
                pair_in_row = tid + pair_round * _THREADS
                output_weight_word = fx.Int32(
                    bo.buffer_load(output_norm_weight_rsrc, pair_in_row, vec_width=1, dtype=T.i32)
                )
                weight_lo = (output_weight_word << 16).bitcast(fx.Float32)
                weight_hi = (output_weight_word & fx.Int32(-65536)).bitcast(fx.Float32)
                value_lo, value_hi = updated_pairs[pair_round]
                store_output(
                    pair_in_row,
                    value_lo * inverse_rms * weight_lo,
                    value_hi * inverse_rms * weight_hi,
                )
            return

        logits = []
        cached_source_pairs = []
        for source in range_constexpr(num_sources):
            square_sum = fx.Float32(0.0)
            weighted_sum = fx.Float32(0.0)
            source_pairs = []
            for pair_round in range_constexpr(pair_rounds):
                pair_in_row = tid + pair_round * _THREADS
                if const_expr(source < num_blocks):
                    if const_expr(source == source_override_idx):
                        source_word = fx.Int32(
                            bo.buffer_load(
                                delta_rsrc,
                                sample * (hidden // 2) + pair_in_row,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                        source_pair = (sample * block_stride + source) * (hidden // 2) + pair_in_row
                        bo.buffer_store(source_word, blocks_rsrc, source_pair)
                    else:
                        source_pair = (sample * block_stride + source) * (hidden // 2) + pair_in_row
                        source_word = fx.Int32(bo.buffer_load(blocks_rsrc, source_pair, vec_width=1, dtype=T.i32))
                    value_lo = (source_word << 16).bitcast(fx.Float32)
                    value_hi = (source_word & fx.Int32(-65536)).bitcast(fx.Float32)
                else:
                    value_lo_bf16, value_hi_bf16 = load_updated(sample * (hidden // 2) + pair_in_row)
                    value_lo = fx.Float32(value_lo_bf16)
                    value_hi = fx.Float32(value_hi_bf16)
                    updated_word = (
                        fx.Vector.from_elements([value_lo, value_hi], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                    )
                    bo.buffer_store(updated_word, updated_out_rsrc, sample * (hidden // 2) + pair_in_row)
                    if const_expr(block_write_idx >= 0):
                        block_pair = (sample * block_stride + block_write_idx) * (hidden // 2) + pair_in_row
                        bo.buffer_store(updated_word, blocks_rsrc, block_pair)

                if const_expr(num_sources <= 2):
                    source_pairs.append((value_lo, value_hi))

                norm_word = fx.Int32(bo.buffer_load(norm_weight_rsrc, pair_in_row, vec_width=1, dtype=T.i32))
                qk_word = fx.Int32(bo.buffer_load(qk_weight_rsrc, pair_in_row, vec_width=1, dtype=T.i32))
                norm_lo = (norm_word << 16).bitcast(fx.Float32)
                norm_hi = (norm_word & fx.Int32(-65536)).bitcast(fx.Float32)
                qk_lo = (qk_word << 16).bitcast(fx.Float32)
                qk_hi = (qk_word & fx.Int32(-65536)).bitcast(fx.Float32)
                square_sum = square_sum + value_lo * value_lo + value_hi * value_hi
                weighted_sum = weighted_sum + value_lo * norm_lo * qk_lo + value_hi * norm_hi * qk_hi

            total_square, total_weighted = block_sums(square_sum, weighted_sum)
            logits.append(total_weighted * rsq(total_square * (1.0 / hidden) + EPS))
            if const_expr(num_sources <= 2):
                cached_source_pairs.append(source_pairs)

        max_logit = logits[0]
        for source in range_constexpr(1, num_sources):
            max_logit = fx.max(max_logit, logits[source])
        probabilities = [exp(logit - max_logit) for logit in logits]
        probability_sum = probabilities[0]
        for source in range_constexpr(1, num_sources):
            probability_sum = probability_sum + probabilities[source]
        inverse_probability_sum = rcp(probability_sum)
        probabilities = [probability * inverse_probability_sum for probability in probabilities]

        mixed_pairs = []
        mixed_square_sum = fx.Float32(0.0)
        for pair_round in range_constexpr(pair_rounds):
            pair_in_row = tid + pair_round * _THREADS
            mixed_lo = fx.Float32(0.0)
            mixed_hi = fx.Float32(0.0)
            for source in range_constexpr(num_sources):
                if const_expr(num_sources <= 2):
                    value_lo, value_hi = cached_source_pairs[source][pair_round]
                else:
                    if const_expr(source < num_blocks):
                        if const_expr(source == source_override_idx):
                            source_word = fx.Int32(
                                bo.buffer_load(
                                    delta_rsrc,
                                    sample * (hidden // 2) + pair_in_row,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                        else:
                            source_pair = (sample * block_stride + source) * (hidden // 2) + pair_in_row
                            source_word = fx.Int32(bo.buffer_load(blocks_rsrc, source_pair, vec_width=1, dtype=T.i32))
                        value_lo = (source_word << 16).bitcast(fx.Float32)
                        value_hi = (source_word & fx.Int32(-65536)).bitcast(fx.Float32)
                    else:
                        value_lo_bf16, value_hi_bf16 = load_updated(sample * (hidden // 2) + pair_in_row)
                        value_lo = fx.Float32(value_lo_bf16)
                        value_hi = fx.Float32(value_hi_bf16)
                mixed_lo = mixed_lo + probabilities[source] * value_lo
                mixed_hi = mixed_hi + probabilities[source] * value_hi
            mixed_pairs.append((mixed_lo, mixed_hi))
            mixed_square_sum = mixed_square_sum + mixed_lo * mixed_lo + mixed_hi * mixed_hi

        total_mixed_square, _ = block_sums(mixed_square_sum, fx.Float32(0.0))
        output_inverse_rms = rsq(total_mixed_square * (1.0 / hidden) + EPS)
        for pair_round in range_constexpr(pair_rounds):
            pair_in_row = tid + pair_round * _THREADS
            output_weight_word = fx.Int32(
                bo.buffer_load(output_norm_weight_rsrc, pair_in_row, vec_width=1, dtype=T.i32)
            )
            weight_lo = (output_weight_word << 16).bitcast(fx.Float32)
            weight_hi = (output_weight_word & fx.Int32(-65536)).bitcast(fx.Float32)
            mixed_lo, mixed_hi = mixed_pairs[pair_round]
            store_output(
                pair_in_row,
                mixed_lo * output_inverse_rms * weight_lo,
                mixed_hi * output_inverse_rms * weight_hi,
            )

    @flyc.jit
    def launch(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        norm_weight: Int64,
        qk_weight: Int64,
        output_norm_weight: Int64,
        updated_out: Int64,
        output: Int64,
        quantized_output: Int64,
        quantized_scale: Int64,
        block_stride: Int32,
        stream: Stream = Stream(None),
    ):
        kimi_k3_attn_res_kernel(
            prefix,
            delta,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight,
            updated_out,
            output,
            quantized_output,
            quantized_scale,
            block_stride,
            value_attrs={"rocdl.flat_work_group_size": f"{_THREADS},{_THREADS}"},
        ).launch(grid=(samples, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    launch.func.__name__ = (
        f"kimi_k3_attn_res_s{samples}_h{hidden}_b{num_blocks}_d{int(has_delta)}"
        f"_w{block_write_idx}_q{int(quantize_output)}_o{source_override_idx}"
    )
    return launch


class KimiK3AttnRes:
    """Torch adapter for one fixed Kimi-K3 AttnRes position."""

    def __init__(
        self,
        samples: int,
        hidden: int,
        num_blocks: int,
        has_delta: bool,
        block_write_idx: int,
        quantize_output: bool = False,
        source_override_idx: int = -1,
    ) -> None:
        self.samples = samples
        self.hidden = hidden
        self.quantize_output = quantize_output
        self.launch = build_kimi_k3_attn_res(
            samples,
            hidden,
            num_blocks,
            has_delta,
            block_write_idx,
            quantize_output,
            source_override_idx,
        )

    def __call__(
        self,
        prefix: torch.Tensor,
        delta: torch.Tensor,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        output_norm_weight: torch.Tensor,
        updated_out: torch.Tensor,
        output: torch.Tensor,
        quantized_output: torch.Tensor | None = None,
        quantized_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = (prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, updated_out, output)
        if any(tensor.dtype != torch.bfloat16 or not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("Kimi-K3 AttnRes inputs and outputs must be contiguous BF16 tensors")
        if prefix.shape != (self.samples, self.hidden) or delta.shape != prefix.shape:
            raise ValueError("prefix and delta must have shape [samples, hidden]")
        if updated_out.shape != prefix.shape or output.shape != prefix.shape:
            raise ValueError("updated_out and output must match prefix")
        if blocks.ndim != 3 or blocks.shape[0] != self.samples or blocks.shape[2] != self.hidden:
            raise ValueError("blocks must have shape [samples, blocks, hidden]")
        for weight in (norm_weight, qk_weight, output_norm_weight):
            if weight.shape != (self.hidden,):
                raise ValueError("AttnRes weights must have shape [hidden]")
        if self.quantize_output:
            if quantized_output is None or quantized_scale is None:
                raise ValueError("quantized output buffers are required")
            if quantized_output.shape != (32, self.hidden) or quantized_output.dtype != torch.uint8:
                raise ValueError("quantized_output must be uint8 [32, hidden]")
            if quantized_scale.numel() != 32 * (self.hidden // 32) or quantized_scale.dtype != torch.uint8:
                raise ValueError("quantized_scale has the wrong packed size or dtype")
        else:
            quantized_output = output
            quantized_scale = output
        self.launch(
            prefix.data_ptr(),
            delta.data_ptr(),
            blocks.data_ptr(),
            norm_weight.data_ptr(),
            qk_weight.data_ptr(),
            output_norm_weight.data_ptr(),
            updated_out.data_ptr(),
            output.data_ptr(),
            quantized_output.data_ptr(),
            quantized_scale.data_ptr(),
            blocks.shape[1],
            stream=torch.cuda.current_stream(),
        )
        return output, updated_out
