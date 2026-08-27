# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4/E8M0 transport decode fused with expert-aware INT8 SmoothQuant."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.extras import types as T
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from kernels.common import buffer_ops

VALUES_PER_THREAD = 16
WAVE_SIZE = 64


@flyc.jit
def fused_prepare_bf16_smoothquant_row_wave(
    source_rsrc,
    smooth_rsrc,
    smooth_scale_rsrc,
    destination_rsrc,
    source_row,
    destination_row,
    expert,
    *,
    cols,
):
    """Quantize one expert-aware BF16 route directly into its INT8 payload.

    This is the row primitive used by fused prepare.  One producer wave owns
    one route, computes the same per-row ``amax / 127`` scale as AITER's
    SmoothQuant front end, and writes the final destination payload directly.
    No route-major front-quant tensor is materialized.

    Source rows are token-major BF16. Destination rows are the final
    route-major INT8 payload chosen by dispatch prepare.
    """
    values_per_lane = 8
    values_per_wave = WAVE_SIZE * values_per_lane
    if cols % values_per_wave:
        raise ValueError(
            f"fused BF16 SmoothQuant requires cols divisible by "
            f"{values_per_wave}, got {cols}"
        )

    lane = fx.thread_idx.x % fx.Int32(WAVE_SIZE)
    rounds = cols // values_per_wave
    local_max = fx.Float32(1e-10)

    for round_index in range_constexpr(rounds):
        column = lane * fx.Int32(values_per_lane) + fx.Int32(
            round_index * values_per_wave
        )
        packed = Vec(
            buffer_ops.buffer_load(
                source_rsrc,
                source_row * fx.Int32(cols // 2) + column // fx.Int32(2),
                vec_width=4,
                dtype=T.i32(),
            )
        )
        act = packed.bitcast(fx.BFloat16).to(fx.Float32)
        smooth = []
        smooth_base = expert * fx.Int32(cols) + column
        for half in range_constexpr(2):
            smooth.extend(
                Vec(
                    buffer_ops.buffer_load(
                        smooth_rsrc,
                        smooth_base + fx.Int32(half * 4),
                        vec_width=4,
                        dtype=T.f32(),
                    )
                )
            )
        for element in range_constexpr(values_per_lane):
            transformed = act[element] * smooth[element]
            local_max = local_max.maximumf(
                transformed.maximumf(fx.Float32(0.0) - transformed)
            )

    for offset in (32, 16, 8, 4, 2, 1):
        local_max = local_max.maximumf(
            local_max.shuffle_xor(fx.Int32(offset), fx.Int32(WAVE_SIZE))
        )

    # Match AITER's compile-time float `1 / numeric<int8_t>::max()` and
    # multiply.  Rewriting this as a runtime divide changes one FP32 ULP.
    row_scale = local_max * fx.Float32(1.0 / 127.0)
    # AITER's SmoothQuant path intentionally uses the gfx reciprocal
    # instruction before its float-to-int8 truncation.  Preserve that exact
    # numerical contract in the fused prepare path.
    inverse_scale = fx.Float32(fx.rocdl.rcp(T.f32(), row_scale))
    for round_index in range_constexpr(rounds):
        column = lane * fx.Int32(values_per_lane) + fx.Int32(
            round_index * values_per_wave
        )
        packed = Vec(
            buffer_ops.buffer_load(
                source_rsrc,
                source_row * fx.Int32(cols // 2) + column // fx.Int32(2),
                vec_width=4,
                dtype=T.i32(),
            )
        )
        act = packed.bitcast(fx.BFloat16).to(fx.Float32)
        smooth = []
        smooth_base = expert * fx.Int32(cols) + column
        for half in range_constexpr(2):
            smooth.extend(
                Vec(
                    buffer_ops.buffer_load(
                        smooth_rsrc,
                        smooth_base + fx.Int32(half * 4),
                        vec_width=4,
                        dtype=T.f32(),
                    )
                )
            )
        quantized = Vec.from_elements(
            [
                act[element] * smooth[element] * inverse_scale
                for element in range_constexpr(values_per_lane)
            ],
            fx.Float32,
        ).to(fx.Int8)
        packed_quantized = quantized.bitcast(fx.Int32)
        buffer_ops.buffer_store(
            packed_quantized,
            destination_rsrc,
            destination_row * fx.Int32(cols) + column,
            offset_is_bytes=True,
        )
    return row_scale




@flyc.jit
def copy_mxfp4_row_and_compute_smoothquant_scale_wave(
    source_row_rsrc,
    destination_row_rsrc,
    mx_scale_rsrc,
    smooth_rsrc,
    source_row,
    expert,
    *,
    cols,
):
    """Copy one FP4 row remotely while reusing its registers for row-max.

    One 128-bit copy is exactly one 32-value MX block.  Reusing those four i32
    registers avoids reading the source FP4 payload a second time for the
    producer-side SmoothQuant reduction.
    """
    if cols % 32:
        raise ValueError(f"cols={cols} must be divisible by 32")
    blocks_per_row = cols // 32
    block_rounds = (blocks_per_row + WAVE_SIZE - 1) // WAVE_SIZE
    vec2_f32_ty = T.VectorType.get([2], T.f32())
    lane = fx.thread_idx.x % fx.Int32(WAVE_SIZE)
    local_max = fx.Float32(1e-10)
    for block_round in range_constexpr(block_rounds):
        block = lane + fx.Int32(block_round * WAVE_SIZE)
        active = block < fx.Int32(blocks_per_row)
        safe_block = active.select(block, fx.Int32(0))
        packed = Vec(
            buffer_ops.buffer_load(
                source_row_rsrc,
                safe_block * fx.Int32(4),
                vec_width=4,
                dtype=T.i32(),
            )
        )
        if active:
            buffer_ops.buffer_store(
                packed, destination_row_rsrc, safe_block * fx.Int32(4)
            )
        scale_byte = buffer_ops.buffer_load(
            mx_scale_rsrc,
            source_row * fx.Int32(blocks_per_row) + safe_block,
            vec_width=1,
            dtype=T.i8(),
        )
        e8m0 = fx.Uint8(scale_byte).to(fx.Int32)
        block_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
        smooth_base = expert * fx.Int32(cols) + safe_block * fx.Int32(32)
        for word in range_constexpr(4):
            smooth = []
            for half in range_constexpr(2):
                smooth.extend(
                    Vec(
                        buffer_ops.buffer_load(
                            smooth_rsrc,
                            smooth_base + fx.Int32(word * 8 + half * 4),
                            vec_width=4,
                            dtype=T.f32(),
                        )
                    )
                )
            for pair in range_constexpr(4):
                decoded = Vec(
                    fx.rocdl.cvt_scalef32_pk_f32_fp4(
                        vec2_f32_ty,
                        packed[word].ir_value(),
                        block_scale.ir_value(),
                        pair,
                    )
                )
                for elem in range_constexpr(2):
                    value = active.select(
                        decoded[elem] * smooth[pair * 2 + elem],
                        fx.Float32(0.0),
                    )
                    local_max = local_max.maximumf(
                        value.maximumf(fx.Float32(0.0) - value)
                    )
    for offset in (32, 16, 8, 4, 2, 1):
        local_max = local_max.maximumf(
            local_max.shuffle_xor(fx.Int32(offset), fx.Int32(WAVE_SIZE))
        )
    return local_max / fx.Float32(127.0)




class MxFp4SmoothInt8ATileLoader:
    """Decode expert-aware MXFP4 directly into a GEMM CTA's INT8 A-LDS.

    Row scales are produced once by the sending producer.  Every GEMM CTA then
    converts only the current 256-K slice while staging A, eliminating the
    global INT8 intermediate and its cross-CTA completion counter.
    """

    def __init__(
        self,
        *,
        payload_rsrc,
        mx_scale_rsrc,
        smooth_rsrc,
        row_scale_rsrc,
        model_dim,
        sort_block_m,
        k_step_bytes,
        total_threads,
        expert_offset=0,
        swizzle=False,
    ):
        if model_dim % 32:
            raise ValueError("MXFP4 A-load requires model_dim divisible by 32")
        if k_step_bytes % 16:
            raise ValueError("MXFP4 A-load requires a 16-value-aligned K step")
        self._payload = payload_rsrc
        self._mx_scale = mx_scale_rsrc
        self._smooth = smooth_rsrc
        self._row_scale = row_scale_rsrc
        self._model_dim = model_dim
        self._sort_block_m = sort_block_m
        self._k_step_bytes = k_step_bytes
        self._total_threads = total_threads
        self._expert_offset = int(expert_offset)
        self._swizzle = bool(swizzle)
        self._tx = fx.thread_idx.x
        self._packed_i32_per_row = model_dim // 8
        self._scales_per_row = model_dim // 32
        self._vec2_f32_ty = T.VectorType.get([2], T.f32())

    def for_tile(self, tile_row_base_i32, expert_i32):
        self._tile_row_base = tile_row_base_i32
        self._expert = expert_i32
        chunks_per_row = self._k_step_bytes // 16
        row_stride_i32 = self._k_step_bytes // 4
        total_chunks = self._sort_block_m * chunks_per_row
        self._chunks = []
        for base in range_constexpr(0, total_chunks, self._total_threads):
            linear = fx.Int32(base) + fx.Int32(self._tx)
            row = linear // fx.Int32(chunks_per_row)
            chunk = linear - row * fx.Int32(chunks_per_row)
            if const_expr(self._swizzle):
                col_i32 = chunk * fx.Int32(4)
                lds_i32 = row * fx.Int32(row_stride_i32) + (
                    col_i32 ^ ((row & fx.Int32(15)) << fx.Int32(2))
                )
            else:
                lds_i32 = linear * fx.Int32(4)
            self._chunks.append((lds_i32 * fx.Int32(4), row, chunk))

    def load_regs(self, k_step_byte_off):
        regs = []
        k_base = fx.Int32(k_step_byte_off)
        for lds_byte, tile_row, chunk in self._chunks:
            physical_row = self._tile_row_base + tile_row
            column = k_base + chunk * fx.Int32(16)
            packed = Vec(
                buffer_ops.buffer_load(
                    self._payload,
                    physical_row * fx.Int32(self._packed_i32_per_row)
                    + column // fx.Int32(8),
                    vec_width=2,
                    dtype=T.i32(),
                )
            )
            scale_byte = buffer_ops.buffer_load(
                self._mx_scale,
                physical_row * fx.Int32(self._scales_per_row)
                + column // fx.Int32(32),
                vec_width=1,
                dtype=T.i8(),
            )
            e8m0 = fx.Uint8(scale_byte).to(fx.Int32)
            block_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
            smooth_expert = self._expert + fx.Int32(self._expert_offset)
            smooth_base = smooth_expert * fx.Int32(self._model_dim) + column
            smooth = []
            for group in range_constexpr(4):
                smooth.extend(
                    Vec(
                        buffer_ops.buffer_load(
                            self._smooth,
                            smooth_base + fx.Int32(group * 4),
                            vec_width=4,
                            dtype=T.f32(),
                        )
                    )
                )
            inv_scale = fx.Float32(1.0) / buffer_ops.buffer_load(
                self._row_scale,
                physical_row,
                vec_width=1,
                dtype=T.f32(),
            )
            transformed = []
            for word in range_constexpr(2):
                for pair in range_constexpr(4):
                    decoded = Vec(
                        fx.rocdl.cvt_scalef32_pk_f32_fp4(
                            self._vec2_f32_ty,
                            packed[word].ir_value(),
                            block_scale.ir_value(),
                            pair,
                        )
                    )
                    for elem in range_constexpr(2):
                        index = word * 8 + pair * 2 + elem
                        transformed.append(decoded[elem] * smooth[index] * inv_scale)
            quant_i32 = Vec.from_elements(transformed, fx.Float32).to(fx.Int8).bitcast(
                fx.Int32
            )
            regs.append((lds_byte, quant_i32))
        return regs

    def store(self, lds_dst, regs, base_i32=0):
        base_bytes = fx.Int32(base_i32) * fx.Int32(4)
        for lds_byte, value in regs:
            dst = fx.make_view(
                fx.add_offset(
                    fx.recast_iter(fx.Int32, lds_dst.ptr),
                    (base_bytes + lds_byte) // fx.Int32(4),
                ),
                fx.make_layout(4, 1),
            )
            fragment = fx.make_rmem_tensor(4, fx.Int32)
            fragment.store(Vec(value))
            fx.copy(
                fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), fragment, dst
            )


@flyc.jit
def convert_mxfp4_smoothquant_row(
    payload_rsrc,
    mx_scale_rsrc,
    smooth_rsrc,
    output_rsrc,
    output_scale_rsrc,
    reduction_scratch,
    row,
    expert,
    *,
    cols,
    total_threads,
):
    """Convert one received route row; all threads in the CTA participate."""
    if cols % 32:
        raise ValueError(f"cols={cols} must be divisible by 32")
    if total_threads % WAVE_SIZE:
        raise ValueError(f"total_threads={total_threads} must be wave aligned")
    active_threads = (cols + VALUES_PER_THREAD - 1) // VALUES_PER_THREAD
    if active_threads > total_threads:
        raise ValueError(
            f"cols={cols} needs {active_threads} threads, got total_threads={total_threads}"
        )
    reduction_waves = total_threads // WAVE_SIZE
    packed_i32_per_row = cols // 8
    scales_per_row = cols // 32
    vec2_f32_ty = T.VectorType.get([2], T.f32())

    tid = fx.thread_idx.x
    active = tid < fx.Int32(active_threads)
    safe_tid = active.select(tid, fx.Int32(0))
    scale_byte = buffer_ops.buffer_load(
        mx_scale_rsrc,
        row * fx.Int32(scales_per_row) + safe_tid // fx.Int32(2),
        vec_width=1,
        dtype=T.i8(),
    )
    e8m0 = fx.Uint8(scale_byte).to(fx.Int32)
    block_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
    packed = Vec(
        buffer_ops.buffer_load(
            payload_rsrc,
            row * fx.Int32(packed_i32_per_row) + safe_tid * fx.Int32(2),
            vec_width=2,
            dtype=T.i32(),
        )
    )
    smooth_base = expert * fx.Int32(cols) + safe_tid * fx.Int32(VALUES_PER_THREAD)
    smooth = []
    for chunk in range_constexpr(VALUES_PER_THREAD // 4):
        smooth.extend(
            Vec(
                buffer_ops.buffer_load(
                    smooth_rsrc,
                    smooth_base + fx.Int32(chunk * 4),
                    vec_width=4,
                    dtype=T.f32(),
                )
            )
        )

    transformed = []
    local_max = fx.Float32(1e-10)
    for word in range_constexpr(2):
        for pair in range_constexpr(4):
            decoded = Vec(
                fx.rocdl.cvt_scalef32_pk_f32_fp4(
                    vec2_f32_ty,
                    packed[word].ir_value(),
                    block_scale.ir_value(),
                    pair,
                )
            )
            for elem in range_constexpr(2):
                index = word * 8 + pair * 2 + elem
                value = active.select(decoded[elem] * smooth[index], fx.Float32(0.0))
                transformed.append(value)
                local_max = local_max.maximumf(value.maximumf(fx.Float32(0.0) - value))

    for offset in (32, 16, 8, 4, 2, 1):
        local_max = local_max.maximumf(
            local_max.shuffle_xor(fx.Int32(offset), fx.Int32(WAVE_SIZE))
        )
    lane = tid % fx.Int32(WAVE_SIZE)
    wave = tid // fx.Int32(WAVE_SIZE)
    if lane == fx.Int32(0):
        fx.ptr_store(Vec.from_elements([local_max], fx.Float32), reduction_scratch + wave)
    fx.barrier()
    if wave == fx.Int32(0):
        in_range = lane < fx.Int32(reduction_waves)
        safe_lane = in_range.select(lane, fx.Int32(0))
        block_max = fx.Float32(fx.ptr_load(reduction_scratch + safe_lane))
        block_max = in_range.select(block_max, fx.Float32(0.0))
        for offset in (32, 16, 8, 4, 2, 1):
            block_max = block_max.maximumf(
                block_max.shuffle_xor(fx.Int32(offset), fx.Int32(WAVE_SIZE))
            )
        if lane == fx.Int32(0):
            fx.ptr_store(Vec.from_elements([block_max], fx.Float32), reduction_scratch)
    fx.barrier()

    row_max = fx.Float32(fx.ptr_load(reduction_scratch))
    row_quant_scale = row_max / fx.Float32(127.0)
    if tid == fx.Int32(0):
        buffer_ops.buffer_store(row_quant_scale, output_scale_rsrc, row)
    inv_scale = fx.Float32(1.0) / row_quant_scale
    quant = Vec.from_elements(
        [transformed[index] * inv_scale for index in range_constexpr(VALUES_PER_THREAD)],
        fx.Float32,
    ).to(fx.Int8)
    quant_i32 = quant.bitcast(fx.Int32)
    output_base = row * fx.Int32(cols) + safe_tid * fx.Int32(VALUES_PER_THREAD)
    if active:
        for word in range_constexpr(VALUES_PER_THREAD // 4):
            buffer_ops.buffer_store(
                quant_i32[word],
                output_rsrc,
                output_base + fx.Int32(word * 4),
                offset_is_bytes=True,
            )
