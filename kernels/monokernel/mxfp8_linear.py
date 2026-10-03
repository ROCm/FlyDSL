# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reusable small-batch MXFP8 quantization and linear projection kernels."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int64, ReductionOp, Stream, T
from kernels.common import buffer_ops as bo
from kernels.monokernel.ops import rsrc
from kernels.monokernel.packing import pack_mxfp8_scale, pack_mxfp8_weight

_GROUP = 32
_QUANT_THREADS = 64
_PROJECT_THREADS = 256
_WAVE_SIZE = 64
_PROJECT_WAVES = 4
_N_TILE = 16
_FP8_INV_MAX_POS_BITS = 0x3B124925


@functools.cache
def build_mxfp8_quantize(rows: int, cols: int):
    """Build BF16 -> row-major MXFP8 quantization with preshuffled E8M0 scales."""

    if rows not in {1, 2, 4, 8}:
        raise ValueError(f"rows must be one of {{1, 2, 4, 8}}, got {rows}")
    if cols <= 0 or cols % 256:
        raise ValueError(f"cols must be a positive multiple of 256, got {cols}")
    scale_cols = cols // _GROUP
    groups = rows * scale_cols
    blocks = (groups + _QUANT_THREADS - 1) // _QUANT_THREADS
    scale_tile_bytes = scale_cols * 32

    @flyc.kernel(known_block_size=[_QUANT_THREADS, 1, 1])
    def quantize_kernel(source: Int64, output: Int64, scale_output: Int64):
        group = fx.block_idx.x * fx.Int32(_QUANT_THREADS) + fx.thread_idx.x
        if group < fx.Int32(groups):
            source_rsrc = rsrc(source)
            output_rsrc = rsrc(output)
            scale_rsrc = rsrc(scale_output)
            source_dw = group * fx.Int32(_GROUP * 2 // 4)
            values = []
            local_max = fx.Float32(1e-10)
            for chunk in range_constexpr(_GROUP // 8):
                raw = fx.Vector(
                    bo.buffer_load(
                        source_rsrc,
                        source_dw + fx.Int32(chunk * 4),
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                chunk_values = raw.bitcast(fx.BFloat16).to(fx.Float32)
                local_max = fx.max(local_max, fmath.absf(chunk_values).reduce(ReductionOp.MAX))
                for element in range_constexpr(8):
                    values.append(chunk_values[element])

            working = (local_max * fx.Int32(_FP8_INV_MAX_POS_BITS).bitcast(fx.Float32)).bitcast(fx.Int32)
            exponent = (working >> fx.Int32(23)) & fx.Int32(0xFF)
            round_up = ((working & fx.Int32(0x400000)) != 0) & (
                ((working & fx.Int32(0x200000)) != 0) | ((working & fx.Int32(0x1FFFFF)) != 0) | (exponent > 0)
            )
            e8m0 = exponent + round_up.select(fx.Int32(1), fx.Int32(0))
            e8m0 = fx.min(e8m0, fx.Int32(255))
            quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)

            output_dw = group * fx.Int32(_GROUP // 4)
            words = []
            for word in range_constexpr(_GROUP // 4):
                base = word * 4
                value0 = fx.min(fx.max(values[base] * quant_scale, -448.0), 448.0)
                value1 = fx.min(fx.max(values[base + 1] * quant_scale, -448.0), 448.0)
                value2 = fx.min(fx.max(values[base + 2] * quant_scale, -448.0), 448.0)
                value3 = fx.min(fx.max(values[base + 3] * quant_scale, -448.0), 448.0)
                packed = rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    value0,
                    value1,
                    fx.Int32(0),
                    False,
                )
                packed = rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    value2,
                    value3,
                    packed,
                    True,
                )
                words.append(packed)
            bo.buffer_store(fx.Vector.from_elements(words[:4], fx.Int32), output_rsrc, output_dw)
            bo.buffer_store(fx.Vector.from_elements(words[4:], fx.Int32), output_rsrc, output_dw + 4)

            row = group // fx.Int32(scale_cols)
            scale_col = group % fx.Int32(scale_cols)
            row_group = row >> fx.Int32(5)
            row_pair = (row >> fx.Int32(4)) & fx.Int32(1)
            row_lane = row & fx.Int32(15)
            col_group = scale_col >> fx.Int32(3)
            col_pair = (scale_col >> fx.Int32(2)) & fx.Int32(1)
            col_lane = scale_col & fx.Int32(3)
            scale_offset = (
                row_group * fx.Int32(scale_tile_bytes)
                + col_group * fx.Int32(256)
                + col_lane * fx.Int32(64)
                + row_lane * fx.Int32(4)
                + col_pair * fx.Int32(2)
                + row_pair
            )
            bo.buffer_store(e8m0.to(fx.Uint8), scale_rsrc, scale_offset, offset_is_bytes=True)

    @flyc.jit
    def launch(source: Int64, output: Int64, scale_output: Int64, stream: Stream = Stream(None)):
        quantize_kernel(source, output, scale_output).launch(
            grid=(blocks, 1, 1),
            block=(_QUANT_THREADS, 1, 1),
            stream=stream,
        )

    launch.func.__name__ = f"mxfp8_quantize_r{rows}_c{cols}"
    return launch


@functools.cache
def build_mxfp8_project(rows: int, n: int, k: int):
    """Build a low-row MXFP8 GEMM specialized for decode projections."""

    if rows not in {1, 2, 4, 8}:
        raise ValueError(f"rows must be one of {{1, 2, 4, 8}}, got {rows}")
    if n <= 0 or n % 32:
        raise ValueError(f"output width must be a positive multiple of 32, got {n}")
    if k <= 0 or k % 256:
        raise ValueError(f"K must be a positive multiple of 256, got {k}")

    n_tiles = n // _N_TILE
    blocks = (n_tiles + _PROJECT_WAVES - 1) // _PROJECT_WAVES
    k_chunks = k // 64
    k_scale_chunks = k // 256
    activation_words = rows * k // 4

    @fx.struct
    class SharedStorage:
        activation: fx.Array[fx.Float32, activation_words, 16]

    @flyc.kernel(known_block_size=[_PROJECT_THREADS, 1, 1])
    def project_kernel(
        activation: Int64,
        activation_scale: Int64,
        weight: Int64,
        weight_scale: Int64,
        output: Int64,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        lane = tid % _WAVE_SIZE
        wave = tid // _WAVE_SIZE
        lane_div16 = lane // 16
        lane_mod16 = lane % 16

        activation_rsrc = rsrc(activation)
        activation_scale_rsrc = rsrc(activation_scale)
        weight_rsrc = rsrc(weight)
        weight_scale_rsrc = rsrc(weight_scale)
        output_rsrc = rsrc(output)
        staged = fx.SharedAllocator().allocate(SharedStorage).peek().activation.ptr

        loads = (activation_words + 4 * _PROJECT_THREADS - 1) // (4 * _PROJECT_THREADS)
        for load_index in range_constexpr(loads):
            word = (tid + load_index * _PROJECT_THREADS) * 4
            if word < activation_words:
                values = fx.Vector(
                    bo.buffer_load(
                        activation_rsrc,
                        word,
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                fx.ptr_store(values.bitcast(fx.Float32), staged + word)
        gpu.barrier()

        row_tile = bid * _PROJECT_WAVES + wave
        if (wave < _PROJECT_WAVES) & (row_tile < n_tiles):
            scale_atoms = [
                fx.make_mma_atom(
                    fx.rocdl.cdna4.MFMA_Scale(
                        16,
                        16,
                        128,
                        fx.Float8E4M3FN,
                        opsel_a=opsel,
                        opsel_b=opsel,
                    )
                )
                for opsel in (0, 2)
            ]
            accumulator = fx.make_rmem_tensor(4, fx.Float32)
            accumulator.store(fx.Vector.filled(4, 0.0, fx.Float32))
            valid_row = lane_mod16 < rows
            input_row = fx.min(lane_mod16, rows - 1)
            scale_lane = lane_div16 * 16 + lane_mod16

            for k256 in range_constexpr(k_scale_chunks):
                input_scale = fx.Int32(
                    bo.buffer_load(
                        activation_scale_rsrc,
                        k256 * 64 + scale_lane,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                output_scale = fx.Int32(
                    bo.buffer_load(
                        weight_scale_rsrc,
                        ((row_tile // 2) * k_scale_chunks + k256) * 64 + scale_lane,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                output_scale = ((row_tile % 2) != 0).select(
                    output_scale.shrui(fx.Int32(8)),
                    output_scale,
                )
                for k128_half in range_constexpr(2):
                    k128 = k256 * 2 + k128_half
                    k_base = k128 * 128 + lane_div16 * 16
                    activation_halves = []
                    for k64_half in range_constexpr(2):
                        loaded = fx.Vector(
                            fx.ptr_load(
                                staged + (input_row * k + k_base + k64_half * 64) // 4,
                                result_type=fx.Vector.make_type(4, fx.Float32),
                            )
                        ).bitcast(fx.Int32)
                        activation_halves.append(
                            fx.Vector.from_elements(
                                [valid_row.select(loaded[index], fx.Int32(0)) for index in range(4)],
                                fx.Int32,
                            )
                        )
                    activation_fragment = fx.make_rmem_tensor(8, fx.Int32)
                    activation_fragment.store(activation_halves[0].shuffle(activation_halves[1], list(range(8))))
                    weight_halves = []
                    for k64_half in range_constexpr(2):
                        k64 = k128 * 2 + k64_half
                        weight_halves.append(
                            fx.Vector(
                                bo.buffer_load(
                                    weight_rsrc,
                                    (((row_tile * k_chunks + k64) * 4 + lane_div16) * 16 + lane_mod16) * 4,
                                    vec_width=4,
                                    dtype=T.i32,
                                )
                            )
                        )
                    weight_fragment = fx.make_rmem_tensor(8, fx.Int32)
                    weight_fragment.store(weight_halves[0].shuffle(weight_halves[1], list(range(8))))
                    fx.gemm(
                        scale_atoms[k128_half],
                        accumulator,
                        activation_fragment,
                        weight_fragment,
                        accumulator,
                        scale_a=input_scale,
                        scale_b=output_scale,
                    )

            values = accumulator.load()
            output_row_base = lane_div16 * 4
            for element in range_constexpr(4):
                output_row = output_row_base + element
                if output_row < rows:
                    bo.buffer_store(
                        values[element].to(fx.BFloat16),
                        output_rsrc,
                        output_row * n + row_tile * _N_TILE + lane_mod16,
                    )

    @flyc.jit
    def launch(
        activation: Int64,
        activation_scale: Int64,
        weight: Int64,
        weight_scale: Int64,
        output: Int64,
        stream: Stream = Stream(None),
    ):
        project_kernel(
            activation,
            activation_scale,
            weight,
            weight_scale,
            output,
            value_attrs={"rocdl.flat_work_group_size": f"{_PROJECT_THREADS},{_PROJECT_THREADS}"},
        ).launch(grid=(blocks, 1, 1), block=(_PROJECT_THREADS, 1, 1), stream=stream)

    launch.func.__name__ = f"mxfp8_project_r{rows}_n{n}_k{k}"
    return launch


class Mxfp8Linear:
    """Graph-safe MXFP8 linear operator with caller-owned output."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        rows: int,
        *,
        workspace_only: bool = False,
    ) -> None:
        if weight.ndim != 2:
            raise ValueError(f"MXFP8 weight must be a matrix, got {tuple(weight.shape)}")
        self.n, self.k = weight.shape
        if scale.shape != (self.n, self.k // _GROUP):
            raise ValueError(f"MXFP8 scale must have shape {(self.n, self.k // _GROUP)}, got {tuple(scale.shape)}")
        self.rows = rows
        self.weight = pack_mxfp8_weight(weight)
        self.scale = pack_mxfp8_scale(scale)
        self._initialize_workspace(rows, weight.device, workspace_only)

    @classmethod
    def from_packed(
        cls,
        weight: torch.Tensor,
        scale: torch.Tensor,
        *,
        n: int,
        k: int,
        rows: int,
        workspace_only: bool = False,
    ) -> "Mxfp8Linear":
        """Reuse immutable packed weights with a bucket-local activation workspace."""

        if weight.dtype is not torch.uint8 or not weight.is_contiguous():
            raise ValueError("packed MXFP8 weight must be contiguous uint8")
        if scale.dtype is not torch.uint8 or not scale.is_contiguous():
            raise ValueError("packed MXFP8 scale must be contiguous uint8")
        if weight.numel() != n * k:
            raise ValueError(f"packed MXFP8 weight needs {n * k} bytes")
        padded_n = (n + 255) // 256 * 256
        if scale.numel() != padded_n * (k // _GROUP):
            raise ValueError("packed MXFP8 scale has the wrong size")
        instance = cls.__new__(cls)
        instance.n, instance.k, instance.rows = n, k, rows
        instance.weight, instance.scale = weight, scale
        instance._initialize_workspace(rows, weight.device, workspace_only)
        return instance

    def _initialize_workspace(
        self,
        rows: int,
        device: torch.device,
        workspace_only: bool = False,
    ) -> None:
        if workspace_only and rows not in {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError(f"unsupported MXFP8 workspace row count {rows}")
        padded_rows = (rows + 31) // 32 * 32
        self.padded_rows = padded_rows
        self.activation = torch.zeros(
            (padded_rows, self.k),
            dtype=torch.uint8,
            device=device,
        )
        self.activation_scale = torch.zeros(
            padded_rows * (self.k // _GROUP),
            dtype=torch.uint8,
            device=device,
        )
        self.quantize = (
            None if workspace_only else build_mxfp8_quantize(rows, self.k)
        )
        self.project = (
            None if workspace_only else build_mxfp8_project(rows, self.n, self.k)
        )

    def __call__(self, source: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if source.shape != (self.rows, self.k) or source.dtype != torch.bfloat16:
            raise ValueError(f"source must be BF16 [{self.rows}, {self.k}]")
        if not source.is_contiguous():
            raise ValueError("MXFP8 linear source must be contiguous")
        self.quantize_input(source)
        return self.project_quantized(self.activation, self.activation_scale, output)

    def quantize_input(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize one input matrix into the reusable row-major/preshuffled operands."""

        if source.shape != (self.rows, self.k) or source.dtype != torch.bfloat16:
            raise ValueError(f"source must be BF16 [{self.rows}, {self.k}]")
        if not source.is_contiguous():
            raise ValueError("MXFP8 linear source must be contiguous")
        stream = torch.cuda.current_stream()
        self.quantize(
            source.data_ptr(),
            self.activation.data_ptr(),
            self.activation_scale.data_ptr(),
            stream=stream,
        )
        return self.activation, self.activation_scale

    def project_quantized(
        self,
        activation: torch.Tensor,
        activation_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Project already-quantized rows, enabling reuse across independent weights."""

        if activation.shape != (self.padded_rows, self.k) or activation.dtype != torch.uint8:
            raise ValueError(f"activation must be uint8 [{self.padded_rows}, {self.k}]")
        if activation_scale.numel() != self.activation_scale.numel() or activation_scale.dtype != torch.uint8:
            raise ValueError("activation_scale has the wrong packed size or dtype")
        if output.shape != (self.rows, self.n) or output.dtype != torch.bfloat16:
            raise ValueError(f"output must be BF16 [{self.rows}, {self.n}]")
        if not activation.is_contiguous() or not activation_scale.is_contiguous() or not output.is_contiguous():
            raise ValueError("MXFP8 linear operands must be contiguous")
        self.project(
            activation.data_ptr(),
            activation_scale.data_ptr(),
            self.weight.data_ptr(),
            self.scale.data_ptr(),
            output.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        return output
