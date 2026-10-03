# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Per-tensor FP8 quantize / dequantize for gfx120x (RDNA4).

Converts tensors between floating-point (bf16/fp16/fp32) and packed FP8 using
``cvt_pk_fp8_f32`` / ``cvt_f32_fp8`` (e4m3) or the bf8 variants (e5m2). Scale
is applied as exact ``1/scale`` (not reciprocal estimate) so round-trip error
stays within the FP8 format budget.

Packed I/O is ``i32`` holding four FP8 bytes. Host builders specialize on
input dtype and e5m2 flag. Microbench on an otherwise idle gfx120x measured about
2.5-3.3x vs the HIP baseline for typical activation shapes.
"""

from collections.abc import Callable
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from kernels.common.gfx120x_buf_helpers import buf_copy_load, buf_copy_store, kernel_signature, ptr_buf_tensor

KERNEL_NAME = "fp8_quant_gfx120x"
BLOCK = 256
VEC_HALF = 8
VEC_F32 = 4
_F8_E4M3_MAX = 448.0
_F8_E5M2_MAX = 57344.0


@lru_cache(maxsize=16)
def build_fp8_quant_module(
    in_dtype: str = "bfloat16", e5m2: bool = False, vec: int | None = None
) -> Callable[..., None]:
    """Build the per-tensor FP8 quant kernel."""
    InTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[in_dtype]
    in_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[in_dtype]
    if vec is None:
        vec = VEC_F32 if in_dtype == "float32" else VEC_HALF
    cvt_pk = fx.rocdl.cvt_pk_bf8_f32 if e5m2 else fx.rocdl.cvt_pk_fp8_f32
    n_words = vec // 4
    tile = BLOCK * vec
    sig = kernel_signature(block=BLOCK, dtype=in_dtype, e5m2=e5m2, vec=vec, op="quant")

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def quant_kernel(
        In: fx.Pointer,
        Out: fx.Pointer,
        Scale: fx.Pointer,
        n_elems: fx.Int32,
        lp_max: fx.Constexpr[float],
    ) -> None:
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
        in_buf = ptr_buf_tensor(
            In,
            elem=InTy,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=fx.Int64(n_elems) * fx.Int64(in_bytes),
        )
        out_buf = ptr_buf_tensor(
            Out,
            elem=fx.Int32,
            n=0x3FFFFFFF,
            unit_elems=n_words,
            num_records_bytes=fx.Int64(n_elems),
        )
        scale_buf = ptr_buf_tensor(
            Scale,
            elem=fx.Float32,
            n=1,
            unit_elems=1,
            num_records_bytes=4,
        )
        inv = fx.Float32(1.0) / fx.Float32(buf_copy_load(scale_buf, fx.Int32(0), elem=fx.Float32, unit_elems=1))
        v = buf_copy_load(in_buf, unit, elem=InTy, unit_elems=vec)
        if in_dtype != "float32":
            v = v.extf(T.vec(vec, T.f32))
        vmax = fx.Float32(lp_max)
        vmin = fx.Float32(-lp_max)
        safe = []
        for i in range_constexpr(vec):
            x = fx.max(fx.min(v[i] * inv, vmax), vmin)
            safe.append(x.ir_value())
        words = []
        for w in range_constexpr(n_words):
            pk = fx.Int32(0).ir_value()
            bi = w * 4
            pk = cvt_pk(T.i32, safe[bi + 0], safe[bi + 1], pk, 0)
            pk = cvt_pk(T.i32, safe[bi + 2], safe[bi + 3], pk, 1)
            words.append(fx.Int32(pk))
        if n_words == 1:
            buf_copy_store(out_buf, unit, words[0], elem=fx.Int32, unit_elems=1)
        else:
            buf_copy_store(
                out_buf,
                unit,
                fx.Vector.from_elements(words, fx.Int32),
                elem=fx.Int32,
                unit_elems=n_words,
            )

    quant_kernel.__name__ = f"{KERNEL_NAME}_{sig}"

    @flyc.jit
    def launch(
        In: fx.Pointer,
        Out: fx.Pointer,
        Scale: fx.Pointer,
        n_elems: fx.Int32,
        lp_max: fx.Constexpr[float],
        stream: fx.Stream,
    ) -> None:
        n64 = fx.Int64(n_elems)
        grid_x = (n64 + fx.Int64(tile - 1)) // fx.Int64(tile)
        quant_kernel(In, Out, Scale, n_elems, lp_max).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch.__name__ = f"launch_{KERNEL_NAME}_{sig}"
    return launch


@lru_cache(maxsize=16)
def build_fp8_dequant_module(
    out_dtype: str = "bfloat16", e5m2: bool = False, vec: int | None = None
) -> Callable[..., None]:
    """Build the per-tensor FP8 dequant kernel."""
    OutTy = {"float32": fx.Float32, "float16": fx.Float16, "bfloat16": fx.BFloat16}[out_dtype]
    out_bytes = {"float32": 4, "float16": 2, "bfloat16": 2}[out_dtype]
    if vec is None:
        vec = VEC_F32 if out_dtype == "float32" else VEC_HALF
    n_words = vec // 4
    tile = BLOCK * vec
    cvt = fx.rocdl.cvt_f32_bf8 if e5m2 else fx.rocdl.cvt_f32_fp8
    sig = kernel_signature(block=BLOCK, dtype=out_dtype, e5m2=e5m2, vec=vec, op="dequant")

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def dequant_kernel(In: fx.Pointer, Out: fx.Pointer, Scale: fx.Pointer, n_elems: fx.Int32) -> None:
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        unit = fx.Int32(bid) * fx.Int32(BLOCK) + fx.Int32(tid)
        in_buf = ptr_buf_tensor(
            In,
            elem=fx.Int32,
            n=0x3FFFFFFF,
            unit_elems=n_words,
            num_records_bytes=fx.Int64(n_elems),
        )
        out_buf = ptr_buf_tensor(
            Out,
            elem=OutTy,
            n=0x3FFFFFFF,
            unit_elems=vec,
            num_records_bytes=fx.Int64(n_elems) * fx.Int64(out_bytes),
        )
        scale_buf = ptr_buf_tensor(
            Scale,
            elem=fx.Float32,
            n=1,
            unit_elems=1,
            num_records_bytes=4,
        )
        s = fx.Float32(buf_copy_load(scale_buf, fx.Int32(0), elem=fx.Float32, unit_elems=1))
        outs = []
        if n_words == 1:
            packed = buf_copy_load(in_buf, unit, elem=fx.Int32, unit_elems=1)
            for bi in range_constexpr(4):
                f = fx.Float32(cvt(T.f32, packed.ir_value(), bi)) * s
                if out_dtype != "float32":
                    f = f.to(OutTy)
                outs.append(f)
        else:
            words_v = buf_copy_load(in_buf, unit, elem=fx.Int32, unit_elems=n_words)
            for w in range_constexpr(n_words):
                packed = words_v[w]
                for bi in range_constexpr(4):
                    f = fx.Float32(cvt(T.f32, packed.ir_value(), bi)) * s
                    if out_dtype != "float32":
                        f = f.to(OutTy)
                    outs.append(f)
        buf_copy_store(
            out_buf,
            unit,
            fx.Vector.from_elements(outs, OutTy),
            elem=OutTy,
            unit_elems=vec,
        )

    dequant_kernel.__name__ = f"{KERNEL_NAME}_deq_{sig}"

    @flyc.jit
    def launch(
        In: fx.Pointer,
        Out: fx.Pointer,
        Scale: fx.Pointer,
        n_elems: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        n64 = fx.Int64(n_elems)
        grid_x = (n64 + fx.Int64(tile - 1)) // fx.Int64(tile)
        dequant_kernel(In, Out, Scale, n_elems).launch(grid=(grid_x, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch.__name__ = f"launch_{KERNEL_NAME}_deq_{sig}"
    return launch
