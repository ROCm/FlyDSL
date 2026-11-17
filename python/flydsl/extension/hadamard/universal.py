# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Portable warp- and block-scope Fast Walsh-Hadamard Transform (FWHT)."""

import math
from collections.abc import Callable, Sequence

from ...compiler.backends import compile_backend_name, current_target
from ...expr.gpu import barrier, lane_id, num_warp_threads, shuffle_xor, thread_idx
from ...expr.numeric import BFloat16, Float16, Float32, Int32, Numeric

# Disabled until ROCm/FlyDSL#1376 (llvm/llvm-project#231721) lands; see warp_fwht_e4m3x4 below.
# from ...expr.numeric import Uint32
from ...expr.primitive import range_constexpr
from ...expr.struct import Empty, Struct
from ...expr.typing import Array, Vector
from ...expr.utils.arith import ArithValue

__all__ = [
    "BlockFWHT",
    "warp_fwht",
    # Disabled until ROCm/FlyDSL#1376 (llvm/llvm-project#231721) lands; see below.
    # "warp_fwht_e4m3x4",
    # "warp_fwht_e5m2x4",
]

_SUPPORTED_DTYPES = (Float32, Float16, BFloat16)
_FWHTValue = Numeric | Vector | tuple[Numeric, ...] | list[Numeric] | ArithValue
_FWHTOutput = Numeric | Vector | tuple[Numeric, ...] | list[Numeric]


def _require_power_of_two(value: int, what: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or (value & (value - 1)):
        raise ValueError(f"{what} must be a positive power-of-two Python int, got {value!r}")


def _resolve_warp_width(width: int | None) -> int:
    warp_threads = num_warp_threads()
    if width is None:
        return warp_threads
    _require_power_of_two(width, "warp_fwht width")
    if width > warp_threads:
        raise ValueError(f"warp_fwht width must not exceed the target's {warp_threads}-lane warp, got {width}")
    return width


def _validate_dtype(dtype: object) -> None:
    if not isinstance(dtype, type) or not issubclass(dtype, _SUPPORTED_DTYPES):
        supported = ", ".join(t.__name__ for t in _SUPPORTED_DTYPES)
        raise TypeError(f"FWHT requires dtype in ({supported}), got {getattr(dtype, '__name__', dtype)!r}")


def _normalize_scalar(value: object) -> object:
    if isinstance(value, ArithValue) and not isinstance(value, Vector):
        dtype = Numeric.from_ir_type(value.type)
        _validate_dtype(dtype)
        return dtype(value)
    return value


def _unpack_items(
    value: _FWHTValue,
    expected_dtype: type[Numeric] | None = None,
) -> tuple[list[Float32], type[Numeric], str]:
    """Extract scalar float items into a list of ``Float32`` registers plus shape metadata."""
    norm_val = _normalize_scalar(value)
    if isinstance(norm_val, Vector):
        elem_dtype = norm_val.dtype
        _validate_dtype(elem_dtype)
        if expected_dtype is not None and elem_dtype is not expected_dtype:
            raise TypeError(f"expected {expected_dtype.__name__} elements, got {elem_dtype.__name__}")
        count = len(norm_val)
        _require_power_of_two(count, "FWHT items_per_thread")
        return [Float32(norm_val[i]) for i in range_constexpr(count)], elem_dtype, "vector"

    if isinstance(norm_val, (tuple, list)):
        if not norm_val:
            raise ValueError("FWHT item sequence must be nonempty")
        count = len(norm_val)
        _require_power_of_two(count, "FWHT items_per_thread")
        first = _normalize_scalar(norm_val[0])
        if not isinstance(first, Numeric):
            raise TypeError(f"FWHT requires Float32, Float16, or BFloat16 items, got {type(first).__name__}")
        elem_dtype = first.dtype if expected_dtype is None else expected_dtype
        _validate_dtype(elem_dtype)
        regs: list[Float32] = []
        for raw_item in norm_val:
            item = _normalize_scalar(raw_item)
            if not isinstance(item, Numeric) or item.dtype is not elem_dtype:
                raise TypeError(f"all FWHT items must have dtype {elem_dtype.__name__}")
            regs.append(Float32(item))
        return regs, elem_dtype, "tuple" if isinstance(norm_val, tuple) else "list"

    if isinstance(norm_val, Numeric):
        elem_dtype = norm_val.dtype
        _validate_dtype(elem_dtype)
        if expected_dtype is not None and elem_dtype is not expected_dtype:
            raise TypeError(f"expected {expected_dtype.__name__}, got {elem_dtype.__name__}")
        return [Float32(norm_val)], elem_dtype, "scalar"

    raise TypeError(f"FWHT requires a Float32/Float16/BFloat16 scalar or sequence, got {type(norm_val).__name__}")


def _repack_items(regs_f32: list[Float32], elem_dtype: type[Numeric], container_kind: str) -> _FWHTOutput:
    casted: list[Numeric] = [r if elem_dtype is Float32 else elem_dtype(r) for r in regs_f32]
    if container_kind == "scalar":
        return casted[0]
    if container_kind == "vector":
        return Vector.from_elements(casted, dtype=elem_dtype)
    if container_kind == "tuple":
        return tuple(casted)
    return casted


def _local_fwht_f32(regs: list[Float32]) -> list[Float32]:
    """In-register Sylvester Walsh-Hadamard butterfly over a power-of-two register list."""
    count = len(regs)
    out = list(regs)
    stride = 1
    while stride < count:
        step = stride << 1
        for base in range(0, count, step):
            for j in range(stride):
                a = out[base + j]
                b = out[base + j + stride]
                out[base + j] = a + b
                out[base + j + stride] = a - b
        stride = step
    return out


def _apply_butterfly_stages_f32(
    regs: list[Float32],
    width: int,
    xor_exchange_fn: Callable[[Float32, int, int], Float32],
) -> list[Float32]:
    """Run cross-lane Sylvester butterflies using ``xor_exchange_fn(val_f32, offset, width)``."""
    out = list(regs)
    if width == 1:
        return out
    lid = lane_id()
    offset = 1
    while offset < width:
        sign = ((lid & Int32(offset)) == Int32(0)).select(Float32(1.0), Float32(-1.0))
        for j in range(len(out)):
            partner = xor_exchange_fn(out[j], offset, width)
            out[j] = partner + sign * out[j]
        offset <<= 1
    return out


def _portable_xor_exchange(val: Float32, offset: int, width: int) -> Float32:
    return shuffle_xor(val, offset, width)


def warp_fwht(
    value: _FWHTValue,
    *,
    width: int | None = None,
    normalize: bool = False,
) -> _FWHTOutput:
    """Compute the Sylvester Fast Walsh-Hadamard Transform across a logical warp.

    Supports power-of-two logical warp widths ``width <= num_warp_threads()`` and
    either one scalar or a power-of-two blocked item sequence per lane, giving total
    transform size ``N = width * items_per_lane``. Butterfly arithmetic runs in
    ``Float32`` to prevent ``Float16`` intermediate overflow and avoid emulated
    scalar ``BFloat16`` arithmetic before casting back to the input element type.

    Args:
        value: ``Float32``, ``Float16``, or ``BFloat16`` scalar, ``Vector``, ``tuple``,
            or ``list`` of power-of-two length owned by this lane in blocked order.
        width: Compile-time power-of-two logical warp width, or ``None`` for the
            target's native warp width.
        normalize: When ``True``, scales the result by ``1 / sqrt(N)`` so the transform
            is orthonormal and self-inverse (``H(H(x)) == x``).

    Returns:
        Transformed value with the same element dtype and container shape as ``value``.

    Raises:
        TypeError: If ``value`` does not hold ``Float32``, ``Float16``, or ``BFloat16``
            elements.
        ValueError: If ``width`` or the per-lane item count is not a positive power of
            two, or if ``width`` exceeds ``num_warp_threads()``.
    """
    resolved_width = _resolve_warp_width(width)
    regs, elem_dtype, kind = _unpack_items(value)
    regs = _local_fwht_f32(regs)
    regs = _apply_butterfly_stages_f32(regs, resolved_width, _portable_xor_exchange)
    if normalize:
        scale = Float32(1.0 / math.sqrt(resolved_width * len(regs)))
        regs = [r * scale for r in regs]
    return _repack_items(regs, elem_dtype, kind)


# NOTE: Packed OCP FP8 warp Hadamard transforms (``warp_fwht_e4m3x4`` and
# ``warp_fwht_e5m2x4``) are intentionally disabled until upstream LLVM PR
# https://github.com/llvm/llvm-project/pull/231721 lands in FlyDSL's pinned LLVM
# commit (tracked in https://github.com/ROCm/FlyDSL/issues/1376).
# Their ROCDL counterparts in ``hadamard/rocdl.py`` unpack four OCP FP8/BF8 bytes
# per lane via 2-instruction ``cvt_pk_f32_{fp8,bf8}``, which in unmodified LLVM
# (``AMDGPU/VOP1Instructions.td``) defines ``HasCvtFP8ByteSel`` ISel patterns only
# for ``V_CVT_PK_F32_{FP8,BF8}_fake16`` and omits ``UseRealTrue16Insts`` (``_t16``)
# patterns, failing MC assembly on ``+real-true16`` targets (``gfx120x``/``gfx125x``).
#
# _F32_ABS_MASK = 0x7FFFFFFF
# _F32_EXP_INF_BITS = 0x7F800000
# _F32_QNAN_BITS = 0x7FC00000
# _E4M3_SUBNORM_MAGIC_F32 = 24576.0
# _E4M3_SUBNORM_MAGIC_BITS = 0x46400000
# _E4M3_MIN_NORM_BITS = 0x3C800000
# _E4M3_RNE_BIAS_BITS = 0x0007FFFF
# _E4M3_SAT_THRESHOLD_BITS = 0x43E00000
# _E4M3_OVF_RNE_BITS = 0x43E80000
# _E5M2_SUBNORM_MAGIC_F32 = 192.0
# _E5M2_SUBNORM_MAGIC_BITS = 0x43400000
# _E5M2_MIN_NORM_BITS = 0x38800000
# _E5M2_RNE_BIAS_BITS = 0x000FFFFF
# _E5M2_SAT_THRESHOLD_BITS = 0x47600000
# _E5M2_OVF_RNE_BITS = 0x47700000
#
#
# def _unpack_ocp_e4m3_byte_f32(byte_val: Uint32) -> Float32:
#     b = byte_val & Uint32(0xFF)
#     sign = (b & Uint32(0x80)) << Uint32(24)
#     mag = b & Uint32(0x7F)
#     exp = mag >> Uint32(3)
#     mant = mag & Uint32(0x7)
#     norm_bits = sign | ((exp + Uint32(120)) << Uint32(23)) | (mant << Uint32(20))
#     sub_bits = (Float32(Int32(mant)) * Float32(1.0 / 512.0)).bitcast(Uint32) | sign
#     out_bits = (exp == Uint32(0)).select(sub_bits, norm_bits)
#     out_bits = (mag == Uint32(0)).select(sign, out_bits)
#     out_bits = (mag == Uint32(0x7F)).select(sign | Uint32(_F32_QNAN_BITS), out_bits)
#     return out_bits.bitcast(Float32)
#
#
# def _unpack_ocp_e5m2_byte_f32(byte_val: Uint32) -> Float32:
#     b = byte_val & Uint32(0xFF)
#     sign = (b & Uint32(0x80)) << Uint32(24)
#     mag = b & Uint32(0x7F)
#     exp = mag >> Uint32(2)
#     mant = mag & Uint32(0x3)
#     norm_bits = sign | ((exp + Uint32(112)) << Uint32(23)) | (mant << Uint32(21))
#     sub_bits = (Float32(Int32(mant)) * Float32(1.0 / 65536.0)).bitcast(Uint32) | sign
#     out_bits = (exp == Uint32(0)).select(sub_bits, norm_bits)
#     out_bits = (mag == Uint32(0)).select(sign, out_bits)
#     out_bits = (mag == Uint32(0x7C)).select(sign | Uint32(_F32_EXP_INF_BITS), out_bits)
#     out_bits = (mag > Uint32(0x7C)).select(sign | Uint32(_F32_QNAN_BITS), out_bits)
#     return out_bits.bitcast(Float32)
#
#
# def _pack_ocp_e4m3_byte_u32(val: Float32, *, saturate: bool) -> Uint32:
#     bits = val.bitcast(Uint32)
#     sign = (bits >> Uint32(24)) & Uint32(0x80)
#     abs_bits = bits & Uint32(_F32_ABS_MASK)
#     is_nan = abs_bits > Uint32(_F32_EXP_INF_BITS)
#     sub_f32 = abs_bits.bitcast(Float32) + Float32(_E4M3_SUBNORM_MAGIC_F32)
#     sub_code = sub_f32.bitcast(Uint32) - Uint32(_E4M3_SUBNORM_MAGIC_BITS)
#     lsb = (abs_bits >> Uint32(20)) & Uint32(1)
#     norm_code = ((abs_bits + Uint32(_E4M3_RNE_BIAS_BITS) + lsb) >> Uint32(20)) - Uint32(120 << 3)
#     mag = (abs_bits < Uint32(_E4M3_MIN_NORM_BITS)).select(sub_code, norm_code)
#     if saturate:
#         mag = (abs_bits >= Uint32(_E4M3_SAT_THRESHOLD_BITS)).select(Uint32(0x7E), mag)
#     else:
#         mag = (abs_bits > Uint32(_E4M3_OVF_RNE_BITS)).select(Uint32(0x7F), mag)
#     mag = is_nan.select(Uint32(0x7F), mag)
#     return sign | (mag & Uint32(0x7F))
#
#
# def _pack_ocp_e5m2_byte_u32(val: Float32, *, saturate: bool) -> Uint32:
#     bits = val.bitcast(Uint32)
#     sign = (bits >> Uint32(24)) & Uint32(0x80)
#     abs_bits = bits & Uint32(_F32_ABS_MASK)
#     is_nan = abs_bits > Uint32(_F32_EXP_INF_BITS)
#     sub_f32 = abs_bits.bitcast(Float32) + Float32(_E5M2_SUBNORM_MAGIC_F32)
#     sub_code = sub_f32.bitcast(Uint32) - Uint32(_E5M2_SUBNORM_MAGIC_BITS)
#     lsb = (abs_bits >> Uint32(21)) & Uint32(1)
#     norm_code = ((abs_bits + Uint32(_E5M2_RNE_BIAS_BITS) + lsb) >> Uint32(21)) - Uint32(112 << 2)
#     mag = (abs_bits < Uint32(_E5M2_MIN_NORM_BITS)).select(sub_code, norm_code)
#     if saturate:
#         mag = (abs_bits >= Uint32(_E5M2_SAT_THRESHOLD_BITS)).select(Uint32(0x7B), mag)
#     else:
#         mag = (abs_bits >= Uint32(_E5M2_OVF_RNE_BITS)).select(Uint32(0x7C), mag)
#     mag = is_nan.select(Uint32(0x7E), mag)
#     return sign | (mag & Uint32(0x7F))
#
#
# def _warp_fwht_fp8x4_portable(
#     packed: Int32 | Uint32 | int | ArithValue,
#     *,
#     is_bf8: bool,
#     width: int | None,
#     normalize: bool,
#     saturate: bool,
# ) -> Uint32:
#     resolved_width = _resolve_warp_width(width)
#     if isinstance(packed, Uint32):
#         word = packed
#     elif isinstance(packed, Int32):
#         word = packed.bitcast(Uint32)
#     else:
#         word = Uint32(packed)
#     decode = _unpack_ocp_e5m2_byte_f32 if is_bf8 else _unpack_ocp_e4m3_byte_f32
#     encode = _pack_ocp_e5m2_byte_u32 if is_bf8 else _pack_ocp_e4m3_byte_u32
#     regs = [decode(word >> Uint32(8 * j)) for j in range(4)]
#     regs = _local_fwht_f32(regs)
#     regs = _apply_butterfly_stages_f32(regs, resolved_width, _portable_xor_exchange)
#     if normalize:
#         scale = Float32(1.0 / math.sqrt(4 * resolved_width))
#         regs = [r * scale for r in regs]
#     b0 = encode(regs[0], saturate=saturate)
#     b1 = encode(regs[1], saturate=saturate)
#     b2 = encode(regs[2], saturate=saturate)
#     b3 = encode(regs[3], saturate=saturate)
#     return b0 | (b1 << Uint32(8)) | (b2 << Uint32(16)) | (b3 << Uint32(24))
#
#
# def warp_fwht_e4m3x4(
#     packed: Int32 | Uint32 | int | ArithValue,
#     *,
#     width: int | None = None,
#     normalize: bool = True,
#     saturate: bool = True,
# ) -> Uint32:
#     """Compute an in-register ``4 * width`` FWHT on four packed OCP ``E4M3FN`` bytes per lane."""
#     return _warp_fwht_fp8x4_portable(packed, is_bf8=False, width=width, normalize=normalize, saturate=saturate)
#
#
# def warp_fwht_e5m2x4(
#     packed: Int32 | Uint32 | int | ArithValue,
#     *,
#     width: int | None = None,
#     normalize: bool = True,
#     saturate: bool = True,
# ) -> Uint32:
#     """Compute an in-register ``4 * width`` FWHT on four packed OCP ``E5M2`` bytes per lane."""
#     return _warp_fwht_fp8x4_portable(packed, is_bf8=True, width=width, normalize=normalize, saturate=saturate)


def _normalize_block_shape(block_size: int | Sequence[int]) -> tuple[int, int, int]:
    dims = tuple(block_size) if isinstance(block_size, (tuple, list)) else (block_size, 1, 1)
    if len(dims) != 3:
        raise TypeError(f"block_size must be an int or a 3-tuple, got {block_size!r}")
    for name, dim in zip(("block_dim_x", "block_dim_y", "block_dim_z"), dims):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise TypeError(f"{name} must be a positive Python int, got {dim!r}")
    return (dims[0], dims[1], dims[2])


def _linear_tid(block_size: tuple[int, int, int]) -> Int32:
    dim_x, dim_y, dim_z = block_size
    tid = thread_idx.x
    if dim_y > 1 or dim_z > 1:
        tid = tid + thread_idx.y * dim_x + thread_idx.z * (dim_x * dim_y)
    return tid


class _BlockFWHTMeta(type):
    """Compile-time specialization metaclass for ``BlockFWHT[dtype, block_size, items_per_thread=1]``."""

    _cache: dict[tuple[object, ...], type["BlockFWHT"]] = {}

    def __getitem__(cls, params: object) -> type["BlockFWHT"]:
        if cls.block_threads is not None:
            raise TypeError(f"{cls.__name__} is already specialized")
        param_tuple = params if isinstance(params, tuple) else (params,)
        if not 2 <= len(param_tuple) <= 3:
            raise TypeError(f"{cls.__name__}[dtype, block_size, items_per_thread=1]")
        dtype, raw_block_size = param_tuple[0], param_tuple[1]
        items_per_thread = param_tuple[2] if len(param_tuple) == 3 else 1
        _validate_dtype(dtype)
        block_size = _normalize_block_shape(raw_block_size)
        block_threads = block_size[0] * block_size[1] * block_size[2]
        _require_power_of_two(block_threads, "BlockFWHT block_size thread count")
        if block_threads > 1024:
            raise ValueError(f"BlockFWHT block_size must not exceed 1024 threads, got {block_threads}")
        _require_power_of_two(items_per_thread, "BlockFWHT items_per_thread")

        phys_warp = num_warp_threads()
        warp_threads = min(block_threads, phys_warp)
        num_warps = block_threads // warp_threads
        transform_size = block_threads * items_per_thread

        key = (cls, dtype, block_size, items_per_thread, warp_threads, current_target())
        cached = _BlockFWHTMeta._cache.get(key)
        if cached is not None:
            return cached

        if num_warps == 1:
            row_stride = 0
            storage_type: type = Empty
        else:
            # Pad each warp row by ``32 // num_warps`` floats (total +32 floats = 128 B
            # across the entire block) so both warp-contiguous and sub-warp transpose
            # accesses map all 32 lanes onto 32 distinct LDS banks.
            pad = max(1, 32 // num_warps)
            row_stride = warp_threads * items_per_thread + pad
            storage_type = Struct["slots" : Array[Float32, num_warps * row_stride]]

        specialized = type(
            f"{cls.__name__}[{dtype.__name__}, {block_threads}, {items_per_thread}]",
            (cls,),
            {
                "dtype": dtype,
                "block_size": block_size,
                "block_threads": block_threads,
                "items_per_thread": items_per_thread,
                "warp_threads": warp_threads,
                "num_warps": num_warps,
                "transform_size": transform_size,
                "_row_stride": row_stride,
                "SharedStorage": storage_type,
            },
        )
        _BlockFWHTMeta._cache[key] = specialized
        return specialized

    def __call__(
        cls,
        value: _FWHTValue,
        *,
        storage: Struct | Empty | None = None,
        normalize: bool = False,
    ) -> _FWHTOutput:
        return cls.fwht(value, storage=storage, normalize=normalize)


class BlockFWHT(metaclass=_BlockFWHTMeta):
    """Cooperative block-wide Sylvester Fast Walsh-Hadamard Transform.

    Specialize with ``BlockFWHT[dtype, block_size, items_per_thread=1]``.

    Args:
        dtype: ``Float32``, ``Float16``, or ``BFloat16``.
        block_size: Power-of-two block thread count (up to 1024), or 3-tuple ``(x, y, z)``
            matching ``fx.known_block_size()``.
        items_per_thread: Optional power-of-two item count owned by each thread in blocked
            order (defaults to ``1``). Total transform length is
            ``N = block_threads * items_per_thread``.

    Single-warp specializations (``block_threads <= num_warp_threads()``) expose
    ``SharedStorage = Empty`` (zero LDS bytes, zero barriers) and execute completely
    in registers. Multi-warp specializations factor ``H_N = kron(H_num_warps, H_warp_tile)``
    with a single bank-conflict-free padded LDS transpose (only ``(N + 32) * 4`` bytes
    and two barriers for any ``num_warps <= 32``).

    Attributes:
        dtype: Specialized element type (``Float32``, ``Float16``, or ``BFloat16``).
        block_size: Specialized ``(x, y, z)`` block shape.
        block_threads: Total number of participating threads in the block.
        items_per_thread: Number of blocked elements owned per thread.
        warp_threads: Logical warp width selected for the target and block.
        num_warps: Number of logical warps in the block.
        transform_size: Total transform length ``block_threads * items_per_thread``.
        SharedStorage: Shared-memory ``Struct`` type (or ``Empty`` for single-warp blocks)
            required by the specialization.

    Examples:
        H = fx.hadamard.BlockFWHT[fx.Float16, fx.known_block_size(), 4]
        storage = fx.SharedAllocator().allocate(H.SharedStorage).peek()
        y = H(x, storage=storage, normalize=True)
    """

    dtype: type[Numeric] | None = None
    block_size: tuple[int, int, int] | None = None
    block_threads: int | None = None
    items_per_thread: int | None = None
    warp_threads: int | None = None
    num_warps: int | None = None
    transform_size: int | None = None
    _row_stride: int = 0
    SharedStorage: type | None = None
    _dispatcher = None

    @classmethod
    def _resolve_warp_fn(cls) -> Callable[..., _FWHTOutput]:
        if cls._dispatcher is not None:
            return cls._dispatcher._resolve(compile_backend_name(), "warp_fwht", warp_fwht)
        return warp_fwht

    @classmethod
    def fwht(
        cls,
        value: _FWHTValue,
        *,
        storage: Struct | Empty | None = None,
        normalize: bool = False,
    ) -> _FWHTOutput:
        """Execute the cooperative block-wide FWHT on blocked per-thread input ``value``.

        Args:
            value: Per-thread scalar or ``items_per_thread``-element sequence matching
                ``cls.dtype`` in blocked order.
            storage: Instance of ``cls.SharedStorage`` (optional when ``num_warps == 1``).
            normalize: When ``True``, scales the result by ``1 / sqrt(transform_size)``.

        Returns:
            Transformed per-thread value with the same dtype and container shape as ``value``.

        Raises:
            TypeError: If ``BlockFWHT`` is unspecialized, ``storage`` does not match
                ``cls.SharedStorage``, or ``value`` does not match ``cls.dtype``.
            ValueError: If ``value`` length does not match ``cls.items_per_thread``.
        """
        if cls.block_threads is None:
            raise TypeError(f"specialize {cls.__name__} first, e.g. {cls.__name__}[fx.Float16, 128]")
        assert cls.num_warps is not None
        assert cls.items_per_thread is not None
        assert cls.warp_threads is not None
        assert cls.block_size is not None
        assert cls.transform_size is not None
        assert cls.SharedStorage is not None
        if cls.num_warps > 1:
            if storage is None or not isinstance(storage, cls.SharedStorage):
                raise TypeError(f"multi-warp {cls.__name__} requires storage allocated from SharedStorage")
        elif storage is not None and not isinstance(storage, cls.SharedStorage):
            raise TypeError(f"storage must be an instance of {cls.__name__}.SharedStorage")

        regs, elem_dtype, kind = _unpack_items(value, expected_dtype=cls.dtype)
        if len(regs) != cls.items_per_thread:
            raise ValueError(f"expected {cls.items_per_thread} items per thread, got {len(regs)}")

        warp_fn = cls._resolve_warp_fn()

        # Phase 1: intra-thread + intra-warp FWHT of size ``warp_threads * items_per_thread``.
        regs = list(warp_fn(regs, width=cls.warp_threads, normalize=False))

        # Phase 2: cross-warp butterfly via bank-conflict-free padded LDS transpose.
        if cls.num_warps > 1:
            tid = _linear_tid(cls.block_size)
            warp_id = tid // cls.warp_threads
            lane = tid % cls.warp_threads
            stride = cls._row_stride
            w_count = cls.num_warps

            for j in range_constexpr(cls.items_per_thread):
                storage.slots[warp_id * stride + j * cls.warp_threads + lane] = regs[j]
            barrier()

            col_group = tid // w_count
            sub_lane = tid % w_count
            transposed = [
                Float32(storage.slots[sub_lane * stride + j * cls.warp_threads + col_group])
                for j in range_constexpr(cls.items_per_thread)
            ]
            transposed = [warp_fn(t_val, width=w_count, normalize=False) for t_val in transposed]
            for j in range_constexpr(cls.items_per_thread):
                storage.slots[sub_lane * stride + j * cls.warp_threads + col_group] = transposed[j]
            barrier()

            regs = [
                Float32(storage.slots[warp_id * stride + j * cls.warp_threads + lane])
                for j in range_constexpr(cls.items_per_thread)
            ]

        if normalize:
            scale = Float32(1.0 / math.sqrt(cls.transform_size))
            regs = [r * scale for r in regs]

        return _repack_items(regs, elem_dtype, kind)
