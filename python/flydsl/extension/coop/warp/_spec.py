# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile-time specialization shared by warp operator classes."""

from ....compiler.backends import compile_backend_name, current_target
from ....expr.struct import Empty
from .._common import _convert_value, _resolve_warp_width
from .._values import _as_items, _leaf_fields

_CACHE = {}


class WarpPrimitiveMeta(type):
    """Specialize operators with the same dimension order as block operators.

    Scalar operators take ``[dtype, width=None]``. Tile operators take
    ``[dtype, width, items_per_thread]``, with an optional final algorithm only
    for operators that offer algorithm selection. ``None`` selects the target
    warp width. Batched reductions use the tile extent as their batch count.
    """

    def __getitem__(cls, params):
        if cls.warp_threads is not None:
            raise TypeError(f"{cls.__name__} is already specialized")
        params = params if isinstance(params, tuple) else (params,)
        minimum = 3 if cls._tile else 1
        maximum = 4 if cls._algorithms is not None else (3 if cls._tile else 2)
        if not minimum <= len(params) <= maximum:
            suffix = ", items_per_thread" if cls._tile else ""
            suffix += ", algorithm=..." if cls._algorithms is not None else ""
            raise TypeError(f"{cls.__name__}[dtype, width{suffix}]")
        dtype = params[0]
        value_dtype = None
        if isinstance(dtype, tuple):
            if not cls._pairs or len(dtype) != 2:
                raise TypeError(f"{cls.__name__} does not accept a key/value dtype pair")
            dtype, value_dtype = dtype
            _leaf_fields(value_dtype)
        _leaf_fields(dtype)
        width = params[1] if len(params) > 1 else None
        if isinstance(width, bool):
            raise TypeError("width must be a Python int or None")
        width = _resolve_warp_width(width, f"{cls.__name__} width")
        count = params[2] if cls._tile else None
        if cls._tile and (not isinstance(count, int) or isinstance(count, bool) or count < cls._minimum_items):
            raise ValueError(f"items_per_thread must be a Python int >= {cls._minimum_items}")
        algorithm = params[3] if len(params) == 4 else cls._default_algorithm
        if cls._algorithms is not None and not isinstance(algorithm, cls._algorithms):
            raise TypeError(f"expected a {cls._algorithms.__name__}, got {algorithm!r}")
        key = (cls, dtype, value_dtype, width, count, algorithm, current_target())
        if key not in _CACHE:
            specialized = type(
                f"{cls.__name__}[{dtype.__name__}, {width}"
                f"{', ' + str(count) if cls._tile else ''}"
                f"{', ' + algorithm.name if algorithm is not None else ''}]",
                (cls,),
                dict(
                    dtype=dtype,
                    key_dtype=dtype,
                    value_dtype=value_dtype,
                    warp_threads=width,
                    items_per_thread=count,
                    algorithm=algorithm,
                ),
            )
            specialized._validate_specialization()
            specialized.SharedStorage = specialized._make_storage()
            _CACHE[key] = specialized
        return _CACHE[key]

    def __call__(cls, *args, **kwargs):
        raise TypeError(f"use a specialized {cls.__name__} member function")


class WarpPrimitive(metaclass=WarpPrimitiveMeta):
    """Metadata and validation for a compile-time warp operator.

    Operator classes contain Python class methods, not runtime Struct methods.
    A specialization fixes the element type, logical width and any tile extent
    or algorithm. All lanes in that logical warp participate in each call.

    Attributes:
        dtype: Input element type, or the key type for pair sorting.
        key_dtype: Identical to dtype.
        value_dtype: Payload type for a pair sort; otherwise None.
        warp_threads: Power-of-two logical width within a physical warp.
        items_per_thread: Static tile extent, or None for scalar operators.
        algorithm: Selected policy, or None when there is no algorithm choice.
        SharedStorage: Scratch type for one logical warp. Register-only
            implementations expose Empty, which can be allocated directly.
            For non-empty scratch, allocate Array[SharedStorage, num_warps]
            and select the calling warp's element so concurrently active
            groups have separate scratch.
    """

    dtype = key_dtype = value_dtype = None
    warp_threads = items_per_thread = algorithm = SharedStorage = None
    _tile = _pairs = False
    _minimum_items = 1
    _algorithms = _default_algorithm = _dispatcher = None

    @classmethod
    def _validate_specialization(cls):
        pass

    @classmethod
    def _make_storage(cls):
        return Empty

    @classmethod
    def _check_storage(cls, storage):
        cls._check()
        if storage is not None and not isinstance(storage, cls.SharedStorage):
            raise TypeError(f"storage must be an instance of {cls.__name__}.SharedStorage")

    @classmethod
    def _check(cls):
        if cls.warp_threads is None:
            raise TypeError(f"specialize {cls.__name__} first")

    @classmethod
    def _prepare(cls, value, *, dtype=None):
        cls._check()
        if cls.items_per_thread == 0 and isinstance(value, (tuple, list)) and not value:
            return value
        items = _as_items(value)
        if cls.items_per_thread is not None and len(items) != cls.items_per_thread:
            raise ValueError(f"expected {cls.items_per_thread} items per thread, got {len(items)}")
        return _convert_value(value, cls.dtype if dtype is None else dtype)

    @classmethod
    def _invoke(cls, implementation, *args, storage=None, **kwargs):
        cls._check_storage(storage)
        if cls._dispatcher is not None:
            implementation = cls._dispatcher._resolve(compile_backend_name(), implementation.__name__, implementation)
        return implementation(*args, width=cls.warp_threads, **kwargs)
