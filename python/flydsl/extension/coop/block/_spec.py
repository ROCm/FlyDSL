# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared specialization machinery for the block-scope collectives.

Every block algorithm is specialized the same way and validates the same things,
so the parsing lives here once and each algorithm supplies only what differs:
its policy enum, its default policy, and the shared storage each policy needs.
"""

from ....compiler.backends import current_target
from ....expr.gpu import num_warp_threads

# Specializations are cached across algorithms, keyed by the root class as well
# as the parameters, so two algorithms cannot collide on an identical key.
_CACHE = {}


def _block_shape(block_size):
    """Normalize a block size to ``(x, y, z)``; a bare int means ``(x, 1, 1)``."""
    dims = tuple(block_size) if isinstance(block_size, (tuple, list)) else (block_size, 1, 1)
    if len(dims) != 3:
        raise TypeError(f"block_size must be an int or a 3-tuple, got {block_size!r}")
    for name, dim in zip(("block_dim_x", "block_dim_y", "block_dim_z"), dims):
        if not isinstance(dim, int) or dim < 1:
            raise TypeError(f"{name} must be a positive Python int, got {dim!r}")
    return dims


class BlockAlgorithmMeta(type):
    """Gives a block collective its ``[...]`` specialization syntax.

    The parameters are ``[dtype, block_size, algorithm]``, of which only the
    first two are required. *block_size* is either the x extent on its own —
    the y and z extents are then ``1`` — or the full ``(x, y, z)``, which is
    what :func:`~flydsl.expr.gpu.known_block_size` hands back.

    The product of the dimensions must be a positive multiple of the target
    physical warp size. A smaller block or an incomplete final warp is rejected.

    A concrete metaclass sets ``_algorithms`` and ``_shared_storage``, and
    implements :meth:`_default_algorithm_for`.
    """

    _algorithms = None
    _shared_storage = None
    _supports_pairs = False

    def _default_algorithm_for(cls, target):
        """The policy to use when the caller does not name one."""
        raise NotImplementedError(f"{cls.__name__} must implement _default_algorithm_for")

    def __getitem__(cls, params) -> type:
        if cls.block_threads is not None:
            raise TypeError(f"{cls.__name__} is already specialized")

        if not isinstance(params, tuple):
            params = (params,)
        if not 2 <= len(params) <= 3:
            raise TypeError(f"{cls.__name__}[dtype, block_size, algorithm=...]")

        dtype, block_size = params[0], params[1]
        algorithm = params[2] if len(params) > 2 else cls._default_algorithm_for(current_target())

        return cls._specialize(dtype, block_size, algorithm)

    def _specialize(cls, dtype, block_size, algorithm, items_per_thread=None):
        if not isinstance(algorithm, cls._algorithms):
            raise TypeError(f"expected a {cls._algorithms.__name__}, got {algorithm!r}")
        block_size = _block_shape(block_size)
        block_threads = block_size[0] * block_size[1] * block_size[2]
        warp_threads = num_warp_threads()
        if block_threads % warp_threads:
            raise ValueError(
                f"{cls.__name__} block_size must contain a multiple of the target warp size "
                f"({warp_threads}) threads, got {block_threads}"
            )
        key_dtype, value_dtype = dtype, None
        if isinstance(dtype, tuple):
            if not cls._supports_pairs or len(dtype) != 2:
                raise TypeError(f"{cls.__name__} does not accept this key/value dtype specification")
            key_dtype, value_dtype = dtype

        key = (cls, dtype, block_size, algorithm, warp_threads, items_per_thread, current_target())
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        make_storage = cls._shared_storage.get(algorithm)
        if make_storage is None:
            raise NotImplementedError(f"{cls._algorithms.__name__}.{algorithm.name} is not implemented yet")

        storage_args = (dtype, block_threads, warp_threads)
        if items_per_thread is not None:
            storage_args += (items_per_thread,)
        specialized = type(
            f"{cls.__name__}[{getattr(dtype, '__name__', dtype)}, {block_threads}, "
            f"{str(items_per_thread) + ', ' if items_per_thread is not None else ''}{algorithm.name}]",
            (cls,),
            {
                "dtype": key_dtype,
                "key_dtype": key_dtype,
                "value_dtype": value_dtype,
                "block_size": block_size,
                "block_threads": block_threads,
                "algorithm": algorithm,
                "warp_threads": warp_threads,
                "num_warps": block_threads // warp_threads,
                "items_per_thread": items_per_thread,
                "SharedStorage": make_storage(*storage_args),
            },
        )
        if items_per_thread is not None:
            setattr(specialized, cls._parameter_name, items_per_thread)
        _CACHE[key] = specialized
        return specialized


class BlockTileAlgorithmMeta(BlockAlgorithmMeta):
    """The same specialization, with a static per-thread tile extent.

    ``[dtype, block_size, items_per_thread, algorithm]`` keeps storage sizing
    explicit, as it is for reduce and scan. Item counts need not be powers of
    two; individual algorithms may impose that additional restriction.
    """

    _parameter_name = "items_per_thread"

    def __getitem__(cls, params) -> type:
        if cls.block_threads is not None:
            raise TypeError(f"{cls.__name__} is already specialized")
        if not isinstance(params, tuple) or not 3 <= len(params) <= 4:
            raise TypeError(f"{cls.__name__}[dtype, block_size, {cls._parameter_name}, algorithm=...]")
        dtype, block_size, count = params[:3]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{cls._parameter_name} must be a positive Python int")
        algorithm = params[3] if len(params) == 4 else cls._default_algorithm_for(current_target())
        return cls._specialize(dtype, block_size, algorithm, count)


class BlockPrimitive:
    """Provide specialization metadata inherited by public block primitives.

    Attributes are populated on the class returned by ``Primitive[...]``.
    Allocate ``SharedStorage`` in shared memory once per block and pass its
    view to collective calls. All participating threads must finish reading
    that storage before it is reused.

    Attributes:
        dtype: Input element type, or key type for a key/value specialization.
        key_dtype: Key type; identical to ``dtype``.
        value_dtype: Payload type for a key/value specialization, else ``None``.
        block_size: Launch dimensions ``(x, y, z)``, with x varying fastest.
        block_threads: Product of the block dimensions.
        items_per_thread: Static tile extent when required by the primitive,
            otherwise ``None``. Some primitives give this extent a specific
            meaning, such as ``bins`` or ``runs_per_thread``.
        algorithm: Selected member of the primitive's algorithm enum.
        warp_threads: Physical warp width of the compilation target.
        num_warps: Number of complete physical warps in the block.
        SharedStorage: DSL type describing the caller-allocated shared memory.
    """

    dtype = None
    key_dtype = None
    value_dtype = None
    block_size = None
    block_threads = None
    items_per_thread = None
    algorithm = None
    warp_threads = None
    num_warps = None
    SharedStorage = None

    @classmethod
    def _check(cls, value=None):
        from .._values import _as_items, _items_dtype

        if cls.block_threads is None:
            raise TypeError(f"specialize {cls.__name__} first")
        if value is not None:
            dtype = _items_dtype(value)
            if dtype is not cls.dtype:
                raise TypeError(f"expected {cls.dtype.__name__}, got {dtype.__name__}")
            count = len(_as_items(value))
            if cls.items_per_thread is not None and count != cls.items_per_thread:
                raise ValueError(f"expected {cls.items_per_thread} items per thread, got {count}")
