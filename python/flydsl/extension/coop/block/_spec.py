# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared specialization machinery for the block-scope collectives.

Every block algorithm is specialized the same way and validates the same things,
so the parsing lives here once and each algorithm supplies only what differs:
its policy enum, its default policy, and the shared storage each policy needs.
"""

from ....compiler.backends import current_target
from ....expr.gpu import num_warp_threads
from .._common import require_power_of_two

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

    A concrete metaclass sets ``_algorithms`` and ``_shared_storage``, and
    implements :meth:`_default_algorithm_for`.
    """

    _algorithms = None
    _shared_storage = None

    def _default_algorithm_for(cls, target):
        """The policy to use when the caller does not name one."""
        raise NotImplementedError(f"{cls.__name__} must implement _default_algorithm_for")

    def __getitem__(cls, params):
        if cls.block_threads is not None:
            raise TypeError(f"{cls.__name__} is already specialized")

        if not isinstance(params, tuple):
            params = (params,)
        if not 2 <= len(params) <= 3:
            raise TypeError(f"{cls.__name__}[dtype, block_size, algorithm=...]")

        dtype, block_size = params[0], params[1]
        algorithm = params[2] if len(params) > 2 else cls._default_algorithm_for(current_target())

        if not isinstance(algorithm, cls._algorithms):
            raise TypeError(f"expected a {cls._algorithms.__name__}, got {algorithm!r}")
        block_size = _block_shape(block_size)
        block_threads = block_size[0] * block_size[1] * block_size[2]
        # A power-of-two product forces every dimension to be one too, so the
        # dimensions need no separate check.
        require_power_of_two(block_threads, "block_dim_x * block_dim_y * block_dim_z")
        # A block narrower than the wave leaves the wave's remaining lanes
        # unlaunched, and a cross-lane read of one of those is undefined — so
        # the logical warp narrows to the block.
        warp_threads = min(num_warp_threads(), block_threads)

        key = (cls, dtype, block_size, algorithm, warp_threads)
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        make_storage = cls._shared_storage.get(algorithm)
        if make_storage is None:
            raise NotImplementedError(f"{cls._algorithms.__name__}.{algorithm.name} is not implemented yet")

        specialized = type(
            f"{cls.__name__}[{getattr(dtype, '__name__', dtype)}, {block_threads}, {algorithm.name}]",
            (cls,),
            {
                "dtype": dtype,
                "block_size": block_size,
                "block_threads": block_threads,
                "algorithm": algorithm,
                "warp_threads": warp_threads,
                "num_warps": block_threads // warp_threads,
                "SharedStorage": make_storage(dtype, block_threads, warp_threads),
            },
        )
        _CACHE[key] = specialized
        return specialized
