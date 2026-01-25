from dataclasses import dataclass
from functools import cached_property, reduce
import numpy as np
from typing import Tuple, Union, List, Any

from ....dialects.linalg.opdsl.lang.emitter import _is_index_type
from .arith import Scalar
from ....ir import DenseElementsAttr, ShapedType, Type, Value, RankedTensorType

S = ShapedType.get_dynamic_size()


# mixin that requires `is_constant`
def ShapedValue(cls):
    @cached_property
    def literal_value(self) -> np.ndarray:
        if not self.is_constant:
            raise ValueError("Can't build literal from non-constant value")
        return np.array(DenseElementsAttr(self.owner.opview.value), copy=False)

    @cached_property
    def _shaped_type(self) -> ShapedType:
        return ShapedType(self.type)

    def has_static_shape(self) -> bool:
        return self._shaped_type.has_static_shape

    def has_rank(self) -> bool:
        return self._shaped_type.has_rank

    @cached_property
    def rank(self) -> int:
        return self._shaped_type.rank

    @cached_property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._shaped_type.shape)

    @cached_property
    def n_elements(self) -> int:
        assert self.has_static_shape()
        return reduce(lambda acc, v: acc * v, self._shaped_type.shape, 1)

    @cached_property
    def dtype(self) -> Type:
        return self._shaped_type.element_type

    setattr(cls, "literal_value", literal_value)
    cls.literal_value.__set_name__(None, "literal_value")
    setattr(cls, "_shaped_type", _shaped_type)
    cls._shaped_type.__set_name__(None, "_shaped_type")

    setattr(cls, "has_static_shape", has_static_shape)
    setattr(cls, "has_rank", has_rank)

    setattr(cls, "rank", rank)
    cls.rank.__set_name__(None, "rank")
    setattr(cls, "shape", shape)
    cls.shape.__set_name__(None, "shape")
    setattr(cls, "n_elements", n_elements)
    cls.n_elements.__set_name__(None, "n_elements")
    setattr(cls, "dtype", dtype)
    cls.dtype.__set_name__(None, "dtype")

    return cls


@dataclass(frozen=True)
class _Indexer:
    indices: Tuple[Union[int, Scalar, slice, "Ellipsis", None]]
    newaxis_dims: Tuple[int, "Ellipsis"]
    in_shape: Tuple[Union[Value, int]]

    def is_constant(self):
        return all(_is_constant_index(i) for i in self.indices)

    def is_full(self):
        return all(
            isinstance(idx, slice)
            # TODO(max): could also work for constant Scalar
            and all([isinstance(x, int) for x in [idx.start, idx.stop, idx.step]])
            and len(range(*idx.indices(self.in_shape[i]))) == self.in_shape[i]
            for i, idx in enumerate(self.indices)
        )

    # waiting on hashable slices in 3.12 https://stackoverflow.com/a/76562346
    # @lru_cache(maxsize=1)
    def static_offsets(self):
        offsets = []
        for i in self.indices:
            if isinstance(i, (int, Scalar)):
                offsets.append(int(i))
            elif isinstance(i, slice):
                offsets.append(int(i.start))
            else:
                raise ValueError(f"idx {i} not supported with static offsets")
        return tuple(offsets)

    # @lru_cache(maxsize=1)
    def static_sizes(self):
        sizes = []
        for i in self.indices:
            if isinstance(i, (int, Scalar)):
                sizes.append(1)
            elif isinstance(i, slice):
                start, stop, step = map(int, (i.start, i.stop, i.step))
                if all(isinstance(j, int) for j in (start, stop, step)):
                    s = ((stop - start) // step) + 1
                    if (stop - start) % step == 0:
                        s -= 1
                    sizes.append(s)
                else:
                    raise ValueError(f"idx {i} not supported with static sizes")

            else:
                raise ValueError(f"idx {i} not supported with static sizes")
        return tuple(sizes)

    # @lru_cache(maxsize=1)
    def static_strides(self):
        strides = []
        for i in self.indices:
            if isinstance(i, (int, Scalar)):
                strides.append(1)
            elif isinstance(i, slice):
                strides.append(int(i.step))
            else:
                raise ValueError(f"idx {i} not supported with static strides")
        return tuple(strides)


def _indices_to_indexer(
    idx: Tuple[Union[Scalar, slice, "Ellipsis", None]], in_shape: Tuple[int]
) -> _Indexer:
    """Processes sequence of index objects and constructs _Indexer with
    corresponding indexing tensor and collapse dims (i.e., scatter/gather dims).

    Args:
      idx: Sequence (list or tuple) of slices, ellipses, Scalar, or Tensors.
      in_shape: The shape of the tensor being indexed into.

    Returns:
      _Indexer object.

    """
    idx = _canonicalize_tuple_index(idx, len(in_shape))

    in_axis = 0  # Current axis in input.
    out_axis = 0  # Current axis in output.
    indices: List[Union[Scalar, slice, Ellipsis, None]] = [slice(None)] * len(in_shape)
    newaxis_dims: List[int] = []

    # nb: idx_e <-> idx_element
    for idx_i, idx_e in enumerate(idx):
        if _is_scalar(idx_e) and _has_index_type(idx_e):
            # Handle basic Scalar indexes.
            indices[in_axis] = idx_e
            in_axis += 1
        # Handle newaxis (None)
        elif idx_e is None:
            newaxis_dims.append(out_axis)
            out_axis += 1
        elif isinstance(idx_e, slice):
            # Normalize the slice to use None when possible
            start, stop, step = idx_e.start, idx_e.stop, idx_e.step
            if isinstance(step, int) and step == 1:
                step = None
            if step is None:
                if start is None or isinstance(start, int) and start == 0:
                    start = None
                if (
                    isinstance(stop, int)
                    and in_shape[in_axis] != ShapedType.get_dynamic_size()
                    and stop >= in_shape[in_axis]
                ):
                    stop = None
            # Handle slice(None) and slice(None, None, -1)
            if (
                start is None
                and stop is None
                and (step is None or isinstance(step, int) and step == -1)
            ):
                if step == -1:
                    raise IndexError(
                        f"Negative step indexing mode not yet supported:\n{idx}"
                    )
                indices[in_axis] = slice(None)
                out_axis += 1
                in_axis += 1

            # Handle slice index (only static shape supported)
            else:
                if (
                    not isinstance(in_shape[in_axis], int)
                    or in_shape[in_axis] == ShapedType.get_dynamic_size()
                ):
                    msg = (
                        "Cannot use NumPy slice indexing on an array dimension whose "
                        f"size is not statically known ({in_shape[in_axis]}). "
                    )
                    raise IndexError(msg)

                if step is None:
                    step = 1
                if stop is None:
                    stop = in_shape[in_axis]

                indices[in_axis] = slice(start, stop, step)

                out_axis += 1
                in_axis += 1
        else:
            raise IndexError(f"Indexing mode not yet supported:\n{idx}")

    for i, idx in enumerate(indices):
        if _is_constant_index(idx) and _is_constant_scalar(in_shape[i]):
            if isinstance(idx, slice):
                indices[i] = slice(*idx.indices(int(in_shape[i])))
            elif isinstance(idx, Scalar):
                indices[i] = int(idx)

    return _Indexer(
        newaxis_dims=tuple(newaxis_dims), indices=tuple(indices), in_shape=in_shape
    )


def _canonicalize_tuple_index(idx: Tuple[Any], rank: int):
    """Helper to
    1. remove Ellipsis and replace with implicit trailing slice(None)s.
    2. cast Python lists of lists or numpy arrays to index Tensors

    Args:
      rank: Rank of tensor.
      idx: Index object (Scalar, Tensor, slice, Ellipse, or None).

    Returns:
      Tuple of index objects with no ellipses.
    """

    len_without_none = 0
    for e in idx:
        if e is None or e is Ellipsis:
            continue
        else:
            len_without_none += 1

    if len_without_none > rank:
        raise IndexError(
            f"Too many indices for shaped type with rank: {len_without_none} "
            f"non-None/Ellipsis indices for dim {rank}."
        )
    ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
    ellipsis_index = next(ellipses, None)
    if ellipsis_index is not None:
        if next(ellipses, None) is not None:
            raise IndexError(
                f"Multiple ellipses (...) not supported: {list(map(type, idx))}."
            )
        colons = (slice(None),) * (rank - len_without_none)
        idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1 :]
    elif len_without_none < rank:
        colons = (slice(None),) * (rank - len_without_none)
        idx = tuple(idx) + colons
    return idx


def _is_int_arraylike(x):
    """Returns True if x is array-like with integer dtype, False otherwise.

    Positive (i.e., return True) examples are e.g., [[0], [1]], [[0, 1]],
    [[[0, 1]], [[0, 1]]].
    """
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        or isinstance(x, (list, tuple))
        and all(_is_int_arraylike(e) for e in x)
    )


def _is_scalar(e: Any) -> bool:
    """Checks whether e is a Scalar or can be used to construct a Scalar.

    Args:
      e: Anything
    """
    return isinstance(e, Scalar) or isinstance(e, (int, float, bool))


def _has_index_type(e: Any) -> bool:
    """Checks whether e has MLIR index type or a Python value that can be used
    to construct an index type.

    Args:
      e: Anything
    """
    return (
        isinstance(e, int)
        or isinstance(e, np.ndarray)
        and e.dtype in {np.intp}
        or isinstance(e, Value)
        and _is_index_type(e.type)
        or isinstance(e.type, RankedTensorType)
        and _is_index_type(e.type.element_type)
    )


def _is_constant_index(e: Any) -> bool:
    return (
        (isinstance(e, Scalar) and e.is_constant())
        or isinstance(e, (int, float, bool))
        or (
            isinstance(e, slice)
            and _is_constant_scalar(e.start)
            and _is_constant_scalar(e.stop)
            and _is_constant_scalar(e.step)
        )
    )


def _is_constant_scalar(e: Any) -> bool:
    return (
        (isinstance(e, Scalar) and e.is_constant())
        or (isinstance(e, (int, float, bool)) and e != ShapedType.get_dynamic_size())
        or e is None
    )


def _maybe_compute_size(start, stop, step):
    from ....dialects import arith

    # TODO(max): figure out how to use actual canonicalizers
    if (
        isinstance(start, Value)
        and isinstance(stop, Value)
        and stop.owner.operands[0]._eq(start)
        and stop.owner.operands[1].is_constant()
    ):
        return stop.owner.operands[1].literal_value
    elif (
        isinstance(start, Value)
        and isinstance(start.owner.opview, arith.MulIOp)
        and isinstance(stop, Value)
        and isinstance(stop.owner.opview, arith.MulIOp)
        and isinstance(stop.owner.operands[0].owner.opview, arith.AddIOp)
        and start.owner.operands[0] == stop.owner.operands[0].owner.operands[0]
        and stop.owner.operands[1].is_constant()
        and isinstance(step, int)
        or (isinstance(step, Scalar) and step.is_constant())
    ):
        # looks like this
        # l = lambda l: l * D
        # r = lambda r: (r + 1) * D
        # a, b, c = (
        #     A[l(i) : r(i), l(j) : r(j)],
        #     B[l(i) : r(i), l(j) : r(j)],
        #     C[l(i) : r(i), l(j) : r(j)],
        # )
        return stop.owner.operands[1]
    else:
        return stop - start
