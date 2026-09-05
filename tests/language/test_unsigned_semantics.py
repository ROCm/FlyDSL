#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Logical signedness of integer dtypes (ROCm/FlyDSL#701).

MLIR integers are signless, so the *storage* descriptor of an unsigned dtype spells its
element type ``uiN``. That is what lets a DSL value reconstructed from IR come back as
``Uint*`` instead of ``Int*``, which in turn selects the unsigned form of every operation
whose signed and unsigned forms differ.
"""

import pytest
from lang_utils import source_ir

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr.numeric import Numeric, _ssa_value_ir_type

pytestmark = pytest.mark.l1a_compile_no_target_dialect


UNSIGNED_SIGNED = [(fx.Uint8, fx.Int8), (fx.Uint16, fx.Int16), (fx.Uint32, fx.Int32), (fx.Uint64, fx.Int64)]


@pytest.mark.parametrize("udtype,sdtype", UNSIGNED_SIGNED)
def test_storage_type_carries_signedness(udtype, sdtype):
    """Storage keeps the signedness that SSA values cannot express; values stay signless."""
    with ir.Context():
        assert str(udtype.storage_ir_type) == f"ui{udtype.width}"
        assert str(udtype.ir_type) == f"i{udtype.width}"
        # Signed integers keep the signless spelling as their canonical storage form.
        assert sdtype.storage_ir_type == sdtype.ir_type


def test_non_integer_storage_type_is_unchanged():
    with ir.Context():
        assert fx.Float32.storage_ir_type == fx.Float32.ir_type
        assert fx.Boolean.storage_ir_type == fx.Boolean.ir_type
        # ``Index`` is flagged unsigned, but ``index`` has no signed and unsigned spellings.
        assert fx.Index.storage_ir_type == fx.Index.ir_type


@pytest.mark.parametrize(
    "storage, value",
    [
        ("ui8", "i8"),
        ("ui64", "i64"),
        ("si16", "si16"),  # no DSL dtype maps to a signed spelling; leave it alone
        ("i32", "i32"),
        ("vector<4xui8>", "vector<4xi8>"),
        ("vector<4xi8>", "vector<4xi8>"),
        ("vector<[4]xui8>", "vector<[4]xi8>"),  # scalable dimensions survive the rewrite
        ("index", "index"),
        ("f32", "f32"),
    ],
)
def test_ssa_value_ir_type_undoes_storage_signedness(storage, value):
    """The inverse of ``storage_ir_type``: what loading a storage element type produces."""
    with ir.Context(), ir.Location.unknown():
        assert str(_ssa_value_ir_type(ir.Type.parse(storage))) == value


def test_register_tensor_round_trips_its_dtype():
    """A fragment declared unsigned reads back unsigned; the loaded value is signless."""
    seen = {}

    def body():
        for dtype in (fx.Uint8, fx.Int8, fx.Uint32, fx.Float32):
            frag = fx.make_rmem_tensor(fx.make_layout(4, 1), dtype)
            seen[dtype] = frag.element_type

    text = source_ir(body)
    assert seen == {fx.Uint8: fx.Uint8, fx.Int8: fx.Int8, fx.Uint32: fx.Uint32, fx.Float32: fx.Float32}
    assert "!fly.memref<ui8, register" in text
    assert "!fly.memref<i8, register" in text


@pytest.mark.parametrize("dtype,shift_op", [(fx.Uint8, "arith.shrui"), (fx.Int8, "arith.shrsi")])
def test_scalar_loads_keep_the_storage_signedness(dtype, shift_op):
    """A scalar element read back out of storage is still ``Uint8``, not ``Int8``.

    ``memref_load`` and ``ptr_load`` produce a signless value, so the dtype has to come
    from the memref / pointer that was loaded, exactly as the vector load already does.
    """

    def body():
        frag = fx.make_rmem_tensor(fx.make_layout(4, 1), dtype)
        frag[1] = frag[0] >> dtype(2)
        ptr = fx.get_iter(frag)
        ptr[3] = ptr[2] >> dtype(2)

    text = source_ir(body)
    other = "arith.shrsi" if shift_op == "arith.shrui" else "arith.shrui"
    assert text.count(shift_op) == 2 and other not in text


def _widen(v, dtype):
    return v.to(fx.Int64)


def _to_float(v, dtype):
    return v.to(fx.Float32)


def _div(v, dtype):
    return v // dtype(3)


def _rem(v, dtype):
    return v % dtype(3)


def _shr(v, dtype):
    return v >> dtype(2)


def _gt(v, dtype):
    return v > dtype(3)


def _gt_select(v, dtype):
    # Ordering, spelled out of the pieces the DSL already has.
    return (v > dtype(3)).select(v, dtype(3))


def _apply_ir(apply, dtype):
    """Trace ``apply`` over a run-time value of *dtype*; return the emitted IR."""

    def body(v: dtype):
        apply(v, dtype)

    return source_ir(body, 1)


@pytest.mark.parametrize(
    "apply,unsigned_op,signed_op",
    [
        (_widen, "arith.extui", "arith.extsi"),
        (_to_float, "arith.uitofp", "arith.sitofp"),
        (_div, "arith.divui", "arith.floordivsi"),
        (_rem, "arith.remui", "arith.remsi"),
        (_shr, "arith.shrui", "arith.shrsi"),
        (_gt, "ugt", "sgt"),
        (_gt_select, "ugt", "sgt"),
    ],
)
def test_operation_form_follows_dtype_signedness(apply, unsigned_op, signed_op):
    unsigned_ir = _apply_ir(apply, fx.Uint8)
    signed_ir = _apply_ir(apply, fx.Int8)
    assert unsigned_op in unsigned_ir and signed_op not in unsigned_ir
    assert signed_op in signed_ir and unsigned_op not in signed_ir


def test_mixed_signedness_promotes_before_choosing_the_predicate():
    """A ``Uint8`` against an ``Int32``: widen unsigned (``extui``), then compare signed."""

    def body(v: fx.Uint8):
        _ = v > fx.Int32(3)

    text = source_ir(body, 1)
    assert "arith.extui" in text and "sgt" in text and "ugt" not in text


@pytest.mark.parametrize(
    "dtype,other,predicate",
    [
        (fx.Uint16, lambda: fx.Uint16(3), "ugt"),
        (fx.Int64, lambda: fx.Int64(3), "sgt"),
        # Neither of these has a sign bit to read, so both are the unsigned predicate:
        # ``i1``'s single bit is its value, and ``index`` has no signed spelling.
        (fx.Boolean, lambda: fx.Boolean(True), "ugt"),
        (fx.Index, lambda: fx.Index(3), "ugt"),
    ],
)
def test_comparison_covers_the_whole_integer_domain(dtype, other, predicate):
    """Every integer dtype, and the comparison predicate its signedness selects."""

    def body(v: dtype):
        _ = v > other()

    text = source_ir(body, 1)
    unwanted = "sgt" if predicate == "ugt" else "ugt"
    assert predicate in text and unwanted not in text


def test_vector_ordering_follows_the_element_signedness():
    """The same rule element-wise: a vector compares with its element dtype's predicate."""
    seen = {}

    def body():
        u = fx.Vector.filled(4, 200, fx.Uint8)
        s = fx.Vector.filled(4, -100, fx.Int8)
        seen["u"] = (u > fx.Uint8(5)).select(u, fx.Uint8(5))
        seen["s"] = (s > fx.Int8(5)).select(s, fx.Int8(5))

    text = source_ir(body)
    assert isinstance(seen["u"], fx.Vector) and seen["u"].dtype is fx.Uint8
    assert isinstance(seen["s"], fx.Vector) and seen["s"].dtype is fx.Int8
    assert "ugt" in text and "sgt" in text


# The boundary itself: torch dtype -> MLIR element type -> DSL ``Numeric``. No kernel is
# traced and no device is needed, so this runs in CPU-only CI.
_ROUND_TRIP = [("Uint8", "uint8"), ("Int8", "int8"), ("Int16", "int16"), ("Int32", "int32"), ("Int64", "int64")]


@pytest.mark.parametrize("dsl_name,torch_name", _ROUND_TRIP)
def test_torch_dtype_round_trips_to_the_dsl_dtype(dsl_name, torch_name):
    """The dtype a kernel reconstructs must never disagree with the tensor it was given."""
    torch = pytest.importorskip("torch")
    from flydsl.compiler.jit_argument import torch_dtype_to_mlir_type

    with ir.Context():
        assert Numeric.from_ir_type(torch_dtype_to_mlir_type(getattr(torch, torch_name))) is getattr(fx, dsl_name)


_DLPACK_ELEMENT_TYPES = [
    ("uint8", "ui8"),
    ("uint16", "ui16"),
    ("uint32", "ui32"),
    ("uint64", "ui64"),
    ("int8", "i8"),
    ("int32", "i32"),
    ("float32", "f32"),
]


@pytest.mark.parametrize("torch_name,spelling", _DLPACK_ELEMENT_TYPES)
def test_dlpack_adaptor_maps_unsigned_codes_to_unsigned_types(torch_name, spelling):
    """The other entry point: ``DLTensorAdaptor`` reading a DLPack capsule directly.

    ``torch_dtype_to_mlir_type`` is a Python table; this is the C++ ``kDLUInt`` arm, which
    a device launch only exercises indirectly. A host tensor is enough -- the adaptor reads
    the capsule's dtype code, not its memory. It reads the width from the capsule, so every
    unsigned width the framework can export is covered, not just the byte.
    """
    torch = pytest.importorskip("torch")
    from flydsl._mlir._mlir_libs._mlirDialectsFly import DLTensorAdaptor

    torch_dtype = getattr(torch, torch_name, None)
    if torch_dtype is None:
        pytest.skip(f"torch has no {torch_name}")
    tensor = torch.zeros(4, dtype=torch_dtype)
    with ir.Context():
        assert str(DLTensorAdaptor(tensor.__dlpack__()).dtype) == spelling


def test_wider_unsigned_torch_dtypes_are_still_rejected():
    """Only ``uint8`` is accepted at the boundary; a wider unsigned tensor is refused.

    Widening that set is out of scope here -- the point is that it fails loudly rather
    than silently arriving as its signed counterpart, which is the bug this fixes.
    """
    torch = pytest.importorskip("torch")
    from flydsl.compiler.jit_argument import torch_dtype_to_mlir_type

    for name in ("uint16", "uint32", "uint64"):
        torch_dtype = getattr(torch, name, None)
        if torch_dtype is None:
            continue
        with pytest.raises(TypeError, match="unsupported torch dtype"):
            with ir.Context():
                torch_dtype_to_mlir_type(torch_dtype)
