# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Machinery shared by the typed ROCDL instruction wrappers.

An :class:`Instruction` wraps one ISA instruction and is named after its mnemonic (``V_MFMA_F32_16X16X16_F16``). It is
declared from a spec (operand types, immediates, result rule) and called like a function. Calls convert DSL values,
Python literals and ``fx.Pointer`` operands to the operand types the ISA takes, build MLIR attributes from Python
literals, infer the result type, and return DSL-typed values. Register operands also take ``fx.Tensor`` register
memory, and a result can be stored to an ``out`` tensor instead of returned. :class:`MatrixInstruction` (MFMA family)
takes exactly its compute types. :class:`MmaInstruction` and :class:`CopyInstruction` also expose the thread/value
layouts of the FlyROCDL atom that lowers to the same instruction.
"""

from __future__ import annotations

import inspect
import sys
from enum import IntEnum

from ....._mlir import ir
from ....._mlir.dialects import _rocdl_ops_gen as _ods_gen
from ....._mlir.dialects import llvm
from ....meta import dsl_loc_tracing
from ....numeric import Numeric
from ....typing import Pointer, Tensor, Vector, as_ir_value
from ....utils.arith import constant, unwrap
from . import _sched

REQUIRED = inspect.Parameter.empty

_OPS = {
    cls.OPERATION_NAME: cls
    for cls in vars(_ods_gen).values()
    if isinstance(cls, type) and issubclass(cls, ir.OpView) and hasattr(cls, "OPERATION_NAME")
}

# ---------------------------------------------------------------------------------------------------------------------
# Types. Specs are strings, resolved lazily because MLIR types need a context:
#   scalars  i1 i8 i16 i32 i64 f16 bf16 f32 f64, and the narrow floats f8E4M3FN ... (MLIR spelling)
#   vectors  "4xf16"
#   pointers "ptr1" (global), "ptr3" (LDS), "ptr8" (buffer resource)


_SCALARS = {
    "i1": lambda: ir.IntegerType.get_signless(1),
    "i8": lambda: ir.IntegerType.get_signless(8),
    "i16": lambda: ir.IntegerType.get_signless(16),
    "i32": lambda: ir.IntegerType.get_signless(32),
    "i64": lambda: ir.IntegerType.get_signless(64),
    "f16": lambda: ir.F16Type.get(),
    "bf16": lambda: ir.BF16Type.get(),
    "f32": lambda: ir.F32Type.get(),
    "f64": lambda: ir.F64Type.get(),
    "f8E4M3FN": lambda: ir.Float8E4M3FNType.get(),
    "f8E5M2": lambda: ir.Float8E5M2Type.get(),
    "f8E4M3FNUZ": lambda: ir.Float8E4M3FNUZType.get(),
    "f8E5M2FNUZ": lambda: ir.Float8E5M2FNUZType.get(),
    "f8E8M0FNU": lambda: ir.Float8E8M0FNUType.get(),
    "f6E2M3FN": lambda: ir.Float6E2M3FNType.get(),
    "f6E3M2FN": lambda: ir.Float6E3M2FNType.get(),
    "f4E2M1FN": lambda: ir.Float4E2M1FNType.get(),
}


def mlir_type(spec: str) -> ir.Type:
    if spec in _SCALARS:
        return _SCALARS[spec]()
    if spec.startswith("ptr"):
        return ir.Type.parse(f"!llvm.ptr<{int(spec[3:])}>")
    count, elem = spec.split("x", 1)
    return ir.VectorType.get([int(count)], _SCALARS[elem]())


def spec_text(spec: str) -> str:
    """Spec as written in MLIR, for documentation."""
    if spec in _SCALARS:
        return spec
    if spec.startswith("ptr"):
        return f"!llvm.ptr<{spec[3:]}>"
    return f"vector<{spec}>"


def bit_width(ty: ir.Type) -> int | None:
    if isinstance(ty, ir.VectorType):
        elem = bit_width(ty.element_type)
        return None if elem is None else elem * ty.shape[0]
    if isinstance(ty, (ir.IntegerType, ir.FloatType)):
        return ty.width
    return None


def _bitcast(value, ty):
    return value if value.type == ty else llvm.bitcast(ty, value)


# ---------------------------------------------------------------------------------------------------------------------
# Specs


class Arg:
    """A register operand.

    ``type`` is a type spec, a tuple of alternatives, or ``None`` for any type. A matrix instruction declares the types
    it computes in and takes exactly those (see :class:`MatrixInstruction`); for other instructions a value of another
    type with the same bit width is bitcast, so packed data (``fx.Vector[fx.Int8, 16]`` stored as ``4xi32``) needs no
    manual conversion.
    """

    __slots__ = ("name", "type", "ods", "default", "like", "op_type")

    def __init__(self, name, type=None, *, ods=None, default=REQUIRED, like=None, op_type=None):
        """``like``: name of an earlier-declared operand whose converted type this operand must have (so a Python
        literal here takes that type). ``op_type``: the type the MLIR op takes when it is not ``type`` (a spec, or
        alternatives told apart by bit width); the operand is bitcast to it."""
        self.name, self.type, self.ods = name, type, ods or _camel(name)
        self.default, self.like, self.op_type = default, like, op_type

    def to_op(self, value):
        """``value`` (of one of ``types()``) as the MLIR op takes it."""
        if self.op_type is None:
            return value
        specs = self.op_type if isinstance(self.op_type, tuple) else (self.op_type,)
        (ty,) = [t for t in map(mlir_type, specs) if bit_width(t) == bit_width(value.type)]
        return _bitcast(value, ty)

    def types(self):
        if self.type is None:
            return None
        specs = self.type if isinstance(self.type, tuple) else (self.type,)
        return [mlir_type(s) for s in specs]

    def text(self):
        if self.type is None:
            return "any"
        specs = self.type if isinstance(self.type, tuple) else (self.type,)
        return " | ".join(spec_text(s) for s in specs)


class Imm:
    """A compile-time immediate: an MLIR attribute, or a constant operand when ``operand=True``.

    ``kind`` is ``"i32"``, ``"i16"``, ``"bool"`` or one of the enum kinds in ``ENUMS``. ``values`` restricts the
    accepted integers (any container). Immediates with a default become keyword-only parameters. A ``fixed``
    immediate is not a parameter: a value that is part of the instruction (e.g. the transfer size of an LDS DMA), or a
    function of the bound arguments (e.g. the data format of a scaled MFMA, from its operand types).
    """

    __slots__ = ("name", "kind", "default", "values", "ods", "operand", "fixed")

    def __init__(self, name, kind="i32", default=REQUIRED, *, values=None, ods=None, operand=False, fixed=None):
        self.name, self.kind, self.default, self.values = name, kind, default, values
        self.ods, self.operand, self.fixed = ods or _camel(name), operand, fixed


class Dtype:
    """A result of type spec ``native`` that the caller may reinterpret through a keyword-only ``dtype`` parameter:
    any DSL type of the same bit width (``fx.Vector[fx.BFloat16, 8]`` for a 128-bit load), applied by a bitcast."""

    __slots__ = ("native",)

    def __init__(self, native):
        self.native = native


def _camel(name):
    head, *rest = name.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


# ---------------------------------------------------------------------------------------------------------------------
# Attributes

# kind: (attribute mnemonic, is bit enum, cases). Values from mlir/Dialect/LLVMIR/ROCDLEnums.td.
ENUMS = {
    "perm_b": (
        "mfma_perm_b",
        False,
        {
            "none": 0,
            "bcast_first_32": 1,
            "bcast_second_32": 2,
            "rotate_16_right": 3,
            "bcast_first_16": 4,
            "bcast_second_16": 5,
            "bcast_third_16": 6,
            "bcast_fourth_16": 7,
        },
    ),
    "neg_mod": ("mfma_neg_modifier", True, {"neg_a": 1, "neg_b": 2, "neg_c": 4}),
    "matrix_format": (
        "matrix_format",
        False,
        {"fp8_e4m3": 0, "fp8_e5m2": 1, "fp6_e2m3": 2, "fp6_e3m2": 3, "fp4_e2m1": 4},
    ),
    # Cache policy of a gfx942/gfx950 memory access (sc0 = GLC, nt = SLC, sc1 = SCC bits of the raw encoding).
    "cpol_gfx942": ("gfx942_cache_policy", True, {"sc0": 0x1, "nt": 0x2, "swz": 0x8, "sc1": 0x10}),
}


def _static(value, what):
    """A Python int/bool from a literal, an IntEnum or a static DSL integer; attributes pass through."""
    if isinstance(value, Numeric):
        if not value.is_static():
            raise TypeError(f"{what} must be a compile-time constant, got a run-time value")
        value = value.value
    if isinstance(value, IntEnum):
        value = int(value)
    if not isinstance(value, (bool, int, str, ir.Attribute)):
        raise TypeError(f"{what} must be a Python int/bool, got {type(value).__name__}")
    return value


def _enum_attr(kind, value, what):
    mnemonic, is_bits, cases = ENUMS[kind]
    if isinstance(value, str):
        names = [n.strip() for n in value.split("|")]
        unknown = [n for n in names if n not in cases and n != "none"]
        if unknown:
            raise ValueError(f"{what}: unknown {mnemonic} value(s) {unknown}; expected {sorted(cases)}")
        if not is_bits and len(names) != 1:
            raise ValueError(f"{what}: {mnemonic} takes a single value, got {value!r}")
    else:
        value = int(value)
        if is_bits:
            names = [n for n, bit in cases.items() if value & bit]
            if sum(cases[n] for n in names) != value:
                raise ValueError(f"{what}: {value:#x} has bits outside {mnemonic} ({sorted(cases)})")
            names = names or ["none"]
        else:
            by_value = {v: n for n, v in cases.items()}
            if value not in by_value:
                raise ValueError(f"{what}: {value} is not a {mnemonic} value; expected one of {sorted(by_value)}")
            names = [by_value[value]]
    return ir.Attribute.parse(f"#rocdl<{mnemonic} {'|'.join(names)}>")


def _immediate(spec: Imm, value, what):
    """The Python value or attribute to pass to the ODS builder for an immediate."""
    value = _static(value, what)
    if isinstance(value, ir.Attribute):
        return value
    if spec.kind in ENUMS:
        if isinstance(value, int) and spec.values is not None and value not in spec.values:
            raise ValueError(f"{what} must be one of {sorted(spec.values)}, got {value}")
        if spec.kind.startswith("cpol") and not value:
            return None  # the op's default (no cache policy bits)
        return _enum_attr(spec.kind, value, what)
    if isinstance(value, str):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    if spec.kind == "bool":
        return bool(value)
    if spec.values is not None and value not in spec.values:
        raise ValueError(f"{what} must be one of {sorted(spec.values)}, got {value}")
    if spec.operand:
        return unwrap(constant(int(value), type=ir.IntegerType.get_signless(32)))
    return int(value)


# ---------------------------------------------------------------------------------------------------------------------
# Operands and results


def _operand(value, types, what, *, exact=False):
    """Convert ``value`` to an ``ir.Value`` of one of ``types`` (``None``: any type).

    ``exact``: the value must have one of ``types``; otherwise a value of another type with the same bit width is
    bitcast. Returns ``(value, type before any bitcast)``.
    """
    if isinstance(value, Pointer):
        if types is not None and any(str(t) == "!llvm.ptr<8>" for t in types):
            from ...universal import get_buffer_rsrc

            value = get_buffer_rsrc(value)
        else:
            value = value.llvm_ptr
    if isinstance(value, (bool, int, float)) and not isinstance(value, IntEnum):
        if types is None or len(types) != 1 or bit_width(types[0]) is None or isinstance(types[0], ir.VectorType):
            raise TypeError(f"{what}: cannot infer the type of Python literal {value!r}; pass a DSL value")
        value = constant(value, type=types[0])
    value = unwrap(as_ir_value(value))
    if not isinstance(value, ir.Value):
        raise TypeError(f"{what}: expected a DSL value or ir.Value, got {type(value).__name__}")
    if types is None or value.type in types:
        return value, value.type
    width = bit_width(value.type)
    same = [] if exact else [t for t in types if width is not None and bit_width(t) == width]
    if len(same) == 1:
        return _bitcast(value, same[0]), value.type
    expected = " | ".join(str(t) for t in types)
    raise TypeError(f"{what}: expected {expected}, got {value.type}")


def _dsl_class(value):
    """The DSL class a caller's argument belongs to, so a same-typed result can be returned as that class."""
    if isinstance(value, (Numeric, Vector)):
        return type(value)
    return None


def wrap_result(value, cls=None):
    """Return ``value`` as a DSL value: ``cls`` when given, else the class implied by its MLIR type."""
    ty = value.type
    if cls is not None and getattr(cls, "ir_type", None) == ty:
        return cls(value)
    if isinstance(ty, ir.VectorType):
        return Vector(value)
    if bit_width(ty) is not None:
        return Numeric.from_ir_type(ty)(value)
    return value


def _dtype_type(dtype, what):
    if isinstance(dtype, ir.Type):
        return dtype, None
    ty = getattr(dtype, "ir_type", None)
    if not isinstance(ty, ir.Type):
        raise TypeError(f"{what}: dtype must be a DSL type such as fx.Float16 or fx.Vector[fx.Float16, 8]")
    return ty, dtype


# ---------------------------------------------------------------------------------------------------------------------
# Instructions


class Instruction:
    """One ISA instruction, called like a function.

    ``name``: the ISA mnemonic (``V_MFMA_F32_16X16X16_F16``), which is also the Python name. ``op``: the MLIR
    operation emitted (``rocdl.mfma.f32.16x16x16f16``).
    ``args`` / ``imms``: :class:`Arg` and :class:`Imm` specs, in signature order (operands, then required immediates,
    then keyword-only immediates with defaults).
    ``result``: ``None`` (no result), a type spec, ``"=name"`` (the type of operand ``name``; when the caller passed a
    value of another same-width type the result is bitcast back to it), or a :class:`Dtype`.
    ``sched``: the ``sched_group_mask`` class the instruction belongs to, or ``None`` when it has none. In a
    :class:`ScheduleRegion` an exclusive class (``"vmem_read"``) orders the instruction, ``"valu"`` keeps it between the
    ordered instructions around it, and ``None`` leaves it free.

    Register memory works in place of values. A register operand also takes an ``fx.Tensor``: it is loaded whole
    (``tensor.load()``) and used as that vector. An instruction with a result takes a keyword-only ``out``, an
    ``fx.Tensor`` the result is stored to whole (``out.store(...)``); the call then returns ``None``. For a result the
    caller may reinterpret (:class:`Dtype`, such as a load), ``out`` also sets that type: the tensor's vector type, of
    the instruction's width. Otherwise ``out`` holds the result's element type and count.
    """

    exact_operands = False  # operands must have their declared types (no same-width bitcast)
    sched = None

    def __init__(self, name, op, *, args=(), imms=(), result=None, doc="", sched=None):
        self.op = op
        self._cls = _OPS.get(op)
        self.args, self.imms, self.result = tuple(args), tuple(imms), result
        if sched is not None:
            self.sched = sched
        self.__name__ = self.__qualname__ = name
        self.__module__ = sys._getframe(1).f_globals.get("__name__", __name__)
        self.__signature__ = self._signature()
        self.__doc__ = self._doc(doc)
        self._emit_traced = dsl_loc_tracing(self._emit)

    def _signature(self):
        P = inspect.Parameter
        imms = [i for i in self.imms if i.fixed is None]
        params = [P(a.name, P.POSITIONAL_OR_KEYWORD, default=a.default) for a in self.args]
        params += [P(i.name, P.POSITIONAL_OR_KEYWORD) for i in imms if i.default is REQUIRED]
        if isinstance(self.result, Dtype):
            params.append(P("dtype", P.KEYWORD_ONLY, default=None))
        params += [P(i.name, P.KEYWORD_ONLY, default=i.default) for i in imms if i.default is not REQUIRED]
        if self.result is not None:
            params.append(P("out", P.KEYWORD_ONLY, default=None))
        return inspect.Signature(params)

    def _result_text(self):
        r = self.result
        if r is None:
            return "None"
        if isinstance(r, Dtype):
            return f"{spec_text(r.native)}, or ``dtype`` of the same width"
        if r.startswith("="):
            return f"type of {r[1:]}"
        return spec_text(r)

    def _doc(self, text):
        lines = [text.strip() or f"``{self.__name__}``.", "", f"MLIR: ``{self.op}``."]
        if self.args:
            lines.append("Operands: " + ", ".join(f"``{a.name}``: {a.text()}" for a in self.args) + ".")
            lines.append("Register operands also take a register ``fx.Tensor`` (its ``load()``).")
        lines.append(f"Result: {self._result_text()}.")
        if self.result is not None:
            lines.append("``out``: a register ``fx.Tensor`` to store the result to instead of returning it.")
        return "\n".join(lines)

    def __repr__(self):
        return f"<rocdl instruction {self.__name__} ({self.op})>"

    def __call__(self, *args, **kwargs):
        return self._emit_traced(*args, **kwargs)

    def _emit(self, *args, **kwargs):
        _sched.check(self)
        try:
            bound = self.__signature__.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"{self.__name__}{self.__signature__}: {exc}") from None
        bound.apply_defaults()
        values = bound.arguments
        out = values.pop("out", None)
        if out is not None:
            if not isinstance(out, Tensor):
                raise TypeError(f"{self.__name__}: out must be an fx.Tensor, got {type(out).__name__}")
            if isinstance(self.result, Dtype):
                if values["dtype"] is not None:
                    raise TypeError(f"{self.__name__}: pass dtype or out, not both; out sets the result type")
                values["dtype"] = _tensor_type(out)
                native = mlir_type(self.result.native)
                if bit_width(values["dtype"]) != bit_width(native):
                    raise TypeError(
                        f"{self.__name__}: out must hold {bit_width(native)} bits like {native}, "
                        f"got a tensor of {values['dtype']}"
                    )
        for a in self.args:
            if isinstance(values[a.name], Tensor) and not _is_pointer(a):
                values[a.name] = values[a.name].load()
        ods_kwargs, given = {}, {}
        by_name = {a.name: a for a in self.args}
        for a in sorted(self.args, key=lambda a: a.like is not None):
            what = f"{self.__name__}: operand '{a.name}'"
            raw, types = values[a.name], a.types()
            if a.like is not None:
                types = [ods_kwargs[by_name[a.like].ods].type]
            value, src_ty = _operand(raw, types, what, exact=self.exact_operands)
            ods_kwargs[a.ods] = a.to_op(value)
            given[a.name] = (_dsl_class(values[a.name]), src_ty)
        for i in self.imms:
            value = values[i.name] if i.fixed is None else i.fixed(values) if callable(i.fixed) else i.fixed
            v = _immediate(i, value, f"{self.__name__}: '{i.name}'")
            if v is not None:
                ods_kwargs[i.ods] = v
        result_cls, bitcast_back = None, None
        if self.result is not None:
            ods_kwargs["res"], result_cls, bitcast_back = self._result_type(values, ods_kwargs, given)
        value = self._build(ods_kwargs)
        _sched.order(self)
        if self.result is None:
            return None
        if bitcast_back is not None:
            value = _bitcast(value, bitcast_back)
        if out is None:
            return wrap_result(value, result_cls)
        _store(value, out, f"{self.__name__}: out")
        return None

    def _build(self, ods_kwargs):
        """Emit the MLIR op from the ODS keyword arguments; return its result value (``None`` without a result)."""
        op = self._cls(**ods_kwargs)
        return None if self.result is None else op.result

    def _result_type(self, values, ods_kwargs, given):
        """(result ir.Type for the builder, DSL class to return, type to bitcast the result back to)."""
        r = self.result
        if isinstance(r, Dtype):
            native = mlir_type(r.native)
            if values["dtype"] is None:
                return native, None, None
            ty, cls = _dtype_type(values["dtype"], f"{self.__name__}: dtype")
            if bit_width(ty) != bit_width(native):
                raise TypeError(f"{self.__name__}: dtype must be {bit_width(native)} bits like {native}, got {ty}")
            return native, cls, (ty if ty != native else None)
        if r.startswith("="):
            arg = next(a for a in self.args if a.name == r[1:])
            ty = ods_kwargs[arg.ods].type
            cls, src_ty = given[arg.name]
            return ty, cls, (src_ty if src_ty != ty else None)
        return mlir_type(r), None, None


class IrInstruction(Instruction):
    """An instruction the backend selects from plain LLVM IR (an LDS load, store or atomic), not from a ROCDL op.

    ``op`` names the LLVM operation, for documentation. ``build(ods_kwargs)`` emits it from the converted operands
    (keyed by :attr:`Arg.ods`, the result type under ``"res"``) and returns the result value, or ``None``.
    """

    def __init__(self, name, op, *, build, **kwargs):
        self._build_ir = build
        super().__init__(name, op, **kwargs)
        self.__module__ = sys._getframe(1).f_globals.get("__name__", __name__)

    def _build(self, ods_kwargs):
        return self._build_ir(ods_kwargs)


class MatrixInstruction(Instruction):
    """A matrix instruction (MFMA, scaled MFMA), typically called on register fragments.

    Operands are declared in the types the instruction computes in (``vector<4xbf16>`` for a BF16 MFMA, even where
    the MLIR op takes the bits as ``vector<4xi16>``) and must have exactly those types: data of another type is not
    reinterpreted, neither as an operand nor through ``out``.
    """

    exact_operands = True
    sched = "mfma_wmma"


def _is_pointer(arg):
    specs = arg.type if isinstance(arg.type, tuple) else (arg.type,)
    return any(s is not None and s.startswith("ptr") for s in specs)


def _tensor_type(tensor):
    """The vector type a register tensor loads / stores whole."""
    return Vector.make_type(tensor.shape.to_py_value(), tensor.dtype)


def _store(value, out, what):
    ty = _tensor_type(out)
    elem = value.type.element_type if isinstance(value.type, ir.VectorType) else value.type
    if ty.element_type != elem or bit_width(ty) != bit_width(value.type):
        raise TypeError(f"{what} must hold {value.type}, got a tensor of {ty}")
    out.store(_bitcast(value, ty))  # a scalar result into a one-element tensor


class MmaInstruction(MatrixInstruction):
    """An MMA instruction whose register layout is the one a FlyROCDL MMA atom implements.

    ``mma_op`` (zero-argument callable) builds the Fly MMA op type that lowers to this instruction. Besides the call,
    the object exposes the atom's layouts under the names :class:`fx.MmaAtom` uses: ``thr_layout``, ``shape_mnk``,
    ``layout_A_tv``, ``layout_B_tv``, ``layout_C_tv`` (static ``fx.Layout`` / ``fx.IntTuple`` values, inside a
    kernel), plus ``mma_op`` and ``atom_type`` (types; outside a kernel they need only an active MLIR context, and
    ``str(instr.atom_type.tv_layout_a)`` prints the layout). A layout maps ``(thread, value)`` to the column-major
    offset in the ``M x K`` (A), ``N x K`` (B) or ``M x N`` (C) tile; the thread index is the lane id.
    """

    def __init__(self, name, op, *, mma_op, **kwargs):
        self._mma_op = mma_op
        super().__init__(name, op, **kwargs)
        self.__module__ = sys._getframe(1).f_globals.get("__name__", __name__)

    def _doc(self, text):
        return super()._doc(text) + "\nRegister layout: ``layout_A_tv`` / ``layout_B_tv`` / ``layout_C_tv``."

    @property
    def mma_op(self):
        return self._mma_op()

    @property
    def atom_type(self):
        from ....primitive import MmaAtomType

        return MmaAtomType.get(self.mma_op)

    @property
    def thr_layout(self):
        return _static_value(self.atom_type.thr_layout)

    @property
    def thr_id(self):
        return self.thr_layout

    @property
    def shape_mnk(self):
        return _static_value(self.atom_type.shape_mnk)

    @property
    def layout_A_tv(self):
        return _static_value(self.atom_type.tv_layout_a)

    @property
    def layout_B_tv(self):
        return _static_value(self.atom_type.tv_layout_b)

    @property
    def layout_C_tv(self):
        return _static_value(self.atom_type.tv_layout_c)


class CopyInstruction(Instruction):
    """A copy instruction whose data layout is the one a FlyROCDL copy atom implements.

    ``copy_op`` builds the Fly copy op type; ``val_bits`` is the element width the layouts count in. Layouts use the
    :class:`fx.CopyAtom` names: ``thr_layout``, ``layout_src_tv``, ``layout_dst_tv``, ``layout_ref_tv``; ``copy_op`` and
    ``atom_type`` give the types.
    """

    def __init__(self, name, op, *, copy_op, val_bits, **kwargs):
        self._copy_op, self.val_bits = copy_op, val_bits
        super().__init__(name, op, **kwargs)
        self.__module__ = sys._getframe(1).f_globals.get("__name__", __name__)

    def _doc(self, text):
        return super()._doc(text) + "\nData layout: ``layout_src_tv`` / ``layout_dst_tv`` (``val_bits`` per value)."

    @property
    def copy_op(self):
        return self._copy_op()

    @property
    def atom_type(self):
        from ....primitive import CopyAtomType

        return CopyAtomType.get(self.copy_op, self.val_bits)

    @property
    def thr_layout(self):
        return _static_value(self.atom_type.thr_layout)

    @property
    def thr_id(self):
        return self.thr_layout

    @property
    def layout_src_tv(self):
        return _static_value(self.atom_type.tv_layout_src)

    @property
    def layout_dst_tv(self):
        return _static_value(self.atom_type.tv_layout_dst)

    @property
    def layout_ref_tv(self):
        return _static_value(self.atom_type.tv_layout_ref)


def _static_value(ty):
    from ....primitive import static

    return static(ty)
