"""Centralized MLIR Python binding imports for the ASM backend.

This is a local copy of the subset of `wave_lang.support.ir_imports` that the
ASM backend relies on. Keeping this local makes `flydsl/exp/asm` self-contained
and robust to missing optional dialect wrappers in various `iree-compiler`
wheel builds.
"""

from __future__ import annotations

# Core MLIR IR bindings (IREE compiler python package)
from iree.compiler.ir import (  # type: ignore
    AffineMap,
    AffineMapAttr,
    Attribute,
    BF16Type,
    Context,
    F16Type,
    F32Type,
    FloatAttr,
    FunctionType,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Type,
    Location,
    MemRefType,
    Module,
    Operation,
    ShapedType,
    StringAttr,
    TypeAttr,
    UnitAttr,
    Value,
    VectorType,
)

# Backwards-compat alias used by some helper modules.
IrType = Type

# OpAttributeMap type varies by build; use a best-effort import.
try:  # pragma: no cover
    from iree.compiler.ir import OpAttributeMap  # type: ignore
except Exception:  # pragma: no cover
    OpAttributeMap = object  # type: ignore[misc,assignment]

# Dialect opview modules. Some are optional in certain wheel builds.
from iree.compiler.dialects import (  # type: ignore
    amdgpu as amdgpu_d,
    arith as arith_d,
    func as func_d,
    gpu as gpu_d,
    memref as memref_d,
    scf as scf_d,
    stream as stream_d,
    vector as vector_d,
)

try:  # pragma: no cover
    from iree.compiler.dialects import affine as affine_d  # type: ignore
except Exception:  # pragma: no cover
    affine_d = None  # type: ignore[assignment]

try:  # pragma: no cover
    from iree.compiler.dialects import rocdl as rocdl_d  # type: ignore
except Exception:  # pragma: no cover
    rocdl_d = None  # type: ignore[assignment]

