# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import sympy

from wave_lang.support.ir_imports import (
    gpu_d,
    VectorType,
    MemRefType,
    Context,
    IrType as Type,
    AffineMap,
    AffineMapAttr,
    IntegerAttr,
    Attribute,
)

if TYPE_CHECKING:
    from .kernel_model import KernelInfo, MemRefInfo


def normalize_wg_size(wg_size: tuple) -> Tuple[int, int, int]:
    """
    Normalize workgroup size tuple to always have 3 dimensions.

    Args:
        wg_size: Workgroup size tuple with 1, 2, or 3 elements (e.g., (64,), (64, 4), or (64, 4, 1))

    Returns:
        Tuple of (wg_size_x, wg_size_y, wg_size_z) with missing dimensions defaulting to 1

    Examples:
        >>> normalize_wg_size((64,))
        (64, 1, 1)
        >>> normalize_wg_size((64, 4))
        (64, 4, 1)
        >>> normalize_wg_size((64, 4, 2))
        (64, 4, 2)
    """
    if len(wg_size) == 3:
        return wg_size
    elif len(wg_size) == 2:
        return (wg_size[0], wg_size[1], 1)
    elif len(wg_size) == 1:
        return (wg_size[0], 1, 1)
    else:
        raise ValueError(f"Invalid workgroup size: {wg_size}")


def parse_vector_type(s: str) -> Tuple[int, int, str]:
    """Parse vector type using MLIR Python bindings instead of string matching."""
    ctx = Context()
    try:
        vector_type = Type.parse(s, context=ctx)
        if not isinstance(vector_type, VectorType):
            raise ValueError(f"Expected VectorType, got {type(vector_type)}")

        return parse_vector_type_from_obj(vector_type)
    except Exception as e:
        raise ValueError(f"Failed to parse vector type '{s}': {e}")


def parse_memref_type(s: str) -> Tuple[List[int], List[int], int]:
    """
    Parse memref type using MLIR Python bindings instead of string matching.
    Parse e.g.: 'memref<16x16xf16, strided<[16, 1], offset: ?>>'
    Return (shape, strides_elems, elem_bytes)
    """
    ctx = Context()
    try:
        memref_type = Type.parse(s, context=ctx)
        if not isinstance(memref_type, MemRefType):
            raise ValueError(f"Expected MemRefType, got {type(memref_type)}")

        shape, strides_elems, elem_bytes = parse_memref_type_from_obj(memref_type)

        # Check if strides were explicitly provided (not defaulted)
        layout = memref_type.layout
        if not (hasattr(layout, "strides") and layout.strides):
            raise ValueError(f"Memref type '{s}' has no strides")

        return shape, strides_elems, elem_bytes
    except Exception as e:
        raise ValueError(f"Failed to parse memref type '{s}': {e}")


def parse_vector_type_from_obj(vector_type_obj) -> Tuple[int, int, str]:
    """Parse vector type directly from MLIR VectorType object."""
    if not isinstance(vector_type_obj, VectorType):
        raise ValueError(f"Expected VectorType, got {type(vector_type_obj)}")

    # Extract shape (first dimension for 1D vectors)
    assert len(vector_type_obj.shape) == 1, "Vector type must be 1D"
    n = vector_type_obj.shape[0] if vector_type_obj.shape else 1

    # Extract element type and bitwidth
    element_type = vector_type_obj.element_type
    eb = element_type.width // 8  # Convert bits to bytes

    # Get element type string representation
    et = str(element_type)

    return n, eb, et


def parse_memref_type_from_obj(memref_type_obj) -> Tuple[List[int], List[int], int]:
    """Parse memref type directly from MLIR MemRefType object."""
    if not isinstance(memref_type_obj, MemRefType):
        raise ValueError(f"Expected MemRefType, got {type(memref_type_obj)}")

    # Extract shape
    shape = list(memref_type_obj.shape)

    # Extract strides from layout
    layout = memref_type_obj.layout
    if hasattr(layout, "strides") and layout.strides:
        strides_elems = list(layout.strides)
    else:
        # Default to row-major strides if no explicit strides
        strides_elems = []
        stride = 1
        for dim in reversed(shape):
            strides_elems.insert(0, stride)
            stride *= dim

    # Extract element type bitwidth
    element_type = memref_type_obj.element_type
    elem_bytes = element_type.width // 8  # Convert bits to bytes

    return shape, strides_elems, elem_bytes


def attrs_to_dict(attrs) -> Dict[str, object]:
    return {x.name: x.attr for x in attrs}


def parse_wg_and_subgroup(
    attr_obj: Attribute,
) -> Tuple[Optional[Tuple[int, int, int]], Optional[int]]:
    """Parse workgroup and subgroup sizes from MLIR attribute using Python bindings."""
    wg = attr_obj.workgroup_size
    sg = attr_obj.subgroup_size
    return wg, sg


def tid_upper_bound_from_thread_id(op: gpu_d.ThreadIdOp) -> Optional[int]:
    """Extract upper bound from GPU thread ID operation using MLIR Python bindings."""
    # Check if the operation has an upper_bound attribute
    if hasattr(op, "upper_bound") and op.upper_bound is not None:
        if isinstance(op.upper_bound, IntegerAttr):
            return int(op.upper_bound.value)

    return None


def _apply_floor_division_pattern(
    expression: sympy.Expr, tid_upper_bound: int
) -> sympy.Expr:
    """Apply floor division simplification pattern: if x < N, then floor(x/N) = 0."""
    if tid_upper_bound is None or tid_upper_bound <= 0:
        return expression

    free_symbols = expression.free_symbols
    for symbol in free_symbols:
        symbol_name = str(symbol)
        if symbol_name == "tid_x":
            # Find all floor divisions involving tid_x and substitute with 0
            # when the divisor is greater than or equal to the upper bound
            floor_substitutions = {}

            # Find floor(x/N) patterns where N >= tid_upper_bound
            def find_floor_substitutions(expr):
                if isinstance(expr, sympy.floor):
                    # Check if this is floor(symbol/N) where N >= upper_bound
                    if len(expr.args) == 1:
                        arg = expr.args[0]
                        if isinstance(arg, sympy.Mul) and len(arg.args) == 2:
                            # Check if it's symbol * (1/N) or (1/N) * symbol
                            for i, factor in enumerate(arg.args):
                                if factor == symbol:
                                    # Get the other factor (should be 1/N)
                                    other_factor = arg.args[1 - i]

                                    # Handle Rational (like 1/64)
                                    if isinstance(other_factor, sympy.Rational):
                                        divisor = other_factor.denominator
                                        if int(divisor) >= tid_upper_bound:
                                            floor_substitutions[expr] = 0

                                    # Handle Pow with -1 exponent (like 1/N)
                                    elif (
                                        isinstance(other_factor, sympy.Pow)
                                        and other_factor.exp == -1
                                    ):
                                        divisor = other_factor.base
                                        if int(divisor) >= tid_upper_bound:
                                            floor_substitutions[expr] = 0
                elif hasattr(expr, "args"):
                    for arg in expr.args:
                        find_floor_substitutions(arg)

            find_floor_substitutions(expression)

            # Apply the floor substitutions
            if floor_substitutions:
                expression = expression.subs(floor_substitutions)

    return expression


def _apply_simplification_patterns(
    expression: sympy.Expr, tid_upper_bound: int
) -> sympy.Expr:
    """Apply a list of simplification patterns to the expression."""
    # List of pattern functions to apply
    pattern_functions = [
        _apply_floor_division_pattern,
        # Add more pattern functions here as needed
    ]

    # Apply each pattern function
    for pattern_function in pattern_functions:
        expression = pattern_function(expression, tid_upper_bound)

    return expression


def _create_substitutions(values: List, prefix: str) -> Dict[sympy.Symbol, sympy.Expr]:
    """
    Create substitutions for affine map parameters.

    Args:
        values: List of values (integers, ID strings like "tid_x", "wgid_x", "s4",
                or SymPy expressions from previous affine.apply results)
        prefix: Prefix for the symbols ("d" for dimensions, "s" for symbols)

    Returns:
        Dictionary mapping parameter symbols (d0, d1, ... or s0, s1, ...) to expressions
    """
    substitutions = {}

    for i, value in enumerate(values):
        param_symbol = sympy.Symbol(f"{prefix}{i}")
        if isinstance(value, sympy.Expr):
            # SymPy expression from previous affine.apply - use directly
            substitutions[param_symbol] = value
        elif isinstance(value, int):
            substitutions[param_symbol] = sympy.Integer(value)
        elif value in ["tid_x", "tid_y", "tid_z", "wgid_x", "wgid_y", "wgid_z"]:
            # Thread IDs and workgroup IDs are symbolic runtime values
            substitutions[param_symbol] = sympy.Symbol(value, nonnegative=True)
        elif isinstance(value, str) and value.startswith("s") and value[1:].isdigit():
            # SGPR references (e.g., "s4" for loop counter) - keep as symbol
            # The expression emitter will handle these as direct SGPR references
            sgpr_symbol = sympy.Symbol(value, nonnegative=True)
            substitutions[param_symbol] = sgpr_symbol

    return substitutions


def simplify_expression(
    map_attr: AffineMapAttr,
    tid_upper_bound: int,
    dim_values: List = None,
    symbol_values: List = None,
) -> Optional[sympy.Expr]:
    """Simplify affine map expression using SymPy and return the simplified expression."""
    if not isinstance(map_attr, AffineMapAttr):
        return None

    affine_map = map_attr.value
    if not isinstance(affine_map, AffineMap):
        return None

    # Get the result expressions
    results = affine_map.results
    if len(results) != 1:
        return None

    result_expr = results[0]

    # Convert MLIR affine expression to SymPy expression
    try:
        sympy_expr = _mlir_affine_to_sympy(result_expr)
        if sympy_expr is None:
            return None

        # Substitute dimension and symbol values if provided
        if dim_values is not None or symbol_values is not None:
            substitutions = {}

            # Create dimension substitutions (d0, d1, ...)
            if dim_values is not None:
                dimension_substitutions = _create_substitutions(dim_values, "d")
                substitutions.update(dimension_substitutions)

            # Create symbol substitutions (s0, s1, ...)
            if symbol_values is not None:
                symbol_substitutions = _create_substitutions(symbol_values, "s")
                substitutions.update(symbol_substitutions)

            # Apply substitutions
            if substitutions:
                sympy_expr = sympy_expr.subs(substitutions)

        # Apply pattern-based simplifications
        sympy_expr = _apply_simplification_patterns(sympy_expr, tid_upper_bound)

        # Return the simplified expression
        return sympy.simplify(sympy_expr)

    except Exception:
        # If SymPy conversion fails, return None
        return None


def _mlir_affine_to_sympy(expr) -> Optional[sympy.Expr]:
    """Convert MLIR affine expression to SymPy expression by parsing the string representation."""
    try:
        expr_str = str(expr)

        # Handle simple cases first
        if expr_str.isdigit():
            return sympy.Integer(int(expr_str))

        # Replace MLIR-specific operators with Python equivalents
        python_expr = (
            expr_str.replace("floordiv", "//")
            .replace("ceildiv", "ceil")
            .replace("mod", "%")
        )

        # Parse the expression using SymPy's parser
        from sympy.parsing.sympy_parser import parse_expr

        parsed_expr = parse_expr(python_expr)

        # Keep the original symbol names (d0, d1, s0, s1, etc.)
        # The caller will substitute actual values for these symbols
        return parsed_expr

    except Exception as e:
        raise ValueError(
            f"Failed to convert MLIR affine expression to SymPy expression: {e}"
        )


def build_memref_byte_offset_expr(
    indices_ssa: List[str], kernel_info: "KernelInfo", memref_info: "MemRefInfo"
) -> sympy.Expr:
    """
    Build a symbolic byte-offset expression from MLIR indices and memref strides.

    Computes: sum(index[i] * stride[i]) * elem_bytes

    Args:
        indices_ssa: List of SSA value strings representing indices
        kernel_info: Kernel information containing index environment
        memref_info: Memref information containing shape, strides, and element size

    Returns:
        SymPy expression representing the byte offset

    Raises:
        ValueError: If an index SSA is not found or has an unsupported type
    """
    # Mapping for thread ID and workgroup ID strings to SymPy symbols
    ID_SYMBOLS = {
        "tid_x": sympy.Symbol("tid_x", nonnegative=True),
        "tid_y": sympy.Symbol("tid_y", nonnegative=True),
        "tid_z": sympy.Symbol("tid_z", nonnegative=True),
        "wgid_x": sympy.Symbol("wgid_x", nonnegative=True),
        "wgid_y": sympy.Symbol("wgid_y", nonnegative=True),
        "wgid_z": sympy.Symbol("wgid_z", nonnegative=True),
    }

    def to_sympy(value) -> sympy.Expr:
        """Convert index value to SymPy expression."""
        from .kernel_ir import KSReg, KPhysSReg

        if isinstance(value, sympy.Expr):
            return value
        if isinstance(value, int):
            return sympy.Integer(value)
        if isinstance(value, str):
            if value in ID_SYMBOLS:
                return ID_SYMBOLS[value]
            # Handle SGPR references like "s24" (loop counters)
            if value.startswith("s") and value[1:].isdigit():
                return sympy.Symbol(value, nonnegative=True)
            raise ValueError(f"Unknown ID string: {value}")
        # Handle virtual SGPR (loop counter) - create symbol using virtual register id
        # The symbol name uses ks{id} to indicate it's a virtual SGPR
        if isinstance(value, KSReg):
            return sympy.Symbol(f"ks{value.id}", nonnegative=True)
        # Handle physical SGPR references
        if isinstance(value, KPhysSReg):
            return sympy.Symbol(f"s{value.index}", nonnegative=True)
        raise ValueError(f"Unsupported index type: {type(value)}")

    # Build element index expression: sum(index[i] * stride[i])
    elem_index_expr = sum(
        (to_sympy(kernel_info.index_env[idx_ssa]) * memref_info.strides_elems[dim])
        for dim, idx_ssa in enumerate(indices_ssa)
        if idx_ssa in kernel_info.index_env
    )

    # Verify all indices were found
    missing = [idx for idx in indices_ssa if idx not in kernel_info.index_env]
    if missing:
        raise ValueError(f"Index SSAs not found in kernel_info.index_env: {missing}")

    # Convert to byte offset
    return elem_index_expr * memref_info.elem_bytes


def split_const_dynamic(
    expr: sympy.Expr, max_immediate: int = 4096
) -> Tuple[int, sympy.Expr]:
    """
    Split a SymPy expression into constant and dynamic parts for buffer addressing.

    For large matrices, buffer instruction immediate offsets can overflow the 16-bit limit (65535).
    We clamp the immediate offset to a safe value and fold any overflow into the VGPR address.

    Args:
        expr: SymPy expression to split (e.g., tid_x * 4 + 128)
        max_immediate: Maximum value for immediate offset (default 4096 for safety)
                       Buffer instructions support 16-bit unsigned (0-65535), but we use
                       a conservative limit to ensure large constants are handled in VGPRs.
                       Set to 0 to force all constants into VGPR (matching LLVM's approach).

    Returns:
        Tuple of (constant_term, dynamic_expr)
        - constant_term: Integer constant for instruction immediate (clamped to max_immediate)
        - dynamic_expr: Dynamic expression with overflow folded in if needed

    Example:
        >>> split_const_dynamic(tid_x * 4 + 128)  # 128 < 4096, stays in immediate
        (128, tid_x * 4)
        >>> split_const_dynamic(tid_x * 4 + 100000)  # 100000 > 4096, folded into vaddr
        (0, tid_x * 4 + 100000)
    """
    # Use SymPy's built-in as_coeff_Add which returns (const, rest)
    const, rest = expr.as_coeff_Add()

    # Convert constant to integer
    const_term = int(const) if const.is_number else 0

    # If rest is just the constant part, dynamic is zero
    dynamic_expr = rest if rest != const else sympy.Integer(0)

    # Clamp immediate to max_immediate and fold remainder into dynamic expression
    if const_term > max_immediate:
        # Fold the constant into the dynamic expression
        if const_term > 0:
            dynamic_expr = dynamic_expr + sympy.Integer(const_term)
        const_term = 0

    return const_term, dynamic_expr


def build_element_byte_offset_exprs(
    value_vector_type,
    indices_ssa: List[str],
    kernel_info: "KernelInfo",
    memref_info: "MemRefInfo",
) -> List[sympy.Expr]:
    """
    Build per-element byte-offset expressions for a vector store.

    For a vector store of vector<NxT> to a memref, this computes the byte offset
    for each of the N elements by deriving everything from MLIR affine maps and indices.

    The MLIR indices already encode any complex layout patterns (e.g., MFMA layouts),
    so we just need to compute the base offset from the indices and add contiguous
    element offsets.

    Args:
        value_vector_type: The MLIR vector type being stored (e.g., vector<4xf32>)
        indices_ssa: List of SSA value strings representing store indices
        kernel_info: Kernel information containing index environment with affine maps
        memref_info: Memref information containing shape, strides, and element size

    Returns:
        List of SymPy expressions, one per vector element, representing byte offsets

    Raises:
        ValueError: If vector is not 1-D or indices are not found in environment
    """
    # Parse vector type to get number of elements
    num_elements, elem_bytes, _ = parse_vector_type_from_obj(value_vector_type)

    # Build base offset from MLIR indices (which include affine maps encoding layout)
    base_expr = build_memref_byte_offset_expr(indices_ssa, kernel_info, memref_info)

    # Determine element stride in bytes (innermost dimension)
    if len(memref_info.strides_elems) > 0:
        elem_stride_elems = memref_info.strides_elems[-1]
    else:
        elem_stride_elems = 1
    elem_stride_bytes = elem_stride_elems * memref_info.elem_bytes

    # Build list of expressions: base + k * elem_stride_bytes for k in 0..num_elements-1
    # The affine maps in indices_ssa already encode any complex access patterns
    exprs = []
    for k in range(num_elements):
        if k == 0:
            exprs.append(base_expr)
        else:
            offset_expr = base_expr + sympy.Integer(k * elem_stride_bytes)
            exprs.append(offset_expr)

    return exprs
