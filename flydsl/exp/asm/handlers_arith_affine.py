# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import operator

import sympy
from wave_lang.support.ir_imports import (
    affine_d,
    arith_d,
    gpu_d,
)

from .handlers_shared import *
from .kernel_ir import KSReg
from .kernel_model import KernelInfo
from .utils import simplify_expression, tid_upper_bound_from_thread_id


class _ArithAffineHandlers:
    def handle_arith_constant_op(
        self, operation: arith_d.ConstantOp, kernel_info: KernelInfo
    ):
        """Handle arith.constant operations - extract constant values for index environment."""
        # Extract integer constants for the index environment
        # Skip non-integer constants (floats, vectors, etc.) without error
        value_attribute = operation.value
        if not hasattr(value_attribute, "value"):
            return

        value = value_attribute.value
        # Only store if it's an integer or can be safely converted to one
        if isinstance(value, int):
            kernel_info.index_env[str(operation.result)] = value
        elif (
            hasattr(value, "is_integer")
            and callable(value.is_integer)
            and value.is_integer()
        ):
            # Handle float-like types that represent exact integers
            kernel_info.index_env[str(operation.result)] = int(value)

    def _handle_arith_binop(self, operation, kernel_info: KernelInfo, op_func):
        """Handle binary arithmetic operations (addi, muli) in index_env.

        Args:
            operation: The MLIR operation (AddIOp or MulIOp)
            kernel_info: Kernel info containing index_env
            op_func: Binary function to apply (e.g., operator.add, operator.mul)
        """
        lhs = kernel_info.index_env.get(str(operation.operands[0]))
        rhs = kernel_info.index_env.get(str(operation.operands[1]))

        # Operands not tracked - can't compute result
        if lhs is None or rhs is None:
            return

        # Convert symbolic strings (tid_x, wgid_x, etc.) to SymPy symbols
        if isinstance(lhs, str):
            lhs = sympy.Symbol(lhs)
        if isinstance(rhs, str):
            rhs = sympy.Symbol(rhs)

        if isinstance(lhs, (int, sympy.Expr)) and isinstance(rhs, (int, sympy.Expr)):
            kernel_info.index_env[str(operation.result)] = op_func(lhs, rhs)

    def handle_arith_addi_op(self, operation: arith_d.AddIOp, kernel_info: KernelInfo):
        """Handle arith.addi - track integer addition in index_env."""
        self._handle_arith_binop(operation, kernel_info, operator.add)

    def handle_arith_muli_op(self, operation: arith_d.MulIOp, kernel_info: KernelInfo):
        """Handle arith.muli - track integer multiplication in index_env."""
        self._handle_arith_binop(operation, kernel_info, operator.mul)

    def handle_arith_index_cast_op(
        self, operation: arith_d.IndexCastOp, kernel_info: KernelInfo
    ):
        """Handle arith.index_cast operations - propagate values through cast.

        Propagates integers, SymPy expressions, and symbolic strings (tid_x, etc.).
        """
        result_ssa = str(operation.result)
        src_ssa = str(operation.operands[0])

        src_val = kernel_info.index_env.get(src_ssa)
        if src_val is None:
            return

        # Propagate numeric values and symbolic strings
        if isinstance(src_val, (int, sympy.Expr, str)):
            kernel_info.index_env[result_ssa] = src_val

    def handle_gpu_thread_id_op(
        self, operation: gpu_d.ThreadIdOp, kernel_info: KernelInfo
    ):
        """Handle gpu.thread_id operations - extract thread ID information."""
        upper_bound = tid_upper_bound_from_thread_id(operation)
        # Get the actual dimension from the operation
        dimension = operation.dimension
        # Extract dimension from MLIR attribute string like "#gpu<dim x>"
        dimension_string = str(dimension)
        if "dim x" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "tid_x"
            if upper_bound is not None:
                kernel_info.tid_ub_x = upper_bound
        elif "dim y" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "tid_y"
            if upper_bound is not None:
                kernel_info.tid_ub_y = upper_bound
        elif "dim z" in dimension_string:
            if upper_bound is not None:
                kernel_info.tid_ub_z = upper_bound
            kernel_info.index_env[str(operation.result)] = "tid_z"

    def handle_gpu_block_id_op(
        self, operation: gpu_d.BlockIdOp, kernel_info: KernelInfo
    ):
        """
        Handle gpu.block_id operations - map to workgroup ID symbols.

        Maps block IDs to symbolic names that the expression emitter can use:
        - block_id x -> wgid_x
        - block_id y -> wgid_y
        - block_id z -> wgid_z
        """
        dimension = operation.dimension
        dimension_string = str(dimension)

        if "dim x" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "wgid_x"
        elif "dim y" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "wgid_y"
        elif "dim z" in dimension_string:
            kernel_info.index_env[str(operation.result)] = "wgid_z"

    def handle_subgroup_broadcast_op(
        self, operation: gpu_d.SubgroupBroadcastOp, kernel_info: KernelInfo
    ):
        """Handle gpu.subgroup_broadcast - propagate value for uniform broadcast.

        The expression is preserved as-is (not evaluated) because each wavefront
        has different tid values. v_readfirstlane is emitted during code generation.
        """
        result_ssa = str(operation.results[0])
        source_ssa = str(operation.src)

        if source_ssa in kernel_info.index_env:
            kernel_info.index_env[result_ssa] = kernel_info.index_env[source_ssa]

    def _extract_dimension_values(
        self,
        operation: affine_d.AffineApplyOp,
        kernel_info: KernelInfo,
        num_dimensions: int,
    ) -> list:
        """Extract dimension values from the first num_dimensions operands."""
        import sympy

        dimension_values = []

        for i in range(num_dimensions):
            if i < len(operation.operands):
                operand_ssa = str(operation.operands[i])
                operand_value = kernel_info.index_env.get(operand_ssa)

                if isinstance(operand_value, int):
                    dimension_values.append(operand_value)
                elif isinstance(operand_value, sympy.Expr):
                    # SymPy expressions from previous affine.apply results
                    dimension_values.append(operand_value)
                elif operand_value in [
                    "tid_x",
                    "tid_y",
                    "tid_z",
                    "wgid_x",
                    "wgid_y",
                    "wgid_z",
                ]:
                    # Thread IDs and workgroup IDs can be represented as symbols in the expression
                    dimension_values.append(operand_value)
                else:
                    # If we can't resolve the dimension value, we can't simplify
                    return None
            else:
                # Not enough operands for the expected number of dimensions
                return None

        return dimension_values

    def _extract_symbol_values(
        self,
        operation: affine_d.AffineApplyOp,
        kernel_info: KernelInfo,
        num_dimensions: int,
        num_symbols: int,
    ) -> list:
        """Extract symbol values from the next num_symbols operands."""
        import sympy

        symbol_values = []

        for i in range(num_symbols):
            operand_index = num_dimensions + i
            if operand_index < len(operation.operands):
                operand_ssa = str(operation.operands[operand_index])
                operand_value = kernel_info.index_env.get(operand_ssa)

                if isinstance(operand_value, int):
                    symbol_values.append(operand_value)
                elif isinstance(operand_value, sympy.Expr):
                    # SymPy expressions from previous affine.apply results
                    symbol_values.append(operand_value)
                elif operand_value in [
                    "tid_x",
                    "tid_y",
                    "tid_z",
                    "wgid_x",
                    "wgid_y",
                    "wgid_z",
                ]:
                    # Thread IDs and workgroup IDs can be used as symbol values
                    symbol_values.append(operand_value)
                elif (
                    isinstance(operand_value, str)
                    and operand_value.startswith("s")
                    and operand_value[1:].isdigit()
                ):
                    # SGPR references (e.g., "s4" for loop counter) can be used as symbol values
                    symbol_values.append(operand_value)
                elif isinstance(operand_value, KSReg):
                    # Virtual SGPR references (loop counters) can be used as symbol values
                    # Convert to ks{id} symbol format for expression evaluation
                    symbol_values.append(
                        sympy.Symbol(f"ks{operand_value.id}", nonnegative=True)
                    )
                else:
                    # If we can't resolve the symbol value, we can't simplify
                    return None
            else:
                # Not enough operands for the expected number of symbols
                return None

        return symbol_values

    def handle_affine_apply_op(
        self, operation: affine_d.AffineApplyOp, kernel_info: KernelInfo
    ):
        """Handle affine.apply operations - simplify affine expressions."""
        # The first operands correspond to dimensions, the rest to symbols
        affine_map_attribute = operation.map
        affine_map = affine_map_attribute.value
        num_dimensions = affine_map.n_dims
        num_symbols = affine_map.n_symbols

        # Extract dimension and symbol values from operands
        dimension_values = self._extract_dimension_values(
            operation, kernel_info, num_dimensions
        )
        symbol_values = self._extract_symbol_values(
            operation, kernel_info, num_dimensions, num_symbols
        )

        # Try to simplify the expression with actual values
        simplified_expression = simplify_expression(
            operation.map, kernel_info.tid_ub_x, dimension_values, symbol_values
        )

        destination_ssa = str(operation.result)
        if simplified_expression is not None:
            # Check if the simplified expression is a constant
            if len(simplified_expression.free_symbols) == 0:
                # Expression has no free symbols, so it's a constant - convert to int
                assert hasattr(simplified_expression, "__int__") or hasattr(
                    simplified_expression, "is_integer"
                ), f"Simplified expression without free symbols should be convertible to int: {simplified_expression}"
                constant_value = int(simplified_expression)
                kernel_info.index_env[destination_ssa] = constant_value
            # Check if the simplified expression is a thread ID symbol
            else:
                import sympy

                # Check for all thread ID types
                thread_id_symbols = {
                    "tid_x": sympy.Symbol("tid_x", nonnegative=True),
                    "tid_y": sympy.Symbol("tid_y", nonnegative=True),
                    "tid_z": sympy.Symbol("tid_z", nonnegative=True),
                }

                matched_tid = False
                for thread_id_name, thread_id_symbol in thread_id_symbols.items():
                    if simplified_expression == thread_id_symbol:
                        # Map back to the original thread ID format
                        original_thread_id = thread_id_name.replace("_", ".")
                        source_ssa = (
                            str(operation.operands[0])
                            if len(operation.operands) > 0
                            else None
                        )
                        if (
                            source_ssa
                            and kernel_info.index_env.get(source_ssa)
                            == original_thread_id
                        ):
                            kernel_info.index_env[destination_ssa] = original_thread_id
                            matched_tid = True
                        break

                if not matched_tid:
                    # Store the simplified SymPy expression for later ASM emission
                    kernel_info.index_env[destination_ssa] = simplified_expression
