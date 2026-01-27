# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Unified instruction emitter for AMDGCN kernels.

This module provides a single class that owns instruction definitions
and emission for kernel IR (whole-program register allocation).

The emitter generates methods dynamically from instruction definitions,
ensuring consistency between the instruction registry and emission API.

All instruction formatting goes through InstructionFormatter, which is
the SINGLE point for physical instruction rendering.

Usage:
    from kernel_compilation_context import KernelCompilationContext
    ctx = KernelCompilationContext()
    # ctx.unified provides access to instruction emission methods
    result = ctx.unified.v_add_u32(src0=v1, src1=v2)  # Returns virtual register
"""

from enum import Enum, auto
from typing import List, Optional, Any, Callable

from .instruction_registry import (
    InstructionRegistry,
    InstructionDef,
    get_registry,
)
from .instruction_categories import InstructionCategory
from .instruction_formatter import InstructionFormatter, get_formatter


# ==============================================================================
# Emission Modes
# ==============================================================================


class EmissionMode(Enum):
    """Mode of instruction emission."""

    KERNEL_IR = auto()  # Emit to KernelProgram via KernelCompilationContext


# ==============================================================================
# Unified Emitter
# ==============================================================================


class UnifiedEmitter:
    """
    Unified instruction emitter for kernel IR emission.

    This class dynamically generates emission methods from the instruction
    registry, providing a consistent API for instruction emission.

    Methods emit to a KernelCompilationContext and return virtual register results.

    All physical instruction formatting goes through InstructionFormatter.
    """

    def __init__(
        self,
        architecture: str = "common",
        mode: EmissionMode = EmissionMode.KERNEL_IR,
        context: Any = None,  # KernelCompilationContext for KERNEL_IR mode
    ):
        self.architecture = architecture
        self.mode = mode
        self.context = context
        self._registry = get_registry(architecture)
        self._formatter = get_formatter(architecture)
        self._lines: List[str] = []

        # Generate emission methods
        self._generate_methods()

    def _generate_methods(self) -> None:
        """Generate emission methods for all instructions in the registry."""
        for instr in self._registry:
            if instr.category != InstructionCategory.PSEUDO:  # Skip pseudo-ops
                method = self._create_emission_method(instr)
                setattr(self, instr.name, method)

    def _create_emission_method(self, instr: InstructionDef) -> Callable:
        """Create an emission method for an instruction."""

        def emit_method(*args, comment: str = None, **kwargs):
            """Emit instruction."""
            return self._emit_kernel_ir(instr, args, kwargs, comment)

        # Set docstring
        emit_method.__doc__ = f"""
        Emit {instr.mnemonic} instruction.

        Mnemonic: {instr.mnemonic}
        Category: {instr.category.name}
        Latency: {instr.latency}
        """

        return emit_method

    def _emit_kernel_ir(
        self,
        instr: InstructionDef,
        args: tuple,
        kwargs: dict,
        comment: str,
    ) -> Any:
        """Emit instruction to kernel IR via context."""
        if self.context is None:
            raise ValueError("KernelCompilationContext required for emission")

        # Map instruction to context method if available
        method_name = instr.name
        if hasattr(self.context, method_name):
            ctx_method = getattr(self.context, method_name)
            # Call context method with provided args
            return ctx_method(*args, comment=comment, **kwargs)

        # Fallback: emit raw using InstructionFormatter
        defs, uses = self._parse_args(instr, args, kwargs)
        line = self._formatter.format(instr.name, defs=defs, uses=uses, comment=comment)
        if line:
            self.context.emit_raw(line)

        return None

    def _parse_args(self, instr: InstructionDef, args: tuple, kwargs: dict):
        """Parse positional and keyword args into defs and uses."""
        defs = []
        uses = []

        arg_idx = 0
        for def_op in instr.defs:
            if arg_idx < len(args):
                defs.append(args[arg_idx])
                arg_idx += 1
            elif def_op.name in kwargs:
                defs.append(kwargs[def_op.name])

        for use_op in instr.uses:
            if arg_idx < len(args):
                uses.append(args[arg_idx])
                arg_idx += 1
            elif use_op.name in kwargs:
                uses.append(kwargs[use_op.name])
            elif use_op.optional:
                uses.append(None)

        return defs, uses

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_instruction_def(self, name: str) -> Optional[InstructionDef]:
        """Get instruction definition by name."""
        return self._registry.get(name)

    def get_latency(self, name: str) -> int:
        """Get instruction latency."""
        instr = self._registry.get(name)
        return instr.latency if instr else 1

    @property
    def registry(self) -> InstructionRegistry:
        """Get the underlying instruction registry."""
        return self._registry

    @property
    def formatter(self) -> InstructionFormatter:
        """Get the underlying instruction formatter."""
        return self._formatter


# ==============================================================================
# Factory Function
# ==============================================================================


def create_kernel_ir_emitter(
    context: Any,
    architecture: str = "common",
) -> UnifiedEmitter:
    """Create an emitter for kernel IR emission."""
    return UnifiedEmitter(
        architecture=architecture,
        mode=EmissionMode.KERNEL_IR,
        context=context,
    )
