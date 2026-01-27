# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .kernel_model import KernelInfo
    from wave_lang.kernel.wave.constraints import MMAType

# Core IR types
from .kernel_ir import (
    KernelProgram,
    KernelBuilder,
    KInstr,
    KVReg,
    KSReg,
    KPhysVReg,
    KPhysSReg,
    KReg,
    KRegRange,
    KImm,
    KMemOffset,
    KernelABI,
    M0,
    is_virtual,
)

# Liveness and allocation

# Code generation
from .unified_emitter import UnifiedEmitter, EmissionMode

# Instruction info
from .instruction_categories import InstructionCategory, categorize_instruction
from .instruction_registry import InstructionDef, get_registry, Instruction

# Ticketing
from .ticketing import Ticketing

# Shared helpers
from .kernel_pipeline_shared import (
    _kernel_ticketing_enabled,
    _get_def_info,
    _parse_waitcnt_threshold,
    _iter_virtual_regs,
)

from .kernel_expr_emitter import KernelIRExprEmitter
from .kernel_loops import _LoopSupport
from .kernel_mfma import _MFMASupport
from .kernel_passes import _CompilationPasses

# ABI Constants
# Number of SGPRs used for kernarg segment pointer (s[0:1])
KERNARG_PTR_SGPRS = 2


@dataclass
class KernelCompilationContext(_LoopSupport, _MFMASupport, _CompilationPasses):
    """
    Context for kernel compilation with whole-program register allocation.

    This context manages:
    - The KernelProgram being built
    - Symbol bindings (MLIR SSA values to virtual registers)
    - ABI register configuration
    - CSE cache for expression deduplication

    Instructions are emitted via dynamic dispatch:
        ctx.v_add_u32(src0, src1)  # Calls _emit_instruction("v_add_u32", ...)

    Usage:
        ctx = KernelCompilationContext(max_vgprs=256, max_sgprs=104)

        # Emit instructions - methods generated dynamically
        v1 = ctx.v_mov_b32(42)
        v2 = ctx.v_add_u32(v1, 100)
        ctx.ds_read_b64(addr)

        # Finalize and get assembly
        asm_lines = ctx.finalize()
    """

    # Configuration
    max_vgprs: int = 256
    max_sgprs: int = 104

    # ABI configuration
    use_flat_tid: bool = True
    use_workgroup_ids: Tuple[bool, bool, bool] = (True, True, True)  # x, y, z

    # Thread ID bounds for algebraic simplification
    # These are set from kernel_info when the context is created
    tid_ub_x: int = 64  # Upper bound for tid_x (exclusive)
    tid_ub_y: int = 1  # Upper bound for tid_y (exclusive)
    tid_ub_z: int = 1  # Upper bound for tid_z (exclusive)
    subgroup_size: int = 64
    wg_size: Tuple[int, int, int] = (64, 1, 1)

    # MMA type for MFMA instruction selection
    mma_type: Optional["MMAType"] = None

    # Internal state
    program: KernelProgram = field(init=False)
    builder: KernelBuilder = field(init=False)
    _unified: UnifiedEmitter = field(init=False)
    _registry: InstructionRegistry = field(init=False)

    # Symbol bindings: MLIR SSA value string -> virtual register
    _symbol_bindings: Dict[str, KReg] = field(default_factory=dict, init=False)

    # SSA to register mapping for MLIR SSA values
    # Maps SSA value string to tuple of virtual registers (for multi-reg results like loads)
    # This is the source of truth for all SSA-to-register tracking
    ssa_to_reg: Dict[str, Tuple[KReg, ...]] = field(default_factory=dict, init=False)

    # CSE cache: expression key -> virtual register
    _cse_cache: Dict[tuple, KVReg] = field(default_factory=dict, init=False)

    # SRD ranges: memref_ssa -> SGPR quad range for buffer descriptor
    srd_ranges: Dict[str, KRegRange] = field(default_factory=dict, init=False)

    # Pending SRD setups: list of (srd_range, arg_idx, limit_bytes)
    # These are queued during IR emission and inserted at program start in finalize()
    _pending_srd_setups: List[Tuple[KRegRange, int, int]] = field(
        default_factory=list, init=False
    )

    # Pending ABI prologue instructions: list of KInstr
    # These define tid_x, tid_y, wgid_* etc in the prologue for proper dominance
    _pending_abi_prologue: List[KInstr] = field(default_factory=list, init=False)

    # Prologue-defined ABI VGPRs (allocated early, defined in prologue)
    _prologue_tid_x: Optional[KVReg] = field(default=None, init=False)
    _prologue_tid_y: Optional[KVReg] = field(default=None, init=False)
    _prologue_wgid_x: Optional[KVReg] = field(default=None, init=False)
    _prologue_wgid_y: Optional[KVReg] = field(default=None, init=False)
    _prologue_wgid_z: Optional[KVReg] = field(default=None, init=False)

    # Loop management
    _loop_counter: int = field(default=0, init=False)
    _loop_stack: List[dict] = field(default_factory=list, init=False)

    # Ticketing / waitcnt coalescing state
    _ticketing: Ticketing = field(default_factory=Ticketing, init=False)

    # Kernarg management: arg_index -> (low_sgpr_reg, high_sgpr_reg) as KRegRange
    _kernarg_pairs: Dict[int, KRegRange] = field(default_factory=dict, init=False)
    _kernargs_emitted: bool = field(default=False, init=False)

    # Kernel argument preloading: if True, use preloaded SGPRs instead of s_load
    # When enabled, hardware preloads args into s[2:3], s[4:5], etc. at kernel start
    use_kernarg_preloading: bool = False

    # Number of kernel arguments (needed for preloading SGPR reservation)
    num_kernargs: int = 0

    # Kernel name (used for generating unique labels)
    kernel_name: str = "kernel"

    # Statistics
    _cse_hits: int = field(default=0, init=False)

    # Precolored SGPRs: virtual SGPR -> physical SGPR index
    # Used to force kernarg SGPRs to their hardware-preloaded locations
    _precolored_sregs: Dict[KSReg, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Calculate preload SGPR count (2 SGPRs per 64-bit pointer arg)
        preload_sgpr_count = self.num_kernargs * 2 if self.use_kernarg_preloading else 0

        # Workgroup IDs come after preloaded args (if any)
        # Layout: s[0:1]=kernarg_ptr, s[2:2+preload-1]=preloaded args, then wgid
        wgid_start = KERNARG_PTR_SGPRS + preload_sgpr_count

        # Initialize ABI
        abi = KernelABI()
        if self.use_flat_tid:
            abi.flat_tid_vreg = KPhysVReg(0)
        if self.use_workgroup_ids[0]:
            abi.workgroup_id_x_sreg = KPhysSReg(wgid_start)
        if self.use_workgroup_ids[1]:
            abi.workgroup_id_y_sreg = KPhysSReg(wgid_start + 1)
        if self.use_workgroup_ids[2]:
            abi.workgroup_id_z_sreg = KPhysSReg(wgid_start + 2)

        # Set preload count in ABI for reservation
        abi.preload_sgpr_count = preload_sgpr_count

        # Create program
        self.program = KernelProgram(
            abi=abi, max_vgprs=self.max_vgprs, max_sgprs=self.max_sgprs
        )
        self.builder = KernelBuilder(self.program)

        # Load instruction registry
        self._registry = get_registry("common")

        # Create unified emitter in KERNEL_IR mode
        # This allows callers to use kernel_ctx.unified.v_add_u32(...) syntax
        self._unified = UnifiedEmitter(
            architecture="common",
            mode=EmissionMode.KERNEL_IR,
            context=self,
        )

        # Expression emitter for sympy expressions (lazy init)
        self._expr_emitter = None

    @property
    def expr_emitter(self) -> "KernelIRExprEmitter":
        """Get the expression emitter for this context."""
        if self._expr_emitter is None:
            self._expr_emitter = KernelIRExprEmitter(self)
        return self._expr_emitter

    @property
    def ticketing(self) -> Ticketing:
        """Ticketing interface for VMEM/LGKM operations (kernel IR path)."""
        return self._ticketing

    def update_bounds_from_kernel_info(self, kernel_info: "KernelInfo") -> None:
        """
        Update thread ID bounds from kernel_info after MLIR parsing.

        This should be called after interpret_func() returns, as that's when
        we know the actual workgroup_size and subgroup_size from the MLIR.

        Args:
            kernel_info: KernelInfo populated by interpret_func()

        Raises:
            ValueError: If kernel_info is missing wg_size or subgroup_size
        """
        if not hasattr(kernel_info, "wg_size"):
            raise ValueError(
                "kernel_info.wg_size is required but not set. "
                "Ensure translation_info is parsed from MLIR."
            )
        if not hasattr(kernel_info, "subgroup_size"):
            raise ValueError(
                "kernel_info.subgroup_size is required but not set. "
                "Ensure translation_info is parsed from MLIR."
            )
        wg_size = kernel_info.wg_size
        subgroup_size = kernel_info.subgroup_size

        self.wg_size = wg_size
        self.subgroup_size = subgroup_size

        # Update use_flat_tid based on actual num_waves
        num_waves = max(1, wg_size[0] * wg_size[1] * wg_size[2] // subgroup_size)
        self.use_flat_tid = num_waves > 1

        # In multi-wave mode, tid_x/tid_y/tid_z are extracted from flat_tid
        # and bounded by the workgroup dimensions wg_size = (x, y, z)
        # In single-wave mode, tid_x = lane_id (0 to subgroup_size-1) and tid_y/z = 0
        if self.use_flat_tid:
            # Multi-wave: unpack wg_size tuple for tid bounds
            self.tid_ub_x = wg_size[0]
            self.tid_ub_y = wg_size[1]
            self.tid_ub_z = wg_size[2] if len(wg_size) > 2 else 1
        else:
            # Single-wave: tid_x = lane_id (bounded by subgroup_size), tid_y/z = 0
            self.tid_ub_x = subgroup_size
            self.tid_ub_y = 1
            self.tid_ub_z = 1

    # =========================================================================
    # Kernarg Loading
    # =========================================================================

    def emit_kernargs(self, num_args: int) -> None:
        """
        Emit kernel argument loading code at the start of the kernel.

        IMPORTANT: We ALWAYS emit s_load_dwordx2 instructions, even when
        hardware preloading is enabled. This follows LLVM's pattern where:
        1. The preloading metadata tells hardware to preload args into SGPRs
        2. The code still issues s_load instructions for compatibility
        3. If hardware preloading worked, the loads complete instantly (or are skipped)
        4. If preloading didn't work (unsupported/disabled), loads still work

        When preloading is enabled, we also "precolor" the allocated SGPRs to
        match the hardware preload locations (s[2:3], s[4:5], etc.). This ensures
        the s_load targets the same registers where hardware preloads, making the
        s_load idempotent (if preloading worked) or functional (if it didn't).

        Args:
            num_args: Number of kernel arguments
        """
        if self._kernargs_emitted:
            return

        # Kernarg pointer is always at s[0:1] (user SGPR)
        kernarg_ptr = KRegRange(KPhysSReg(0), KERNARG_PTR_SGPRS)

        for i in range(num_args):
            # Allocate SGPR pair for this kernel argument
            pair = self.program.alloc_sreg_range(2, alignment=2)
            self._kernarg_pairs[i] = pair

            # If preloading is enabled, pin this virtual SGPR pair to the
            # hardware-preloaded physical location: s[2+i*2 : 3+i*2]
            # This ensures our s_load targets the same regs hardware uses.
            if self.use_kernarg_preloading:
                phys_base = KERNARG_PTR_SGPRS + (i * 2)  # s[2:3], s[4:5], s[6:7], ...
                # Precolor both SGPRs in the pair
                base_sreg = pair.base_reg
                if isinstance(base_sreg, KSReg):
                    self._precolored_sregs[base_sreg] = phys_base
                    # For the second SGPR in the pair
                    second_sreg = KSReg(base_sreg.id + 1)
                    self._precolored_sregs[second_sreg] = phys_base + 1
                # NOTE: We still emit s_load below for compatibility!
                # If preloading works, the s_load completes instantly.
                # If preloading doesn't work, the s_load ensures correctness.

            # ALWAYS emit s_load_dwordx2 to load the pointer (for compatibility)
            offset = i * 8
            self.program.emit(
                KInstr(
                    "s_load_dwordx2",
                    defs=(pair,),
                    uses=(kernarg_ptr, KImm(offset)),
                    comment=f"Load kernarg at offset {offset}",
                )
            )

        # ALWAYS emit s_waitcnt to wait for all kernarg loads
        if num_args > 0:
            self.program.emit(
                KInstr(
                    "s_waitcnt",
                    defs=(),
                    uses=("lgkmcnt(0)",),
                    comment="wait for all kernarg loads",
                )
            )

        # Emit empty line for readability
        self.program.emit(KInstr(Instruction._COMMENT, defs=(), uses=(), comment=""))

        self._kernargs_emitted = True

    def get_kernarg_pair(self, arg_index: int) -> Optional[KRegRange]:
        """
        Get the SGPR pair holding the kernel argument pointer.

        Args:
            arg_index: Kernel argument index

        Returns:
            KRegRange for the SGPR pair, or None if not yet emitted
        """
        return self._kernarg_pairs.get(arg_index)

    # =========================================================================
    # Dynamic instruction dispatch
    # =========================================================================

    def __getattr__(self, name: str) -> Any:
        """
        Dynamic dispatch for instruction methods.

        When ctx.v_add_u32(...) is called and v_add_u32 isn't explicitly defined,
        this method handles it by:
        1. Looking up the instruction in the registry
        2. Allocating destination registers based on operand types
        3. Emitting a KInstr with the instruction name
        """
        # Avoid recursion on internal attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Check if it's an instruction in the registry
        instr_def = self._registry.get(name)
        if instr_def is not None:
            # Create and return an emission method
            def emit_method(*args, comment: str = None, **kwargs):
                return self._emit_instruction(name, instr_def, args, kwargs, comment)

            return emit_method

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def _emit_instruction(
        self,
        name: str,
        instr_def: InstructionDef,
        args: tuple,
        kwargs: dict,
        comment: str,
    ) -> Optional[Any]:
        """
        Emit an instruction with automatic register allocation.

        This method:
        1. Allocates destination registers based on operand types
        2. Emits a KInstr to the program with the instruction name
        3. Returns the destination register(s) if any
        """
        # Determine destination type and allocate
        dst = None
        defs = ()

        if instr_def.defs:
            # Get the first def's type info
            def_op = instr_def.defs[0]
            reg_class, count, alignment = _get_def_info(def_op.types)

            if reg_class == "v":
                if count == 1:
                    dst = self.vreg()
                else:
                    dst = self.program.alloc_vreg_range(count, alignment=alignment)
                defs = (dst,)
            elif reg_class == "s":
                if count == 1:
                    dst = self.sreg()
                else:
                    dst = self.program.alloc_sreg_range(count, alignment=alignment)
                defs = (dst,)

        # Build uses from args and kwargs
        # Parse positional args first, then fill in missing operands from kwargs
        uses = []
        arg_idx = 0

        for use_op in instr_def.uses:
            if arg_idx < len(args):
                # Use positional arg
                value = args[arg_idx]
                arg_idx += 1
            elif use_op.name in kwargs:
                # Use kwarg
                value = kwargs[use_op.name]
            elif use_op.optional:
                # Skip optional operands with no value
                continue
            else:
                # Required operand missing
                value = None

            if value is not None:
                # Handle special operand types
                from .instruction_registry import OperandType

                if OperandType.OFFSET in use_op.types and isinstance(value, int):
                    uses.append(KMemOffset(value))
                elif isinstance(value, int):
                    # Let kernel_generator handle raw ints
                    uses.append(value)
                else:
                    uses.append(value)

        # Also add any remaining positional args (for instructions not fully defined)
        while arg_idx < len(args):
            value = args[arg_idx]
            uses.append(value)
            arg_idx += 1

        # Issue tickets for memory operations (for later waitcnt placement/coalescing).
        mnemonic = instr_def.mnemonic if instr_def else name
        category = categorize_instruction(mnemonic)
        if category == InstructionCategory.VMEM:
            self.ticketing.next_vmem_ticket()
        elif category == InstructionCategory.LGKM:
            self.ticketing.next_lgkm_ticket()

        # Emit the instruction using name string
        self.program.emit(KInstr(name, defs, tuple(uses), comment=comment))

        return dst

    # =========================================================================
    # Ticketing for memory operations
    # =========================================================================

    def get_vmem_ticket(self) -> int:
        """Get the last issued VMEM ticket (or -1 if none issued)."""
        return self.ticketing._vmem_last_ticket

    def get_lgkm_ticket(self) -> int:
        """Get the last issued LGKM ticket (or -1 if none issued)."""
        return self.ticketing._lgkm_last_ticket

    def wait_vmem(self, count: int = 0):
        """Emit s_waitcnt vmcnt(count) to wait for VMEM operations."""
        self.program.emit(
            KInstr(
                Instruction.S_WAITCNT, (), (f"vmcnt({count})",), comment="wait for VMEM"
            )
        )

    def wait_lgkm(self, count: int = 0):
        """Emit s_waitcnt lgkmcnt(count) to wait for LDS/scalar operations."""
        self.program.emit(
            KInstr(
                Instruction.S_WAITCNT,
                (),
                (f"lgkmcnt({count})",),
                comment="wait for LGKM",
            )
        )

    # =========================================================================
    # Ticketing-driven waitcnt placement (optional)
    # =========================================================================

    def _apply_ticketing_waitcnt_placement(self) -> None:
        """
        Insert/coalesce s_waitcnt using ticketing across the KernelProgram.

        This pass is intentionally conservative:
        - It only inserts waits when an instruction *uses* a register defined by
          a VMEM/LGKM instruction.
        - It respects existing s_waitcnt instructions by updating ticketing
          coalescing state via observe_* calls.

        Enabled by default; set WAVE_KERNEL_TICKETING=0 to disable.
        """
        if not _kernel_ticketing_enabled():
            return

        ticketing = Ticketing()
        vmem_def_ticket: Dict[KReg, int] = {}
        lgkm_def_ticket: Dict[KReg, int] = {}

        def _record_defs(
            defs: Tuple[Any, ...], ticket: int, table: Dict[KReg, int]
        ) -> None:
            for d in defs:
                if isinstance(d, KRegRange):
                    base = d.base_reg
                    if is_virtual(base):
                        table[base] = ticket
                        # Record components so later scalar uses of range members resolve.
                        for i in range(d.count):
                            table[type(base)(base.id + i)] = ticket
                elif isinstance(d, (KVReg, KSReg)):
                    if is_virtual(d):
                        table[d] = ticket

        new_instructions: List[KInstr] = []

        for instr in self.program.instructions:
            # Respect externally emitted waits (e.g. from rocdl.s.waitcnt lowering).
            if instr.name == Instruction.S_WAITCNT:
                for u in instr.uses:
                    parsed = _parse_waitcnt_threshold(u)
                    if parsed is None:
                        continue
                    kind, threshold = parsed
                    if kind == "vmem":
                        ticketing.observe_vmem_wait(threshold)
                    elif kind == "lgkm":
                        ticketing.observe_lgkm_wait(threshold)

            # Insert waits before consumers of ticketed defs.
            required_vmem: List[int] = []
            required_lgkm: List[int] = []
            for u in instr.uses:
                for reg in _iter_virtual_regs(u):
                    vt = vmem_def_ticket.get(reg)
                    if vt is not None:
                        required_vmem.append(vt)
                    lt = lgkm_def_ticket.get(reg)
                    if lt is not None:
                        required_lgkm.append(lt)

            if required_vmem:
                # Wait for the newest required op (also implies all earlier).
                threshold = ticketing.compute_vmem_wait(max(required_vmem))
                if threshold is not None:
                    new_instructions.append(
                        KInstr(
                            "s_waitcnt",
                            defs=(),
                            uses=(f"vmcnt({threshold})",),
                            comment="ticketing: wait for VMEM defs",
                        )
                    )

            if required_lgkm:
                threshold = ticketing.compute_lgkm_wait(max(required_lgkm))
                if threshold is not None:
                    new_instructions.append(
                        KInstr(
                            "s_waitcnt",
                            defs=(),
                            uses=(f"lgkmcnt({threshold})",),
                            comment="ticketing: wait for LGKM defs",
                        )
                    )

            # Keep original instruction.
            new_instructions.append(instr)

            # Issue tickets for this instruction and record defs.
            instr_def = self._registry.get(instr.name)
            mnemonic = instr_def.mnemonic if instr_def else instr.name
            category = categorize_instruction(mnemonic)
            if category == InstructionCategory.VMEM:
                ticket = ticketing.next_vmem_ticket()
                _record_defs(instr.defs, ticket, vmem_def_ticket)
            elif category == InstructionCategory.LGKM:
                ticket = ticketing.next_lgkm_ticket()
                _record_defs(instr.defs, ticket, lgkm_def_ticket)

        self.program.instructions = new_instructions

    # =========================================================================
    # Symbol binding (for MLIR SSA value tracking)
    # =========================================================================

    def bind_symbol(self, symbol: str, reg: KReg) -> None:
        """Bind an MLIR SSA value name to a virtual register."""
        self._symbol_bindings[symbol] = reg

    def get_binding(self, symbol: str) -> Optional[KReg]:
        """Get the virtual register bound to an MLIR SSA value."""
        return self._symbol_bindings.get(symbol)

    def require_binding(self, symbol: str) -> KReg:
        """Get the virtual register bound to an MLIR SSA value, or raise."""
        if symbol not in self._symbol_bindings:
            raise ValueError(f"Symbol '{symbol}' not bound to any register")
        return self._symbol_bindings[symbol]

    # =========================================================================
    # SSA-to-register mapping (for vector results)
    # =========================================================================

    def bind_ssa(self, ssa_value: str, regs: Tuple[KReg, ...]) -> None:
        """
        Bind an MLIR SSA value to a tuple of virtual registers.

        Used for operations that produce multi-register results like:
        - vector loads (dwordx2/x4 → 2/4 regs)
        - MFMA results (4 regs)

        Args:
            ssa_value: The SSA value string (e.g., "%12")
            regs: Tuple of virtual registers
        """
        self.ssa_to_reg[ssa_value] = regs

    def bind_ssa_single(self, ssa_value: str, reg: KReg) -> None:
        """Bind an SSA value to a single register (convenience method)."""
        self.ssa_to_reg[ssa_value] = (reg,)

    def bind_ssa_range(self, ssa_value: str, reg_range: KRegRange) -> None:
        """
        Bind an SSA value to a register range.

        Extracts individual registers from the range for proper tracking.
        """
        # For ranges, we track the base_reg and count
        # The actual tuple will be derived during allocation
        base = reg_range.base_reg
        if isinstance(base, KVReg):
            regs = tuple(KVReg(base.id + i) for i in range(reg_range.count))
        elif isinstance(base, KSReg):
            regs = tuple(KSReg(base.id + i) for i in range(reg_range.count))
        else:
            # Physical regs - store as-is
            regs = (reg_range,)
        self.ssa_to_reg[ssa_value] = regs

    def get_ssa_regs(self, ssa_value: str) -> Optional[Tuple[KReg, ...]]:
        """Get the virtual registers bound to an SSA value."""
        return self.ssa_to_reg.get(ssa_value)

    def require_ssa_regs(self, ssa_value: str) -> Tuple[KReg, ...]:
        """Get the virtual registers bound to an SSA value, or raise."""
        regs = self.ssa_to_reg.get(ssa_value)
        if regs is None:
            raise ValueError(
                f"SSA value '{ssa_value}' not bound to any registers. "
                f"Available: {list(self.ssa_to_reg.keys())}"
            )
        return regs

    # =========================================================================
    # CSE support
    # =========================================================================

    def cse_lookup(self, key: tuple) -> Optional[KVReg]:
        """Look up a value in the CSE cache."""
        return self._cse_cache.get(key)

    def cse_insert(self, key: tuple, reg: KVReg) -> None:
        """Insert a value into the CSE cache."""
        self._cse_cache[key] = reg

    def cse_get_or_emit(self, key: tuple, emit_fn) -> KVReg:
        """Get from CSE cache or emit using the provided function."""
        if key in self._cse_cache:
            self._cse_hits += 1
            return self._cse_cache[key]
        result = emit_fn()
        self._cse_cache[key] = result
        return result

    # =========================================================================
    # Register allocation helpers
    # =========================================================================

    def vreg(self) -> KVReg:
        """Allocate a new virtual VGPR."""
        return self.builder.vreg()

    def sreg(self) -> KSReg:
        """Allocate a new virtual SGPR."""
        return self.builder.sreg()

    def vreg_pair(self) -> KRegRange:
        """Allocate a pair of virtual VGPRs."""
        return self.program.alloc_vreg_range(2, alignment=2)

    def vreg_quad(self) -> KRegRange:
        """Allocate a quad of virtual VGPRs."""
        return self.program.alloc_vreg_range(4, alignment=4)

    def sreg_pair(self) -> KRegRange:
        """Allocate a pair of virtual SGPRs."""
        return self.program.alloc_sreg_range(2, alignment=2)

    def sreg_quad(self) -> KRegRange:
        """Allocate a quad of virtual SGPRs."""
        return self.program.alloc_sreg_range(4, alignment=4)

    # =========================================================================
    # Special emission methods (not auto-generated)
    # =========================================================================

    def emit(self, instr: KInstr):
        """Emit a raw instruction."""
        self.program.emit(instr)

    def emit_raw(self, asm_line: str):
        """Emit a raw assembly line (escape hatch)."""
        self.program.emit(KInstr(Instruction._RAW_ASM, (), (), comment=asm_line))

    def emit_label(self, label: str):
        """Emit a label."""
        self.program.emit(KInstr(Instruction._LABEL, (), (), comment=label))

    def comment(self, text: str):
        """Emit a comment."""
        self.builder.comment(text)

    def s_mov_b32_to_m0(self, src, comment: str = None):
        """Emit s_mov_b32 m0, src (special: destination is M0)."""
        self.program.emit(KInstr(Instruction.S_MOV_B32, (M0,), (src,), comment=comment))

    def s_cbranch_scc1(self, label: str, comment: str = None):
        """Emit s_cbranch_scc1 with explicit branch target."""
        self.program.emit(
            KInstr(Instruction.S_CBRANCH_SCC1, (), (), target=label, comment=comment)
        )

    def s_branch(self, label: str, comment: str = None):
        """Emit s_branch with explicit branch target."""
        self.program.emit(
            KInstr(Instruction.S_BRANCH, (), (), target=label, comment=comment)
        )

    # =========================================================================
    # SRD Management (for kernel IR path)
    # =========================================================================

    def ensure_srd(self, memref_ssa: str, arg_idx: int, limit_bytes: int) -> KRegRange:
        """
        Ensure SRD is set up for a memref and return the SGPR range.

        If already set up, returns the cached range.
        Otherwise, allocates a new SGPR quad and queues SRD setup for the program prologue.

        SRD setup instructions are queued and emitted at the START of the program
        (during finalize) to ensure they're not inside loops.
        """
        if memref_ssa in self.srd_ranges:
            return self.srd_ranges[memref_ssa]

        # Allocate SGPR quad for SRD
        srd_range = self.sreg_quad()
        self.srd_ranges[memref_ssa] = srd_range

        # Queue the SRD setup for prologue emission
        self._pending_srd_setups.append((srd_range, arg_idx, limit_bytes))

        return srd_range

    def get_srd(self, memref_ssa: str) -> Optional[KRegRange]:
        """Get the SRD range for a memref if it exists."""
        return self.srd_ranges.get(memref_ssa)

    # =========================================================================
    # Prologue-Hoisted ABI Values
    # =========================================================================

    def ensure_tid_x(self) -> KVReg:
        """
        Get a VGPR containing tid_x, defined in the prologue.

        The VGPR is allocated immediately but the defining instruction
        is queued for prologue emission, ensuring it dominates all uses.
        """
        if self._prologue_tid_x is not None:
            return self._prologue_tid_x

        result = self.vreg()
        self._prologue_tid_x = result

        if self.use_flat_tid:
            # Multi-wave: extract tid_x from flat_tid (v0[0:9])
            self._pending_abi_prologue.append(
                KInstr(
                    "v_bfe_u32",
                    (result,),
                    (KPhysVReg(0), KImm(0), KImm(10)),
                    comment="extract tid_x from flat_tid",
                )
            )
        else:
            # Single-wave: compute lane_id using v_mbcnt
            lo_result = self.vreg()
            self._pending_abi_prologue.append(
                KInstr(
                    "v_mbcnt_lo_u32_b32",
                    (lo_result,),
                    (KImm(-1), KImm(0)),
                    comment="lane_id low",
                )
            )
            self._pending_abi_prologue.append(
                KInstr(
                    "v_mbcnt_hi_u32_b32",
                    (result,),
                    (KImm(-1), lo_result),
                    comment="lane_id = tid_x for single-wave",
                )
            )

        return result

    def ensure_tid_y(self) -> KVReg:
        """Get a VGPR containing tid_y, defined in the prologue."""
        if self._prologue_tid_y is not None:
            return self._prologue_tid_y

        result = self.vreg()
        self._prologue_tid_y = result

        if self.use_flat_tid:
            # Multi-wave: extract tid_y from flat_tid (v0[10:19])
            self._pending_abi_prologue.append(
                KInstr(
                    "v_bfe_u32",
                    (result,),
                    (KPhysVReg(0), KImm(10), KImm(10)),
                    comment="extract tid_y from flat_tid",
                )
            )
        else:
            # Single-wave: tid_y is 0
            self._pending_abi_prologue.append(
                KInstr(
                    "v_mov_b32",
                    (result,),
                    (KImm(0),),
                    comment="tid_y = 0 for single-wave",
                )
            )

        return result

    def _emit_wgid_prologue(self, sreg: Optional[KPhysSReg], axis: str) -> KVReg:
        """Helper to emit workgroup ID prologue for a given axis.

        Args:
            sreg: The ABI-defined SGPR for this workgroup ID, or None if not enabled.
            axis: The axis name ('x', 'y', or 'z') for error messages and comments.

        Returns:
            A VGPR containing the workgroup ID.

        Raises:
            RuntimeError: If the workgroup ID is not enabled in ABI.
        """
        if sreg is None:
            raise RuntimeError(f"wgid_{axis} requested but not enabled in ABI")

        result = self.vreg()
        self._pending_abi_prologue.append(
            KInstr(
                Instruction.V_MOV_B32,
                (result,),
                (sreg,),
                comment=f"wgid_{axis} from s{sreg.index}",
            )
        )
        return result

    def ensure_wgid_x(self) -> KVReg:
        """Get a VGPR containing wgid_x, defined in the prologue."""
        if self._prologue_wgid_x is not None:
            return self._prologue_wgid_x
        self._prologue_wgid_x = self._emit_wgid_prologue(
            self.program.abi.workgroup_id_x_sreg, "x"
        )
        return self._prologue_wgid_x

    def ensure_wgid_y(self) -> KVReg:
        """Get a VGPR containing wgid_y, defined in the prologue."""
        if self._prologue_wgid_y is not None:
            return self._prologue_wgid_y
        self._prologue_wgid_y = self._emit_wgid_prologue(
            self.program.abi.workgroup_id_y_sreg, "y"
        )
        return self._prologue_wgid_y

    def ensure_wgid_z(self) -> KVReg:
        """Get a VGPR containing wgid_z, defined in the prologue."""
        if self._prologue_wgid_z is not None:
            return self._prologue_wgid_z
        self._prologue_wgid_z = self._emit_wgid_prologue(
            self.program.abi.workgroup_id_z_sreg, "z"
        )
        return self._prologue_wgid_z

    def _emit_srd_prologue(self) -> None:
        """
        Emit all queued SRD setup and ABI prologue instructions at the start of the program.

        This should be called at the beginning of finalize() to ensure
        SRD setup and ABI value definitions happen before any loops.
        """
        prologue_instrs = []

        # First, emit ABI prologue (tid_x, tid_y, wgid_x, etc.)
        # These must come first since they define values used throughout
        if self._pending_abi_prologue:
            prologue_instrs.extend(self._pending_abi_prologue)
            self._pending_abi_prologue = []

        # Then, emit SRD setup if any
        if self._pending_srd_setups:
            kernarg_pair = KRegRange(
                self.program.abi.kernarg_ptr_sreg_lo, KERNARG_PTR_SGPRS
            )  # s[0:1]

            if self.use_kernarg_preloading:
                # With preloading enabled, follow LLVM's pattern:
                # 1. Load args into PRELOAD LOCATIONS (s[2:3], s[4:5], etc.)
                # 2. s_waitcnt
                # 3. s_branch + alignment
                # 4. Copy from preload locations to SRD ranges + fill SRD

                # First, emit s_load into preload locations (before branch)
                for srd_range, arg_idx, limit_bytes in self._pending_srd_setups:
                    preloaded_base = KERNARG_PTR_SGPRS + (
                        arg_idx * 2
                    )  # s[2:3], s[4:5], etc.
                    preloaded_pair = KRegRange(KPhysSReg(preloaded_base), 2)
                    kernarg_offset = arg_idx * 8

                    # Load into preload location (matches LLVM pattern)
                    prologue_instrs.append(
                        KInstr(
                            "s_load_dwordx2",
                            defs=(preloaded_pair,),
                            uses=(kernarg_pair, KImm(kernarg_offset)),
                            comment=f"Load arg{arg_idx} into preload location",
                        )
                    )

                # Wait for all preload-location loads
                prologue_instrs.append(
                    KInstr(
                        "s_waitcnt",
                        (),
                        ("lgkmcnt(0)",),
                        comment="wait for preload loads",
                    )
                )

                # Branch to aligned entry + padding
                # Use kernel-specific label name for uniqueness when multiple kernels
                # are emitted into a single assembly file
                preload_label = f".L_{self.kernel_name}_main"

                # Use target field for branch destination (cleaner than comment)
                prologue_instrs.append(
                    KInstr(
                        "_preload_branch",
                        (),
                        (),
                        target=preload_label,  # Explicit branch target field
                    )
                )
                # Use _raw_asm for alignment directive (not control flow)
                prologue_instrs.append(
                    KInstr(
                        Instruction._RAW_ASM,
                        (),
                        (),
                        comment=".p2align 8",  # 256-byte alignment
                    )
                )
                # Use standard _label so CFG recognizes it via is_label
                prologue_instrs.append(
                    KInstr(
                        Instruction._LABEL,
                        (),
                        (),
                        comment=preload_label,  # Label name in comment
                    )
                )

                # Now copy from preload locations to SRD ranges
                for srd_range, arg_idx, limit_bytes in self._pending_srd_setups:
                    preloaded_base = KERNARG_PTR_SGPRS + (arg_idx * 2)
                    preloaded_pair = KRegRange(
                        KPhysSReg(preloaded_base), KERNARG_PTR_SGPRS
                    )

                    # Copy from preload location to SRD base
                    # defs=(srd_range,) because we WRITE to SRD base registers
                    # uses=(preloaded_pair,) because we READ from preload locations
                    # Note: _srd_copy_base now defines the SRD range, so no separate
                    # _srd_define is needed (that would cause SSA violation)
                    prologue_instrs.append(
                        KInstr(
                            "_srd_copy_base",
                            defs=(srd_range,),
                            uses=(preloaded_pair,),
                            comment=f"Copy arg{arg_idx} to SRD base",
                        )
                    )
            else:
                # Without preloading, load directly into SRD ranges
                for srd_range, arg_idx, limit_bytes in self._pending_srd_setups:
                    kernarg_offset = arg_idx * 8

                    # Define the full SRD range
                    prologue_instrs.append(
                        KInstr(
                            "_srd_define",
                            (srd_range,),
                            (),
                            comment=f"Define SRD range for arg{arg_idx}",
                        )
                    )

                    # Load base address into first 2 regs of the range
                    prologue_instrs.append(
                        KInstr(
                            "_srd_load_base",
                            (),
                            (srd_range, kernarg_pair, KImm(kernarg_offset)),
                            comment=f"Load base addr for arg{arg_idx}",
                        )
                    )

                # Wait for all SRD loads
                prologue_instrs.append(
                    KInstr(
                        "s_waitcnt",
                        (),
                        ("lgkmcnt(0)",),
                        comment="wait for all SRD loads",
                    )
                )

            # Fill SRD[2] and SRD[3] for each (same for both preload and non-preload)
            for srd_range, arg_idx, limit_bytes in self._pending_srd_setups:
                prologue_instrs.append(
                    KInstr(
                        "_srd_fill_size",
                        (),
                        (srd_range, KImm(limit_bytes)),
                        comment=f"SRD size for arg{arg_idx}",
                    )
                )
                prologue_instrs.append(
                    KInstr(
                        "_srd_fill_stride",
                        (),
                        (srd_range, KImm(0x20000)),
                        comment=f"SRD stride for arg{arg_idx}",
                    )
                )

            # Clear the pending list
            self._pending_srd_setups = []

        if not prologue_instrs:
            return

        # Insert prologue at the start of the program (after any comments)
        insert_pos = 0
        while insert_pos < len(self.program.instructions):
            instr = self.program.instructions[insert_pos]
            if instr.name != Instruction._COMMENT:
                break
            insert_pos += 1

        self.program.instructions[insert_pos:insert_pos] = prologue_instrs

    def emit_buffer_load(
        self,
        memref_ssa: str,
        vector_bytes: int,
        voffset: KVReg,
        inst_offset: int,
    ) -> Tuple[KRegRange, ...]:
        """
        Emit buffer load instruction(s).

        Returns tuple of destination register ranges.
        """
        srd_range = self.srd_ranges.get(memref_ssa)
        if srd_range is None:
            raise RuntimeError(f"SRD not set up for {memref_ssa}")

        result_ranges = []
        current_offset = inst_offset

        # Handle different vector sizes
        bytes_remaining = vector_bytes
        while bytes_remaining > 0:
            if bytes_remaining >= 16:
                # Use buffer_load_dwordx4
                dst_range = self.vreg_quad()
                self.program.emit(
                    KInstr(
                        "buffer_load_dwordx4",
                        (dst_range,),
                        (voffset, srd_range, KImm(0), KMemOffset(current_offset)),
                        comment=f"load 16B @ offset {current_offset}",
                    )
                )
                result_ranges.append(dst_range)
                bytes_remaining -= 16
                current_offset += 16
            elif bytes_remaining >= 8:
                # Use buffer_load_dwordx2
                dst_range = self.vreg_pair()
                self.program.emit(
                    KInstr(
                        "buffer_load_dwordx2",
                        (dst_range,),
                        (voffset, srd_range, KImm(0), KMemOffset(current_offset)),
                        comment=f"load 8B @ offset {current_offset}",
                    )
                )
                result_ranges.append(dst_range)
                bytes_remaining -= 8
                current_offset += 8
            else:
                # Use buffer_load_dword
                dst = self.vreg()
                self.program.emit(
                    KInstr(
                        "buffer_load_dword",
                        (dst,),
                        (voffset, srd_range, KImm(0), KMemOffset(current_offset)),
                        comment=f"load 4B @ offset {current_offset}",
                    )
                )
                result_ranges.append(KRegRange(dst, 1))
                bytes_remaining -= 4
                current_offset += 4

        return tuple(result_ranges)

    def emit_buffer_store(
        self,
        memref_ssa: str,
        src_ranges: Tuple[KRegRange, ...],
        voffset: KVReg,
        inst_offset: int,
    ):
        """Emit buffer store instruction(s)."""
        srd_range = self.srd_ranges.get(memref_ssa)
        if srd_range is None:
            raise RuntimeError(f"SRD not set up for {memref_ssa}")

        current_offset = inst_offset
        for src_range in src_ranges:
            count = src_range.count
            if count == 4:
                self.program.emit(
                    KInstr(
                        "buffer_store_dwordx4",
                        (),
                        (
                            src_range,
                            voffset,
                            srd_range,
                            KImm(0),
                            KMemOffset(current_offset),
                        ),
                        comment=f"store 16B @ offset {current_offset}",
                    )
                )
                current_offset += 16
            elif count == 2:
                self.program.emit(
                    KInstr(
                        "buffer_store_dwordx2",
                        (),
                        (
                            src_range,
                            voffset,
                            srd_range,
                            KImm(0),
                            KMemOffset(current_offset),
                        ),
                        comment=f"store 8B @ offset {current_offset}",
                    )
                )
                current_offset += 8
            else:
                self.program.emit(
                    KInstr(
                        "buffer_store_dword",
                        (),
                        (
                            src_range,
                            voffset,
                            srd_range,
                            KImm(0),
                            KMemOffset(current_offset),
                        ),
                        comment=f"store 4B @ offset {current_offset}",
                    )
                )
                current_offset += 4

    # =========================================================================
    # LDS Read/Write Operations
    # =========================================================================

    def emit_lds_read_b64(
        self,
        dst_range: KRegRange,
        addr_vreg: KVReg,
        offset: int = 0,
    ):
        """Emit ds_read_b64 (LDS load of 8 bytes)."""
        self.program.emit(
            KInstr(
                "ds_read_b64",
                (dst_range,),
                (addr_vreg, KMemOffset(offset)),
                comment=f"LDS load 8B @ offset {offset}",
            )
        )

    def emit_lds_read_b128(
        self,
        dst_range: KRegRange,
        addr_vreg: KVReg,
        offset: int = 0,
    ):
        """Emit ds_read_b128 (LDS load of 16 bytes)."""
        self.program.emit(
            KInstr(
                "ds_read_b128",
                (dst_range,),
                (addr_vreg, KMemOffset(offset)),
                comment=f"LDS load 16B @ offset {offset}",
            )
        )

    def emit_lds_write_b64(
        self,
        addr_vreg: KVReg,
        src_range: KRegRange,
        offset: int = 0,
    ):
        """Emit ds_write_b64 (LDS store of 8 bytes)."""
        self.program.emit(
            KInstr(
                "ds_write_b64",
                (),
                (addr_vreg, src_range, KMemOffset(offset)),
                comment=f"LDS store 8B @ offset {offset}",
            )
        )

    def emit_lds_write_b128(
        self,
        addr_vreg: KVReg,
        src_range: KRegRange,
        offset: int = 0,
    ):
        """Emit ds_write_b128 (LDS store of 16 bytes)."""
        self.program.emit(
            KInstr(
                "ds_write_b128",
                (),
                (addr_vreg, src_range, KMemOffset(offset)),
                comment=f"LDS store 16B @ offset {offset}",
            )
        )
