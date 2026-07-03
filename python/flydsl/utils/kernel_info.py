# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Parse the LLVM AMDGPU AsmPrinter "Kernel info" comment block.

When LLVM lowers a kernel to AMDGPU assembly, it emits an exact,
compiler-computed resource-usage summary as a comment block, e.g.::

    ; Kernel info:
    ; NumVgprs: 6
    ; NumAgprs: 0
    ; Occupancy: 8
    ; LDSByteSize: 0 bytes/workgroup (compile time only)
    ...

This is only emitted when the target machine is built with ``AsmVerbose``
on. FlyDSL's ROCDL serializer (``mlir/lib/Target/LLVM/ROCDL/Target.cpp``,
``SerializeGPUModuleBase::getTargetOptions()``) supports enabling it via
``-asm-verbose`` in ``gpu-module-to-binary``'s ``opts`` argument, which
``jit_function._dump_isa`` passes so its ``.s`` output carries this block.

This module parses that block into a plain dict, so callers don't have to
re-derive occupancy/register usage from device properties themselves.
"""

from typing import Dict

KERNEL_INFO_HEADER = "; Kernel info:"


def parse_kernel_info(isa_text: str) -> Dict[str, str]:
    """Parse the ``; Kernel info:`` comment block out of AMDGPU ISA text.

    Returns an empty dict if the block is not present (e.g. non-AMDGPU ISA,
    or a source that didn't come from asm-verbose codegen).
    """
    block_start = isa_text.find(KERNEL_INFO_HEADER)
    if block_start == -1:
        return {}

    block_end = isa_text.find("\n\t", block_start)
    if block_end == -1:
        block_end = len(isa_text)

    info: Dict[str, str] = {}
    for line in isa_text[block_start:block_end].splitlines()[1:]:
        line = line.lstrip(";").strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        info[key.strip()] = value.strip()
    return info
