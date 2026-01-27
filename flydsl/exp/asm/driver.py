# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Driver for MLIR to AMDGCN assembly emitter.

This module implements a single-path architecture where all instructions
(including kernarg loading, s_endpgm) go through the Kernel IR path:

    ┌─────────────────────────────────────────────────────────────┐
    │  MetadataEmitter.emit_prologue()                            │
    │  - .amdgcn_target, .amdhsa_kernel, etc.                     │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  KernelCompilationContext + KernelGenerator                 │
    │  - emit_kernargs() -> s_load_dwordx2                        │
    │  - MLIR walker -> kernel body                               │
    │  - finalize() adds s_endpgm                                 │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  MetadataEmitter.emit_epilogue()                            │
    │  - .amdgpu_metadata                                         │
    └─────────────────────────────────────────────────────────────┘
"""

import argparse
import sys

from .kernel_module_compiler import KernelModuleCompiler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="-", help="Input MLIR")
    ap.add_argument("--out", dest="out", default="-", help="Output .s")
    ap.add_argument("--targetid", default="gfx942", help="Target for .amdgcn_target")
    ap.add_argument(
        "--codeobj",
        choices=["4", "5"],
        default="5",
        help="Code object version for metadata",
    )
    args = ap.parse_args()

    mlir_text = sys.stdin.read() if args.inp == "-" else open(args.inp, "r").read()

    # Single canonical pipeline:
    compiler = KernelModuleCompiler(targetid=args.targetid, codeobj=args.codeobj)
    asm_text = compiler.compile_mlir_string(mlir_text)

    # Output
    if args.out == "-":
        print(asm_text)
    else:
        with open(args.out, "w") as f:
            f.write(asm_text)


if __name__ == "__main__":
    main()
