# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Kernel model and environment data structures for MLIR to AMDGCN assembly emitter.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class MemRefInfo:
    shape: List[int]
    strides_elems: List[int]
    elem_bytes: int


IndexVal = Union[int, str]  # int for constants, "tid_x" marker for per-lane


@dataclass
class BindingUse:
    memref_ssa: str
    arg_index: int
    memref_info: Optional[MemRefInfo] = None
    srd_quads: Optional[Tuple[int, int, int, int]] = None


@dataclass
class VecAccess:
    kind: str  # "load" or "store"
    memref_ssa: str
    vec_elems: int
    elem_bytes: int
    indices: List[str]  # SSA names for indices (e.g., ["%1","%c0"])

    @property
    def vec_bytes(self) -> int:
        return self.vec_elems * self.elem_bytes


@dataclass
class KernelInfo:
    name: str
    arg_ssa_order: List[str] = field(default_factory=list)
    # SSA env for simple indices
    index_env: Dict[str, IndexVal] = field(
        default_factory=dict
    )  # e.g., {"%thread_id_x": "tid_x", "%c0": 0, "%1": "tid_x"}
    subspans: Dict[str, BindingUse] = field(
        default_factory=dict
    )  # memref_ssa -> BindingUse
    accesses: List[VecAccess] = field(default_factory=list)
    wg_size: Tuple[int, int, int] = (64, 1, 1)
    subgroup_size: int = 64
    tid_ub_x: int = 64
    tid_ub_y: int = 1  # Upper bound for thread ID in Y dimension
    # optional: remember last affine map for comments
    affine_map_attr: Optional[object] = None
    lds_size_bytes: int = 0
