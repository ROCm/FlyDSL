# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir


def test_memref_type_get_accepts_dsl_numeric_and_address_space_enum():
    with ir.Context():
        memref_ty = fx.MemRefType.get(
            fx.BFloat16,
            fx.LayoutType.get(1, 1),
            fx.AddressSpace.Register,
        )

        assert memref_ty.element_type == fx.BFloat16.ir_type
        assert memref_ty.address_space == int(fx.AddressSpace.Register)
