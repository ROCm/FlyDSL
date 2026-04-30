# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from typing import Any

from .._mlir import ir


def convert_to_mlir_attr(value: Any) -> ir.Attribute:
    if isinstance(value, ir.Attribute):
        return value
    if isinstance(value, bool):
        return ir.BoolAttr.get(value)
    if isinstance(value, int):
        return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)
    if isinstance(value, str):
        return ir.StringAttr.get(value)
    if isinstance(value, dict):
        return ir.DictAttr.get({k: convert_to_mlir_attr(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return ir.ArrayAttr.get([convert_to_mlir_attr(v) for v in value])
    raise TypeError(
        f"unsupported attribute value type {type(value).__name__}, expected bool, int, str, list, dict, or ir.Attribute"
    )
