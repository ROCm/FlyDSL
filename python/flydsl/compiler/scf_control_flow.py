# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .._mlir import ir
from .._mlir.dialects import scf
from ..expr.numeric import _unwrap_value, _wrap_like
from ..utils import log


class ControlFlowError(RuntimeError):
    """Raised when Python control-flow cannot be represented as SCF."""


@dataclass(frozen=True)
class TreeSpec:
    kind: str
    children: tuple["TreeSpec", ...] = ()
    exemplar: Any = None


def is_dynamic_value(value: Any) -> bool:
    return isinstance(value, ir.Value) or (hasattr(value, "value") and isinstance(value.value, ir.Value))


def to_i1(cond: Any) -> ir.Value:
    raw = _unwrap_value(cond)
    if not isinstance(raw, ir.Value):
        raise TypeError(f"dynamic if condition must lower to ir.Value, got {type(raw).__name__}")
    return raw


def normalize_to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _wrap_value_like(value: ir.Value, exemplar: Any) -> Any:
    ctor = getattr(exemplar, "__fly_construct_from_values__", None)
    if ctor is not None:
        try:
            return ctor([value])
        except Exception as e:
            log().error(f"failed to construct {type(exemplar)} from SCF result {value}: {e!r}")
    return _wrap_like(value, exemplar)


def flatten_tree(value: Any, label: str = "value") -> tuple[list[ir.Value], TreeSpec]:
    if isinstance(value, tuple):
        flat: list[ir.Value] = []
        specs = []
        for idx, item in enumerate(value):
            item_flat, item_spec = flatten_tree(item, f"{label}[{idx}]")
            flat.extend(item_flat)
            specs.append(item_spec)
        return flat, TreeSpec("tuple", tuple(specs))

    if isinstance(value, list):
        flat = []
        specs = []
        for idx, item in enumerate(value):
            item_flat, item_spec = flatten_tree(item, f"{label}[{idx}]")
            flat.extend(item_flat)
            specs.append(item_spec)
        return flat, TreeSpec("list", tuple(specs))

    raw = _unwrap_value(value)
    if not isinstance(raw, ir.Value):
        raise TypeError(
            f"{label} is {type(raw).__name__}, not an MLIR Value. "
            "Dynamic control-flow results must be MLIR-backed values; "
            "initialize values before the if or mark the condition with const_expr(...)."
        )
    return [raw], TreeSpec("leaf", exemplar=value)


def clone_value_tree(value: Any) -> Any:
    if value is None:
        return None
    flat, spec = flatten_tree(value)
    return pack_tree(list(flat), spec)


def _compare_specs(expected: TreeSpec, actual: TreeSpec, label: str):
    if expected.kind != actual.kind:
        raise TypeError(
            f"{label} changed structure across dynamic control flow: expected {expected.kind}, got {actual.kind}. "
            "Keep the same tuple/list/scalar shape in every branch or use const_expr(...)."
        )
    if len(expected.children) != len(actual.children):
        raise TypeError(
            f"{label} changed structure across dynamic control flow: expected {len(expected.children)} "
            f"children, got {len(actual.children)}. Keep branch result structure identical."
        )
    for idx, (expected_child, actual_child) in enumerate(zip(expected.children, actual.children)):
        _compare_specs(expected_child, actual_child, f"{label}[{idx}]")


def flatten_tree_against(value: Any, spec: TreeSpec, expected_types: Iterable[ir.Type], label: str) -> list[ir.Value]:
    flat, actual_spec = flatten_tree(value, label)
    _compare_specs(spec, actual_spec, label)

    for idx, (got, expected_type) in enumerate(zip(flat, expected_types)):
        if got.type != expected_type:
            raise TypeError(
                f"{label}[{idx}] type mismatch across dynamic control flow: "
                f"expected {expected_type}, got {got.type}. "
                "Keep branch result types identical or use const_expr(...)."
            )
    return flat


def pack_tree(values: list[ir.Value], spec: TreeSpec) -> Any:
    if spec.kind == "leaf":
        if not values:
            raise ControlFlowError("not enough SCF results to pack leaf")
        return _wrap_value_like(values.pop(0), spec.exemplar)

    packed = [pack_tree(values, child) for child in spec.children]
    if spec.kind == "tuple":
        return tuple(packed)
    if spec.kind == "list":
        return packed
    raise ControlFlowError(f"unknown tree spec kind {spec.kind!r}")


def flatten_sequence(values: Iterable[Any], names: Iterable[str]) -> tuple[list[ir.Value], list[TreeSpec], list[list[ir.Type]]]:
    all_flat: list[ir.Value] = []
    specs: list[TreeSpec] = []
    type_groups: list[list[ir.Type]] = []
    for name, value in zip(names, values):
        flat, spec = flatten_tree(value, name)
        all_flat.extend(flat)
        specs.append(spec)
        type_groups.append([v.type for v in flat])
    return all_flat, specs, type_groups


def flatten_sequence_against(
    values: Iterable[Any],
    names: Iterable[str],
    specs: Iterable[TreeSpec],
    type_groups: Iterable[list[ir.Type]],
    branch_label: str,
) -> list[ir.Value]:
    all_flat: list[ir.Value] = []
    for name, value, spec, expected_types in zip(names, values, specs, type_groups):
        all_flat.extend(flatten_tree_against(value, spec, expected_types, f"{branch_label} variable '{name}'"))
    return all_flat


def pack_sequence(results: Iterable[ir.Value], specs: Iterable[TreeSpec]) -> list[Any]:
    remaining = list(results)
    packed = [pack_tree(remaining, spec) for spec in specs]
    if remaining:
        raise ControlFlowError(f"too many SCF results while packing: {len(remaining)} unused")
    return packed


class FlyScfBuilder:
    """Shared SCF construction helpers for FlyDSL dynamic control flow."""

    @staticmethod
    def execute_if(
        cond: Any,
        state_names: tuple[str, ...],
        state_values: tuple[Any, ...],
        then_builder: Callable[[], list[Any]],
        else_builder: Callable[[], list[Any]] | None = None,
    ) -> list[Any]:
        cond_i1 = to_i1(cond)

        if not state_names:
            if_op = scf.IfOp(cond_i1, [], has_else=else_builder is not None, loc=ir.Location.unknown())
            with ir.InsertionPoint(if_op.regions[0].blocks[0]):
                then_builder()
                scf.YieldOp([])
            if else_builder is not None:
                if len(if_op.regions[1].blocks) == 0:
                    if_op.regions[1].blocks.append(*[])
                with ir.InsertionPoint(if_op.regions[1].blocks[0]):
                    else_builder()
                    scf.YieldOp([])
            return []

        state_flat, specs, type_groups = flatten_sequence(state_values, state_names)
        result_types = [value.type for value in state_flat]
        if_op = scf.IfOp(cond_i1, result_types, has_else=True, loc=ir.Location.unknown())

        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            then_values = then_builder()
            then_flat = flatten_sequence_against(then_values, state_names, specs, type_groups, "then-branch")
            scf.YieldOp(then_flat)

        if len(if_op.regions[1].blocks) == 0:
            if_op.regions[1].blocks.append(*[])
        with ir.InsertionPoint(if_op.regions[1].blocks[0]):
            else_values = else_builder() if else_builder is not None else list(state_values)
            else_flat = flatten_sequence_against(else_values, state_names, specs, type_groups, "else-branch")
            scf.YieldOp(else_flat)

        return pack_sequence(list(if_op.results), specs)

    @staticmethod
    def execute_if_expression(cond: Any, then_fn: Callable[[], Any], else_fn: Callable[[], Any]) -> Any:
        if not is_dynamic_value(cond):
            return then_fn() if cond else else_fn()

        cond_i1 = to_i1(cond)
        result_types, spec = FlyScfBuilder._infer_if_expression_result(then_fn, else_fn)
        if_op = scf.IfOp(cond_i1, result_types, has_else=True, loc=ir.Location.unknown())

        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            then_flat = flatten_tree_against(then_fn(), spec, result_types, "then if-expression")
            scf.YieldOp(then_flat)

        if len(if_op.regions[1].blocks) == 0:
            if_op.regions[1].blocks.append(*[])
        with ir.InsertionPoint(if_op.regions[1].blocks[0]):
            else_flat = flatten_tree_against(else_fn(), spec, result_types, "else if-expression")
            scf.YieldOp(else_flat)

        return pack_tree(list(if_op.results), spec)

    @staticmethod
    def _infer_if_expression_result(then_fn: Callable[[], Any], else_fn: Callable[[], Any]) -> tuple[list[ir.Type], TreeSpec]:
        execution_region = scf.ExecuteRegionOp(result=[])
        execution_region.region.blocks.append()
        try:
            with ir.InsertionPoint(execution_region.region.blocks[0]):
                then_flat, then_spec = flatten_tree(then_fn(), "then if-expression")
                else_flat, else_spec = flatten_tree(else_fn(), "else if-expression")
                _compare_specs(then_spec, else_spec, "if-expression")
                result_types = [value.type for value in then_flat]
                for idx, (then_value, else_value) in enumerate(zip(then_flat, else_flat)):
                    if then_value.type != else_value.type:
                        raise TypeError(
                            f"if-expression result[{idx}] type mismatch: "
                            f"then has {then_value.type}, else has {else_value.type}. "
                            "Keep both arms the same type or use const_expr(...)."
                        )
        finally:
            execution_region.operation.erase()
        return result_types, then_spec
