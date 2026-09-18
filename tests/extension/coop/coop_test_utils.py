# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared launch, value, record and documentation helpers for coop tests."""

import ast
import inspect
import math
import textwrap
from dataclasses import dataclass, field

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

# Shared launch helpers, layout references and compilation targets for coop tests.

ARCHES = ("gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1100", "gfx1151", "gfx1201")


def as_items(value):
    """Read the scalar, Vector and tuple tiles used by these test kernels."""
    if isinstance(value, (fx.Vector, tuple, list)):
        return tuple(value[i] for i in range(len(value)))
    return (value,)


def make_items(items):
    """Build test inputs using only expr types, independently of coop helpers."""
    items = tuple(items)
    return fx.Vector.from_elements(items) if items and isinstance(items[0], fx.Numeric) else items


def items_dtype(value):
    """Get the numeric or plain Struct element type of a test tile."""
    item = as_items(value)[0]
    return item.dtype if isinstance(item, fx.Numeric) else type(item)


def run_tile(transform, values, block, count, outputs=1, dtype=None, output_dtype=None):
    shape = (block, 1, 1) if isinstance(block, int) else block
    threads = math.prod(shape)

    def apply(a, out):
        tid = fx.thread_idx.x + fx.thread_idx.y * shape[0] + fx.thread_idx.z * shape[0] * shape[1]
        value = fx.Vector.from_elements([a[tid * count + i] for i in range(count)])
        if dtype is not None:
            value = value.to(dtype)
        result = transform(value)
        result = (result,) if outputs == 1 else result
        for j, vector in enumerate(result):
            for i in range(count):
                out[j * threads * count + tid * count + i] = vector[i].to(
                    value.dtype if output_dtype is None else output_dtype[0]
                )

    @flyc.kernel(known_block_size=list(shape))
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out).launch(grid=(1, 1, 1), block=shape, stream=stream)

    out = torch.empty(
        values.numel() * outputs,
        dtype=values.dtype if output_dtype is None else getattr(torch, output_dtype[1]),
        device="cuda",
    )
    launch(values, out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out.cpu().reshape(outputs, -1)


def run_kernel(apply, values, out_size, block, *, compile_only=False):
    """Run a kernel, or compile CPU signatures with COMPILE_ONLY set by the caller."""
    shape = (block, 1, 1) if isinstance(block, int) else block

    @flyc.kernel(known_block_size=list(shape))
    def kernel(a: fx.Tensor, out: fx.Tensor):
        apply(a, out)

    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(a, out).launch(grid=(1, 1, 1), block=shape, stream=stream)

    out = torch.full((out_size,), -99, device="cpu" if compile_only else "cuda", dtype=values.dtype)
    argument = values if values.stride() == (1,) else flyc.from_dlpack(values)
    if compile_only:
        launch(argument, out)
        return out
    launch(argument, out, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()
    return out.cpu()


def warp_indices(threads, count, width, inverse=False):
    out = []
    for first in range(0, threads, width):
        extent = min(width, threads - first)
        for lane in range(extent):
            for item in range(count):
                position = lane * count + item
                relative = (position % extent) * count + position // extent if inverse else lane + item * extent
                out.append(first * count + relative)
    return torch.tensor(out, device="cpu")


def dtype_entry(entry):
    return entry[0], entry[1].removeprefix("torch.")


def batched_columns(namespace, value, op, *, width=None, sync_physical_warp=False):
    """Collect distributed batch results with the public broadcast primitive."""

    group = width or fx.num_warp_threads()
    distributed = namespace.warp_reduce_batched_to_striped(
        value, op, width=width, sync_physical_warp=sync_physical_warp
    )
    return make_items(
        [namespace.warp_broadcast(distributed[i // group], i % group, width=width) for i in range(len(value))]
    )


def warp_valid_items(valid, width, count):
    """Convert a caller's block prefix to this logical warp's element count."""
    if valid is None:
        return None
    group = width or fx.num_warp_threads()
    remaining = fx.Int32(valid) - (fx.thread_idx.x // group) * group * count
    return fx.min(fx.max(remaining, 0), group * count)


@pytest.fixture(params=["cpu", "cuda"], ids=["default-cpu", "default-cuda"])
def coop_default_device(request):
    """Exercise host references under both defaults without leaking state."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    with torch.device(request.param):
        yield


# Keep existing warp tests on the same scoped default-device coverage.
warp_default_device = coop_default_device


def warp_storage(operator, allocator=None):
    """Allocate separate operator scratch for every logical warp in the block."""
    if operator.SharedStorage is None:
        return None
    shape = fx.known_block_size()
    threads = math.prod(shape)
    width = operator.warp_threads
    assert threads % width == 0
    allocator = fx.SharedAllocator() if allocator is None else allocator
    scratch = allocator.allocate(fx.Array[operator.SharedStorage, threads // width]).peek()
    tid = fx.thread_idx.x + shape[0] * (fx.thread_idx.y + shape[1] * fx.thread_idx.z)
    return scratch[tid // width]


# Record types shared by movement and storage checks.

PAYLOAD = fx.Struct["fraction" : fx.Float32, "tag" : fx.Int64]
RECORD = fx.Struct["key" : fx.Int32, "payload":PAYLOAD]


@fx.struct
class Columns:
    keys: fx.Tensor
    fractions: fx.Tensor
    tags: fx.Tensor

    @property
    def dtype(self):
        return RECORD

    def __getitem__(self, index):
        return RECORD(self.keys[index], PAYLOAD(self.fractions[index], self.tags[index]))

    def __setitem__(self, index, value):
        self.keys[index] = value.key
        self.fractions[index] = value.payload.fraction
        self.tags[index] = value.payload.tag


# Execute the actual Python statements in primitive docstring examples on a GPU.


@dataclass
class Example:
    module: str
    symbol: str
    inputs: dict
    checks: list
    block: int = 4
    # Replace documented symbolic inputs when a later example starts a new scenario.
    changes: dict = field(default_factory=dict)
    # Named writable tensor arguments, initialized to -9.
    buffers: dict = field(default_factory=dict)


def flatten(value):
    if isinstance(value, (list, tuple)):
        return [x for row in value for x in flatten(row)]
    return [value]


def _validate_results(case, source, nodes, checks):
    """Keep documented output cells and every warp result tied to executed checks."""
    checked_nodes = {}
    for index, (line, expression, _) in enumerate(checks):
        node = next(node for node in nodes if node.lineno <= line <= node.end_lineno)
        checked_nodes.setdefault(node.lineno, {})[expression] = index

    for node in nodes:
        if not isinstance(node, (ast.Assign, ast.Expr)) or not isinstance(node.value, ast.Call):
            continue
        call = ast.unparse(node.value.func)
        name = call.rsplit(".", 1)[-1]
        if not call.startswith("fx.coop.") or not name.startswith("warp_") or name == "warp_exchange_storage":
            continue
        covered = checked_nodes.get(node.lineno, {})
        assert covered, f"{case.symbol}: unchecked example call {call} on line {node.lineno}"
        if isinstance(node, ast.Assign):
            results = [n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)]
            for result in results:
                assert any(
                    expr == result or expr.startswith(result + "[") for expr in covered
                ), f"{case.symbol}: unchecked result {result} from {call}"

    aliases = {"keys out": "sorted_keys", "values out": "sorted_values", "out": "y"}
    for line_number, line in enumerate(source.splitlines(), 1):
        if not line.lstrip().startswith("# ") or "|" not in line:
            continue
        cells = [cell.strip() for cell in line.split("# ", 1)[1].split("|") if cell.strip()]
        label = cells[0].split(" (", 1)[0]
        if label.endswith(" in") or label in ("in", "Data", "Group"):
            continue
        expression = aliases.get(label, label.split()[0])
        previous = [
            checked_nodes[node.lineno][expression]
            for node in nodes
            if node.end_lineno < line_number and expression in checked_nodes.get(node.lineno, {})
        ]
        if not previous:
            assert "out" not in label.split(), f"{case.symbol}: unchecked output table row {label!r}"
            continue
        reference = flatten([ast.literal_eval(cell.replace("?", "None")) for cell in cells[1:]])
        expected = flatten(checks[previous[-1]][2])
        assert reference == expected[: len(reference)], (
            f"{case.symbol}: documented row {label!r} disagrees with its GPU check: "
            f"table={reference}, check={expected[: len(reference)]}"
        )


def prepare_example(case, universal=False, *, namespace=None):
    """Parse literal docstring calls and validate their result coverage before launching."""
    namespace = fx.coop if namespace is None else namespace
    doc = inspect.getdoc(getattr(namespace, case.symbol))
    source = textwrap.dedent(doc.split("Examples:", 1)[1])
    if universal:
        universal_doc = inspect.getdoc(getattr(fx.coop.universal, case.symbol))
        source = textwrap.dedent(universal_doc.split("Examples:", 1)[1])
        source = source.replace("fx.coop.universal.", "fx.coop.")
        source = source.replace("fx.coop.", "fx.coop.universal.")
    nodes = ast.parse(source).body

    def line_of(needle, executable=False):
        # A tuple selects a later occurrence of an otherwise identical statement.
        needle, occurrence = needle if isinstance(needle, tuple) else (needle, 1)
        if universal:
            needle = needle.replace("fx.coop.", "fx.coop.universal.")
        start = -1
        matched = 0
        while matched < occurrence:
            start = source.index(needle, start + 1)
            line = source.count("\n", 0, start) + 1
            if not executable or any(node.lineno <= line <= node.end_lineno for node in nodes):
                matched += 1
        return line

    checks = [(line_of(needle, executable=True), expr, expected) for needle, expr, expected in case.checks]
    changes = [(line_of(needle), inputs) for needle, inputs in case.changes.items()]
    _validate_results(case, source, nodes, checks)
    return nodes, checks, changes


def run_example(case, universal=False, *, namespace=None):
    nodes, checks, changes = prepare_example(case, universal, namespace=namespace)
    data = []

    def prepare(inputs):
        prepared = {}
        for name, rows in inputs.items():
            if isinstance(rows, str):
                prepared[name] = rows
                continue
            offset = len(data)
            data.extend(flatten(rows))
            prepared[name] = (offset, rows)
        return prepared

    initial = prepare(case.inputs)
    changes = [(line, prepare(inputs)) for line, inputs in changes]
    buffer_offsets = {}
    size = 0
    for name, count in case.buffers.items():
        buffer_offsets[name] = size
        size += count
    check_offsets = []
    expected_flat = []
    for _, _, expected in checks:
        values = flatten(expected)
        assert len(values) % case.block == 0
        check_offsets.append(size)
        size += len(values)
        expected_flat.extend(values)
    programs = [compile(ast.Module(body=[node], type_ignores=[]), "<coop docstring>", "exec") for node in nodes]

    def apply(a, out):
        env = {"fx": fx}
        tid = fx.thread_idx.x

        def bind(inputs):
            for name, specification in inputs.items():
                if isinstance(specification, str):
                    env[name] = env[specification]
                    continue
                offset, rows = specification
                if name == "source":
                    env[name] = fx.make_view(a.iter + offset, fx.make_layout(len(rows), 1))
                elif isinstance(rows[0], list):
                    count = len(rows[0])
                    env[name] = fx.Vector.from_elements(
                        [a[offset + (tid % len(rows)) * count + i] for i in range(count)]
                    )
                else:
                    env[name] = a[offset + tid % len(rows)]

        bind(initial)
        for name, count in case.buffers.items():
            env[name] = fx.make_view(out.iter + buffer_offsets[name], fx.make_layout(count, 1))
        applied_changes = set()
        for node, program in zip(nodes, programs):
            for index, (line, inputs) in enumerate(changes):
                if line <= node.lineno and index not in applied_changes:
                    bind(inputs)
                    applied_changes.add(index)
            exec(program, env)
            for index, (line, expression, expected) in enumerate(checks):
                if node.lineno <= line <= node.end_lineno:
                    # Observe a shared/global buffer only after all writers finish.
                    # Host-side inspection after kernel completion has the same visibility.
                    if expression.startswith("["):
                        fx.barrier()
                    result = eval(expression, env)
                    items = list(result) if isinstance(result, (fx.Vector, tuple, list)) else [result]
                    # Tuples of scan results are captured one expression at a time.
                    width = len(flatten(expected)) // case.block
                    assert len(items) == width, (expression, len(items), width)
                    for i, item in enumerate(items):
                        out[check_offsets[index] + tid * width + i] = fx.Int32(item)

    values = torch.tensor(data or [0], dtype=torch.int32, device="cuda")
    # run_kernel initializes outputs to -99. Examples explicitly require -9 buffers.
    buffer_size = sum(case.buffers.values())

    @flyc.jit
    def initialize_buffers(out):
        for i in range(fx.thread_idx.x, buffer_size, case.block):
            out[i] = -9
        fx.barrier()

    def initialized(a, out):
        initialize_buffers(out)
        apply(a, out)

    actual = run_kernel(initialized, values, size, case.block)[sum(case.buffers.values()) :].tolist()
    assert len(actual) == len(expected_flat)
    offset = 0
    for _, expression, expected in checks:
        reference = flatten(expected)
        observed = actual[offset : offset + len(reference)]
        assert all(want is None or got == want for got, want in zip(observed, reference)), (
            case.symbol,
            expression,
            observed,
            reference,
        )
        offset += len(reference)
