# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Wave tracing with a static fast path and a dynamic-control-flow fallback.

Static schedules keep the original low-overhead one-word records.  Kernels with
runtime loops, branches, payloads, or launch dimensions automatically use a
bounded dynamic recorder.  Dynamic records carry their site and payload, which
makes them suitable for persistent and heterogeneous mega kernels.
"""

import json
import math
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .. import expr as fx
from .._mlir import ir
from .._mlir.dialects import llvm
from ..compiler.kernel_function import CompilationContext
from ..expr.meta import dsl_loc_tracing

_active = ContextVar("flytrace_capture", default=None)
_MAX_EVENTS = 1 << 20
_DEFAULT_BLOCK = object()
_DYNAMIC_HEADER_WORDS = 16
_DYNAMIC_RECORD_WORDS = 3
_SUPPORTED_ARCHES = ("gfx942", "gfx950")


def _normalize_blocks(block):
    if block is None:
        return None
    if isinstance(block, (tuple, list)) and len(block) == 3 and all(type(n) is int for n in block):
        blocks = (tuple(block),)
    else:
        try:
            blocks = tuple(tuple(item) for item in block)
        except (TypeError, ValueError) as exc:
            raise ValueError("flytrace block must be an (x, y, z) tuple, a sequence of tuples, or None") from exc
    if not blocks:
        raise ValueError("flytrace block selection cannot be empty")
    if any(len(item) != 3 or any(type(n) is not int or n < 0 for n in item) for item in blocks):
        raise ValueError("flytrace blocks must contain non-negative compile-time (x, y, z) tuples")
    if len(set(blocks)) != len(blocks):
        raise ValueError("flytrace block selection contains duplicates")
    return blocks


def _option(options, name):
    # v3: (version, mode, block selector, excluded names, max events,
    #      maximum all-grid blocks, hardware identity)
    fields = dict(zip(("version", "mode", "blocks", "exclude", "max_events", "max_blocks", "hardware"), options))
    return fields[name]


@contextmanager
def configure(*, block):
    """Scope block selection to kernel launches inside a with block in @jit.

    Pass a compile-time (x, y, z) tuple, a sequence of tuples, or None for all
    blocks. Each selected block records all its waves. An explicit capture(block=...)
    overrides this choice.
    Nested contexts restore the outer selection on exit, including exceptions.
    Without an active capture, this emits no instrumentation or extra arguments.
    """
    from ..compiler.kernel_function import KernelFunction

    ctx = CompilationContext.get_current()
    if ctx is None or KernelFunction.get_current() is not None:
        raise RuntimeError("flytrace.configure belongs in a @jit before kernel launch")
    if ir.InsertionPoint.current.block.owner.operation.name != "func.func":
        raise ValueError("flytrace.configure requires top-level JIT or compile-time control flow")
    block = _normalize_blocks(block)
    previous = ctx.trace_blocks
    ctx.trace_blocks = block
    try:
        yield
    finally:
        ctx.trace_blocks = previous


def _selected_blocks(ctx):
    override = _option(ctx.trace_spec["options"], "blocks")
    return ctx.trace_blocks if override == "jit" else override


@dsl_loc_tracing
def _event(name, kind, payload=None):
    from ..compiler.kernel_function import KernelFunction

    ctx = CompilationContext.get_current()
    if ctx is None or KernelFunction.get_current() is None:
        raise RuntimeError("flytrace annotations belong inside a @kernel")
    if ctx.trace_spec is None:
        return
    if not isinstance(name, str) or not name:
        raise ValueError("flytrace event name must be a nonempty compile-time string")
    if name in _option(ctx.trace_spec["options"], "exclude"):
        return
    operands = [] if payload is None else [fx.Int32(payload).ir_value()]
    ir.Operation.create(
        "fly.trace_event",
        operands=operands,
        attributes={
            "event_name": ir.StringAttr.get(name),
            "kind": ir.StringAttr.get(kind),
        },
    )


def mark(name, payload=None):
    _event(name, "mark", payload)


def boundary(name, payload=None):
    _event(name, "boundary", payload)


def end():
    _event("__end", "end")


def push(name, payload=None):
    _event(name, "push", payload)


def pop():
    _event("__pop", "pop")


def _op(value):
    owner = value.owner
    return owner.operation if hasattr(owner, "operation") else owner


def _integer(value, env):
    # FlyDSL casts integer IR values to ArithValue, whose == emits arith.cmpi.
    # Bypass that overload when comparing compiler identities on the host.
    for key, number in env:
        if ir.Value.__eq__(value, key):
            return number
    op = _op(value)
    if not hasattr(op, "name"):
        raise ValueError("flytrace requires constant bounds and statically reconstructible integer payloads")
    if op.name == "arith.constant":
        return int(ir.IntegerAttr(op.attributes["value"]).value)
    args = list(op.operands)
    if op.name in (
        "arith.index_cast",
        "arith.index_castui",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
    ):
        result = _integer(args[0], env)
        if ir.IntegerType.isinstance(value.type):
            bits = ir.IntegerType(value.type).width
            result &= (1 << bits) - 1
            if op.name != "arith.extui" and result >= 1 << (bits - 1):
                result -= 1 << bits
        return result
    operations = {
        "arith.addi": lambda a, b: a + b,
        "arith.subi": lambda a, b: a - b,
        "arith.muli": lambda a, b: a * b,
    }
    if op.name in operations:
        return operations[op.name](*[_integer(a, env) for a in args])
    raise ValueError(f"flytrace cannot reconstruct payload from {op.name}; use constants or loop indices")


def _contains(block):
    return any(
        op.operation.name == "fly.trace_event" or any(_contains(b) for r in op.regions for b in r.blocks)
        for op in block.operations
    )


def _layout(block):
    items, total = [], 0
    for view in list(block.operations):
        op = view.operation
        if op.name == "fly.trace_event":
            items.append(("event", op, total))
            total += 1
        elif op.name == "scf.for":
            child, count = _layout(op.regions[0].blocks[0])
            if not count:
                continue
            lo, hi, step = (_integer(v, {}) for v in list(op.operands)[:3])
            if step <= 0:
                raise ValueError("flytrace requires positive constant loop steps")
            trips = max(0, (hi - lo + step - 1) // step)
            if total + trips * count > _MAX_EVENTS:
                raise ValueError("flytrace static schedule exceeds 2**20 events per wave")
            items.append(("loop", op, total, lo, hi, step, child, count))
            total += trips * count
        elif any(_contains(b) for r in op.regions for b in r.blocks):
            raise ValueError(f"flytrace annotations inside {op.name} are unsupported; use uniform static for loops")
    if total > _MAX_EVENTS:
        raise ValueError("flytrace static schedule exceeds 2**20 events per wave")
    return items, total


def _schema(items, env=None):
    env = () if env is None else env
    result = []
    for item in items:
        if item[0] == "event":
            op = item[1]
            payload = _integer(op.operands[0], env) if len(op.operands) else None
            if payload is not None and not -(1 << 31) <= payload < (1 << 31):
                raise ValueError("flytrace reconstructed payload must fit signed int32")
            result.append(
                dict(
                    name=ir.StringAttr(op.attributes["event_name"]).value,
                    kind=ir.StringAttr(op.attributes["kind"]).value,
                    payload=payload,
                )
            )
        else:
            _, op, _, lo, hi, step, child, _ = item
            iv = op.regions[0].blocks[0].arguments[0]
            for n in range(lo, hi, step):
                result.extend(_schema(child, env + ((iv, n),)))
    return result


def _dynamic_sites(block):
    """Return trace operations in stable source/region order without expanding loops."""
    result = []
    for view in list(block.operations):
        op = view.operation
        if op.name == "fly.trace_event":
            result.append(op)
        for region in op.regions:
            for child in region.blocks:
                result.extend(_dynamic_sites(child))
    return result


def _dynamic_schema(sites):
    result = []
    for site, op in enumerate(sites):
        result.append(
            dict(
                site=site,
                name=ir.StringAttr(op.attributes["event_name"]).value,
                kind=ir.StringAttr(op.attributes["kind"]).value,
                has_payload=bool(len(op.operands)),
            )
        )
    return result


def _asm(code, operands=(), constraints=""):
    llvm.inline_asm(
        None,
        [v.ir_value() for v in operands],
        code,
        constraints + ("," if constraints else "") + "~{vcc},~{scc},~{memory}",
        has_side_effects=True,
    )


def _header(base, offset, selected):
    guard = "s_cmp_eq_u64 $0, 0\ns_cbranch_scc1 .Lheader_done_${:uid}" if selected else ""
    _asm(
        f"""
        {guard}
        s_memrealtime vcc
        s_waitcnt lgkmcnt(0)
        s_store_dwordx2 vcc, $0, {offset} glc
        s_waitcnt lgkmcnt(0)
        .Lheader_done_${{:uid}}:
    """,
        (base,),
        "s",
    )


def _dynamic_header(base, block, wave, selected, hardware):
    guard = "s_cmp_eq_u64 $0, 0\ns_cbranch_scc1 .Ldynamic_header_done_${:uid}" if selected else ""
    zero = fx.Uint32(0)
    _asm(
        f"""
        {guard}
        s_memrealtime vcc
        s_waitcnt lgkmcnt(0)
        s_store_dwordx2 vcc, $0, 0 glc
        s_store_dword $1, $0, 16 glc
        s_store_dword $2, $0, 20 glc
        s_store_dword $3, $0, 24 glc
        s_store_dword $4, $0, 28 glc
        s_store_dword $5, $0, 32 glc
        s_waitcnt lgkmcnt(0)
        .Ldynamic_header_done_${{:uid}}:
    """,
        (base, zero, *block, wave),
        "s,s,s,s,s,s",
    )
    if hardware:
        _hardware_identity(base, 36, selected)


def _dynamic_record(op, site, base, max_events, selected):
    payload = fx.Uint32(0)
    if len(op.operands):
        payload = fx.Uint32(fx.rocdl.readfirstlane(fx.Uint32.ir_type, fx.Uint32(op.operands[0])))
    guard = "s_cmp_eq_u64 $1, 0\ns_cbranch_scc1 .Ldynamic_done_${:uid}" if selected else ""
    # The scalar atomic is one operation per wave, even when a trace site sits
    # in divergent control flow.  It also serializes the per-wave cursor across
    # loop backedges without adding values to scf.for/scf.while signatures.
    llvm.inline_asm(
        fx.Uint32.ir_type,
        [v.ir_value() for v in (base, payload, fx.Uint32(site))],
        f"""
            s_mov_b32 $0, 1
            {guard}
            s_atomic_add $0, $1, 16 glc
            s_waitcnt lgkmcnt(0)
            s_cmp_ge_u32 $0, {max_events}
            s_cbranch_scc1 .Ldynamic_done_${{:uid}}
            s_mul_i32 $0, $0, {_DYNAMIC_RECORD_WORDS * 4}
            s_memrealtime vcc
            s_waitcnt lgkmcnt(0)
            s_store_dword vcc_lo, $1, $0 offset:{_DYNAMIC_HEADER_WORDS * 4} glc
            s_store_dword $3, $1, $0 offset:{(_DYNAMIC_HEADER_WORDS + 1) * 4} glc
            s_store_dword $2, $1, $0 offset:{(_DYNAMIC_HEADER_WORDS + 2) * 4} glc
            .Ldynamic_done_${{:uid}}:
        """,
        "=&s,s,s,s,~{vcc},~{scc},~{memory}",
        has_side_effects=True,
    )


def _hardware_identity(base, offset, selected):
    guard = "s_cmp_eq_u64 $0, 0\ns_cbranch_scc1 .Lidentity_done_${:uid}" if selected else ""
    _asm(
        f"""
        {guard}
        s_getreg_b32 vcc_lo, hwreg(HW_REG_HW_ID)
        s_getreg_b32 vcc_hi, hwreg(HW_REG_XCC_ID)
        s_store_dwordx2 vcc, $0, $1 glc
        s_waitcnt lgkmcnt(0)
        .Lidentity_done_${{:uid}}:
        """,
        (base, fx.Uint32(offset)),
        "s,s",
    )


def _store_sample(base, dynamic, immediate):
    # Keep the clock intrinsic visible to the backend's wait-count analysis.
    # The scalar store has no public LLVM intrinsic; this one-instruction asm
    # preserves the uniform address and never changes EXEC, M0, SCC, or VCC.
    stamp = fx.Uint32(fx.Uint64(llvm.call_intrinsic(fx.Uint64.ir_type, "llvm.amdgcn.s.memrealtime", [], [], [])))
    if immediate > 0xFFFFF:
        dynamic = fx.Uint32(immediate) if dynamic is None else dynamic + immediate
        immediate = 0
    address = f"$2 offset:{immediate}" if dynamic is not None else str(immediate)
    operands = (stamp, base) if dynamic is None else (stamp, base, dynamic)
    llvm.inline_asm(
        None,
        [v.ir_value() for v in operands],
        f"s_store_dword $0, $1, {address} glc",
        ",".join(["s"] * len(operands)) + ",~{memory}",
        has_side_effects=True,
    )


def _is_dense_loop(loop, children):
    """Buffer trace loops with only scalar bookkeeping and at most two sites.

    Keep MFMA/LDS/VMEM pipelines on the short direct path. Dense timestamp
    streams need buffering: waiting for the next scalar clock would otherwise
    also drain the preceding scalar store. This test deliberately excludes
    opaque operations and nested regions; they retain the general direct path.
    """
    if not 1 <= len(children) <= 2 or any(item[0] != "event" for item in children):
        return False
    allowed = {
        "fly.trace_event",
        "scf.yield",
        "arith.constant",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.index_cast",
        "arith.index_castui",
        "arith.extsi",
        "arith.extui",
        "arith.trunci",
    }
    return all(op.operation.name in allowed for op in loop.regions[0].blocks[0].operations)


def _buffer_dense_loop(item, base, selected, start):
    _, loop, offset, lo, hi, step, children, count = item
    trips = max(0, (hi - lo + step - 1) // step)
    if not trips:
        return False
    iv = loop.regions[0].blocks[0].arguments[0]
    parent_ops = list(loop.parent.regions[0].blocks[0].operations)
    after = parent_ops[next(i for i, op in enumerate(parent_ops) if op.operation == loop) + 1]
    with ir.InsertionPoint(loop), loop.location:
        # The whitelist above excludes all M0 consumers. Preserve it once for
        # this whole scalar-bookkeeping loop instead of at every event.
        saved_m0 = fx.Uint32(
            llvm.inline_asm(fx.Uint32.ir_type, [], "s_mov_b32 $0, m0", "=s,~{memory}", has_side_effects=True)
        )
        lane_offset = fx.Uint32(fx.thread_idx.x % 64) * (count * 4)
        caches = []
        for _ in children:
            cache = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Uint32)
            cache.fill(0)
            caches.append(cache)
    for event, cache in zip(children, caches):
        _, op, event_offset = event
        immediate = 16 + 4 * (start + offset + event_offset)
        # Large offsets use the direct path. All the ordinary page offsets are
        # derived from a bounded induction variable, with no mutable cursor.
        if immediate > 0xFFFFF:
            raise ValueError("dense flytrace loop starts beyond the 1 MiB immediate-address window")
        with ir.InsertionPoint(op), op.location:
            index = (fx.Uint32(iv) - lo) // step
            previous = fx.Uint32(cache[0])
            guard = "s_cmp_eq_u64 $4, 0\ns_cbranch_scc1 .Ldense_done_${:uid}" if selected else ""
            if trips > 64:
                body = f"""
                    s_and_b32 m0, $3, 63
                    v_writelane_b32 $0, vcc_lo, m0
                    s_cmp_eq_u32 m0, 63
                    s_cbranch_scc0 .Ldense_done_${{:uid}}
                    s_and_b32 vcc_lo, $3, -64
                    s_mul_i32 vcc_lo, vcc_lo, {count * 4}
                    v_add_u32 $1, vcc_lo, $5
                    global_store_dword $1, $0, $4 offset:{immediate}
                    s_waitcnt vmcnt(0)
                """
            else:
                body = """
                    s_mov_b32 m0, $3
                    v_writelane_b32 $0, vcc_lo, m0
                """
            result = llvm.inline_asm(
                ir.Type.parse("!llvm.struct<(i32, i32)>"),
                [v.ir_value() for v in (previous, index, base, lane_offset)],
                f"""
                    {guard}
                    s_memrealtime vcc
                    s_waitcnt lgkmcnt(0)
                    {body}
                    .Ldense_done_${{:uid}}:
                """,
                "=&v,=&v,0,s,s,v,~{m0},~{vcc},~{scc},~{memory}",
                has_side_effects=True,
            )
            cache[0] = fx.Uint32(llvm.extractvalue(fx.Uint32.ir_type, result, [0]))
        op.erase()
    with ir.InsertionPoint(after), loop.location:
        llvm.inline_asm(None, [saved_m0.ir_value()], "s_mov_b32 m0, $0", "s,~{m0},~{memory}", has_side_effects=True)
    tail = trips if trips <= 64 else trips % 64
    if tail:
        page_start = 0 if trips <= 64 else trips - tail
        with ir.InsertionPoint(after), loop.location:
            # Materialize the exact 64-bit mask in SGPRs; a 32-bit literal is
            # insufficient for tails such as 32 or 33 lanes.
            mask = fx.Uint64((1 << tail) - 1)
            guard = "s_cmp_eq_u64 $1, 0\ns_cbranch_scc1 .Ldense_flush_done_${:uid}" if selected else ""
            for event, cache in zip(children, caches):
                immediate = 16 + 4 * (start + offset + event[2] + page_start * count)
                address = lane_offset
                if immediate > 0xFFFFF:
                    address = lane_offset + immediate
                    immediate = 0
                llvm.inline_asm(
                    fx.Uint64.ir_type,
                    [v.ir_value() for v in (base, address, fx.Uint32(cache[0]), mask)],
                    f"""
                        {guard}
                        s_and_saveexec_b64 $0, $4
                        global_store_dword $2, $3, $1 offset:{immediate}
                        s_mov_b64 exec, $0
                        s_waitcnt vmcnt(0)
                        .Ldense_flush_done_${{:uid}}:
                    """,
                    "=&s,s,v,v,s,~{scc},~{memory}",
                    has_side_effects=True,
                )
    return True


def _lower_sites(items, base, selected, start=0, terms=()):
    for item in items:
        if item[0] == "loop":
            _, op, offset, lo, hi, step, child, count = item
            if not terms and _is_dense_loop(op, child) and 16 + 4 * (start + offset + count) <= 0xFFFFF:
                if _buffer_dense_loop(item, base, selected, start):
                    continue
            iv = op.regions[0].blocks[0].arguments[0]
            _lower_sites(child, base, selected, start + offset, terms + ((iv, lo, step, count),))
            continue
        _, op, offset = item
        with ir.InsertionPoint(op), op.location:
            immediate = 16 + 4 * (start + offset)
            dynamic = None
            for iv, lo, step, count in terms:
                term = fx.Uint32(iv) - lo
                term = term * (count * 4 // step) if count * 4 % step == 0 else (term // step) * (count * 4)
                dynamic = term if dynamic is None else dynamic + term
            if selected:
                # Keep the sampling guard out of the computation CFG. On gfx942
                # a native SCF guard changes loop liveness and raises VGPR use.
                if immediate > 0xFFFFF:
                    dynamic = fx.Uint32(immediate) if dynamic is None else dynamic + immediate
                    immediate = 0
                address = f"$1 offset:{immediate}" if dynamic is not None else str(immediate)
                _asm(
                    f"""
                    s_cmp_eq_u64 $0, 0
                    s_cbranch_scc1 .Ltrace_done_${{:uid}}
                    s_memrealtime vcc
                    s_waitcnt lgkmcnt(0)
                    s_store_dword vcc_lo, $0, {address} glc
                    .Ltrace_done_${{:uid}}:
                """,
                    (base,) if dynamic is None else (base, dynamic),
                    "s" if dynamic is None else "s,s",
                )
            else:
                _store_sample(base, dynamic, immediate)
        op.erase()


def _grid_metadata(grid):
    from ..expr.typing import as_ir_value

    values = tuple(as_ir_value(v, keep_static=True) for v in grid)
    return values if all(isinstance(v, int) for v in values) else None


def _wave_base(entry, offset, stride, waves_per_block, blocks, capacity_blocks):
    bx, by, bz = (fx.Uint32(x) for x in fx.block_idx)
    wave = fx.Uint32(fx.rocdl.readfirstlane(fx.Uint32.ir_type, fx.Uint32(fx.thread_idx.x // 64)))
    selected = blocks is not None or capacity_blocks is not None
    if blocks is not None:
        slot = fx.Uint32(0)
        matched = None
        for number, (x, y, z) in enumerate(blocks):
            pred = (bx == x) & (by == y) & (bz == z)
            slot = pred.select(fx.Uint32(number), slot)
            matched = pred if matched is None else matched | pred
        block_index = slot
        predicate = matched
    else:
        gx, gy, _ = (fx.Uint32(x) for x in fx.grid_dim)
        block_index = (bz * gy + by) * gx + bx
        predicate = None if capacity_blocks is None else block_index < fx.Uint32(capacity_blocks)
    index = block_index * waves_per_block + wave
    base = fx.Uint64(entry.arguments[-1]) + offset * 4 + fx.Uint64(index) * (stride * 4)
    if predicate is not None:
        base = fx.Uint64(
            llvm.inline_asm(
                fx.Uint64.ir_type,
                [base.ir_value(), fx.Uint32(predicate).ir_value()],
                "s_cmp_lg_u32 $2, 0\ns_cselect_b64 $0, $1, 0",
                "=s,s,s,~{scc}",
                has_side_effects=True,
            )
        )
    return base, (bx, by, bz), wave, selected


def lower_kernel(func, ctx, grid, block, stream):
    """Plan annotations before Fly/LLVM lowering; append only a hidden ABI pointer."""
    from ..runtime.device import get_rocm_arch

    arch = get_rocm_arch().split(":")[0]
    if arch not in _SUPPORTED_ARCHES:
        raise ValueError(f"flytrace requires one of {_SUPPORTED_ARCHES}, got {arch}")
    if block is None or tuple(block[1:]) != (1, 1) or block[0] % 64:
        raise ValueError("flytrace requires a one-dimensional block of complete wave64 waves")
    entry = func.regions[0].blocks[0]
    sites = _dynamic_sites(entry)
    if not sites:
        return
    options = ctx.trace_spec["options"]
    requested_mode = _option(options, "mode")
    selected_blocks = _selected_blocks(ctx)
    max_events = _option(options, "max_events")
    max_blocks = _option(options, "max_blocks")
    hardware = _option(options, "hardware")
    if hardware and arch != "gfx942":
        raise ValueError("flytrace hardware identity / ATT merging currently requires gfx942")

    grid_static = _grid_metadata(grid)
    if grid_static is not None and any(v <= 0 for v in grid_static):
        raise ValueError("flytrace requires positive launch dimensions")
    if selected_blocks is not None and grid_static is not None:
        for selected in selected_blocks:
            if any(not 0 <= b < n for b, n in zip(selected, grid_static)):
                raise ValueError(f"flytrace sampled block {selected} is outside launch grid {grid_static}")

    static_error = None
    try:
        items, count = _layout(entry)
        schema = _schema(items)
        if len(schema) != count:
            raise RuntimeError("flytrace internal schema/layout mismatch")
    except ValueError as exc:
        static_error = exc
        items = schema = None
        count = 0
    mode = requested_mode
    if mode == "auto":
        mode = (
            "static" if static_error is None and (grid_static is not None or selected_blocks is not None) else "dynamic"
        )
    elif mode == "static" and static_error is not None:
        raise static_error
    if mode == "static" and selected_blocks is None and grid_static is None:
        raise ValueError(
            "flytrace mode='static' with block=None requires static launch dimensions; "
            "use mode='auto'/'dynamic' or select explicit blocks"
        )

    if selected_blocks is not None:
        capacity_blocks = len(selected_blocks)
        guard_capacity = None
    elif grid_static is not None:
        total_blocks = math.prod(grid_static)
        capacity_blocks = min(total_blocks, max_blocks) if max_blocks is not None else total_blocks
        guard_capacity = capacity_blocks if capacity_blocks < total_blocks else None
    else:
        if max_blocks is None:
            raise ValueError("dynamic launch dimensions with block=None require capture(max_blocks=...)")
        capacity_blocks = max_blocks
        guard_capacity = capacity_blocks

    waves_per_block = block[0] // 64
    waves = waves_per_block * capacity_blocks
    if mode == "static":
        stride = ((4 + count + (2 if hardware else 0) + 15) // 16) * 16
    else:
        schema = _dynamic_schema(sites)
        stride = ((_DYNAMIC_HEADER_WORDS + max_events * _DYNAMIC_RECORD_WORDS + 15) // 16) * 16
    offset = ctx.trace_spec["words"]
    ctx.trace_spec["words"] += waves * stride
    ctx.trace_spec["kernels"].append(
        dict(
            name=ir.StringAttr(func.attributes["sym_name"]).value,
            mode=mode,
            grid=grid_static,
            block=tuple(block),
            selected=selected_blocks,
            capacity_blocks=capacity_blocks,
            waves=waves,
            stride=stride,
            offset=offset,
            events=schema,
            hardware=hardware,
            max_events=max_events if mode == "dynamic" else None,
        )
    )
    with ir.InsertionPoint.at_block_begin(entry), func.location:
        base, block_id, wave, selected = _wave_base(
            entry,
            offset,
            stride,
            waves_per_block,
            selected_blocks,
            guard_capacity,
        )
        if mode == "static":
            if hardware:
                _hardware_identity(base, (4 + count) * 4, selected)
            _header(base, 0, selected)
        else:
            _dynamic_header(base, block_id, wave, selected, hardware)
    if mode == "static":
        _lower_sites(items, base, selected)
    else:
        for site, op in enumerate(sites):
            with ir.InsertionPoint(op), op.location:
                _dynamic_record(op, site, base, max_events, selected)
            op.erase()
    with ir.InsertionPoint(entry.operations[-1]), func.location:
        _header(base, 8, selected)


class TraceCallState:
    def __init__(self, state, spec):
        self.state, self.spec = state, spec

    def __call__(self, args):
        cap = _active.get()
        if cap is None:
            raise RuntimeError("a trace-enabled compiled function must run inside flytrace.capture()")
        if cap.options != self.spec["options"]:
            raise ValueError("compiled trace configuration differs from the active capture")
        return self.state(args)


def fill_buffer(spec, _ignored, storage):
    cap = _active.get()
    if cap is None:
        raise RuntimeError("trace launch requires an active flytrace.capture()")
    storage.value = cap._buffer(spec).data_ptr()


class capture:
    """Capture the latest launch of each compiled JIT specialization.

    Allocations and the extra ABI argument are managed internally. Repeated
    launches overwrite the same fixed schedule; call export after completion.
    A configure() context in the JIT selects blocks; its default is (0, 0, 0).
    Supplying block explicitly here overrides JIT selection, including None for
    all blocks or a sequence of block-coordinate tuples. ``mode="auto"`` uses the compact static
    format when possible and falls back to a bounded dynamic recorder for runtime
    loops/branches. ``max_blocks`` bounds all-grid capture for a dynamic grid.
    """

    def __init__(
        self,
        path=None,
        *,
        block=_DEFAULT_BLOCK,
        exclude=(),
        hardware=False,
        mode="auto",
        max_events=65536,
        max_blocks=None,
    ):
        if mode not in ("auto", "static", "dynamic"):
            raise ValueError("flytrace mode must be 'auto', 'static', or 'dynamic'")
        if type(max_events) is not int or not 0 < max_events <= _MAX_EVENTS:
            raise ValueError("flytrace max_events must be in [1, 2**20]")
        if max_blocks is not None and (type(max_blocks) is not int or max_blocks <= 0):
            raise ValueError("flytrace max_blocks must be a positive integer or None")
        blocks = "jit" if block is _DEFAULT_BLOCK else _normalize_blocks(block)
        self.path = path
        self.options = (
            "flytrace-v3",
            mode,
            blocks,
            tuple(sorted(exclude)),
            max_events,
            max_blocks,
            bool(hardware),
        )
        self._records = {}
        self._buffers = []
        self._entered = False

    def __enter__(self):
        import torch

        if _active.get() is not None or self._entered:
            raise RuntimeError("nested/reentrant flytrace captures are unsupported")
        self.device = torch.cuda.current_device()
        self._entered = True
        self._token = _active.set(self)
        self._hints = CompilationContext.compile_hints({"flytrace": self.options})
        self._hints.__enter__()
        return self

    def __exit__(self, typ, value, tb):
        import torch

        try:
            torch.cuda.synchronize(self.device)
            if typ is None and self.path is not None:
                self.export(self.path)
        finally:
            self._hints.__exit__(typ, value, tb)
            _active.reset(self._token)
            self._entered = False

    def _buffer(self, spec):
        import torch

        if torch.cuda.current_device() != self.device:
            raise ValueError("cannot change device inside flytrace capture")
        key = id(spec)
        if spec["words"] * 4 > 512 * 1024**2:
            raise ValueError("flytrace capture exceeds the 512 MiB allocation limit; select fewer blocks/events")
        # A fresh buffer makes repeated launches deterministic even when a
        # dynamic grid shrinks. Synchronizing here establishes initialization
        # before a kernel launched on an arbitrary user stream consumes it.
        buffer = torch.zeros(spec["words"], dtype=torch.int32, device=f"cuda:{self.device}")
        torch.cuda.synchronize(self.device)
        self._buffers.append(buffer)
        self._records[key] = (spec, buffer)
        return buffer

    def decode(self):
        import torch

        torch.cuda.synchronize(self.device)
        waves = []
        for spec, buffer in self._records.values():
            words = [v & 0xFFFFFFFF for v in buffer.cpu().tolist()]
            for kernel in spec["kernels"]:
                for i in range(kernel["waves"]):
                    pos = kernel["offset"] + i * kernel["stride"]
                    row = words[pos : pos + kernel["stride"]]
                    epoch = row[0] | row[1] << 32
                    end_tick = row[2] | row[3] << 32
                    if not epoch and kernel.get("mode") == "dynamic":
                        continue
                    if not epoch or not 0 <= end_tick - epoch < 1 << 32:
                        raise ValueError("invalid/incomplete flytrace header or capture exceeded 42.95 seconds")
                    records = []
                    previous = epoch
                    if kernel.get("mode") == "dynamic":
                        attempted = row[4]
                        count = min(attempted, kernel["max_events"])
                        for n in range(count):
                            record_pos = _DYNAMIC_HEADER_WORDS + n * _DYNAMIC_RECORD_WORDS
                            tick = epoch + ((row[record_pos] - epoch) & 0xFFFFFFFF)
                            site = row[record_pos + 1]
                            if site >= len(kernel["events"]):
                                raise ValueError(f"invalid flytrace site {site} at wave {i}, event {n}")
                            if not previous <= tick <= end_tick:
                                raise ValueError(f"invalid timestamp at wave {i}, event {n}")
                            previous = tick
                            event = kernel["events"][site]
                            payload = row[record_pos + 2]
                            if payload >= 1 << 31:
                                payload -= 1 << 32
                            records.append(
                                {
                                    "name": event["name"],
                                    "kind": event["kind"],
                                    "payload": payload if event["has_payload"] else None,
                                    "site": site,
                                    "tick": tick,
                                    "ordinal": n,
                                }
                            )
                        block = tuple(row[5:8])
                        wave = row[8]
                        overflow = attempted > kernel["max_events"]
                    else:
                        for n, event in enumerate(kernel["events"]):
                            tick = epoch + ((row[4 + n] - epoch) & 0xFFFFFFFF)
                            if not previous <= tick <= end_tick:
                                raise ValueError(f"invalid timestamp at wave {i}, event {n}")
                            previous = tick
                            records.append({**event, "tick": tick, "ordinal": n})
                        cta, wave = divmod(i, kernel["block"][0] // 64)
                        if kernel["selected"] is not None:
                            block = kernel["selected"][cta]
                        else:
                            gx, gy, _ = kernel["grid"]
                            block = (cta % gx, (cta // gx) % gy, cta // (gx * gy))
                        attempted = len(records)
                        overflow = False
                    waves.append(
                        dict(
                            kernel=kernel["name"],
                            block=block,
                            wave=wave,
                            epoch=epoch,
                            end_tick=end_tick,
                            events=records,
                            attempted_events=attempted,
                            overflow=overflow,
                        )
                    )
                    if kernel.get("hardware"):
                        hardware_pos = 9 if kernel.get("mode") == "dynamic" else 4 + len(records)
                        hw_id, xcc = row[hardware_pos : hardware_pos + 2]
                        waves[-1]["hardware"] = dict(
                            raw_hw_id=hw_id,
                            xcc=xcc & 15,
                            se=(hw_id >> 13) & 3,
                            sh=(hw_id >> 12) & 1,
                            cu=(hw_id >> 8) & 15,
                            simd=(hw_id >> 4) & 3,
                            slot=hw_id & 15,
                        )
        return waves

    def save(self, path):
        """Save absolute realtime ticks and wave identities for offline ATT merging.

        Use hardware=True and collect ATT during the same launch. The default
        export() remains a standalone Perfetto trace with a relative origin.
        """
        from ..runtime.device import get_rocm_arch

        waves = self.decode()
        Path(path).write_text(
            json.dumps(
                dict(
                    format="flytrace.raw.v1",
                    arch=get_rocm_arch(),
                    clock_hz=100_000_000,
                    waves=waves,
                )
            )
            + "\n"
        )
        return dict(
            waves=len(waves),
            records=sum(len(w["events"]) for w in waves),
            dropped=sum(max(0, w["attempted_events"] - len(w["events"])) for w in waves),
        )

    def export(self, path):
        waves = self.decode()
        origin = min((w["epoch"] for w in waves), default=0)
        events = []
        for tid, wave in enumerate(waves, 1):
            events.append(
                dict(
                    ph="M",
                    name="thread_name",
                    pid=1,
                    tid=tid,
                    args=dict(name=f"{wave['kernel']} Block {wave['block']} / wave {wave['wave']}"),
                )
            )
            stack = []
            current = None

            if wave.get("overflow"):
                events.append(
                    dict(
                        ph="i",
                        s="t",
                        name="flytrace overflow",
                        cat="flytrace.warning",
                        pid=1,
                        tid=tid,
                        ts=(wave["end_tick"] - origin) / 100,
                        args=dict(
                            attempted=wave["attempted_events"],
                            captured=len(wave["events"]),
                        ),
                    )
                )

            def interval(a, b):
                args = dict(site=a.get("site")) if "site" in a else {}
                if a["payload"] is not None:
                    args["payload"] = a["payload"]
                events.append(
                    dict(
                        ph="X",
                        name=a["name"],
                        pid=1,
                        tid=tid,
                        ts=(a["tick"] - origin) / 100,
                        dur=(b["tick"] - a["tick"]) / 100,
                        args=args,
                    )
                )

            for event in wave["events"]:
                kind = event["kind"]
                if kind == "mark":
                    args = dict(site=event.get("site")) if "site" in event else {}
                    if event["payload"] is not None:
                        args["payload"] = event["payload"]
                    events.append(
                        dict(
                            ph="i",
                            s="t",
                            name=event["name"],
                            pid=1,
                            tid=tid,
                            ts=(event["tick"] - origin) / 100,
                            args=args,
                        )
                    )
                elif kind == "push":
                    stack.append(event)
                elif kind == "pop":
                    if not stack:
                        raise ValueError("unmatched flytrace.pop()")
                    interval(stack.pop(), event)
                else:
                    if current is not None:
                        interval(current, event)
                    current = event if kind == "boundary" else None
            if (stack or current) and not wave.get("overflow"):
                raise ValueError("unfinished flytrace range")
        Path(path).write_text(json.dumps(dict(traceEvents=events, displayTimeUnit="ns")) + "\n")
        return dict(
            waves=len(waves),
            records=sum(len(w["events"]) for w in waves),
            dropped=sum(max(0, w["attempted_events"] - len(w["events"])) for w in waves),
        )
