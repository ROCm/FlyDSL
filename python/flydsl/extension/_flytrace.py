# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Static-schedule wave tracing: compiler planning and scoped capture runtime.

The fast path requires full wave64, static launch dimensions, and positive-step
constant-bound scf.for loops. Trace sites inside runtime branches are rejected.
Names, kinds, and affine loop payloads live in the host schema, not GPU records.
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


@contextmanager
def configure(*, block):
    """Scope CTA selection to kernel launches inside a with block in @jit.

    Pass a compile-time (x, y, z) tuple, or None for all CTAs. Each selected CTA
    records all its waves. An explicit capture(block=...) overrides this choice.
    Nested contexts restore the outer selection on exit, including exceptions.
    Without an active capture, this emits no instrumentation or extra arguments.
    """
    from ..compiler.kernel_function import KernelFunction

    ctx = CompilationContext.get_current()
    if ctx is None or KernelFunction.get_current() is not None:
        raise RuntimeError("flytrace.configure belongs in a @jit before kernel launch")
    if ir.InsertionPoint.current.block.owner.operation.name != "func.func":
        raise ValueError("flytrace.configure requires top-level JIT or compile-time control flow")
    if block is not None:
        if not isinstance(block, (tuple, list)) or len(block) != 3 or any(type(n) is not int or n < 0 for n in block):
            raise ValueError("flytrace.configure block must be a compile-time (x, y, z) tuple or None")
        block = tuple(block)
    previous = ctx.trace_block
    ctx.trace_block = block
    try:
        yield
    finally:
        ctx.trace_block = previous


def _selected_block(ctx):
    override = ctx.trace_spec["options"][1]
    return ctx.trace_block if override == "jit" else override


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
    if name in ctx.trace_spec["options"][2]:
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


def lower_kernel(func, ctx, grid, block, stream):
    """Plan annotations before Fly/LLVM lowering; append only a hidden ABI pointer."""
    from ..expr.typing import as_ir_value
    from ..runtime.device import get_rocm_arch

    if get_rocm_arch().split(":")[0] != "gfx942":
        raise ValueError("flytrace static recorder currently requires gfx942")
    grid = tuple(as_ir_value(v, keep_static=True) for v in grid)
    if any(not isinstance(v, int) or v <= 0 for v in grid):
        raise ValueError("flytrace requires static positive launch dimensions")
    if block is None or block[1:] != [1, 1] or block[0] % 64:
        raise ValueError("flytrace requires a one-dimensional block of complete wave64 waves")
    # Capture currently supports the default stream and a directly forwarded
    # JIT stream argument. Do not silently accept an internal opaque stream.
    if (
        stream is not None
        and isinstance(as_ir_value(stream, keep_static=True), int)
        and as_ir_value(stream, keep_static=True) != 0
    ):
        raise ValueError("flytrace capture currently requires the default stream")
    entry = func.regions[0].blocks[0]
    items, count = _layout(entry)
    schema = _schema(items)
    assert len(schema) == count
    if not count:
        return
    selected = _selected_block(ctx)
    if selected is not None and (len(selected) != 3 or any(not 0 <= b < n for b, n in zip(selected, grid))):
        raise ValueError("flytrace sampled CTA is outside the launch grid")
    waves = block[0] // 64 * (math.prod(grid) if selected is None else 1)
    hardware = ctx.trace_spec["options"][0] == "static-v2-hardware"
    stride = ((4 + count + (2 if hardware else 0) + 15) // 16) * 16
    offset = ctx.trace_spec["words"]
    ctx.trace_spec["words"] += waves * stride
    ctx.trace_spec["kernels"].append(
        dict(
            name=ir.StringAttr(func.attributes["sym_name"]).value,
            grid=grid,
            block=tuple(block),
            selected=selected,
            waves=waves,
            stride=stride,
            offset=offset,
            events=schema,
            hardware=hardware,
        )
    )
    with ir.InsertionPoint.at_block_begin(entry), func.location:
        bx, by, bz = (fx.Uint32(x) for x in fx.block_idx)
        wave = fx.Uint32(fx.rocdl.readfirstlane(fx.Uint32.ir_type, fx.Uint32(fx.thread_idx.x // 64)))
        index = ((bz * grid[1] + by) * grid[0] + bx) * (block[0] // 64) + wave if selected is None else wave
        base = fx.Uint64(entry.arguments[-1]) + offset * 4 + fx.Uint64(index) * (stride * 4)
        if selected is not None:
            pred = fx.Uint32((bx == selected[0]) & (by == selected[1]) & (bz == selected[2]))
            base = fx.Uint64(
                llvm.inline_asm(
                    fx.Uint64.ir_type,
                    [base.ir_value(), pred.ir_value()],
                    "s_cmp_lg_u32 $2, 0\ns_cselect_b64 $0, $1, 0",
                    "=s,s,s,~{scc}",
                    has_side_effects=True,
                )
            )
        if hardware:
            _hardware_identity(base, (4 + count) * 4, selected is not None)
        _header(base, 0, selected is not None)
    _lower_sites(items, base, selected is not None)
    with ir.InsertionPoint(entry.operations[-1]), func.location:
        _header(base, 8, selected is not None)


class TraceCallState:
    def __init__(self, state, spec, streams):
        self.state, self.spec, self.streams = state, spec, streams

    def __call__(self, args):
        cap = _active.get()
        if cap is None:
            raise RuntimeError("a trace-enabled compiled function must run inside flytrace.capture()")
        if cap.options != self.spec["options"]:
            raise ValueError("compiled trace configuration differs from the active capture")
        for i in self.streams:
            raw = args[i].value if getattr(args[i], "_is_stream_param", False) else args[i]
            pointer = raw.cuda_stream if hasattr(raw, "cuda_stream") else raw
            if pointer not in (None, 0):
                raise ValueError("flytrace capture currently requires the default stream")
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
    A configure() context in the JIT selects the CTA; its default is (0, 0, 0). Supplying
    block explicitly here overrides JIT selection, including None for all CTAs.
    Only the default stream is supported in this experimental implementation.
    """

    def __init__(self, path=None, *, block=_DEFAULT_BLOCK, exclude=(), hardware=False):
        self.path = path
        self.options = (
            "static-v2-hardware" if hardware else "static-v1",
            "jit" if block is _DEFAULT_BLOCK else None if block is None else tuple(block),
            tuple(sorted(exclude)),
        )
        self._records = {}
        self._entered = False

    def __enter__(self):
        import torch

        if _active.get() is not None or self._entered:
            raise RuntimeError("nested/reentrant flytrace captures are unsupported")
        if torch.cuda.current_stream().cuda_stream != 0:
            raise ValueError("flytrace capture currently requires the default stream")
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
        if key not in self._records:
            if spec["words"] * 4 > 512 * 1024**2:
                raise ValueError("flytrace capture exceeds the 512 MiB allocation limit; sample one CTA")
            self._records[key] = (
                spec,
                torch.zeros(spec["words"], dtype=torch.int32, device=f"cuda:{self.device}"),
            )
        return self._records[key][1]

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
                    if not epoch or not 0 <= end_tick - epoch < 1 << 32:
                        raise ValueError("invalid/incomplete flytrace header or capture exceeded 42.95 seconds")
                    records = []
                    previous = epoch
                    for n, event in enumerate(kernel["events"]):
                        tick = epoch + ((row[4 + n] - epoch) & 0xFFFFFFFF)
                        if not previous <= tick <= end_tick:
                            raise ValueError(f"invalid timestamp at wave {i}, event {n}")
                        previous = tick
                        records.append({**event, "tick": tick, "ordinal": n})
                    cta, wave = divmod(i, kernel["block"][0] // 64)
                    gx, gy, _ = kernel["grid"]
                    block = kernel["selected"] or (
                        cta % gx,
                        (cta // gx) % gy,
                        cta // (gx * gy),
                    )
                    waves.append(
                        dict(
                            kernel=kernel["name"],
                            block=block,
                            wave=wave,
                            epoch=epoch,
                            end_tick=end_tick,
                            events=records,
                        )
                    )
                    if kernel.get("hardware"):
                        hw_id, xcc = row[4 + len(records) : 6 + len(records)]
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
        return dict(waves=len(waves), records=sum(len(w["events"]) for w in waves))

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
                    args=dict(name=f"{wave['kernel']} CTA {wave['block']} / wave {wave['wave']}"),
                )
            )
            stack = []
            current = None

            def interval(a, b):
                events.append(
                    dict(
                        ph="X",
                        name=a["name"],
                        pid=1,
                        tid=tid,
                        ts=(a["tick"] - origin) / 100,
                        dur=(b["tick"] - a["tick"]) / 100,
                        args={} if a["payload"] is None else dict(payload=a["payload"]),
                    )
                )

            for event in wave["events"]:
                kind = event["kind"]
                if kind == "mark":
                    events.append(
                        dict(
                            ph="i",
                            s="t",
                            name=event["name"],
                            pid=1,
                            tid=tid,
                            ts=(event["tick"] - origin) / 100,
                            args=({} if event["payload"] is None else dict(payload=event["payload"])),
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
            if stack or current:
                raise ValueError("unfinished flytrace range")
        Path(path).write_text(json.dumps(dict(traceEvents=events, displayTimeUnit="ns")) + "\n")
        return dict(waves=len(waves), records=sum(len(w["events"]) for w in waves))
