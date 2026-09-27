# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""ROCm trace lowering with a static fast path and dynamic fallback.

Static schedules keep the original low-overhead one-word records.  Kernels with
runtime loops, branches, payloads, or launch dimensions automatically use a
bounded dynamic recorder.  Dynamic records carry their site and payload, which
makes them suitable for persistent and heterogeneous mega kernels.
"""

import math

from .. import expr as fx
from .._mlir import ir
from .._mlir.dialects import llvm
from ..compiler.backends import current_target
from ._flytrace_backend import TraceBackend, register_trace_backend
from ._flytrace_schema import (
    dynamic_schema,
    dynamic_sites,
    grid_metadata,
    option,
    selected_blocks,
    static_layout,
    static_schema,
)

_DYNAMIC_HEADER_WORDS = 16
_DYNAMIC_RECORD_WORDS = 3
_SUPPORTED_ARCHES = ("gfx942", "gfx950")


def _waves_per_block(block, wave_size):
    return (math.prod(block) + wave_size - 1) // wave_size


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


def _lower_sites(items, base, selected, start=0, terms=(), *, dense_loops=True):
    for item in items:
        if item[0] == "loop":
            _, op, offset, lo, hi, step, child, count = item
            if dense_loops and not terms and _is_dense_loop(op, child) and 16 + 4 * (start + offset + count) <= 0xFFFFF:
                if _buffer_dense_loop(item, base, selected, start):
                    continue
            iv = op.regions[0].blocks[0].arguments[0]
            _lower_sites(
                child,
                base,
                selected,
                start + offset,
                terms + ((iv, lo, step, count),),
                dense_loops=dense_loops,
            )
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


def _wave_base(entry, offset, stride, block, wave_size, blocks, capacity_blocks):
    bx, by, bz = (fx.Uint32(x) for x in fx.block_idx)
    tx, ty, tz = (fx.Uint32(x) for x in fx.thread_idx)
    linear_thread = (tz * block[1] + ty) * block[0] + tx
    wave = fx.Uint32(fx.rocdl.readfirstlane(fx.Uint32.ir_type, linear_thread // wave_size))
    waves_per_block = math.prod(block) // wave_size
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


def _lower_kernel(func, ctx, grid, block, stream):
    """Plan annotations before device lowering; append only a hidden ABI pointer."""

    entry = func.regions[0].blocks[0]
    sites = dynamic_sites(entry)
    target = current_target()
    arch = target.arch.split(":")[0]
    if arch not in _SUPPORTED_ARCHES:
        raise ValueError(f"flytrace requires one of {_SUPPORTED_ARCHES}, got {arch}")
    if target.warp_size != 64:
        raise ValueError(f"flytrace ROCm backend requires wave64, got wave{target.warp_size} on {arch}")
    if block is None or any(type(extent) is not int or extent <= 0 for extent in block):
        raise ValueError("flytrace requires positive static block dimensions")
    options = ctx.trace_spec["options"]
    requested_mode = option(options, "mode")
    selected = selected_blocks(ctx)
    max_events = option(options, "max_events")
    max_blocks = option(options, "max_blocks")
    hardware = option(options, "hardware")
    if hardware and arch != "gfx942":
        raise ValueError("flytrace hardware identity / ATT merging currently requires gfx942")

    grid_static = grid_metadata(grid)
    if grid_static is not None and any(v <= 0 for v in grid_static):
        raise ValueError("flytrace requires positive launch dimensions")
    if selected is not None and grid_static is not None:
        for coordinate in selected:
            if any(not 0 <= value < extent for value, extent in zip(coordinate, grid_static)):
                raise ValueError(f"flytrace sampled block {coordinate} is outside launch grid {grid_static}")

    static_error = None
    try:
        items, count = static_layout(entry)
        schema = static_schema(items)
        if len(schema) != count:
            raise RuntimeError("flytrace internal schema/layout mismatch")
    except ValueError as exc:
        static_error = exc
        items = schema = None
        count = 0
    mode = requested_mode
    if mode == "auto":
        mode = (
            "static" if static_error is None and (grid_static is not None or selected is not None) else "dynamic"
        )
    elif mode == "static" and static_error is not None:
        raise static_error
    if mode == "static" and selected is None and grid_static is None:
        raise ValueError(
            "flytrace mode='static' with block=None requires static launch dimensions; "
            "use mode='auto'/'dynamic' or select explicit blocks"
        )

    if selected is not None:
        capacity_blocks = len(selected)
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

    waves_per_block = _waves_per_block(block, target.warp_size)
    waves = waves_per_block * capacity_blocks
    if mode == "static":
        stride = ((4 + count + (2 if hardware else 0) + 15) // 16) * 16
    else:
        schema = dynamic_schema(sites)
        record_capacity = max_events if sites else 0
        stride = ((_DYNAMIC_HEADER_WORDS + record_capacity * _DYNAMIC_RECORD_WORDS + 15) // 16) * 16
    offset = ctx.trace_spec["words"]
    ctx.trace_spec["words"] += waves * stride
    existing_backend = ctx.trace_spec.setdefault("backend", "rocm")
    if existing_backend != "rocm":
        raise RuntimeError(f"trace spec mixes backend {existing_backend!r} with 'rocm'")
    existing_arch = ctx.trace_spec.setdefault("arch", arch)
    if existing_arch != arch:
        raise RuntimeError(f"trace spec mixes architectures {existing_arch!r} and {arch!r}")
    ctx.trace_spec["clock_hz"] = 100_000_000
    ctx.trace_spec["kernels"].append(
        dict(
            name=ir.StringAttr(func.attributes["sym_name"]).value,
            mode=mode,
            grid=grid_static,
            block=tuple(block),
            wave_size=target.warp_size,
            selected=selected,
            capacity_blocks=capacity_blocks,
            waves=waves,
            stride=stride,
            offset=offset,
            events=schema,
            hardware=hardware,
            max_events=record_capacity if mode == "dynamic" else None,
        )
    )
    with ir.InsertionPoint.at_block_begin(entry), func.location:
        base, block_id, wave, guarded = _wave_base(
            entry,
            offset,
            stride,
            block,
            target.warp_size,
            selected,
            guard_capacity,
        )
        if mode == "static":
            if hardware:
                _hardware_identity(base, (4 + count) * 4, guarded)
            _header(base, 0, guarded)
        else:
            _dynamic_header(base, block_id, wave, guarded, hardware)
    if mode == "static":
        _lower_sites(items, base, guarded, dense_loops=tuple(block[1:]) == (1, 1))
    else:
        for site, op in enumerate(sites):
            with ir.InsertionPoint(op), op.location:
                _dynamic_record(op, site, base, max_events, guarded)
            op.erase()
    with ir.InsertionPoint(entry.operations[-1]), func.location:
        _header(base, 8, guarded)


class RocmTraceBackend(TraceBackend):
    """AMDGPU wave64 recorder and HIP-backed capture storage."""

    name = "rocm"
    clock_hz = 100_000_000

    def lower_kernel(self, func, ctx, grid, block, stream):
        _lower_kernel(func, ctx, grid, block, stream)

    def current_device(self):
        import torch

        return torch.cuda.current_device()

    def synchronize(self, device):
        import torch

        torch.cuda.synchronize(device)

    def allocate_buffer(self, words, device):
        import torch

        return torch.zeros(words, dtype=torch.int32, device=f"cuda:{device}")

    def buffer_pointer(self, buffer):
        return buffer.data_ptr()

    def buffer_words(self, buffer):
        return [value & 0xFFFFFFFF for value in buffer.cpu().tolist()]

    def decode(self, spec, words):
        waves = []
        for kernel in spec["kernels"]:
            for index in range(kernel["waves"]):
                pos = kernel["offset"] + index * kernel["stride"]
                row = words[pos : pos + kernel["stride"]]
                if len(row) != kernel["stride"]:
                    raise ValueError("flytrace buffer is shorter than its compiled record layout")
                epoch = row[0] | row[1] << 32
                end_tick = row[2] | row[3] << 32
                if not epoch and kernel["mode"] == "dynamic":
                    continue
                if not epoch or not 0 <= end_tick - epoch < 1 << 32:
                    raise ValueError("invalid/incomplete flytrace header or capture exceeded 42.95 seconds")
                records = []
                previous = epoch
                if kernel["mode"] == "dynamic":
                    attempted = row[4]
                    count = min(attempted, kernel["max_events"])
                    for ordinal in range(count):
                        record_pos = _DYNAMIC_HEADER_WORDS + ordinal * _DYNAMIC_RECORD_WORDS
                        tick = epoch + ((row[record_pos] - epoch) & 0xFFFFFFFF)
                        site = row[record_pos + 1]
                        if site >= len(kernel["events"]):
                            raise ValueError(f"invalid flytrace site {site} at wave {index}, event {ordinal}")
                        if not previous <= tick <= end_tick:
                            raise ValueError(f"invalid timestamp at wave {index}, event {ordinal}")
                        previous = tick
                        event = kernel["events"][site]
                        payload = row[record_pos + 2]
                        if payload >= 1 << 31:
                            payload -= 1 << 32
                        record = {
                            "name": event["name"],
                            "kind": event["kind"],
                            "payload": payload if event["has_payload"] else None,
                            "site": site,
                            "tick": tick,
                            "ordinal": ordinal,
                        }
                        if "range_id" in event:
                            record["range_id"] = event["range_id"]
                        records.append(record)
                    block = tuple(row[5:8])
                    wave = row[8]
                    overflow = attempted > kernel["max_events"]
                else:
                    for ordinal, event in enumerate(kernel["events"]):
                        tick = epoch + ((row[4 + ordinal] - epoch) & 0xFFFFFFFF)
                        if not previous <= tick <= end_tick:
                            raise ValueError(f"invalid timestamp at wave {index}, event {ordinal}")
                        previous = tick
                        records.append({**event, "tick": tick, "ordinal": ordinal})
                    block_number, wave = divmod(index, _waves_per_block(kernel["block"], kernel["wave_size"]))
                    if kernel["selected"] is not None:
                        block = kernel["selected"][block_number]
                    else:
                        gx, gy, _ = kernel["grid"]
                        block = (
                            block_number % gx,
                            (block_number // gx) % gy,
                            block_number // (gx * gy),
                        )
                    attempted = len(records)
                    overflow = False
                waves.append(
                    {
                        "kernel": kernel["name"],
                        "block": block,
                        "wave": wave,
                        "epoch": epoch,
                        "end_tick": end_tick,
                        "events": records,
                        "attempted_events": attempted,
                        "overflow": overflow,
                    }
                )
                if kernel.get("hardware"):
                    hardware_pos = 9 if kernel["mode"] == "dynamic" else 4 + len(records)
                    hw_id, xcc = row[hardware_pos : hardware_pos + 2]
                    waves[-1]["hardware"] = {
                        "raw_hw_id": hw_id,
                        "xcc": xcc & 15,
                        "se": (hw_id >> 13) & 3,
                        "sh": (hw_id >> 12) & 1,
                        "cu": (hw_id >> 8) & 15,
                        "simd": (hw_id >> 4) & 3,
                        "slot": hw_id & 15,
                    }
        return waves

    def raw_metadata(self, specs):
        architectures = {spec.get("arch") for spec in specs}
        architectures.discard(None)
        if len(architectures) > 1:
            raise ValueError(f"flytrace capture mixes architectures: {sorted(architectures)}")
        arch = next(iter(architectures), current_target().arch)
        return {"backend": self.name, "arch": arch, "clock_hz": self.clock_hz}



register_trace_backend("rocm", RocmTraceBackend)
