# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ROCm trace provider for the shared compiler and capture API."""

from __future__ import annotations

import sys

from ...expr.rocdl import ktrace_emit
from ...runtime.device_runtime import rocm_ktrace_buffer as ktrace_buffer
from . import _rocm_ktrace_trace as ktrace_trace


class RocmKtraceBackend:
    def ensure_buffer(self):
        return ktrace_emit.ensure_buffer()

    def current_buffer(self):
        return ktrace_emit.current_buffer()

    def require_dispatchable_buffer(self) -> None:
        ktrace_emit.require_dispatchable_buffer()

    def buffer_config(self) -> int:
        """The record-slot bound compiled into this target's device stores."""
        buffer = self.current_buffer()
        return buffer.capacity_slots if buffer is not None else ktrace_emit._buffer_capacity_slots()

    def require_compatible_buffer(self, compiled_config: int) -> None:
        """A cached binary must use the bound of the live allocation."""
        buffer = self.ensure_buffer()
        if buffer.capacity_slots != compiled_config:
            raise RuntimeError(
                f"ktrace: this kernel was compiled to bound its stores at {compiled_config} slots but the "
                f"process holds a {buffer.capacity_slots}-slot buffer. The bound is baked into the "
                "binary and the buffer is allocated once, so re-run with a single "
                "FLYDSL_KTRACE_BUFFER_BYTES, or with FLYDSL_RUNTIME_ENABLE_CACHE=0 to rebuild "
                "against the buffer in use."
            )

    def configure_module(self, module, *, target_arch, known_names, blocks) -> None:
        """Publish the settings consumed by convert-fly-ktrace-to-rocdl."""
        from ..._mlir import ir

        i32 = ir.IntegerType.get_signless(32)
        attrs = module.operation.attributes
        attrs["fly_ktrace.capacity_slots"] = ir.IntegerAttr.get(i32, self.buffer_config())
        attrs["fly_ktrace.chip"] = ir.StringAttr.get(target_arch)
        if known_names:
            attrs["fly_ktrace.known_event_names"] = ir.DictAttr.get(
                {name: ir.IntegerAttr.get(i32, ident) for name, ident in known_names.items()}
            )
        if blocks:
            attrs["fly_ktrace.blocks"] = ir.StringAttr.get(blocks)

    def check_outside_capture(self) -> None:
        """Buffer copies and resets must not occur inside stream capture."""
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("ktrace: enter, clear and read a trace outside torch.cuda.graph()")

    def summarize(self, records, names):
        return ktrace_trace.summarize(records, names)

    def group_launches(self, records):
        return ktrace_trace.group_launches(records)

    def launch_metadata(self, identity):
        return {"queue_id": identity[0], "dispatch_id": identity[1]}

    def write_trace(self, records, names, *, kernel, out_dir):
        return ktrace_trace.write_trace(records, names, kernel=kernel, out_dir=out_dir)

    def dump_dir(self):
        return ktrace_buffer.dump_dir()


ROCM_KTRACE_BACKEND = RocmKtraceBackend()
