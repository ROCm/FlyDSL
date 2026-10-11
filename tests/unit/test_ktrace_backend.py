# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A non-ROCm trace provider can use the shared JIT and capture paths."""

import pytest

from flydsl._mlir import ir
from flydsl.compiler import backends, jit_function
from flydsl.compiler.backends.base import BaseBackend, GPUTarget
from flydsl.expr.experimental import ktrace

pytestmark = pytest.mark.l1a_compile_no_target_dialect


def test_target_provider_owns_trace_config_buffer_and_records(monkeypatch):
    class Buffer:
        device_ptr = 0x1234
        capacity_slots = 31

        def __init__(self):
            self.records = []

        def synchronize(self):
            pass

        def launch_count(self):
            return 0

        def read_records(self):
            return list(self.records)

        def reset(self):
            self.records.clear()

    class Provider:
        def __init__(self):
            self.buffer = Buffer()
            self.config = None
            self.checked_config = None

        def ensure_buffer(self):
            return self.buffer

        def current_buffer(self):
            return self.buffer

        def require_dispatchable_buffer(self):
            pass

        def buffer_config(self):
            return ("sim-buffer", self.buffer.capacity_slots)

        def require_compatible_buffer(self, compiled_config):
            self.checked_config = compiled_config

        def configure_module(self, module, **settings):
            self.config = settings

        def check_outside_capture(self):
            pass

        def summarize(self, records, names):
            return {"records": len(records), "phases": {}}

        def group_launches(self, records):
            return [(7, records)] if records else []

        def launch_metadata(self, identity):
            return {"device_launch": identity}

        def write_trace(self, records, names, *, kernel, out_dir):
            return f"{out_dir}/{kernel}.json"

        def dump_dir(self):
            return "/tmp"

    provider = Provider()

    class FakeBackend(BaseBackend):
        @staticmethod
        def supports_target(target):
            return target.backend == "test_trace"

        @staticmethod
        def detect_target():
            return GPUTarget("test_trace", "sim", 32)

        def pipeline_fragments(self, *, compile_hints):
            return []

        def gpu_module_targets(self):
            return []

        def native_lib_patterns(self):
            return []

        def jit_runtime_lib_basenames(self):
            return []

        @classmethod
        def ktrace_backend(cls):
            return provider

    backends.register_backend("test_trace", FakeBackend)
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "test_trace")
    monkeypatch.setenv("FLYDSL_KTRACE_ENABLE", "1")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    try:
        args = []
        assert jit_function._ensure_ktrace_buffer_arg(args)
        assert args[0].pointer.value == provider.buffer.device_ptr

        with ir.Context() as context, ir.Location.unknown(context):
            module = ir.Module.create()
            jit_function._attach_ktrace_config(module, True)
        assert provider.config["target_arch"] == "sim"
        assert "capacity_slots" not in provider.config
        assert provider.buffer_config() == ("sim-buffer", 31)

        artifact = type("Artifact", (), {"_ktrace_traced": True, "_ktrace_buffer_config": ("sim-buffer", 31)})()
        ktrace.require_compatible_buffer(artifact)
        assert provider.checked_config == ("sim-buffer", 31)
        stale_artifact = type("Artifact", (), {"_ktrace_traced": True})()
        with pytest.raises(RuntimeError, match="no buffer settings"):
            ktrace.require_compatible_buffer(stale_artifact)

        with ktrace.capture("model") as trace:
            provider.buffer.records.append({"event": "recorded"})

        assert trace.summary()["records"] == 1
        assert trace.launches() == [{"launch_id": 1, "device_launch": 7, "records": 1, "phases": {}}]
        assert trace.export() == "/tmp/model.json"
        assert trace.export_launches(names=["layer_0"]) == ["/tmp/layer_0.json"]
    finally:
        backends._registry.pop("test_trace")
