# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors


import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from flydsl.runtime.device import get_rocm_arch

pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(
        not (get_rocm_arch() or "").startswith(("gfx942", "gfx950")),
        reason="ktrace requires gfx942 or gfx950",
    ),
]


def _run_probe(tmp_path, source, *args):
    probe = tmp_path / "probe.py"
    probe.write_text(source)
    probe_env = {
        **os.environ,
        "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "cache"),
        "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
        "FLYDSL_RUNTIME_RUN_ONLY": "0",
        "FLYDSL_KTRACE_ENABLE": "1",
        "FLYDSL_KTRACE_BLOCKS": "",
        "COMPILE_ONLY": "0",
    }
    result = subprocess.run(
        [sys.executable, str(probe), *args], env=probe_env, text=True, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize("compile_only_at_dispatch", ["0", "1"])
@pytest.mark.parametrize("buffer_bytes", ["65536", "16384"])
def test_deferred_compiled_function_validates_trace_buffer(tmp_path, compile_only_at_dispatch, buffer_bytes):
    result = _run_probe(
        tmp_path,
        """
import json, os, sys
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import ktrace
from flydsl.expr.rocdl import ktrace_emit

@flyc.kernel
def kernel(x: fx.Tensor):
    ktrace.mark("deferred")

@flyc.jit
def launch(x: fx.Tensor):
    kernel(x).launch(grid=(1, 1, 1), block=(64, 1, 1))

os.environ["FLYDSL_KTRACE_BUFFER_BYTES"] = "65536"
compiled = flyc.compile(launch, torch.empty(64))
assert ktrace_emit.current_buffer() is None
os.environ["FLYDSL_KTRACE_BUFFER_BYTES"] = sys.argv[2]
os.environ["COMPILE_ONLY"] = sys.argv[1]
if sys.argv[2] == "65536":
    os.environ["FLYDSL_KTRACE_ENABLE"] = "0"
    compiled(torch.empty(64, device="cuda"))
    assert ktrace.summary()["phases"]["deferred"]["count"] == 1
else:
    try:
        compiled(torch.empty(64, device="cuda"))
    except RuntimeError as exc:
        assert "compiled to bound its stores at 2047 slots" in str(exc)
        assert "511-slot buffer" in str(exc)
    else:
        raise AssertionError("launched with an incompatible trace buffer")
print(json.dumps({"launches": ktrace_emit.current_buffer().launch_count()}))
""",
        compile_only_at_dispatch,
        buffer_bytes,
    )
    assert result["launches"] == 0


_CACHE_COMPOSITION_PROBE = """
import json, sys
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import flydsl.compiler as flyc
from flydsl.compiler import jit_function
from flydsl.expr import ktrace

@flyc.kernel
def kernel_a():
    ktrace.mark("phase_a")

@flyc.kernel
def kernel_b():
    ktrace.mark("phase_b")

@flyc.jit
def launch_a():
    kernel_a().launch(grid=(1, 1, 1), block=(64, 1, 1))

@flyc.jit
def launch_b():
    kernel_b().launch(grid=(1, 1, 1), block=(64, 1, 1))

with patch.object(jit_function.MlirCompiler, "compile", wraps=jit_function.MlirCompiler.compile) as compile:
    if sys.argv[1] == "concurrent":
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda fn: fn(), [launch_a, launch_b]))
    else:
        for name in sys.argv[1]:
            (launch_a if name == "a" else launch_b)()
    print(json.dumps({"compiles": compile.call_count, "phases": ktrace.summary()["phases"]}))
"""


def test_independently_cached_traced_kernels_compose(tmp_path):
    assert _run_probe(tmp_path, _CACHE_COMPOSITION_PROBE, "a")["compiles"] == 1
    assert _run_probe(tmp_path, _CACHE_COMPOSITION_PROBE, "b")["compiles"] == 1
    combined = _run_probe(tmp_path, _CACHE_COMPOSITION_PROBE, "ab")
    assert combined["compiles"] == 1
    assert set(combined["phases"]) == {"phase_a", "phase_b"}
    assert all(phase["count"] == 1 for phase in combined["phases"].values())
    warm = _run_probe(tmp_path, _CACHE_COMPOSITION_PROBE, "ba")
    assert warm["compiles"] == 0
    assert set(warm["phases"]) == {"phase_a", "phase_b"}


def test_concurrent_first_compilations_keep_distinct_event_names(tmp_path):
    result = _run_probe(tmp_path, _CACHE_COMPOSITION_PROBE, "concurrent")
    assert result["compiles"] == 2
    assert set(result["phases"]) == {"phase_a", "phase_b"}
    assert all(phase["count"] == 1 for phase in result["phases"].values())


def test_numeric_payloads_survive_trace_export(tmp_path):
    result = _run_probe(
        tmp_path,
        """
import json, math, sys
from flydsl.compiler.backends._rocm_ktrace_trace import build_timeline
from pathlib import Path
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import ktrace

@flyc.kernel
def kernel():
    ktrace.mark("value", fx.Float32(1.5))
    ktrace.mark("value", -2.5)
    ktrace.mark("value", fx.Int32(-1))
    ktrace.mark("value", -1)
    ktrace.mark("value", fx.Uint64(18446744073709551615))
    ktrace.mark("value", fx.Float16(0.5))
    ktrace.mark("value", fx.BFloat16(0.5))
    ktrace.mark("value", True)
    ktrace.mark("value", fx.Boolean(False))
    ktrace.mark("value", 0)
    ktrace.mark("value")
    ktrace.mark("value", float("inf"))
    ktrace.mark("value", float("-inf"))
    ktrace.mark("value", float("nan"))

@flyc.jit
def launch():
    kernel().launch(grid=(1, 1, 1), block=(64, 1, 1))

with ktrace.capture("payloads", out_dir=sys.argv[1]) as trace:
    launch()
records = trace._snapshot()
payloads = [str(p) if isinstance(p, float) and not math.isfinite(p) else p for p in (r["payload"] for r in records)]
print(json.dumps({
    "path": trace.export(),
    "count": trace.summary()["phases"]["value"]["count"],
    "payloads": payloads,
    "payload_types": [r["payload_type"] for r in records],
    "instants": [i["payload"] for i in build_timeline(records, trace._names)["instants"]],
}))
""",
        str(tmp_path),
    )
    assert result["count"] == 14
    expected = [1.5, -2.5, -1, -1, (1 << 64) - 1, 0.5, 0.5, True, False, 0, None, "inf", "-inf", "nan"]
    assert result["payloads"] == expected
    assert result["payload_types"] == [
        "float32",
        "float64",
        "int64",
        "int64",
        "uint64",
        "float32",
        "float32",
        "bool",
        "bool",
        "int64",
        None,
        "float64",
        "float64",
        "float64",
    ]
    assert Path(result["path"]).stat().st_size > 0
    assert result["instants"][:11] == expected[:11]
    assert len(result["instants"]) == len(expected)


_OVERFLOW_PROBE = """
import json, os, re, sys
os.environ["FLYDSL_KTRACE_BUFFER_BYTES"] = sys.argv[1]
import flydsl.compiler as flyc
from flydsl.expr import ktrace

@flyc.kernel
def kernel():
    ktrace.mark("w")

@flyc.jit
def launch():
    kernel().launch(grid=(64, 1, 1), block=(256, 1, 1))

try:
    with ktrace.capture("grid") as trace:
        launch()
    print(json.dumps({"waves": len({(r["block"], r["wave_id"]) for r in trace._snapshot()})}))
except RuntimeError as exc:
    print(json.dumps({"error": str(exc)}))
"""


def test_buffer_overflow_suggests_a_size_that_records_the_grid(tmp_path):
    overflowed = _run_probe(tmp_path, _OVERFLOW_PROBE, str(32 << 10))
    assert "trace overflowed" in overflowed.get("error", ""), overflowed
    suggested = re.search(r"FLYDSL_KTRACE_BUFFER_BYTES>=(\d+)", overflowed["error"]).group(1)
    assert _run_probe(tmp_path, _OVERFLOW_PROBE, suggested) == {"waves": 256}
