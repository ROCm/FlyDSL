# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""IR-addressed artifact reuse across traces, processes and configurations."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from array import array
from dataclasses import dataclass

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function as jf
from flydsl.compiler.backends import get_backend
from flydsl.compiler.jit_argument import JitArgumentRegistry
from flydsl.compiler.kernel_function import CompilationContext

pytestmark = pytest.mark.l1a_compile_no_target_dialect


@pytest.fixture
def frontend(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLYDSL_KTRACE_ENABLE", "0")
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "0")
    monkeypatch.setenv("COMPILE_ONLY", "0")
    monkeypatch.setenv("FLYDSL_COMPILE_LLVM_DIR", "")
    monkeypatch.setattr(jf, "_flydsl_key_cached", lambda *args: "test-toolchain")
    calls = []

    def compile_noop(cls, module, **kwargs):
        calls.append(kwargs.get("config"))
        return module

    monkeypatch.setattr(jf.MlirCompiler, "compile", classmethod(compile_noop))
    return calls


def _compile(launch, *args):
    launch._compile_only(*args)
    return launch._last_compiled


def _load(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    spec = importlib.util.spec_from_file_location("cache_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_launches(monkeypatch):
    launched = []
    monkeypatch.setattr(jf, "_build_call_state", lambda sig, args, artifact: lambda args: launched.append(artifact))
    return launched


@dataclass
class Traits:
    amount: int


def _factory(traits):
    @flyc.kernel
    def body(X: fx.Pointer):
        X[0] = X[0] + traits.amount

    @flyc.jit
    def launch(X: fx.Pointer):
        body(X).launch(grid=(1, 1, 1), block=(64, 1, 1))

    return launch


@pytest.mark.parametrize("disk_cache", [False, True])
def test_captured_traits_retrace_and_reuse_equal_ir(frontend, monkeypatch, disk_cache):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", str(int(disk_cache)))
    ptr = flyc.from_c_void_p(fx.Int32, 0)
    traits = Traits(1)
    launch = _factory(traits)
    first_key, first = _compile(launch, ptr)
    traits.amount = 7
    second_key, second = _compile(launch, ptr)
    assert first_key != second_key
    assert "arith.constant 1" in first.source_ir
    assert "arith.constant 7" in second.source_ir
    traits.amount = 1
    assert _compile(launch, ptr) == (first_key, first)
    assert len(frontend) == 2


@pytest.mark.parametrize("nested", [False, True])
def test_helper_rebinding_retraces(frontend, monkeypatch, tmp_path, nested):
    body = "return helper(x)" if not nested else "def inner(y):\n        return helper(y)\n    return inner(x)"
    module = _load(
        tmp_path / "helpers.py",
        f"""
import flydsl.compiler as flyc
import flydsl.expr as fx
def helper(x):
    return x + 1
def replacement(x):
    return x + 7
@flyc.jit
def launch(x: fx.Int32):
    {body}
""",
    )
    executed = _record_launches(monkeypatch)
    module.launch(3)
    first = executed[-1]
    module.helper = module.replacement
    module.launch(3)
    assert executed[-1] is not first
    assert "arith.constant 7" in executed[-1].source_ir
    module.launch(9)
    assert len(frontend) == 2


@pytest.mark.parametrize("expression", ["VALUE", "CONFIG['amount']"])
def test_live_globals_recompile(frontend, tmp_path, expression):
    module = _load(
        tmp_path / "globals.py",
        f"""
import flydsl.compiler as flyc
import flydsl.expr as fx
VALUE = 1
CONFIG = {{'amount': 1}}
@flyc.jit
def launch(x: fx.Int32):
    y = x + {expression}
""",
    )
    first, _ = _compile(module.launch, 3)
    module.VALUE = 7
    module.CONFIG["amount"] = 7
    second, artifact = _compile(module.launch, 3)
    assert first != second
    assert "arith.constant 7" in artifact.source_ir


def test_receivers_and_mutation(frontend):
    class Program:
        def __init__(self, amount):
            self.amount = amount

        @flyc.jit
        def run(self, x: fx.Int32):
            _ = x + self.amount

    a, b = Program(1), Program(7)
    first, _ = _compile(Program.run, a, 3)
    second, _ = _compile(Program.run, b, 3)
    assert first != second
    a.amount = 7
    assert _compile(Program.run, a, 3)[0] == second
    assert len(frontend) == 2


def test_host_launch_configuration_is_part_of_artifact(frontend):
    blocks = 1

    @flyc.kernel
    def kernel():
        pass

    @flyc.jit
    def launch():
        kernel().launch(grid=(blocks, 1, 1), block=(64, 1, 1))

    first, _ = _compile(launch)
    blocks = 2
    second, _ = _compile(launch)
    assert first != second


@pytest.mark.parametrize("dtype", [fx.Float16, fx.BFloat16])
def test_captured_dtype_is_in_ir(frontend, dtype):
    def factory(kind):
        @flyc.jit
        def launch():
            _ = kind(1.0).ir_value()

        return launch

    baseline, _ = _compile(factory(fx.Float32))
    other, _ = _compile(factory(dtype))
    assert baseline != other


def test_resource_constant_contents_are_not_elided(frontend):
    values = array("i", range(256))

    @flyc.jit
    def launch():
        tensor = jf.ir.RankedTensorType.get((256,), jf.ir.IntegerType.get_signless(32))
        data = jf.ir.DenseResourceElementsAttr.get_from_buffer(values, "weights", tensor)
        jf.ir.Operation.create("arith.constant", results=[tensor], attributes={"value": data})

    first, _ = _compile(launch)
    values[-1] = 4096
    second, _ = _compile(launch)
    assert first != second
    values[-1] = 255
    assert _compile(launch)[0] == first
    assert len(frontend) == 2


class _FakeCudaStream:
    cuda_stream = 1234


JitArgumentRegistry.register(_FakeCudaStream)(fx.Stream)


def test_runtime_values_stay_out_of_the_ir(frontend, tmp_path):
    @flyc.jit
    def stream_launch(stream: fx.Stream = fx.Stream(None)):
        pass

    # CPU AOT can use raw 0 while GPU runtime passes a stream object.
    streams = ((), (0,), (fx.Stream(0),), (_FakeCudaStream(),))
    assert len({_compile(stream_launch, *stream)[0] for stream in streams}) == 1

    # `from __future__ import annotations` stringifies fx.Int32; it must still bind a runtime value.
    module = _load(
        tmp_path / "future.py",
        """
from __future__ import annotations
import flydsl.compiler as flyc
import flydsl.expr as fx
@flyc.jit
def launch(n: fx.Int32):
    _ = n + 1
""",
    )
    assert _compile(module.launch, 1)[0] == _compile(module.launch, 2)[0]


def test_constexpr_values_split_artifacts_only_through_the_ir(frontend):
    @flyc.jit
    def used(x: fx.Int32, value: fx.Constexpr[int]):
        _ = x + value

    @flyc.jit
    def unused(value: fx.Constexpr[int]):
        pass

    assert _compile(used, 3, 1)[0] != _compile(used, 3, 2)[0]
    assert _compile(unused, 1)[0] == _compile(unused, 2)[0]


@pytest.mark.parametrize("debug", [False, True])
def test_disk_key_location_policy(frontend, monkeypatch, tmp_path, debug):
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", str(int(debug)))
    source = """
import flydsl.compiler as flyc
import flydsl.expr as fx
@flyc.jit
def launch(x: fx.Int32):
    y = x + 3
"""
    a = _load(tmp_path / "a" / "launch.py", source)
    b = _load(tmp_path / "b" / "launch.py", "\n\n" + source)
    first, _ = _compile(a.launch, 1)
    second, _ = _compile(b.launch, 8)
    assert (first == second) is not debug
    assert len(frontend) == (2 if debug else 1)
    assert a.launch.cache_manager.cache_dir == tmp_path / "cache" / "launch"


def test_lowering_inputs_outside_the_ir_split_artifacts(frontend, monkeypatch):
    monkeypatch.delenv("LLVM_GFX1250_ANY_XNACK_CODEGEN", raising=False)

    @flyc.jit
    def launch(x: fx.Int32):
        _ = x + 1

    keys = [_compile(launch, 3)[0]]
    with CompilationContext.compile_hints({"llvm_options": {"enable-post-misched": False}}):
        keys.append(_compile(launch, 3)[0])
    # Patch the class: undoing an instance patch would leave a bound method on the cached
    # backend instance and shadow later class-level patches.
    backend_type = type(get_backend())
    original = backend_type.pipeline_fragments
    monkeypatch.setattr(backend_type, "pipeline_fragments", lambda self, **kw: ["cse", *original(self, **kw)])
    keys.append(_compile(launch, 3)[0])
    monkeypatch.setattr(backend_type, "hash", lambda self: "updated-backend-toolchain")
    keys.append(_compile(launch, 3)[0])
    monkeypatch.setenv("ARCH", "gfx1250")
    keys.append(_compile(launch, 3)[0])
    for xnack in ("0", "1"):
        monkeypatch.setenv("LLVM_GFX1250_ANY_XNACK_CODEGEN", xnack)
        keys.append(_compile(launch, 3)[0])
    assert len(set(keys)) == len(keys) == len(frontend)

    # The loaded toolchain cannot change in-process; it namespaces disk entries.
    disk = jf._disk_cache_key(keys[0])
    monkeypatch.setattr(jf, "_flydsl_key", lambda: "other-toolchain")
    assert jf._disk_cache_key(keys[0]) != disk


def test_memory_only_cache_never_fingerprints_the_toolchain(frontend, monkeypatch):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    def unexpected():
        raise AssertionError("hashing the toolchain is only needed for disk entries")

    monkeypatch.setattr(jf, "_flydsl_key", unexpected)

    @flyc.jit
    def launch(x: fx.Int32):
        _ = x + 1

    first, _ = _compile(launch, 3)
    assert _compile(launch, 3)[0] == first
    assert len(frontend) == 1


def test_waves_per_eu_hint_reaches_the_ir(frontend):
    launch = _factory(Traits(1))
    pointer = flyc.from_c_void_p(fx.Int32, 0)

    def key(waves):
        with CompilationContext.compile_hints({"waves_per_eu": waves}):
            return _compile(launch, pointer)[0]

    launch.compile_hints = {"waves_per_eu": 4}
    persistent = _compile(launch, pointer)[0]
    with CompilationContext.compile_hints({"waves_per_eu": 1}):
        outer = _compile(launch, pointer)[0]
        inner = key(2)  # The innermost thread-local overlay wins.
        assert launch._effective_compile_hints()["waves_per_eu"] == 1
    assert launch._effective_compile_hints()["waves_per_eu"] == 4
    assert key("2") == inner  # Equal codegen reuses one artifact.
    assert len({persistent, outer, inner, key((2, 2)), key((1, 2))}) == 5


def test_run_only_traces_every_call_and_never_lowers(frontend, monkeypatch):
    launched = _record_launches(monkeypatch)

    @flyc.jit
    def launch(x: fx.Int32, scale: fx.Constexpr[int]):
        _ = x * scale

    launch(1, 2)
    launch(1, 3)
    launch._mem_cache.clear()
    launch.cache_manager.memory_cache.clear()
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")  # Run-only still reads the disk store.
    traced = []
    trace = launch._trace
    monkeypatch.setattr(launch, "_trace", lambda *a, **k: traced.append(1) or trace(*a, **k))
    for x in range(4):
        launch(x, 2)
    launch(5, 3)
    launch(6, 3)
    assert len(traced) == 6  # Run-only calls trace like ordinary calls.
    assert len(frontend) == 2  # Only the normal runs lowered.
    assert launched[-1] is launched[-2] is not launched[-3]
    with pytest.raises(RuntimeError, match="RUN_ONLY=1.*no usable AOT cache"):
        launch(7, 4)


def test_extern_linked_artifacts_stay_off_disk(frontend, monkeypatch):
    def initializer(module):
        pass

    @flyc.jit
    def launch():
        CompilationContext.get_current().post_load_processors.append(initializer)

    first = _compile(launch)
    assert _compile(launch) == first  # Reused in-process.
    assert not list(launch.cache_manager.cache_dir.glob("*.pkl"))
    launch._mem_cache.clear()
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "1")
    with pytest.raises(RuntimeError, match="never cached on disk"):
        _compile(launch)


_PROCESS_PROBE = r"""
import json, sys
from pathlib import Path
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function as jf

helper = Path(__file__).parent / 'common' / 'util.py'
helper.parent.mkdir(exist_ok=True)
helper.write_text('def scale(x):\n    return x + ' + sys.argv[1] + '\n')
from common.util import scale

@flyc.kernel
def kernel(X: fx.Pointer):
    def inner(x):
        return scale(x)
    if fx.thread_idx.x == 0:
        X[0] = inner(X[0])

@flyc.jit
def launch(X: fx.Pointer):
    kernel(X).launch(grid=(1,1,1), block=(64,1,1))

compiled = []
original = jf.MlirCompiler.compile
gpu = sys.argv[2] == 'gpu'
def compile_spy(cls, module, **kw):
    compiled.append(1)
    return original(module, **kw) if gpu else module
jf.MlirCompiler.compile = classmethod(compile_spy)
if not gpu:
    jf._flydsl_key = lambda: 'subprocess-toolchain'
    launch._compile_only(flyc.from_c_void_p(fx.Int32, 0))
    value = None
else:
    import torch
    X = torch.zeros(1, device='cuda', dtype=torch.int32)
    launch(flyc.from_c_void_p(fx.Int32, X.data_ptr()))
    torch.cuda.synchronize()
    value = X.item()
print(json.dumps({'key': launch._last_compiled[0], 'compiles': len(compiled), 'value': value}))
"""


@pytest.mark.parametrize("gpu", [False, pytest.param(True, marks=pytest.mark.l2_device)])
def test_shared_disk_cache_observes_cross_directory_nested_helper(tmp_path, gpu):
    if gpu:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("requires AMDGPU")
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent(_PROCESS_PROBE))
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
        "FLYDSL_RUNTIME_RUN_ONLY": "0",
        "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "cache"),
        "FLYDSL_KTRACE_ENABLE": "0",
        "COMPILE_ONLY": "0",
        "FLYDSL_DUMP_IR": "0",
        "FLYDSL_COMPILE_LLVM_DIR": "",
    }
    if not gpu:
        env["ARCH"] = "gfx942"  # Compile-only; a GPU run targets its own device.
    results = []
    for value in (1, 7, 7):
        child = subprocess.run(
            [sys.executable, str(script), str(value), "gpu" if gpu else "cpu"],
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        results.append(json.loads(child.stdout.splitlines()[-1]))
    assert [r["compiles"] for r in results] == [1, 1, 0]
    assert results[0]["key"] != results[1]["key"] == results[2]["key"]
    if gpu:
        assert [r["value"] for r in results] == [1, 7, 7]


@pytest.mark.l2_device
def test_gpu_retrace_and_explicit_compiled_snapshot(monkeypatch, tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires AMDGPU")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_RUN_ONLY", "0")
    monkeypatch.setenv("FLYDSL_KTRACE_ENABLE", "0")
    monkeypatch.setenv("COMPILE_ONLY", "0")
    traits = Traits(1)

    @flyc.kernel
    def kernel(X: fx.Pointer):
        if fx.thread_idx.x == 0:
            X[0] = X[0] + traits.amount

    @flyc.jit
    def launch(X: fx.Pointer):
        kernel(X).launch(grid=(1, 1, 1), block=(64, 1, 1))

    X = torch.zeros(1, device="cuda", dtype=torch.int32)
    ptr = flyc.from_c_void_p(fx.Int32, X.data_ptr())
    launch(ptr)
    traits.amount = 7
    launch(ptr)
    fixed = flyc.compile(launch, ptr)
    traits.amount = 11
    fixed(ptr)  # Still adds 7, without tracing.
    flyc.compile(launch, ptr)  # New compile request observes 11.
    torch.cuda.synchronize()
    assert X.item() == 33

    def unexpected(*args, **kwargs):
        raise AssertionError("an unchanged IR cache hit must not lower again")

    monkeypatch.setattr(jf.MlirCompiler, "compile", unexpected)
    launch(ptr)
    torch.cuda.synchronize()
    assert X.item() == 44
    monkeypatch.setattr(launch, "_trace", unexpected)
    fixed(ptr)
    torch.cuda.synchronize()
    assert X.item() == 51


def test_autotune_times_jit_candidates_through_compiled_handles(frontend, monkeypatch, tmp_path):
    from flydsl.autotune import Autotuner, Config

    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "autotune"))
    launched = _record_launches(monkeypatch)
    traced = []
    trace = jf.JitFunction._trace
    monkeypatch.setattr(jf.JitFunction, "_trace", lambda self, *a, **k: traced.append(1) or trace(self, *a, **k))

    @flyc.jit
    def launch(x: fx.Int32, scale: fx.Constexpr[int]):
        _ = x * scale

    def bench(call, warmup, rep):
        for _ in range(warmup + rep):
            call()
        return 1.0

    configs = [Config(scale=2), Config(scale=3)]
    Autotuner(launch, configs, key=["x"], warmup=2, rep=3, do_bench_fn=bench).resolve_config(5)
    assert len(traced) == len(frontend) == 2  # One trace and compile per config.
    assert len(launched) == 10 and len({id(artifact) for artifact in launched}) == 2
