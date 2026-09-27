# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from flydsl.compiler.kernel_function import CompilationContext
from kernels.common import tensor_shim


def test_run_compiled_keeps_ambient_hint_variants_separate(monkeypatch):
    class Launcher:
        pass

    launcher = Launcher()
    calls = []
    compiled = []

    def fake_compile(exe, *args):
        assert exe is launcher
        hint_snapshot = dict(CompilationContext.get_compile_hints())
        calls.append(("compile", hint_snapshot, args))

        def compiled_function(*run_args):
            calls.append(("dispatch", hint_snapshot, run_args))

        compiled.append(compiled_function)
        return compiled_function

    monkeypatch.setattr(tensor_shim.flyc, "compile", fake_compile)

    tensor_shim._run_compiled(launcher, "normal-cold")
    normal = launcher._cf
    tensor_shim._run_compiled(launcher, "normal-hot")

    trace_options = ("flytrace-v3", "dynamic", None, (), 32, 4, False)
    with CompilationContext.compile_hints({"flytrace": trace_options}):
        tensor_shim._run_compiled(launcher, "trace-cold")
        traced = next(iter(launcher._cf_hint_variants.values()))
        tensor_shim._run_compiled(launcher, "trace-hot")

    assert launcher._cf is normal
    assert traced is not normal
    assert len(launcher._cf_hint_variants) == 1

    tensor_shim._run_compiled(launcher, "normal-again")
    assert [call[0] for call in calls] == ["compile", "dispatch", "compile", "dispatch", "dispatch"]
    assert calls[1][2] == ("normal-hot",)
    assert calls[3][2] == ("trace-hot",)
    assert calls[4][2] == ("normal-again",)
    assert len(compiled) == 2
