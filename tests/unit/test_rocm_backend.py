# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for the ROCm backend (RocmBackend) that do not require GPU
execution or the external LLVM toolchain."""

import pytest

from flydsl._mlir import ir
from flydsl.compiler.backends.rocm import RocmBackend

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


# ──────────────────────────────────────────────────────────────
# waves_per_eu validation
# ──────────────────────────────────────────────────────────────


def test_rocm_lower_wpe_preserves_source_default_and_overrides_kernel_entries():
    backend = RocmBackend(RocmBackend.make_target("gfx942"))
    src = r"""module {
      gpu.module @m {
        gpu.func @a() kernel attributes {
          rocdl.waves_per_eu = 3 : i32,
          passthrough = [["keep", "yes"]]
        } { gpu.return }
        gpu.func @b() kernel attributes {
          passthrough = [["amdgpu-waves-per-eu", "1,1"]]
        } { gpu.return }
        gpu.func @helper() { gpu.return }
      }
    }"""

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        ctx.load_all_available_dialects()
        baseline = ir.Module.parse(src)
        baseline_asm = str(baseline)
        backend.lower_compile_hints(baseline, compile_hints={"waves_per_eu": 0})
        assert str(baseline) == baseline_asm

        module = ir.Module.parse(src)
        backend.lower_compile_hints(module, compile_hints={"waves_per_eu": 2})
        funcs = {
            ir.StringAttr(op.attributes["sym_name"]).value: str(op)
            for op in module.body.operations[0].regions[0].blocks[0].operations
            if op.operation.name == "gpu.func"
        }

    for name in ("a", "b"):
        assert 'rocdl.waves_per_eu = "2"' in funcs[name]
    assert '"keep", "yes"' in funcs["a"]
    assert "rocdl.waves_per_eu" not in funcs["helper"]


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (-1, ValueError),
        ((1, 2, 3), TypeError),
        ((2, 1), ValueError),
        ((-1, 1), ValueError),
        ("2,1", ValueError),
        ("abc", ValueError),
        ("-1", ValueError),
        ("1,2,3", ValueError),
    ],
)
def test_rocm_lower_wpe_rejects_invalid_values(value, error):
    backend = RocmBackend(RocmBackend.make_target("gfx942"))

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.create()
        with pytest.raises(error, match="waves_per_eu"):
            backend.lower_compile_hints(module, compile_hints={"waves_per_eu": value})
