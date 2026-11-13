# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for the ROCm backend (RocmBackend) that do not require GPU
execution or the external LLVM toolchain."""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.backends.rocm import RocmBackend, _normalize_waves_per_eu

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
    ("value", "expected"),
    [
        (0, None),
        (2, "2"),
        ((1, 1), "1,1"),
        ((0, 3), "0,3"),
        ("1,1", "1,1"),
        ("3", "3"),
        (" 1 , 2 ", "1,2"),
    ],
)
def test_normalize_waves_per_eu_valid(value, expected):
    assert _normalize_waves_per_eu(value) == expected


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


@pytest.mark.parametrize(
    ("arch", "expect_chipset", "expect_chip"),
    [
        ("gfx1250", "gfx1250", "gfx1250"),
        ("gfx1250-strict", "gfx1250", "gfx1250-strict"),
    ],
)
def test_pipeline_strips_chipset_suffix_but_keeps_llvm_chip(arch, expect_chipset, expect_chip):
    """convert-gpu-to-rocdl needs bare gfxNNNN; the gpu.module target keeps full arch."""
    backend = RocmBackend(RocmBackend.make_target(arch))
    fragments = backend.pipeline_fragments(compile_hints={})
    joined = ",".join(fragments)
    assert f"convert-gpu-to-rocdl{{chipset={expect_chipset} " in joined
    assert f'chip = "{expect_chip}"' in backend.gpu_module_targets(compile_hints={})[0]
    # Suffix must not leak into the MLIR chipset option.
    assert "chipset=gfx1250-strict" not in joined


@flyc.kernel
def _swap_vector_rows_kernel(src: fx.Pointer, dst: fx.Pointer):
    off = fx.thread_idx.x * 4
    rows = fx.Vector((src + off).load(fx.Vector.make_type(4, fx.Float32)), (4,), fx.Float32).reshape((2, 2))
    (dst + off).store(rows[1, None].shuffle(rows[0, None], [0, 1, 2, 3]))


@flyc.jit
def _swap_vector_rows(src: fx.Pointer, dst: fx.Pointer):
    _swap_vector_rows_kernel(src, dst).launch(grid=(1, 1, 1), block=(64, 1, 1))


def test_pipeline_lowers_vector_row_indexing(monkeypatch, tmp_path):
    """Row indexing goes through an n-D vector.shape_cast, which convert-vector-to-llvm expands into
    an insert chain seeded with ub.poison; the pipeline must still lower that ub.poison."""
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    null = flyc.from_c_void_p(fx.Float32, 0)
    flyc.compile(_swap_vector_rows, null, null, compile_only=True)
