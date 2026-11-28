# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for the ROCm backend (RocmBackend) that do not require GPU
execution or the external LLVM toolchain."""

import re

import numpy as np
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
          rocdl.waves_per_eu = 1 : i32,
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
    ("arch", "hint", "expected"),
    [
        (
            "gfx942",
            0,
            {"wpe2": True, "wpe_range": True, "wpe1": False, "unset": False, "explicit": False}
            | {"block512": True, "fwg512": True, "block256_wpe1": False},
        ),
        (
            "gfx950",
            0,
            {"wpe2": True, "wpe_range": True, "wpe1": False, "unset": False, "explicit": False}
            | {"block512": True, "fwg512": True, "block256_wpe1": False},
        ),
        (
            "gfx950",
            2,
            {"wpe2": True, "wpe_range": True, "wpe1": True, "unset": True, "explicit": False}
            | {"block512": True, "fwg512": True, "block256_wpe1": True},
        ),
        (
            "gfx1250",
            2,
            {"wpe2": False, "wpe_range": False, "wpe1": False, "unset": False, "explicit": False}
            | {"block512": False, "fwg512": False, "block256_wpe1": False},
        ),
    ],
)
def test_rocm_lower_disables_agprs_at_two_or_more_waves_per_eu(arch, hint, expected):
    backend = RocmBackend(RocmBackend.make_target(arch))
    src = r"""module {
      gpu.module @m {
        gpu.func @wpe2() kernel attributes {rocdl.waves_per_eu = 2 : i32} { gpu.return }
        gpu.func @wpe_range() kernel attributes {rocdl.waves_per_eu = "2,4"} { gpu.return }
        gpu.func @wpe1() kernel attributes {rocdl.waves_per_eu = 1 : i32} { gpu.return }
        gpu.func @unset() kernel { gpu.return }
        gpu.func @block512() kernel attributes {known_block_size = array<i32: 512, 1, 1>} { gpu.return }
        gpu.func @fwg512() kernel attributes {rocdl.flat_work_group_size = "512,512"} { gpu.return }
        gpu.func @block256_wpe1() kernel attributes {
          known_block_size = array<i32: 256, 1, 1>,
          rocdl.waves_per_eu = 1 : i32
        } { gpu.return }
        gpu.func @explicit() kernel attributes {
          rocdl.waves_per_eu = 2 : i32,
          llvm.passthrough = [["amdgpu-agpr-alloc", "64"]]
        } { gpu.return }
        gpu.func @helper() attributes {rocdl.waves_per_eu = 2 : i32} { gpu.return }
      }
    }"""

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        ctx.load_all_available_dialects()
        module = ir.Module.parse(src)
        backend.lower_compile_hints(module, compile_hints={"waves_per_eu": hint})
        funcs = {
            ir.StringAttr(op.attributes["sym_name"]).value: str(op)
            for op in module.body.operations[0].regions[0].blocks[0].operations
        }

    for name, added in expected.items():
        assert ('["amdgpu-agpr-alloc", "0"]' in funcs[name]) == added, name
    assert '["amdgpu-agpr-alloc", "64"]' in funcs["explicit"]
    assert "amdgpu-agpr-alloc" not in funcs["helper"]


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
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx942")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    null = flyc.from_c_void_p(fx.Float32, 0)
    flyc.compile(_swap_vector_rows, null, null, compile_only=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_resource_limit_check_runs_only_with_its_hint(enabled):
    backend = RocmBackend(RocmBackend.make_target("gfx942"))
    hints = {"check_resource_limits": True} if enabled else {}
    fragments = backend.pipeline_fragments(compile_hints=hints)
    assert ("gpu.module(fly-check-resource-limits)" in fragments) == enabled


# ──────────────────────────────────────────────────────────────
# max_flat_work_group_size
# ──────────────────────────────────────────────────────────────


def test_rocm_lower_max_flat_work_group_size_only_on_kernels_without_a_fixed_size():
    backend = RocmBackend(RocmBackend.make_target("gfx942"))
    src = r"""module {
      gpu.module @m {
        gpu.func @open() kernel { gpu.return }
        gpu.func @known() kernel attributes { known_block_size = array<i32: 64, 1, 1> } { gpu.return }
        gpu.func @own() kernel attributes { rocdl.flat_work_group_size = "1,64" } { gpu.return }
        gpu.func @maxflat() kernel attributes { rocdl.max_flat_work_group_size = 64 : i32 } { gpu.return }
        gpu.func @reqd() kernel attributes { rocdl.reqd_work_group_size = array<i32: 64, 1, 1> } { gpu.return }
        gpu.func @helper() { gpu.return }
      }
    }"""

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        ctx.load_all_available_dialects()
        module = ir.Module.parse(src)
        baseline_asm = str(module)
        backend.lower_compile_hints(module, compile_hints={})
        assert str(module) == baseline_asm

        backend.lower_compile_hints(module, compile_hints={"max_flat_work_group_size": 512, "waves_per_eu": 2})
        funcs = {
            ir.StringAttr(op.attributes["sym_name"]).value: str(op)
            for op in module.body.operations[0].regions[0].blocks[0].operations
            if op.operation.name == "gpu.func"
        }

    assert 'rocdl.flat_work_group_size = "1,512"' in funcs["open"]
    assert 'rocdl.flat_work_group_size = "1,64"' in funcs["own"]
    for name in ("known", "maxflat", "reqd", "helper"):
        assert "rocdl.flat_work_group_size" not in funcs[name], name
    # waves_per_eu still reaches every kernel, whatever its size.
    for name in ("open", "known", "own", "maxflat", "reqd"):
        assert 'rocdl.waves_per_eu = "2"' in funcs[name], name


@pytest.mark.parametrize(
    ("arch", "hint", "agprs_disabled"),
    [
        ("gfx942", 512, True),
        ("gfx942", 1024, True),
        ("gfx950", 512, True),
        ("gfx942", 256, False),
        ("gfx1250", 1024, False),
    ],
)
def test_rocm_lower_max_flat_work_group_size_above_256_disables_agprs(arch, hint, agprs_disabled):
    """A maximum above 256 work-items forces two or more waves per EU, like any workgroup of that size."""
    backend = RocmBackend(RocmBackend.make_target(arch))
    src = r"""module {
      gpu.module @m {
        gpu.func @open() kernel { gpu.return }
        gpu.func @own() kernel attributes { rocdl.flat_work_group_size = "1,64" } { gpu.return }
        gpu.func @explicit() kernel attributes { llvm.passthrough = [["amdgpu-agpr-alloc", "64"]] } { gpu.return }
      }
    }"""

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        ctx.load_all_available_dialects()
        module = ir.Module.parse(src)
        backend.lower_compile_hints(module, compile_hints={"max_flat_work_group_size": hint})
        funcs = {
            ir.StringAttr(op.attributes["sym_name"]).value: str(op)
            for op in module.body.operations[0].regions[0].blocks[0].operations
        }

    assert f'rocdl.flat_work_group_size = "1,{hint}"' in funcs["open"]
    assert ('["amdgpu-agpr-alloc", "0"]' in funcs["open"]) == agprs_disabled
    # A kernel that keeps its own smaller size, or its own AGPR split, is left as it is.
    assert "amdgpu-agpr-alloc" not in funcs["own"]
    assert '["amdgpu-agpr-alloc", "64"]' in funcs["explicit"] and '"amdgpu-agpr-alloc", "0"' not in funcs["explicit"]


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (True, TypeError),
        (np.True_, TypeError),
        ("1024", TypeError),
        (512.0, TypeError),
        (0, ValueError),
        (1025, ValueError),
    ],
)
def test_rocm_lower_max_flat_work_group_size_rejects_invalid_values(value, error):
    backend = RocmBackend(RocmBackend.make_target("gfx942"))

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.create()
        with pytest.raises(error, match="max_flat_work_group_size"):
            backend.lower_compile_hints(module, compile_hints={"max_flat_work_group_size": value})


_OPEN_KERNEL_MODULE = "module { gpu.module @m { gpu.func @open() kernel { gpu.return } } }"


def test_rocm_lower_max_flat_work_group_size_accepts_integral_types():
    class Integral:  # stands in for numpy.int64 and other types with __index__
        def __index__(self):
            return 512

    backend = RocmBackend(RocmBackend.make_target("gfx942"))

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        ctx.load_all_available_dialects()
        module = ir.Module.parse(_OPEN_KERNEL_MODULE)
        backend.lower_compile_hints(module, compile_hints={"max_flat_work_group_size": Integral()})
        assert 'rocdl.flat_work_group_size = "1,512"' in str(module)


def test_rocm_lower_compile_hints_rejects_maxnreg_before_setting_a_size():
    backend = RocmBackend(RocmBackend.make_target("gfx942"))

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        ctx.load_all_available_dialects()
        module = ir.Module.parse(_OPEN_KERNEL_MODULE)
        baseline_asm = str(module)
        with pytest.raises(ValueError, match="maxnreg is not supported"):
            backend.lower_compile_hints(module, compile_hints={"maxnreg": 64, "max_flat_work_group_size": 512})
        assert str(module) == baseline_asm


@flyc.kernel
def _empty_kernel():
    pass


@flyc.jit
def _runtime_block_launch(threads: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _empty_kernel().launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)


@flyc.jit
def _own_size_launch(threads: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    kernel = _empty_kernel(value_attrs={"rocdl.flat_work_group_size": "1,64"})
    kernel.launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)


def _max_flat_workgroup_sizes(compiled):
    """The max flat work-group size in the metadata note of every code object."""
    sizes = []
    for binary in compiled.binaries().values():
        # msgpack: the key, then the value as a uint16 (0xcd), a uint8 (0xcc) or a positive fixint.
        for match in re.finditer(rb"\.max_flat_workgroup_size(?:\xcd(..)|\xcc(.)|([\x00-\x7f]))", binary, re.S):
            sizes.append(int.from_bytes(next(g for g in match.groups() if g is not None), "big"))
    return sizes


@pytest.mark.parametrize(
    ("hints", "launch", "expected"),
    [
        ({}, _runtime_block_launch, 256),  # the ROCDL translation's default for a kernel without a size
        ({"max_flat_work_group_size": 1024}, _runtime_block_launch, 1024),
        ({"max_flat_work_group_size": 512}, _runtime_block_launch, 512),
        ({"max_flat_work_group_size": 128}, _runtime_block_launch, 128),
        ({"max_flat_work_group_size": 1024}, _own_size_launch, 64),  # the kernel's own size wins
    ],
)
def test_max_flat_work_group_size_reaches_the_code_object(monkeypatch, tmp_path, hints, launch, expected):
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx942")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    # flyc.compile[hints] stores its hints on the launcher; restore them after the test.
    monkeypatch.setattr(launch, "compile_hints", {})
    compiled = flyc.compile[hints](launch, 64, compile_only=True)
    assert _max_flat_workgroup_sizes(compiled) == [expected]
