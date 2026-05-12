# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import json
from pathlib import Path

from flydsl._mlir import ir
from flydsl.compiler.backends.rocm import RocmBackend
from flydsl.compiler.external_llvm import (
    _format_llvm_cli_options,
    external_llvm_fingerprint,
    run_external_binary_codegen,
)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def _make_fake_llvm(tmp_path: Path) -> Path:
    prefix = tmp_path / "llvm"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)

    _write_executable(
        bin_dir / "mlir-opt",
        r"""#!/usr/bin/env python3
import json
import pathlib
import sys

if "--version" in sys.argv:
    print("fake mlir-opt 1.2.3")
    raise SystemExit(0)

input_path = pathlib.Path(sys.argv[1])
output_path = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])

# Dump argv to a JSON sidecar for test inspection.
sidecar = output_path.with_suffix(".argv.json")
sidecar.write_text(json.dumps(sys.argv), encoding="utf-8")

output_path.write_text(
    r'''module attributes {gpu.container_module} {
  gpu.binary @kernels [#gpu.object<#rocdl.target<chip = "gfx942">, bin = "\7FELF\00external">]
}
''',
    encoding="utf-8",
)
""",
    )
    return prefix


def test_rocm_external_pipeline_split_matches_full_pipeline():
    backend = RocmBackend(RocmBackend.make_target("gfx942"))
    hints = {"waves_per_eu": 2, "maxnreg": 128}

    full = backend.pipeline_fragments(compile_hints=hints)
    pre_binary, binary = backend.external_binary_pipeline_fragments(compile_hints=hints)

    assert full == [*pre_binary, binary]
    assert pre_binary[-1] == "reconcile-unrealized-casts"
    assert any(fragment.startswith("gpu.module(") for fragment in pre_binary)
    assert binary.startswith("gpu-module-to-binary")
    assert "--amdgpu-waves-per-eu=2" in binary
    assert "--amdgpu-num-vgpr=128" in binary


def test_external_llvm_fingerprint_uses_configured_tools(tmp_path, monkeypatch):
    llvm_dir = _make_fake_llvm(tmp_path)
    monkeypatch.setenv("FLYDSL_COMPILE_LLVM_DIR", str(llvm_dir))
    external_llvm_fingerprint.cache_clear()

    try:
        fingerprint = external_llvm_fingerprint()
    finally:
        external_llvm_fingerprint.cache_clear()

    assert f"external-binary:{llvm_dir.resolve()}:" in fingerprint
    # Fingerprint should contain a hex SHA-256 hash of the mlir-opt binary.
    hash_part = fingerprint.split(":")[-1]
    assert len(hash_part) == 64 and all(c in "0123456789abcdef" for c in hash_part)


def test_run_external_binary_codegen_embeds_external_bin(tmp_path, monkeypatch):
    llvm_dir = _make_fake_llvm(tmp_path)
    monkeypatch.setenv("FLYDSL_COMPILE_LLVM_DIR", str(llvm_dir))

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        ctx.load_all_available_dialects()
        module = ir.Module.parse(
            '''module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    llvm.func @kernel() attributes {gpu.kernel, rocdl.kernel} {
      llvm.return
    }
  }
  llvm.func @host() {
    llvm.return
  }
}
''',
            context=ctx,
        )

        run_external_binary_codegen(
            module,
            'gpu-module-to-binary{format=fatbin opts=""}',
            work_dir=tmp_path / "external-work",
            stage_prefix="test_stage",
        )

        result = module.operation.get_asm(enable_debug_info=True)

        assert "gpu.binary @kernels" in result
        assert 'bin = "\\7FELF\\00external"' in result
        assert "gpu.module @kernels" not in result
        assert "llvm.func @host" in result
        module.operation.verify()

    assert (tmp_path / "external-work" / "test_stage_input.mlir").is_file()
    assert (tmp_path / "external-work" / "test_stage_external_output.mlir").is_file()
    assert (tmp_path / "external-work" / "test_stage_output.mlir").is_file()


_MODULE_WITH_HOST = '''\
module attributes {gpu.container_module} {
  gpu.module @kernels [#rocdl.target<chip = "gfx942">] {
    llvm.func @kernel() attributes {gpu.kernel, rocdl.kernel} {
      llvm.return
    }
  }
  llvm.func @host() {
    llvm.return
  }
}
'''


def test_llvm_options_forwarded_to_external_mlir_opt(tmp_path, monkeypatch):
    """llvm_options dict should appear as CLI flags in the subprocess argv."""
    llvm_dir = _make_fake_llvm(tmp_path)
    monkeypatch.setenv("FLYDSL_COMPILE_LLVM_DIR", str(llvm_dir))

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        ctx.load_all_available_dialects()
        module = ir.Module.parse(_MODULE_WITH_HOST, context=ctx)

        run_external_binary_codegen(
            module,
            'gpu-module-to-binary{format=fatbin opts=""}',
            llvm_options={"enable-post-misched": False, "lsr-drop-solution": 4},
            work_dir=tmp_path / "work",
            stage_prefix="opts_test",
        )

    sidecar = tmp_path / "work" / "opts_test_external_output.argv.json"
    assert sidecar.is_file(), "fake mlir-opt should write argv sidecar"
    argv = json.loads(sidecar.read_text(encoding="utf-8"))
    assert "--enable-post-misched=false" in argv
    assert "--lsr-drop-solution=4" in argv


def test_input_mlir_contains_only_gpu_module(tmp_path, monkeypatch):
    """The _input.mlir sent to external mlir-opt should NOT contain host-side IR."""
    llvm_dir = _make_fake_llvm(tmp_path)
    monkeypatch.setenv("FLYDSL_COMPILE_LLVM_DIR", str(llvm_dir))

    with ir.Context() as ctx, ir.Location.unknown(ctx):
        ctx.load_all_available_dialects()
        module = ir.Module.parse(_MODULE_WITH_HOST, context=ctx)

        run_external_binary_codegen(
            module,
            'gpu-module-to-binary{format=fatbin opts=""}',
            work_dir=tmp_path / "work",
            stage_prefix="gpu_only",
        )

    input_mlir = (tmp_path / "work" / "gpu_only_input.mlir").read_text(encoding="utf-8")
    assert "gpu.module @kernels" in input_mlir
    assert "llvm.func @host" not in input_mlir


def test_format_llvm_cli_options():
    assert _format_llvm_cli_options({"enable-post-misched": False}) == ["--enable-post-misched=false"]
    assert _format_llvm_cli_options({"enable-post-misched": True}) == ["--enable-post-misched=true"]
    assert _format_llvm_cli_options({"lsr-drop-solution": 4}) == ["--lsr-drop-solution=4"]
    assert _format_llvm_cli_options({}) == []
