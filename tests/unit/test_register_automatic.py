# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Hard register-class constraints with compiler-selected physical numbers."""

import os
import re
import subprocess
import sys

import pytest

from tests.unit.test_register_placement import ROOT, _compile_machine_module

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def automatic_assignments(path):
    text = (path / "placement_test.rewritten-registers.txt").read_text()
    return re.findall(
        r"; FlyDSL automatic .* class (\w+) bitOffset (-?\d+) alignment (\d+) assigned (\w+) index (\d+)", text
    )


@pytest.mark.parametrize("bank", ["VGPR", "AGPR"])
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("alignment", [1, 2, 4])
def test_automatic_class_and_alignment(tmp_path, bank, width, alignment):
    ty = "i32" if width == 1 else f"vector<{width}xi32>"
    body = f"""
      %x = llvm.load volatile %p : !llvm.ptr<1> -> {ty}
      %fixed = fly.register_value %x {{regClass = #fly.register_class<"amdgcn", "{bank}_32">,
        registerAlignment = {alignment} : i64, bitOffset = 0 : i64, storageBits = {width * 32} : i64}} : {ty}
      llvm.store volatile %fixed, %p : {ty}, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body)
    assert result.returncode == 0, result.stdout + result.stderr
    records = automatic_assignments(tmp_path)
    assert records
    for cls, offset, align, physical, index in records:
        assert cls == bank + "_32"
        assert int(align) == alignment
        assert (int(index) - int(offset) // 32) % alignment == 0
        assert physical.lower().startswith(bank.lower())
    isa = (tmp_path / "final.s").read_text()
    assert ".vgpr_spill_count: 0" in isa
    # Check the actual load destinations, beyond the pre-emission assignment dump.
    prefix = "a" if bank == "AGPR" else "v"
    loaded = re.findall(rf"(?:global|buffer|flat)_load_\w+\s+{prefix}(?:\[(\d+):\d+\]|(\d+)\b)", isa)
    assert loaded, isa
    assert all(int(first or scalar) % alignment == 0 for first, scalar in loaded)


@pytest.mark.parametrize("alignment", [1, 2, 4])
def test_automatic_sgpr_before_scalar_rewriting(tmp_path, alignment):
    body = f"""
      %x = llvm.call_intrinsic "llvm.amdgcn.workgroup.id.x"() : () -> i32
      %fixed = fly.register_value %x {{regClass = #fly.register_class<"amdgcn", "SGPR_32">,
        registerAlignment = {alignment} : i64, bitOffset = 0 : i64, storageBits = 32 : i64}} : i32
      llvm.store volatile %fixed, %p : i32, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body)
    assert result.returncode == 0, result.stdout + result.stderr
    records = automatic_assignments(tmp_path)
    assert records and all(cls == "SGPR_32" and int(index) % alignment == 0 for cls, _, _, _, index in records)
    isa = (tmp_path / "final.s").read_text()
    for _, _, _, _, index in records:
        assert re.search(rf"\bs{index}\b", isa), isa


def test_automatic_alignment_preserves_slice_residue(tmp_path):
    body = """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> i32
      %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">,
        registerAlignment = 4 : i64, bitOffset = 32 : i64, storageBits = 128 : i64} : i32
      llvm.store volatile %fixed, %p : i32, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body)
    assert result.returncode == 0, result.stdout + result.stderr
    records = automatic_assignments(tmp_path)
    assert records and all(int(index) % 4 == 1 for _, _, _, _, index in records)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("alignment", [1, 4])
def test_automatic_agpr_mfma_has_no_writeback(tmp_path, arch, alignment):
    body = f"""
      %a = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf16>
      %b = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf16>
      %c = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf32>
      %cin = fly.register_value %c {{regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        registerAlignment = {alignment} : i64, bitOffset = 0 : i64, storageBits = 128 : i64}} : vector<4xf32>
      %zero = llvm.mlir.constant(0 : i32) : i32
      %d = llvm.call_intrinsic "llvm.amdgcn.mfma.f32.16x16x16f16"(%a, %b, %cin, %zero, %zero, %zero)
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
      %fixed = fly.register_value %d {{regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        registerAlignment = {alignment} : i64, bitOffset = 0 : i64, storageBits = 128 : i64}} : vector<4xf32>
      %half = llvm.fptrunc %fixed : vector<4xf32> to vector<4xf16>
      llvm.store volatile %half, %p : vector<4xf16>, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body, arch=arch)
    assert result.returncode == 0, result.stdout + result.stderr
    isa = (tmp_path / "final.s").read_text()
    mfma = next(line for line in isa.splitlines() if "v_mfma_" in line)
    match = re.search(r"v_mfma_\S+ a\[(\d+):\d+\],.*a\[(\d+):\d+\]", mfma)
    assert match, isa
    assert all(int(index) % alignment == 0 for index in match.groups())
    assert "v_accvgpr_write" not in isa
    assert isa.count("v_accvgpr_read") == 4
    assert isa.index("v_accvgpr_read") > isa.index(mfma)


def test_automatic_class_rejects_unsupported_arithmetic(tmp_path):
    body = """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> f32
      %y = llvm.fadd %x, %x : f32
      %fixed = fly.register_value %y {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        bitOffset = 0 : i64, storageBits = 32 : i64} : f32
      llvm.store volatile %fixed, %p : f32, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body)
    assert result.returncode != 0
    assert "automatic write-back copies are disabled" in result.stderr
    assert not (tmp_path / "final.s").exists()
    result = _compile_machine_module(tmp_path, body, strip_placement=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "arch,c_bank,alignment",
    [("gfx942", "AGPR", 1), ("gfx942", "AGPR", 4), ("gfx950", "AGPR", 4), ("gfx942", "VGPR", 2)],
)
def test_example04_automatic_registers(tmp_path, arch, c_bank, alignment):
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(sys.path),
        COMPILE_ONLY="1",
        ARCH=arch,
        FLYDSL_DUMP_IR="1",
        FLYDSL_DUMP_DIR=str(tmp_path / "ir"),
        FLYDSL_REGISTER_DUMP_DIR=str(tmp_path / "mir"),
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
    )
    script = f"""
import importlib.util
import resource
import sys
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
spec = importlib.util.spec_from_file_location("automatic_gemm", {str(ROOT / "examples/04-preshuffle_gemm.py")!r})
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)
example.EXPLICIT_REGISTERS = True
example.MMA_REG_A = (example.fx.rocdl.AGPR, None)
example.MMA_REG_B = (example.fx.rocdl.AGPR, None)
example.MMA_REG_C = (example.fx.rocdl.{c_bank}, None)
example.MMA_REG_ALIGNMENT = {alignment}
example.main()
"""
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    files = list((tmp_path / "ir").rglob("*_final_isa.s"))
    assert len(files) == 1
    isa = files[0].read_text()
    lines = isa.splitlines()
    positions = [i for i, line in enumerate(lines) if "v_mfma_" in line]
    assert len(positions) == 256
    prefix = "a" if c_bank == "AGPR" else "v"
    for i in positions:
        assert re.search(rf"v_mfma_\S+ {prefix}\[\d+:\d+\], a\[\d+:\d+\], a\[\d+:\d+\], {prefix}\[\d+:\d+\]", lines[i])
    for i in positions:
        dest = re.search(rf"v_mfma_\S+ {prefix}\[(\d+):\d+\]", lines[i])
        # C's materialized slices start at multiples of four words in this GEMM.
        assert int(dest[1]) % alignment == 0, lines[i]
    assert not any("v_accvgpr_" in line for line in lines[positions[0] : positions[-1] + 1])
    assert re.search(r"\.vgpr_spill_count:\s*0\b", isa)
    dumps = list((tmp_path / "mir").glob("*.rewritten-registers.txt"))
    assert dumps
    records = re.findall(
        r"; FlyDSL automatic .* bitOffset (-?\d+) alignment (\d+) assigned \w+ index (\d+)", dumps[0].read_text()
    )
    assert records and all(
        int(align) == alignment and (int(index) - int(offset) // 32) % alignment == 0
        for offset, align, index in records
    )


@pytest.mark.parametrize("fixed_start", [64, 65])
def test_mixed_constraints_on_the_same_value(tmp_path, fixed_start):
    body = f"""
      %x = llvm.load volatile %p : !llvm.ptr<1> -> i32
      %auto = fly.register_value %x {{regClass = #fly.register_class<"amdgcn", "VGPR_32">,
        registerAlignment = 4 : i64, bitOffset = 0 : i64, storageBits = 32 : i64}} : i32
      %fixed = fly.register_value %auto {{regClass = #fly.register_class<"amdgcn", "VGPR_32">,
        start = {fixed_start} : i64, bitOffset = 0 : i64, storageBits = 32 : i64}} : i32
      llvm.store volatile %fixed, %p : i32, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body)
    if fixed_start == 64:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "v64" in (tmp_path / "final.s").read_text()
    else:
        assert result.returncode != 0
        assert "conflicting FlyDSL register placements" in result.stderr
        assert not (tmp_path / "final.s").exists()


@pytest.mark.parametrize("fixed_count", [3, 4])
def test_automatic_alignment_respects_live_fixed_values(tmp_path, fixed_count):
    # Every aligned slot except v12 is fixed in the positive case; the negative
    # case occupies v12 as well. Other unaligned slots remain available.
    body = ""
    for i in range(fixed_count):
        body += f"""
          %x{i} = llvm.load volatile %p : !llvm.ptr<1> -> i32
          %f{i} = fly.register_value %x{i} {{regClass = #fly.register_class<"amdgcn", "VGPR_32">,
            start = {4 * i} : i64, bitOffset = 0 : i64, storageBits = 32 : i64}} : i32
        """
    body += """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> i32
      %auto = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">,
        registerAlignment = 4 : i64, bitOffset = 0 : i64, storageBits = 32 : i64} : i32
    """
    for i in range(fixed_count):
        body += f"llvm.store volatile %f{i}, %p : i32, !llvm.ptr<1>\n"
    body += "llvm.store volatile %auto, %p : i32, !llvm.ptr<1>\n"
    result = _compile_machine_module(tmp_path, body, vgpr_limit=16)
    if fixed_count == 3:
        assert result.returncode == 0, result.stdout + result.stderr
        assert any(int(index) == 12 for _, _, _, _, index in automatic_assignments(tmp_path))
    else:
        assert result.returncode != 0
        assert "no non-interfering register assignment" in result.stderr
        assert not (tmp_path / "final.s").exists()
        result = _compile_machine_module(tmp_path, body, vgpr_limit=16, strip_placement=True)
        assert result.returncode == 0, result.stdout + result.stderr
