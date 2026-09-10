# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Instruction class constraints remain valid after explicit physical placement."""

import re

import pytest

from tests.unit.test_register_placement import _compile_machine_module

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


@pytest.mark.parametrize("asm_text", ["flydsl_invalid_instruction", "v_cvt_f16_f32 v0, a0"])
@pytest.mark.parametrize("output_format", ["isa", "fatbin"])
def test_assembler_failure_returns_to_python(asm_text, output_format):
    from flydsl._mlir import ir
    from flydsl._mlir.passmanager import PassManager
    from flydsl.compiler.jit_function import _create_mlir_context

    # Inline assembly is only a fault injection here. The placement machinery
    # itself generates standard COPYs and has no opcode-specific repair rules.
    source = """module {
      gpu.module @m [#rocdl.target<O = 3, chip = "gfx942">] {
        llvm.func @bad(%p: !llvm.ptr<1>) attributes {gpu.kernel, rocdl.kernel} {
          %x = llvm.load volatile %p : !llvm.ptr<1> -> i32
          %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">,
            start = 64 : i64, bitOffset = 0 : i64, storageBits = 32 : i64} : i32
          llvm.inline_asm has_side_effects "flydsl_invalid_instruction", "" : () -> !llvm.void
          llvm.store volatile %fixed, %p : i32, !llvm.ptr<1>
          llvm.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source.replace("flydsl_invalid_instruction", asm_text))
        with pytest.raises(ir.MLIRError, match="failed LLVM assembler validation"):
            PassManager.parse(
                f"builtin.module(convert-fly-to-rocdl,fly-serialize-register-kernels{{format={output_format}}})"
            ).run(module.operation)
        assert "gpu.binary" not in str(module)
        assert ir.Module.parse("module {}").operation.verify()


def test_checked_serialization_leaves_unannotated_modules_unchanged():
    from flydsl._mlir import ir
    from flydsl._mlir.passmanager import PassManager
    from flydsl.compiler.jit_function import _create_mlir_context

    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse("module { gpu.module @unannotated {} }")
        before = str(module)
        PassManager.parse("builtin.module(fly-serialize-register-kernels)").run(module.operation)
        assert str(module) == before


@pytest.mark.parametrize("arch", ["gfx908", "gfx942", "gfx950"])
@pytest.mark.parametrize("bank", ["AGPR", "VGPR"])
def test_partial_mfma_direct_placement_after_llvm_rewriting(tmp_path, arch, bank):
    # Only D is fixed. Do not accept a write-back copy as successful placement.
    # LLVM can reclassify the unplaced C as well on gfx942/950. The native
    # rewrite is unavailable on gfx908, which retains AGPR-only C/D.
    body = f"""
      %a = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf16>
      %b = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf16>
      %c = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf32>
      %zero = llvm.mlir.constant(0 : i32) : i32
      %d = llvm.call_intrinsic "llvm.amdgcn.mfma.f32.16x16x16f16"(%a, %b, %c, %zero, %zero, %zero)
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
      %fixed = fly.register_value %d {{regClass = #fly.register_class<"amdgcn", "{bank}_32">,
        start = 64 : i64, bitOffset = 0 : i64, storageBits = 128 : i64}} : vector<4xf32>
      llvm.store volatile %fixed, %p : vector<4xf32>, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body, arch=arch)
    native_bank = "AGPR" if arch == "gfx908" else "VGPR"
    if arch == "gfx908" and bank != native_bank:
        assert result.returncode != 0
        assert "explicit register definition cannot use" in result.stderr
        assert "automatic write-back copies are disabled" in result.stderr
        assert not (tmp_path / "final.s").exists()
        result = _compile_machine_module(tmp_path, body, arch=arch, strip_placement=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert "error:" not in result.stderr.lower(), result.stderr
    asm = (tmp_path / "final.s").read_text()
    assert "Invalid register" not in asm
    fixed_inst = next(line for line in asm.splitlines() if "v_mfma_" in line)
    plain = tmp_path / "plain"
    plain.mkdir()
    result = _compile_machine_module(plain, body, arch=arch, strip_placement=True)
    assert result.returncode == 0, result.stdout + result.stderr
    plain_inst = next(line for line in (plain / "final.s").read_text().splitlines() if "v_mfma_" in line)
    assert fixed_inst.split()[0] == plain_inst.split()[0]
    # The destination directly names the requested tuple. Merely seeing
    # a64/v64 in a later COPY is insufficient.
    prefix = "a" if bank == "AGPR" else "v"
    assert re.search(rf"v_mfma_\S+ {prefix}\[64:67\],", fixed_inst), fixed_inst


@pytest.mark.parametrize("start", [4, 8])
def test_native_rewrite_does_not_evict_live_implicit_accumulator(tmp_path, start):
    # C remains live after MFMA. LLVM's native rewrite keeps it in a[4:7];
    # forcing D onto that same range must fail, not overwrite or move C.
    body = f"""
      %a = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf16>
      %b = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf16>
      %c = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf32>
      %zero = llvm.mlir.constant(0 : i32) : i32
      %d = llvm.call_intrinsic "llvm.amdgcn.mfma.f32.16x16x16f16"(%a, %b, %c, %zero, %zero, %zero)
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
      %fixed = fly.register_value %d {{regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = {start} : i64, bitOffset = 0 : i64, storageBits = 128 : i64}} : vector<4xf32>
      llvm.store volatile %c, %p : vector<4xf32>, !llvm.ptr<1>
      llvm.store volatile %fixed, %p : vector<4xf32>, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body)
    if start == 4:
        assert result.returncode != 0
        assert "interferes with a live value after LLVM allocation" in result.stderr
        assert not (tmp_path / "final.s").exists()
        result = _compile_machine_module(tmp_path, body, strip_placement=True)
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        asm = (tmp_path / "final.s").read_text()
        assert re.search(r"v_mfma_\S+ a\[8:11\],.*a\[4:7\]", asm), asm


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_llvm_rewrites_fixed_agpr_accumulator(tmp_path, arch):
    # Both the incoming accumulator and the result request AGPRs. FlyDSL
    # supplies COPY boundaries; the target owns all MFMA opcode conversion.
    body = """
      %a = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf16>
      %b = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf16>
      %c = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf32>
      %cin = fly.register_value %c {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = 0 : i64, bitOffset = 0 : i64, storageBits = 128 : i64} : vector<4xf32>
      %zero = llvm.mlir.constant(0 : i32) : i32
      %d = llvm.call_intrinsic "llvm.amdgcn.mfma.f32.16x16x16f16"(%a, %b, %cin, %zero, %zero, %zero)
        : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
      %fixed = fly.register_value %d {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = 0 : i64, bitOffset = 0 : i64, storageBits = 128 : i64} : vector<4xf32>
      %half = llvm.fptrunc %fixed : vector<4xf32> to vector<4xf16>
      llvm.store volatile %half, %p : vector<4xf16>, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body, arch=arch)
    assert result.returncode == 0, result.stdout + result.stderr
    asm = (tmp_path / "final.s").read_text()
    mfma = next(line for line in asm.splitlines() if "v_mfma_" in line)
    assert re.search(r"v_mfma_\S+ a\[0:3\],.*a\[0:3\]", mfma), asm
    assert "v_accvgpr_write_b32" not in asm
    assert asm.count("v_accvgpr_read_b32") == 4
    assert asm.index("v_accvgpr_read_b32") > asm.index(mfma)
    before = (tmp_path / "placement_test.registers.txt").read_text()
    after = (tmp_path / "placement_test.rewritten-registers.txt").read_text()
    assert "V_MFMA_F32_16X16X16F16_vgprcd_e64" in before
    assert "V_MFMA_F32_16X16X16F16_vgprcd_e64" not in after
    assert "V_MFMA_F32_16X16X16F16_e64" in after


def test_agpr_tuple_read_for_float_conversion(tmp_path):
    result = _compile_machine_module(
        tmp_path,
        """
      %input = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf32>
      %fixed = fly.register_value %input {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = 128 : i64, bitOffset = 0 : i64, storageBits = 128 : i64} : vector<4xf32>
      %half = llvm.fptrunc %fixed : vector<4xf32> to vector<4xf16>
      llvm.store volatile %half, %p : vector<4xf16>, !llvm.ptr<1>
    """,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    asm = (tmp_path / "final.s").read_text()
    assert "Invalid register" not in asm
    assert "v_cvt" in asm
    for reg in range(128, 132):
        assert re.search(rf"v_accvgpr_read_b32 v\d+, a{reg}\b", asm), asm


@pytest.mark.parametrize("bank", ["AGPR", "VGPR"])
def test_arithmetic_requires_direct_fixed_result(tmp_path, bank):
    body = """
      %input = llvm.load volatile %p : !llvm.ptr<1> -> i32
      %fixed = fly.register_value %input {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = 32 : i64, bitOffset = 0 : i64, storageBits = 32 : i64} : i32
      %c = llvm.mlir.constant(7 : i32) : i32
      %sum = llvm.add %fixed, %c : i32
      %updated = fly.register_value %sum {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = 32 : i64, bitOffset = 0 : i64, storageBits = 32 : i64} : i32
      llvm.store volatile %updated, %p : i32, !llvm.ptr<1>
    """.replace("AGPR_32", f"{bank}_32")
    result = _compile_machine_module(tmp_path, body)
    if bank == "AGPR":
        assert result.returncode != 0
        assert "automatic write-back copies are disabled" in result.stderr
        result = _compile_machine_module(tmp_path, body, strip_placement=True)
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        asm = (tmp_path / "final.s").read_text()
        assert re.search(r"v_add\S* v32,", asm), asm
        assert "v_accvgpr_" not in asm


@pytest.mark.parametrize("bank", ["AGPR", "VGPR"])
def test_tied_read_modify_write_requires_direct_placement(tmp_path, bank):
    body = """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> f32
      %y = llvm.load volatile %p : !llvm.ptr<1> -> f32
      %z = llvm.load volatile %p : !llvm.ptr<1> -> f32
      %fixed = fly.register_value %z {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = 32 : i64, bitOffset = 0 : i64, storageBits = 32 : i64} : f32
      %sum = llvm.intr.fma(%x, %y, %fixed) : (f32, f32, f32) -> f32
      %updated = fly.register_value %sum {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = 32 : i64, bitOffset = 0 : i64, storageBits = 32 : i64} : f32
      llvm.store volatile %updated, %p : f32, !llvm.ptr<1>
    """.replace("AGPR_32", f"{bank}_32")
    result = _compile_machine_module(tmp_path, body)
    if bank == "AGPR":
        assert result.returncode != 0
        assert "automatic write-back copies are disabled" in result.stderr
        result = _compile_machine_module(tmp_path, body, strip_placement=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    mir = (tmp_path / "placement_test.registers.txt").read_text()
    assert re.search(r"\$vgpr32 = [^\n]*V_FMAC_F32[^\n]*\$vgpr32\(tied-def 0\)", mir), mir
    asm = (tmp_path / "final.s").read_text()
    assert "v_fmac_f32" in asm and "Invalid register" not in asm
    assert "v_accvgpr_" not in asm


def test_fixed_result_cannot_hide_incompatible_fixed_input(tmp_path):
    # The VGPR result is legal, but accepting an AGPR input via a new COPY
    # would still hide transfers inside explicitly placed dataflow.
    body = """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> i32
      %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = 32 : i64, bitOffset = 0 : i64, storageBits = 32 : i64} : i32
      %seven = llvm.mlir.constant(7 : i32) : i32
      %sum = llvm.add %fixed, %seven : i32
      %result = fly.register_value %sum {regClass = #fly.register_class<"amdgcn", "VGPR_32">,
        start = 64 : i64, bitOffset = 0 : i64, storageBits = 32 : i64} : i32
      llvm.store volatile %result, %p : i32, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body)
    assert result.returncode != 0
    assert "must also accept their fixed inputs directly" in result.stderr
    result = _compile_machine_module(tmp_path, body, strip_placement=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bank", ["AGPR", "VGPR"])
def test_scalar_consumer_does_not_implicitly_select_a_lane(tmp_path, bank):
    body = f"""
      %id = llvm.call_intrinsic "llvm.amdgcn.workgroup.id.x"() : () -> i32
      %fixed = fly.register_value %id {{regClass = #fly.register_class<"amdgcn", "{bank}_32">,
        start = 64 : i64, bitOffset = 0 : i64, storageBits = 32 : i64}} : i32
      %seven = llvm.mlir.constant(7 : i32) : i32
      %sum = llvm.add %fixed, %seven : i32
      llvm.store volatile %sum, %p : i32, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body)
    assert result.returncode != 0
    assert "vector-to-scalar transfer requires an explicit uniform conversion" in result.stderr
    result = _compile_machine_module(tmp_path, body, strip_placement=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_bridges_under_register_pressure_do_not_borrow_fixed_storage(tmp_path):
    result = _compile_machine_module(
        tmp_path,
        """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<32xf32>
      %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = 32 : i64, bitOffset = 0 : i64, storageBits = 1024 : i64} : vector<32xf32>
      %y = llvm.load volatile %p : !llvm.ptr<1> -> vector<96xi32>
      %half = llvm.fptrunc %fixed : vector<32xf32> to vector<32xf16>
      llvm.store volatile %half, %p : vector<32xf16>, !llvm.ptr<1>
      llvm.store volatile %y, %p : vector<96xi32>, !llvm.ptr<1>
    """,
        vgpr_limit=64,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    asm = (tmp_path / "final.s").read_text()
    assert "scratch_store" in asm and "scratch_load" in asm
    assert int(re.search(r"\.vgpr_spill_count:\s*(\d+)", asm)[1]) > 0
    fixed_instructions = []
    for line in asm.splitlines():
        registers = set()
        for first, last, scalar in re.findall(r"\ba(?:\[(\d+):(\d+)\]|(\d+)\b)", line):
            registers.update(range(int(first), int(last) + 1) if first else [int(scalar)])
        if registers.intersection(range(32, 64)):
            fixed_instructions.append(line.strip())
    # Fixed storage is defined by eight tuple loads and read by 32 bridges.
    # Spill/reload and register scavenging must not reuse any of these AGPRs.
    assert len(fixed_instructions) == 40, fixed_instructions
    assert sum(line.startswith("global_load_dwordx4") for line in fixed_instructions) == 8
    assert sum(line.startswith("v_accvgpr_read_b32") for line in fixed_instructions) == 32
