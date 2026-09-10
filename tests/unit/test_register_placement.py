# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Explicit register storage: promotion offsets and real AMDGPU code generation."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler.jit_function import _create_mlir_context

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]
ROOT = Path(__file__).resolve().parents[2]


def test_register_pointer_slice_keeps_physical_bit_offset():
    # A half-precision pointer advances in elements, while register numbering
    # advances in 32-bit words. Four f16 elements advance a[32] to a[34].
    source = """module {
      gpu.module @m {
        gpu.func @slice(%input: vector<8xf16>, %out: !fly.ptr<f16, global>) kernel {
          %p = fly.make_ptr() {dictAttrs = {
            allocSize = 8 : i64
          }} : () -> !fly.ptr<f16, register>
          fly.set_register %p {regClass = #fly.register_class<"amdgcn", "AGPR_32">, start = 32 : i64} : !fly.ptr<f16, register>
          fly.ptr.store(%input, %p) : (vector<8xf16>, !fly.ptr<f16, register>) -> ()
          %offset = fly.make_int_tuple() : () -> !fly.int_tuple<4>
          %q = fly.add_offset(%p, %offset) : (!fly.ptr<f16, register>, !fly.int_tuple<4>) -> !fly.ptr<f16, register>
          %v = fly.ptr.load(%q) : (!fly.ptr<f16, register>) -> vector<4xf16>
          fly.ptr.store(%v, %out) : (vector<4xf16>, !fly.ptr<f16, global>) -> ()
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        PassManager.parse("builtin.module(fly-promote-regmem-to-vectorssa)").run(module.operation)
        asm = str(module)
        assert "fly.make_ptr" not in asm
        assert re.search(
            r'fly.register_value .*bitOffset = 0 : i64, regClass = #fly.register_class<"amdgcn", "AGPR_32">, start = 32 : i64, storageBits = 128 : i64.*vector<8xf16>',
            asm,
        )
        assert re.search(
            r'fly.register_value .*bitOffset = 64 : i64, regClass = #fly.register_class<"amdgcn", "AGPR_32">, start = 32 : i64, storageBits = 128 : i64.*vector<4xf16>',
            asm,
        )


@pytest.mark.parametrize(
    "arch,a_start,b_start,c_start",
    [("gfx942", 0, 64, 64), ("gfx942", 32, 128, 128), ("gfx950", 0, 64, 64)],
)
def test_example04_exact_registers_in_final_isa(tmp_path, arch, a_start, b_start, c_start):
    run_env = dict(os.environ)
    run_env.update(
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
import sys
spec = importlib.util.spec_from_file_location("register_example", {str(ROOT / "examples/04-preshuffle_gemm.py")!r})
example = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = example
spec.loader.exec_module(example)
example.EXPLICIT_REGISTERS = True
example.MMA_REG_A = (example.fx.rocdl.AGPR, {a_start})
example.MMA_REG_B = (example.fx.rocdl.AGPR, {b_start})
example.MMA_REG_C = (example.fx.rocdl.VGPR, {c_start})
example.main()
"""
    result = subprocess.run([sys.executable, "-c", script], env=run_env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    files = list((tmp_path / "ir").rglob("*_final_isa.s"))
    assert len(files) == 1
    asm = files[0].read_text()
    instructions = re.findall(r"^\s*v_mfma_.*$", asm, re.MULTILINE)
    assert len(instructions) == 256
    used_a, used_b, used_c = set(), set(), set()
    for inst in instructions:
        match = re.search(r" v\[(\d+):(\d+)\], a\[(\d+):(\d+)\], a\[(\d+):(\d+)\], v\[(\d+):(\d+)\]", inst)
        assert match, inst
        d0, d1, a0, a1, b0, b1, c0, c1 = map(int, match.groups())
        assert (d0, d1) == (c0, c1)
        assert c_start <= c0 <= c1 < c_start + 64
        assert a_start <= a0 <= a1 < a_start + 64
        assert b_start <= b0 <= b1 < b_start + 32
        used_a.update(range(a0, a1 + 1))
        used_b.update(range(b0, b1 + 1))
        used_c.update(range(c0, c1 + 1))
    assert used_a == set(range(a_start, a_start + 64))
    assert used_b == set(range(b_start, b_start + 32))
    assert used_c == set(range(c_start, c_start + 64))
    assert re.search(rf"ds_read_b128 a\[{a_start}:{a_start + 3}\]", asm)
    assert re.search(rf"buffer_load_dwordx4 a\[{b_start}:{b_start + 3}\]", asm)
    assert "v_accvgpr_read" not in asm
    assert "v_accvgpr_write" not in asm


def test_explicit_register_storage_does_not_silently_fall_back_to_private_memory():
    source = """module {
      gpu.module @m {
        gpu.func @dynamic(%index: i32) kernel {
          %p = fly.make_ptr() {dictAttrs = {
            allocSize = 8 : i64
          }} : () -> !fly.ptr<f32, register>
          fly.set_register %p {regClass = #fly.register_class<"amdgcn", "AGPR_32">, start = 32 : i64} : !fly.ptr<f32, register>
          %offset = fly.make_int_tuple(%index) : (i32) -> !fly.int_tuple<?>
          %q = fly.add_offset(%p, %offset) : (!fly.ptr<f32, register>, !fly.int_tuple<?>) -> !fly.ptr<f32, register>
          %v = arith.constant 0.0 : f32
          fly.ptr.store(%v, %q) : (f32, !fly.ptr<f32, register>) -> ()
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        with pytest.raises(ir.MLIRError, match="explicit register storage requires static offsets"):
            PassManager.parse("builtin.module(fly-promote-regmem-to-vectorssa)").run(module.operation)


def _compile_machine_module(tmp_path, body, *, vgpr_limit=None, opt_level=3, strip_placement=False, declarations=""):
    attrs = f', passthrough = [["amdgpu-num-vgpr", "{vgpr_limit}"]]' if vgpr_limit else ""
    source = f"""module {{
      gpu.module @m [#rocdl.target<O = {opt_level}, chip = "gfx942">] {{
        {declarations}
        llvm.func @placement_test(%p: !llvm.ptr<1>) attributes {{gpu.kernel, rocdl.kernel{attrs}}} {{
          {body}
          llvm.return
        }}
      }}
    }}"""
    if strip_placement:
        # register_value is an identity; deleting it replaces its result uses.
        pattern = r"%(\w+) = fly\.register_value %(\w+) \{[^}]*\} : [^\n]+"
        while match := re.search(pattern, source):
            result_name, operand_name = match.group(1, 2)
            source = source[: match.start()] + source[match.end() :]
            source = re.sub(rf"%{result_name}\b", f"%{operand_name}", source)
    script = f"""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from pathlib import Path
from flydsl.compiler.jit_function import _create_mlir_context, _extract_isa_text
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
with _create_mlir_context(), ir.Location.unknown():
    module = ir.Module.parse({source!r})
    PassManager.parse("builtin.module(reconcile-unrealized-casts,convert-fly-to-rocdl,gpu-module-to-binary{{format=isa}})").run(module.operation)
    Path({str(tmp_path / 'final.s')!r}).write_text(_extract_isa_text(str(module)))
"""
    run_env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), FLYDSL_REGISTER_DUMP_DIR=str(tmp_path))
    return subprocess.run([sys.executable, "-c", script], env=run_env, capture_output=True, text=True, timeout=90)


def test_implicit_spills_do_not_borrow_explicit_registers(tmp_path):
    # Volatile accesses keep all implicit values live across the fixed value.
    # gfx942 interprets the attribute in its target-specific register units.
    result = _compile_machine_module(
        tmp_path,
        """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<32xi32>
      %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, start = 32 : i64, bitOffset = 0 : i64, storageBits = 1024 : i64} : vector<32xi32>
      %y = llvm.load volatile %p : !llvm.ptr<1> -> vector<96xi32>
      llvm.store volatile %fixed, %p : vector<32xi32>, !llvm.ptr<1>
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
        regs = set()
        for first, last, scalar in re.findall(r"\bv(?:\[(\d+):(\d+)\]|(\d+)\b)", line):
            regs.update(range(int(first), int(last) + 1) if first else [int(scalar)])
        if regs.intersection(range(32, 64)):
            fixed_instructions.append(line.strip())
    # Eight loads define v[32:63], eight stores consume it. No other instruction,
    # including spill/reload or scavenging, borrows any register in this range.
    assert len(fixed_instructions) == 16, fixed_instructions
    assert sum(line.startswith("global_load_dwordx4") for line in fixed_instructions) == 8
    assert sum(line.startswith("global_store_dwordx4") for line in fixed_instructions) == 8
    mir = (tmp_path / "placement_test.registers.txt").read_text()
    assert "STACKMAP" not in mir
    assert "ADJCALLSTACK" not in mir
    assert "DBG_VALUE" not in mir


@pytest.mark.parametrize("overlap", [True, False])
def test_two_explicit_allocations_must_not_overlap_while_live(tmp_path, overlap):
    first = """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed_x = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, start = 32 : i64, bitOffset = 0 : i64, storageBits = 128 : i64} : vector<4xi32>
    """
    second = """
      %y = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed_y = fly.register_value %y {regClass = #fly.register_class<"amdgcn", "VGPR_32">, start = 32 : i64, bitOffset = 0 : i64, storageBits = 128 : i64} : vector<4xi32>
    """
    store_x = "llvm.store volatile %fixed_x, %p : vector<4xi32>, !llvm.ptr<1>\n"
    store_y = "llvm.store volatile %fixed_y, %p : vector<4xi32>, !llvm.ptr<1>\n"
    body = first + second + store_x + store_y if overlap else first + store_x + second + store_y
    result = _compile_machine_module(tmp_path, body)
    if overlap:
        assert result.returncode != 0
        assert "overlapping live intervals in explicit register storage" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_path / "final.s").read_text().count("global_load_dwordx4 v[32:35]") == 2


@pytest.mark.parametrize("start,diagnostic", [(64, "conflicting declarations")])
def test_invalid_register_declarations_are_rejected(start, diagnostic):
    source = f"""module {{
      gpu.module @m {{
        gpu.func @bad() kernel {{
          %p = fly.make_ptr() {{dictAttrs = {{allocSize = 8 : i64}}}} : () -> !fly.ptr<f32, register>
          fly.set_register %p {{regClass = #fly.register_class<"amdgcn", "AGPR_32">, start = 32 : i64}} : !fly.ptr<f32, register>
          fly.set_register %p {{regClass = #fly.register_class<"amdgcn", "AGPR_32">, start = {start} : i64}} : !fly.ptr<f32, register>
          gpu.return
        }}
      }}
    }}"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        with pytest.raises(ir.MLIRError, match=diagnostic):
            PassManager.parse("builtin.module(fly-promote-regmem-to-vectorssa)").run(module.operation)


def test_python_set_register_emits_a_separate_op():
    import flydsl.expr as fx

    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            tensor = fx.make_rmem_tensor(8, fx.Float16)
            allocation_before = str(tensor.owner)
            assert fx.set_register(tensor, register_class=fx.rocdl.AGPR, start=32) is None
            assert str(tensor.owner) == allocation_before
            attrs = ir.DictAttr.get({"allocSize": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 8)})
            ptr = fx.make_ptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Register), [], dict_attrs=attrs)
            pointer_before = str(ptr.owner)
            fx.set_register(ptr, register_class=fx.rocdl.VGPR, start=64)
            assert str(ptr.owner) == pointer_before
            with pytest.raises(ValueError, match="register-memory"):
                fx.set_register(fx.Int32(3), register_class=fx.rocdl.VGPR, start=0)
        assert module.operation.verify()
        assert str(module).count("fly.set_register") == 2
        assert "fly_rocdl.set_register" not in str(module)


def test_unsupported_codegen_pipeline_rejects_instead_of_dropping_placement(tmp_path):
    result = _compile_machine_module(
        tmp_path,
        """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">, start = 32 : i64, bitOffset = 0 : i64, storageBits = 128 : i64} : vector<4xi32>
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
    """,
        opt_level=0,
    )
    assert result.returncode != 0
    assert "explicit register placement requires optimized SelectionDAG codegen" in result.stderr


@pytest.mark.parametrize("descriptor", [("amdgcn", "SGPR_32"), ("nvptx64", "Int32Regs"), ("x86_64", "GR32")])
def test_register_class_representation_is_target_independent(descriptor):
    import flydsl.expr as fx

    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            tensor = fx.make_rmem_tensor(4, fx.Int32)
            fx.set_register(tensor, register_class=fx.RegisterClass(*descriptor), start=4)
        assert module.operation.verify()
        assert str(ir.Module.parse(str(module))) == str(module)
    assert fx.rocdl.SGPR == fx.RegisterClass("amdgcn", "SGPR_32")


@pytest.mark.parametrize(
    "name,start,diagnostic",
    [
        ("AGPR_32", 255, "exceeds LLVM register class members"),
        ("NoSuchClass", 0, "unsupported AMDGPU register class"),
        ("SReg_64", 0, "unsupported AMDGPU register class"),
        ("SGPR_32", 40, "requires a compatible uniform LLVM register class"),
    ],
)
def test_backend_class_constraints_are_checked(tmp_path, name, start, diagnostic):
    result = _compile_machine_module(
        tmp_path,
        f"""
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value %x {{regClass = #fly.register_class<"amdgcn", "{name}">,
        start = {start} : i64, bitOffset = 0 : i64, storageBits = 128 : i64}} : vector<4xi32>
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
    """,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr, result.stdout + result.stderr


def test_uniform_sgpr_placement(tmp_path):
    result = _compile_machine_module(
        tmp_path,
        """
      %x = llvm.call_intrinsic "llvm.amdgcn.workgroup.id.x"() : () -> i32
      %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "SGPR_32">,
        start = 40 : i64, bitOffset = 0 : i64, storageBits = 32 : i64} : i32
      llvm.store volatile %fixed, %p : i32, !llvm.ptr<1>
    """,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    asm = (tmp_path / "final.s").read_text()
    assert re.search(r"s_mov_b32 s40, s[0-9]+", asm), asm
    assert re.search(r"v_mov_b32[^\n]*s40", asm), asm


def test_wrong_backend_is_rejected_by_lowering(tmp_path):
    result = _compile_machine_module(
        tmp_path,
        """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value %x {regClass = #fly.register_class<"nvptx64", "Int32Regs">,
        start = 0 : i64, bitOffset = 0 : i64, storageBits = 128 : i64} : vector<4xi32>
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
    """,
    )
    assert result.returncode != 0
    assert "requires target amdgcn" in result.stderr


@pytest.mark.parametrize("start,diagnostic", [(32, "unavailable on this target"), (4, "precolored machine operand")])
def test_sgpr_placement_protects_target_and_abi_registers(tmp_path, start, diagnostic):
    result = _compile_machine_module(
        tmp_path,
        f"""
      %x = llvm.call_intrinsic "llvm.amdgcn.workgroup.id.x"() : () -> i32
      %fixed = fly.register_value %x {{regClass = #fly.register_class<"amdgcn", "SGPR_32">,
        start = {start} : i64, bitOffset = 0 : i64, storageBits = 32 : i64}} : i32
      llvm.store volatile %fixed, %p : i32, !llvm.ptr<1>
    """,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("opt_level", [1, 2, 3])
def test_optimized_pipelines_honor_placement(tmp_path, opt_level):
    result = _compile_machine_module(
        tmp_path,
        """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">,
        start = 64 : i64, bitOffset = 0 : i64, storageBits = 128 : i64} : vector<4xi32>
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
    """,
        opt_level=opt_level,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "global_load_dwordx4 v[64:67]" in (tmp_path / "final.s").read_text()


@pytest.mark.parametrize(
    "body,diagnostic",
    [
        (
            """
      %x = llvm.mlir.constant(42 : i32) : i32
      %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">,
        start = 64 : i64, bitOffset = 0 : i64, storageBits = 32 : i64} : i32
      llvm.store volatile %fixed, %p : i32, !llvm.ptr<1>
    """,
            "constant-only placement is unsupported",
        ),
        (
            """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "AGPR_32">,
        start = 255 : i64, bitOffset = 0 : i64, storageBits = 128 : i64} : vector<4xi32>
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
    """,
            "exceeds LLVM register class members",
        ),
    ],
)
def test_removing_invalid_placement_restores_compilation(tmp_path, body, diagnostic):
    result = _compile_machine_module(tmp_path, body)
    assert result.returncode != 0
    assert diagnostic in result.stderr, result.stdout + result.stderr
    result = _compile_machine_module(tmp_path, body, strip_placement=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "class_name,element_type,storage_bits,diagnostic,opt_level",
    [
        ("NoSuchClass", "i32", 32, "unsupported AMDGPU register class", 3),
        ("VGPR_32", "i16", 16, "requires whole 32-bit registers", 3),
        ("VGPR_32", "i32", 32, "requires optimized SelectionDAG codegen", 0),
    ],
)
def test_static_errors_return_to_python(class_name, element_type, storage_bits, diagnostic, opt_level):
    source = f"""module {{
      gpu.module @m [#rocdl.target<O = {opt_level}, chip = "gfx942">] {{
        llvm.func @bad(%x: {element_type}) -> {element_type} {{
          %fixed = fly.register_value %x {{regClass = #fly.register_class<"amdgcn", "{class_name}">,
            start = 0 : i64, bitOffset = 0 : i64, storageBits = {storage_bits} : i64}} : {element_type}
          llvm.return %fixed : {element_type}
        }}
      }}
    }}"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        with pytest.raises(ir.MLIRError, match=diagnostic):
            PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)
        # The Python process and context are still usable after the failure.
        assert ir.Module.parse("module {}").operation.verify()


def test_unused_register_storage_is_removed_without_backend_carriers():
    source = """module {
      gpu.module @m {
        gpu.func @unused() kernel {
          %p = fly.make_ptr() {dictAttrs = {allocSize = 4 : i64}} : () -> !fly.ptr<i32, register>
          fly.set_register %p {regClass = #fly.register_class<"amdgcn", "VGPR_32">, start = 64 : i64} : !fly.ptr<i32, register>
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        PassManager.parse("builtin.module(fly-promote-regmem-to-vectorssa,canonicalize)").run(module.operation)
        assert "fly.set_register" not in str(module)
        assert "fly.register_value" not in str(module)
        assert "fly.make_ptr" not in str(module)


@pytest.mark.parametrize("user_id", [0, 0x464C000000000000])
def test_existing_stackmaps_are_not_claimed_by_placement(tmp_path, user_id):
    result = _compile_machine_module(
        tmp_path,
        f"""
      %id = llvm.mlir.constant({user_id} : i64) : i64
      %zero = llvm.mlir.constant(0 : i32) : i32
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      llvm.call_intrinsic "llvm.experimental.stackmap"(%id, %zero, %x) : (i64, i32, vector<4xi32>) -> ()
      %fixed = fly.register_value %x {{regClass = #fly.register_class<"amdgcn", "VGPR_32">,
        start = 64 : i64, bitOffset = 0 : i64, storageBits = 128 : i64}} : vector<4xi32>
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
    """,
    )
    # Foreign STACKMAP support is LLVM-target-dependent. Inspect our pass's
    # output even if AMDGPU's final assembler does not support the foreign op.
    dump = tmp_path / "placement_test.registers.txt"
    assert dump.exists(), result.stdout + result.stderr
    mir = dump.read_text()
    assert mir.count("STACKMAP") == 1
    assert f"STACKMAP {user_id}," in mir
    assert "ADJCALLSTACK" in mir
    assert "$vgpr64_vgpr65_vgpr66_vgpr67" in mir


def test_codegen_registration_does_not_change_unannotated_compilation(tmp_path):
    script = r'''
import json
from flydsl.compiler.jit_function import _create_mlir_context, _extract_isa_text
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager

def compile(start=None):
    marker = ""
    value = "%x"
    if start is not None:
        marker = f"""%fixed = fly.register_value %x {{regClass = #fly.register_class<"amdgcn", "VGPR_32">,
          start = {start} : i64, bitOffset = 0 : i64, storageBits = 128 : i64}} : vector<4xi32>"""
        value = "%fixed"
    source = f"""module {{
      gpu.module @m [#rocdl.target<O = 3, chip = "gfx942">] {{
        llvm.func @placement_test(%p: !llvm.ptr<1>) attributes {{gpu.kernel, rocdl.kernel}} {{
          %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
          {marker}
          llvm.store volatile {value}, %p : vector<4xi32>, !llvm.ptr<1>
          llvm.return
        }}
      }}
    }}"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        PassManager.parse("builtin.module(convert-fly-to-rocdl,gpu-module-to-binary{format=isa})").run(module.operation)
        return _extract_isa_text(str(module))

before = compile()
placed = compile(64)
after = compile()
second = compile(96)
assert before == after
assert "global_load_dwordx4 v[64:67]" in placed
assert "global_load_dwordx4 v[96:99]" in second
print("baseline unchanged; placements isolated across compilations")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_example04_defaults_to_ordinary_allocation(tmp_path):
    run_env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(sys.path),
        COMPILE_ONLY="1",
        ARCH="gfx942",
        FLYDSL_DUMP_IR="1",
        FLYDSL_DUMP_DIR=str(tmp_path / "ir"),
        FLYDSL_REGISTER_DUMP_DIR=str(tmp_path / "mir"),
        FLYDSL_RUNTIME_CACHE_DIR=str(tmp_path / "cache"),
    )
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/04-preshuffle_gemm.py")],
        env=run_env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "explicit registers: False" in result.stdout
    assert list((tmp_path / "ir").rglob("*_final_isa.s"))
    for path in (tmp_path / "ir").rglob("*.mlir"):
        assert "fly.set_register" not in path.read_text()
        assert "__flydsl_register_value_" not in path.read_text()
    assert not list((tmp_path / "mir").glob("*.registers.txt"))


def test_calls_cannot_clobber_reserved_storage(tmp_path):
    body = """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value %x {regClass = #fly.register_class<"amdgcn", "VGPR_32">,
        start = 64 : i64, bitOffset = 0 : i64, storageBits = 128 : i64} : vector<4xi32>
      llvm.call @callee() : () -> ()
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body, declarations="llvm.func @callee()")
    assert result.returncode != 0
    assert "conflicts with a call clobber" in result.stderr, result.stdout + result.stderr
    result = _compile_machine_module(tmp_path, body, declarations="llvm.func @callee()", strip_placement=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_existing_symbol_cannot_silently_replace_register_carrier():
    name = "__flydsl_register_value_" + "amdgcn:VGPR_32".encode().hex() + "_i32"
    source = f"""module {{
      gpu.module @m {{
        llvm.func @{name}(i32, i64, i64, i64) -> i32
        llvm.func @test(%x: i32) -> i32 {{
          %fixed = fly.register_value %x {{regClass = #fly.register_class<"amdgcn", "VGPR_32">,
            start = 64 : i64, bitOffset = 0 : i64, storageBits = 32 : i64}} : i32
          llvm.return %fixed : i32
        }}
      }}
    }}"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        with pytest.raises(ir.MLIRError, match="marker symbol conflicts with existing symbol"):
            PassManager.parse("builtin.module(convert-fly-to-rocdl)").run(module.operation)
