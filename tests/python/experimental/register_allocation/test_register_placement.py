# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Explicit register storage: promotion offsets and real AMDGPU code generation."""

import os
import re
import subprocess
import sys

import pytest

from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler.jit_function import _create_mlir_context

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def test_register_pointer_slice_keeps_physical_bit_offset():
    # A half-precision pointer advances in elements, while register numbering
    # advances in 32-bit words. Four f16 elements advance a[32] to a[34].
    source = """module {
      gpu.module @m {
        gpu.func @slice(%input: vector<8xf16>, %out: !fly.ptr<f16, global>) kernel {
          %p = fly.make_ptr() {dictAttrs = {
            allocSize = 8 : i64
          }} : () -> !fly.ptr<f16, register>
          fly.register_place(%p) {registerClass = #fly.register_class<"amdgcn", "AGPR_32", 32>, startIdx = 32 : i32} : (!fly.ptr<f16, register>) -> ()
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
            r'fly.register_value.*bitOffset = 0 : i32, registerClass = #fly.register_class<"amdgcn", "AGPR_32", 32>, startIdx = 32 : i32, storageBits = 128 : i32.*vector<8xf16>',
            asm,
        )
        assert re.search(
            r'fly.register_value.*bitOffset = 64 : i32, registerClass = #fly.register_class<"amdgcn", "AGPR_32", 32>, startIdx = 32 : i32, storageBits = 128 : i32.*vector<4xf16>',
            asm,
        )


def test_explicit_register_storage_does_not_silently_fall_back_to_private_memory():
    source = """module {
      gpu.module @m {
        gpu.func @dynamic(%index: i32) kernel {
          %p = fly.make_ptr() {dictAttrs = {
            allocSize = 8 : i64
          }} : () -> !fly.ptr<f32, register>
          fly.register_place(%p) {registerClass = #fly.register_class<"amdgcn", "AGPR_32", 32>, startIdx = 32 : i32} : (!fly.ptr<f32, register>) -> ()
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


def _compile_machine_module(
    tmp_path,
    body,
    *,
    vgpr_limit=None,
    opt_level=3,
    strip_placement=False,
    declarations="",
    arch="gfx942",
    llvm_options=None,
):
    attrs = f', passthrough = [["amdgpu-num-vgpr", "{vgpr_limit}"]]' if vgpr_limit else ""
    source = f"""module {{
      gpu.module @m [#rocdl.target<O = {opt_level}, chip = "{arch}">] {{
        {declarations}
        llvm.func @placement_test(%p: !llvm.ptr<1>) attributes {{gpu.kernel, rocdl.kernel{attrs}}} {{
          {body}
          llvm.return
        }}
      }}
    }}"""
    if strip_placement:
        # register_value is an identity; deleting it replaces its result uses.
        pattern = r"%(\w+) = fly\.register_value\(%(\w+)\) \{[^}]*\} : [^\n]+"
        while match := re.search(pattern, source):
            result_name, operand_name = match.group(1, 2)
            source = source[: match.start()] + source[match.end() :]
            source = re.sub(rf"%{result_name}\b", f"%{operand_name}", source)
    options = llvm_options or {}
    script = f"""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from pathlib import Path
from flydsl.compiler.jit_function import _create_mlir_context, _extract_isa_text
from flydsl.compiler.llvm_options import llvm_options
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
with _create_mlir_context(), ir.Location.unknown(), llvm_options({options!r}):
    module = ir.Module.parse({source!r})
    PassManager.parse("builtin.module(reconcile-unrealized-casts,convert-fly-to-rocdl,rocdl-serialize-register-kernels{{format=isa}},gpu-module-to-binary{{format=isa}})").run(module.operation)
    Path({str(tmp_path / 'final.s')!r}).write_text(_extract_isa_text(str(module)))
"""
    run_env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    return subprocess.run([sys.executable, "-c", script], env=run_env, capture_output=True, text=True, timeout=90)


def _compile_and_print_placement(tmp_path, body, **kwargs):
    """Compile with LLVM's -print-after-all and return the MIR after placement."""
    result = _compile_machine_module(tmp_path, body, llvm_options={"print-after-all": True}, **kwargs)
    for section in result.stderr.split("*** IR Dump After ")[1:]:
        if section.startswith("ApplyRegisterPlacement "):
            return result, section
    raise AssertionError(result.stdout + result.stderr)


def test_implicit_spills_do_not_borrow_explicit_registers(tmp_path):
    # Volatile accesses keep all implicit values live across the fixed value.
    # gfx942 interprets the attribute in its target-specific register units.
    result, mir = _compile_and_print_placement(
        tmp_path,
        """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<32xi32>
      %fixed = fly.register_value(%x) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>, startIdx = 32 : i32, bitOffset = 0 : i32, storageBits = 1024 : i32} : (vector<32xi32>) -> vector<32xi32>
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
    assert "STACKMAP" not in mir
    assert "ADJCALLSTACK" not in mir
    assert "DBG_VALUE" not in mir


@pytest.mark.parametrize("overlap", [True, False])
def test_two_explicit_allocations_must_not_overlap_while_live(tmp_path, overlap):
    first = """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed_x = fly.register_value(%x) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>, startIdx = 32 : i32, bitOffset = 0 : i32, storageBits = 128 : i32} : (vector<4xi32>) -> vector<4xi32>
    """
    second = """
      %y = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed_y = fly.register_value(%y) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>, startIdx = 32 : i32, bitOffset = 0 : i32, storageBits = 128 : i32} : (vector<4xi32>) -> vector<4xi32>
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
          fly.register_place(%p) {{registerClass = #fly.register_class<"amdgcn", "AGPR_32", 32>, startIdx = 32 : i32}} : (!fly.ptr<f32, register>) -> ()
          fly.register_place(%p) {{registerClass = #fly.register_class<"amdgcn", "AGPR_32", 32>, startIdx = {start} : i32}} : (!fly.ptr<f32, register>) -> ()
          gpu.return
        }}
      }}
    }}"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        with pytest.raises(ir.MLIRError, match=diagnostic):
            PassManager.parse("builtin.module(fly-promote-regmem-to-vectorssa)").run(module.operation)


def test_python_register_place_emits_a_separate_op():
    import flydsl.expr as fx
    from flydsl.experimental import register_allocation as fx_regalloc

    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            tensor = fx.make_rmem_tensor(8, fx.Float16)
            allocation_before = str(tensor.owner)
            assert fx_regalloc.register_place(tensor, fx_regalloc.rocdl.AGPR32, start_idx=32) is None
            assert str(tensor.owner) == allocation_before
            attrs = ir.DictAttr.get({"allocSize": ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 8)})
            ptr = fx.make_ptr(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Register), [], dict_attrs=attrs)
            pointer_before = str(ptr.owner)
            fx_regalloc.register_place(ptr, fx_regalloc.rocdl.VGPR32, start_idx=64)
            assert str(ptr.owner) == pointer_before
            with pytest.raises(ValueError, match="register-memory"):
                fx_regalloc.register_place(fx.Int32(3), fx_regalloc.rocdl.VGPR32, start_idx=0)
            # startIdx is an i32 attribute; a wider index must not wrap to another register.
            with pytest.raises(ValueError, match="int32"):
                fx_regalloc.register_place(tensor, fx_regalloc.rocdl.AGPR32, start_idx=2**32 + 32)
        assert module.operation.verify()
        assert str(module).count("fly.register_place") == 2


def test_unsupported_codegen_pipeline_rejects_instead_of_dropping_placement(tmp_path):
    result = _compile_machine_module(
        tmp_path,
        """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value(%x) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>, startIdx = 32 : i32, bitOffset = 0 : i32, storageBits = 128 : i32} : (vector<4xi32>) -> vector<4xi32>
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
    """,
        opt_level=0,
    )
    assert result.returncode != 0
    assert "explicit register placement requires optimized SelectionDAG codegen" in result.stderr


@pytest.mark.parametrize(
    "descriptor", [("amdgcn", "SGPR_32", 32), ("nvptx64", "Int32Regs", 32), ("x86_64", "GR32", 32)]
)
def test_register_class_representation_is_target_independent(descriptor):
    import flydsl.expr as fx
    from flydsl.experimental import register_allocation as fx_regalloc

    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            tensor = fx.make_rmem_tensor(4, fx.Int32)
            fx_regalloc.register_place(tensor, fx_regalloc.RegisterClass(*descriptor), start_idx=4)
        assert module.operation.verify()
        assert str(ir.Module.parse(str(module))) == str(module)
    assert fx_regalloc.rocdl.SGPR32 == fx_regalloc.RegisterClass("amdgcn", "SGPR_32", 32)


@pytest.mark.parametrize(
    "name,bits,start,diagnostic",
    [
        ("AGPR_32", 32, 255, "exceeds LLVM register class members"),
        ("NoSuchClass", 32, 0, "unsupported AMDGPU register class"),
        ("SReg_64", 64, 0, "unsupported AMDGPU register class"),
        ("SGPR_32", 32, 40, "requires a compatible uniform LLVM register class"),
        ("VGPR_32", 64, 0, "register class width does not match LLVM"),
    ],
)
def test_backend_class_constraints_are_checked(tmp_path, name, bits, start, diagnostic):
    result = _compile_machine_module(
        tmp_path,
        f"""
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value(%x) {{registerClass = #fly.register_class<"amdgcn", "{name}", {bits}>,
        startIdx = {start} : i32, bitOffset = 0 : i32, storageBits = 128 : i32}} : (vector<4xi32>) -> vector<4xi32>
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
      %fixed = fly.register_value(%x) {registerClass = #fly.register_class<"amdgcn", "SGPR_32", 32>,
        startIdx = 40 : i32, bitOffset = 0 : i32, storageBits = 32 : i32} : (i32) -> i32
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
      %fixed = fly.register_value(%x) {registerClass = #fly.register_class<"nvptx64", "Int32Regs", 32>,
        startIdx = 0 : i32, bitOffset = 0 : i32, storageBits = 128 : i32} : (vector<4xi32>) -> vector<4xi32>
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
    """,
    )
    assert result.returncode != 0
    assert "requires target amdgcn" in result.stderr


@pytest.mark.parametrize(
    "start,diagnostic",
    [
        (32, "unavailable on this target"),
        (4, "precolored machine operand"),
        # SGPR_32 members 106/107 are VCC_LO/VCC_HI, which the compiler writes implicitly.
        (106, "which instructions read or write implicitly"),
        (107, "which instructions read or write implicitly"),
    ],
)
def test_sgpr_placement_protects_target_and_abi_registers(tmp_path, start, diagnostic):
    result = _compile_machine_module(
        tmp_path,
        f"""
      %x = llvm.call_intrinsic "llvm.amdgcn.workgroup.id.x"() : () -> i32
      %fixed = fly.register_value(%x) {{registerClass = #fly.register_class<"amdgcn", "SGPR_32", 32>,
        startIdx = {start} : i32, bitOffset = 0 : i32, storageBits = 32 : i32}} : (i32) -> i32
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
      %fixed = fly.register_value(%x) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>,
        startIdx = 64 : i32, bitOffset = 0 : i32, storageBits = 128 : i32} : (vector<4xi32>) -> vector<4xi32>
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
      %fixed = fly.register_value(%x) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>,
        startIdx = 64 : i32, bitOffset = 0 : i32, storageBits = 32 : i32} : (i32) -> i32
      llvm.store volatile %fixed, %p : i32, !llvm.ptr<1>
    """,
            "constant-only placement is unsupported",
        ),
        (
            """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value(%x) {registerClass = #fly.register_class<"amdgcn", "AGPR_32", 32>,
        startIdx = 255 : i32, bitOffset = 0 : i32, storageBits = 128 : i32} : (vector<4xi32>) -> vector<4xi32>
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
        ("VGPR_32", "i32", 32, "requires optimized SelectionDAG codegen", 0),
    ],
)
def test_static_errors_return_to_python(class_name, element_type, storage_bits, diagnostic, opt_level):
    source = f"""module {{
      gpu.module @m [#rocdl.target<O = {opt_level}, chip = "gfx942">] {{
        llvm.func @bad(%x: {element_type}) -> {element_type} {{
          %fixed = fly.register_value(%x) {{registerClass = #fly.register_class<"amdgcn", "{class_name}", 32>,
            startIdx = 0 : i32, bitOffset = 0 : i32, storageBits = {storage_bits} : i32}} : ({element_type}) -> {element_type}
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
          fly.register_place(%p) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>, startIdx = 64 : i32} : (!fly.ptr<i32, register>) -> ()
          gpu.return
        }
      }
    }"""
    with _create_mlir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        PassManager.parse("builtin.module(fly-promote-regmem-to-vectorssa,canonicalize)").run(module.operation)
        assert "fly.register_place" not in str(module)
        assert "fly.register_value" not in str(module)
        assert "fly.make_ptr" not in str(module)


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
        marker = f"""%fixed = fly.register_value(%x) {{registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>,
          startIdx = {start} : i32, bitOffset = 0 : i32, storageBits = 128 : i32}} : (vector<4xi32>) -> vector<4xi32>"""
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


def test_calls_cannot_clobber_reserved_storage(tmp_path):
    body = """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value(%x) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>,
        startIdx = 64 : i32, bitOffset = 0 : i32, storageBits = 128 : i32} : (vector<4xi32>) -> vector<4xi32>
      llvm.call @callee() : () -> ()
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body, declarations="llvm.func @callee()")
    assert result.returncode != 0
    assert "conflicts with a call clobber" in result.stderr, result.stdout + result.stderr
    result = _compile_machine_module(tmp_path, body, declarations="llvm.func @callee()", strip_placement=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fixed_registers_are_reused_outside_the_live_range(tmp_path):
    # The binding covers the fixed value's live range only. After its last
    # read, ordinary allocation may reuse the registers.
    body = """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xi32>
      %fixed = fly.register_value(%x) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>,
        startIdx = 32 : i32, bitOffset = 0 : i32, storageBits = 128 : i32} : (vector<4xi32>) -> vector<4xi32>
      llvm.store volatile %fixed, %p : vector<4xi32>, !llvm.ptr<1>
      %y = llvm.load volatile %p : !llvm.ptr<1> -> vector<64xi32>
      llvm.store volatile %y, %p : vector<64xi32>, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [line.strip() for line in (tmp_path / "final.s").read_text().splitlines()]
    touching = [i for i, line in enumerate(lines) if _vgprs(line) & set(range(32, 36))]
    assert lines[touching[0]].startswith("global_load_dwordx4 v[32:35]"), lines[touching[0]]
    assert lines[touching[1]].startswith("global_store_dwordx4") and "v[32:35]" in lines[touching[1]]
    # The 64-word value is allocated from v0 upwards and reuses v32-v35.
    assert len(touching) > 2


def test_fixed_registers_are_protected_across_a_loop(tmp_path):
    # The fixed value is live across a loop with heavy register pressure.
    # Spills and reloads inside the loop must not use v32-v63.
    body = """
      %x = llvm.load volatile %p : !llvm.ptr<1> -> vector<32xi32>
      %fixed = fly.register_value(%x) {registerClass = #fly.register_class<"amdgcn", "VGPR_32", 32>,
        startIdx = 32 : i32, bitOffset = 0 : i32, storageBits = 1024 : i32} : (vector<32xi32>) -> vector<32xi32>
      %n = llvm.load volatile %p : !llvm.ptr<1> -> i32
      %c0 = llvm.mlir.constant(0 : i32) : i32
      %c1 = llvm.mlir.constant(1 : i32) : i32
      llvm.br ^loop(%c0 : i32)
    ^loop(%i: i32):
      %y = llvm.load volatile %p : !llvm.ptr<1> -> vector<96xi32>
      llvm.store volatile %y, %p : vector<96xi32>, !llvm.ptr<1>
      %i1 = llvm.add %i, %c1 : i32
      %c = llvm.icmp "slt" %i1, %n : i32
      llvm.cond_br %c, ^loop(%i1 : i32), ^exit
    ^exit:
      llvm.store volatile %fixed, %p : vector<32xi32>, !llvm.ptr<1>
    """
    result = _compile_machine_module(tmp_path, body.replace("llvm.return", ""), vgpr_limit=64)
    assert result.returncode == 0, result.stdout + result.stderr
    asm = (tmp_path / "final.s").read_text()
    assert int(re.search(r"\.vgpr_spill_count:\s*(\d+)", asm)[1]) > 0
    touching = [line.strip() for line in asm.splitlines() if _vgprs(line) & set(range(32, 64))]
    # Eight tuple loads before the loop and eight stores after it, nothing else.
    assert len(touching) == 16, touching
    assert sum(line.startswith("global_load_dwordx4") for line in touching) == 8
    assert sum(line.startswith("global_store_dwordx4") for line in touching) == 8


def test_rejected_placement_returns_to_python_and_process_continues(tmp_path):
    source = """module {{
      gpu.module @m [#rocdl.target<O = 3, chip = "gfx942">] {{
        llvm.func @k(%p: !llvm.ptr<1>) attributes {{gpu.kernel, rocdl.kernel}} {{
          %a = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf16>
          %c = llvm.load volatile %p : !llvm.ptr<1> -> vector<4xf32>
          %zero = llvm.mlir.constant(0 : i32) : i32
          %d = llvm.call_intrinsic "llvm.amdgcn.mfma.f32.16x16x16f16"(%a, %a, %c, %zero, %zero, %zero)
            : (vector<4xf16>, vector<4xf16>, vector<4xf32>, i32, i32, i32) -> vector<4xf32>
          %fixed = fly.register_value(%d) {{registerClass = #fly.register_class<"amdgcn", "{bank}_32", 32>,
            startIdx = 64 : i32, bitOffset = 0 : i32, storageBits = 128 : i32}} : (vector<4xf32>) -> vector<4xf32> loc("kernel.py":42:7)
          llvm.store volatile %fixed, %p : vector<4xf32>, !llvm.ptr<1>
          llvm.return
        }}
      }}
    }}"""
    script = f"""
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
from flydsl.compiler.jit_function import _create_mlir_context, _extract_isa_text
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
pipeline = "builtin.module(convert-fly-to-rocdl,rocdl-serialize-register-kernels{{format=isa}},gpu-module-to-binary{{format=isa}})"
with _create_mlir_context(), ir.Location.unknown():
    module = ir.Module.parse({source.format(bank="AGPR")!r})
    try:
        PassManager.parse(pipeline).run(module.operation)
        raise SystemExit("AGPR accumulator placement unexpectedly compiled")
    except ir.MLIRError as exc:
        print("REJECTED:", str(exc).replace("\\n", " "))
    module = ir.Module.parse({source.format(bank="VGPR")!r})
    PassManager.parse(pipeline).run(module.operation)
    print("COMPILED:", [l.strip() for l in _extract_isa_text(str(module)).splitlines() if "v_mfma" in l])
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LLVM ERROR" not in result.stderr
    rejected = next(line for line in result.stdout.splitlines() if line.startswith("REJECTED:"))
    assert '"kernel.py":42:7' in rejected, rejected
    assert "explicit register definition cannot use" in rejected
    assert "v_mfma_f32_16x16x16_f16 v[64:67]" in result.stdout


def _vgprs(line):
    registers = set()
    for first, last, scalar in re.findall(r"\bv(?:\[(\d+):(\d+)\]|(\d+)\b)", line):
        registers.update(range(int(first), int(last) + 1) if first else [int(scalar)])
    return registers
