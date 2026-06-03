# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""End-to-end test for the custom-MIR-codegen JIT path (fly-llc).

Builds a legacy MachineFunction (MIR) pass plugin (``.so``) registering a pass
``fly-mir-pass`` that runs during codegen (pre-emit) and prints the machine
function name.  The kernel is driven through
``@flyc.jit(llvm_codegen_passes=["fly-mir-pass"], llvm_codegen_plugins=[...])`` so
the full chain is exercised:

    device .ll -> fly-llc (--load + --pre-emit-pass) -> obj -> ld.lld -> HSACO
                -> gpu.binary -> splice -> run

Requires a ROCm GPU, ``FLYDSL_COMPILE_LLVM_DIR`` (LLVM headers + llvm-config), a
host C++ compiler, the ``fly-llc`` tool, and ``ld.lld``; skipped otherwise.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

torch = pytest.importorskip("torch")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.compiler.external_llvm import ExternalLLVMError  # noqa: E402

# Legacy MachineFunctionPass plugin: registers "fly-mir-pass" via RegisterPass,
# runs pre-emit during codegen, prints the MF name (observable under `pytest -s`).
PLUGIN_SRC = r"""
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
namespace {
struct FlyMirPass : public MachineFunctionPass {
  static char ID;
  FlyMirPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override {
    errs() << "fly-mir-pass: ran on " << MF.getName() << "\n";
    return false;
  }
  StringRef getPassName() const override { return "Fly demo MIR pass"; }
};
char FlyMirPass::ID = 0;
} // namespace
static RegisterPass<FlyMirPass> X("fly-mir-pass", "Fly demo MIR pass", false, false);
"""

# A MIR pass that *modifies* the machine code: inserts 8 ``s_nop`` at the entry
# of every kernel.  The opcode is found by name so no AMDGPU target headers are
# needed.  8 nops/function is a distinctive, measurable change in the ASM.
PLUGIN_SRC_NOP = r"""
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Pass.h"
using namespace llvm;
namespace {
struct FlyInsertNopPass : public MachineFunctionPass {
  static char ID;
  FlyInsertNopPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    unsigned NopOpc = ~0u;
    for (unsigned i = 0, e = TII->getNumOpcodes(); i < e; ++i)
      if (TII->getName(i) == "S_NOP") { NopOpc = i; break; }
    if (NopOpc == ~0u || MF.empty()) return false;
    MachineBasicBlock &MBB = MF.front();
    auto It = MBB.begin();
    for (int k = 0; k < 8; ++k)
      BuildMI(MBB, It, DebugLoc(), TII->get(NopOpc)).addImm(0);
    return true;
  }
  StringRef getPassName() const override { return "Fly insert NOP MIR pass"; }
};
char FlyInsertNopPass::ID = 0;
} // namespace
static RegisterPass<FlyInsertNopPass> X("fly-insert-nop", "Fly insert NOP", false, false);
"""

# A MIR pass that *schedules* (reorders) instructions: within each block it swaps
# adjacent instructions whenever that is provably safe — neither has memory/side
# effects and no def of one overlaps any register operand (def or use, explicit
# or implicit) of the other.  Semantics are preserved (results stay correct) but
# the emitted instruction order changes.
PLUGIN_SRC_REORDER = r"""
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Pass.h"
using namespace llvm;
namespace {
static bool unsafe(const MachineInstr &MI) {
  return MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects() || MI.isCall() ||
         MI.isTerminator() || MI.isBranch() || MI.isInlineAsm() || MI.isMetaInstruction();
}
static bool canSwap(const MachineInstr &A, const MachineInstr &B, const TargetRegisterInfo *TRI) {
  if (unsafe(A) || unsafe(B)) return false;
  auto defConflicts = [&](const MachineInstr &X, const MachineInstr &Y) {
    for (const MachineOperand &dx : X.operands()) {
      if (!dx.isReg() || !dx.getReg() || !dx.isDef()) continue;
      for (const MachineOperand &oy : Y.operands())
        if (oy.isReg() && oy.getReg() && TRI->regsOverlap(dx.getReg(), oy.getReg())) return true;
    }
    return false;
  };
  return !defConflicts(A, B) && !defConflicts(B, A);
}
struct FlyReorderPass : public MachineFunctionPass {
  static char ID;
  FlyReorderPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override {
    const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
    bool changed = false;
    for (MachineBasicBlock &MBB : MF) {
      for (auto it = MBB.begin(); it != MBB.end();) {
        auto nxt = std::next(it);
        if (nxt != MBB.end() && canSwap(*it, *nxt, TRI)) {
          MBB.splice(it, &MBB, nxt); // move B before A
          changed = true;
          it = std::next(it);        // it still == A; skip past the swapped pair
        } else {
          ++it;
        }
      }
    }
    return changed;
  }
  StringRef getPassName() const override { return "Fly reorder MIR pass"; }
};
char FlyReorderPass::ID = 0;
} // namespace
static RegisterPass<FlyReorderPass> X("fly-reorder", "Fly reorder", false, false);
"""


def _gpu_available() -> bool:
    try:
        from flydsl.runtime.device import get_rocm_device_count

        return get_rocm_device_count() > 0
    except Exception:
        return False


def _resolve_tool(env_var: str, name: str):
    raw = os.environ.get(env_var, "").strip()
    if raw and Path(raw).expanduser().is_file():
        return Path(raw).expanduser()
    llvm_dir = os.environ.get("FLYDSL_COMPILE_LLVM_DIR", "").strip()
    if llvm_dir:
        cand = Path(llvm_dir).expanduser() / "bin" / name
        if cand.is_file():
            return cand
    return None


def _build_codegen_plugin(tmp_path_factory, *, src: str, name: str) -> str:
    """Skip unless the codegen toolchain is present, then compile a plugin .so."""
    raw = os.environ.get("FLYDSL_COMPILE_LLVM_DIR", "").strip()
    if not raw:
        pytest.skip("FLYDSL_COMPILE_LLVM_DIR not set; required to build/load a codegen pass plugin")
    prefix = Path(raw).expanduser().resolve()
    llvm_config = prefix / "bin" / "llvm-config"
    cxx = shutil.which("clang++") or shutil.which("g++")
    if not llvm_config.is_file() or cxx is None:
        pytest.skip("llvm-config or a C++ compiler not available for plugin build")
    if _resolve_tool("FLYDSL_COMPILE_FLY_LLC", "fly-llc") is None:
        pytest.skip("fly-llc not found; set FLYDSL_COMPILE_FLY_LLC or build it into <FLYDSL_COMPILE_LLVM_DIR>/bin")
    if _resolve_tool("FLYDSL_COMPILE_LLD", "ld.lld") is None:
        pytest.skip("ld.lld not found; set FLYDSL_COMPILE_LLD or place ld.lld in <FLYDSL_COMPILE_LLVM_DIR>/bin")

    cxxflags = subprocess.check_output([str(llvm_config), "--cxxflags"], text=True).split()
    work = tmp_path_factory.mktemp("codegen_plugin")
    cpp = work / (name + ".cpp")
    cpp.write_text(src, encoding="utf-8")
    so = work / ("lib" + name + ".so")
    subprocess.run([cxx, "-shared", "-fPIC", *cxxflags, str(cpp), "-o", str(so)], check=True)
    assert so.is_file()
    return str(so)


@pytest.fixture(scope="module")
def mir_pass_plugin(tmp_path_factory) -> str:
    """Compile the print-only MIR pass plugin (and ensure fly-llc + ld.lld exist)."""
    return _build_codegen_plugin(tmp_path_factory, src=PLUGIN_SRC, name="FlyMir")


@pytest.fixture(scope="module")
def nop_pass_plugin(tmp_path_factory) -> str:
    """Compile the s_nop-inserting MIR pass plugin (modifies the machine code)."""
    return _build_codegen_plugin(tmp_path_factory, src=PLUGIN_SRC_NOP, name="FlyNop")


@pytest.fixture(scope="module")
def reorder_pass_plugin(tmp_path_factory) -> str:
    """Compile the instruction-reordering (scheduling) MIR pass plugin."""
    return _build_codegen_plugin(tmp_path_factory, src=PLUGIN_SRC_REORDER, name="FlyReorder")


@flyc.kernel
def _add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, block_dim: fx.Constexpr[int]):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    A = fx.rocdl.make_buffer_tensor(A)
    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))
    tC = fx.logical_divide(tC, fx.make_layout(1, 1))
    ca = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    cab = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    rA = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    rB = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    rC = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    fx.copy_atom_call(cab, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(ca, fx.slice(tB, (None, tid)), rB)
    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)
    fx.copy_atom_call(ca, rC, fx.slice(tC, (None, tid)))


def _make_add_jit(**jit_kwargs):
    @flyc.jit(**jit_kwargs)
    def add(A: fx.Tensor, B: fx.Tensor, C, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        block_dim = 64
        grid_x = (n + block_dim - 1) // block_dim
        _add_kernel(A, B, C, block_dim).launch(grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream)

    return add


@pytest.mark.skipif(not _gpu_available(), reason="requires a ROCm GPU")
def test_codegen_mir_pass_compiles_and_runs(mir_pass_plugin, monkeypatch):
    """Positive: a custom MIR pass injected pre-emit via fly-llc; the kernel
    codegens through fly-llc + ld.lld, links, and runs correctly.

    The pass's ``fly-mir-pass: ran on ...`` line is printed to stderr during
    codegen; run with ``pytest -s`` to view it."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    add = _make_add_jit(llvm_codegen_passes=["fly-mir-pass"], llvm_codegen_plugins=[mir_pass_plugin])

    n = 64
    A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    C = torch.zeros(n, dtype=torch.float32).cuda()
    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
    add(tA, B, C, n, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    assert torch.allclose(C, A + B)


def test_codegen_unknown_mir_pass_fails(mir_pass_plugin, monkeypatch):
    """Negative: naming the MIR pass without loading the plugin must fail in
    fly-llc (unknown pass) — proving the plugin provides the pass."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    add = _make_add_jit(llvm_codegen_passes=["fly-mir-pass"])  # no plugins

    n = 64
    A = torch.zeros(n, dtype=torch.float32)
    B = torch.zeros(n, dtype=torch.float32)
    C = torch.zeros(n, dtype=torch.float32)
    if _gpu_available():
        A, B, C = A.cuda(), B.cuda(), C.cuda()
    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)

    with pytest.raises(ExternalLLVMError) as excinfo:
        add(tA, B, C, n, stream=torch.cuda.Stream() if _gpu_available() else fx.Stream(None))
    assert "fly-mir-pass" in str(excinfo.value)


def _max_entry_nop_run(disasm: str) -> int:
    """Largest run of leading ``s_nop`` immediately after any ``<func>:`` label.

    Anchoring at the function entry avoids counting trailing alignment padding
    (which the disassembler also renders as ``s_nop``) — the injected sled lands
    at the entry, so this isolates the pass's effect."""
    import re

    lines = disasm.splitlines()
    best = 0
    for i, ln in enumerate(lines):
        if re.match(r"^[0-9a-fA-F]+ <.+>:", ln):
            run = 0
            for j in range(i + 1, len(lines)):
                if "s_nop" in lines[j]:
                    run += 1
                elif lines[j].strip() == "":
                    continue
                else:
                    break
            best = max(best, run)
    return best


def _disasm(objdump: Path, mcpu: str, obj: Path) -> str:
    return subprocess.check_output([str(objdump), "-d", f"--mcpu={mcpu}", str(obj)], text=True)


def _unescape_mlir_bytes(s: str) -> bytes:
    """Decode an MLIR string-attribute body (``\\XX`` hex + ``\\\\`` / ``\\"``)."""
    out = bytearray()
    i, n = 0, len(s)
    while i < n:
        if s[i] == "\\":
            nxt = s[i + 1]
            if nxt in ("\\", '"'):
                out.append(ord(nxt))
                i += 2
            elif nxt == "n":
                out.append(0x0A)
                i += 2
            elif nxt == "t":
                out.append(0x09)
                i += 2
            else:
                out.append(int(s[i + 1 : i + 3], 16))
                i += 3
        else:
            out.append(ord(s[i]))
            i += 1
    return bytes(out)


def _extract_gpu_binary(dump_dir: Path) -> bytes:
    """Pull the embedded device HSACO bytes out of a dumped ``gpu.binary`` op."""
    import re

    for mlir in sorted(dump_dir.rglob("*.mlir")):
        txt = mlir.read_text(encoding="utf-8", errors="replace")
        if "gpu.binary" not in txt:
            continue
        m = re.search(r'bin = "((?:[^"\\]|\\.)*)"', txt, re.S)
        if m:
            return _unescape_mlir_bytes(m.group(1))
    raise AssertionError(f"no gpu.binary found under {dump_dir}")


def _jit_run_add(add, dump: Path, monkeypatch) -> None:
    """Compile+run the add kernel into *dump*, asserting the result is correct."""
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump))
    n = 64
    A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    C = torch.zeros(n, dtype=torch.float32).cuda()
    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
    add(tA, B, C, n, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    assert torch.allclose(C, A + B)  # the codegen pass must preserve correctness


def _kernel_instr_seq(disasm: str) -> list:
    """Ordered list of disassembled instructions (mnemonic + operands, with the
    trailing ``// addr: encoding`` comment stripped) across all functions."""
    import re

    seq = []
    in_func = False
    for ln in disasm.splitlines():
        if re.match(r"^[0-9a-fA-F]+ <.+>:", ln):
            in_func = True
            continue
        if in_func and "\t" in ln:
            ins = ln.split("//")[0].strip()
            if ins and not ins.startswith("."):
                seq.append(ins)
    return seq


@pytest.mark.skipif(not _gpu_available(), reason="requires a ROCm GPU")
def test_codegen_pass_modifies_asm(nop_pass_plugin, monkeypatch, tmp_path):
    """A codegen pass can change the emitted ASM.  Compile the same kernel through
    the JIT twice — with the s_nop-inserting pass and without it — and disassemble
    the device binary each produced.  Only the with-pass binary begins with the
    NOP_PER_FUNC sled at the function entry, proving the pass modified the ASM."""
    objdump = _resolve_tool("FLYDSL_COMPILE_LLVM_OBJDUMP", "llvm-objdump")
    if objdump is None:
        pytest.skip("llvm-objdump not found; set FLYDSL_COMPILE_LLVM_OBJDUMP or place it in <llvm_dir>/bin")

    from flydsl.compiler.backends.rocm import RocmBackend

    mcpu = RocmBackend.detect_target().arch
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")

    mod_dump = tmp_path / "mod"
    base_dump = tmp_path / "base"
    add_mod = _make_add_jit(llvm_codegen_passes=["fly-insert-nop"], llvm_codegen_plugins=[nop_pass_plugin])
    _jit_run_add(add_mod, mod_dump, monkeypatch)
    _jit_run_add(_make_add_jit(), base_dump, monkeypatch)  # baseline: no codegen pass

    mod_hsaco = next(mod_dump.rglob("fly_llc.hsaco"), None)
    assert mod_hsaco is not None, "fly-llc HSACO dump not found"
    base_hsaco = tmp_path / "base.hsaco"
    base_hsaco.write_bytes(_extract_gpu_binary(base_dump))

    # The pass injects an NOP_PER_FUNC sled at each function entry; the baseline
    # kernel does not begin with s_nop.  (Total s_nop count / object size are NOT
    # reliable: the entry sled merely displaces trailing alignment-padding nops.)
    base_run = _max_entry_nop_run(_disasm(objdump, mcpu, base_hsaco))
    mod_run = _max_entry_nop_run(_disasm(objdump, mcpu, mod_hsaco))

    assert mod_run > base_run, f"codegen pass did not modify the ASM: entry s_nop run base={base_run} mod={mod_run}"


@pytest.mark.skipif(not _gpu_available(), reason="requires a ROCm GPU")
def test_codegen_pass_reorders_instructions(reorder_pass_plugin, mir_pass_plugin, monkeypatch, tmp_path):
    """A custom codegen *scheduling* pass can reorder instructions.  Both sides go
    through the same fly-llc codegen driver (the baseline uses the no-op print
    pass, so the *only* difference is the reordering — not regalloc/ISel that would
    differ across codegen drivers).  The reorder run must (a) keep results correct,
    (b) emit the *same multiset* of instructions (pure reorder), yet (c) in a
    *different order*."""
    objdump = _resolve_tool("FLYDSL_COMPILE_LLVM_OBJDUMP", "llvm-objdump")
    if objdump is None:
        pytest.skip("llvm-objdump not found; set FLYDSL_COMPILE_LLVM_OBJDUMP or place it in <llvm_dir>/bin")

    from flydsl.compiler.backends.rocm import RocmBackend

    mcpu = RocmBackend.detect_target().arch
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")

    mod_dump = tmp_path / "mod"
    base_dump = tmp_path / "base"
    add_mod = _make_add_jit(llvm_codegen_passes=["fly-reorder"], llvm_codegen_plugins=[reorder_pass_plugin])
    add_base = _make_add_jit(llvm_codegen_passes=["fly-mir-pass"], llvm_codegen_plugins=[mir_pass_plugin])
    _jit_run_add(add_mod, mod_dump, monkeypatch)  # correctness preserved under reordering
    _jit_run_add(add_base, base_dump, monkeypatch)  # same fly-llc driver, no-op pass

    mod_hsaco = next(mod_dump.rglob("fly_llc.hsaco"), None)
    base_hsaco = next(base_dump.rglob("fly_llc.hsaco"), None)
    assert mod_hsaco is not None and base_hsaco is not None, "fly-llc HSACO dump not found"

    base_seq = _kernel_instr_seq(_disasm(objdump, mcpu, base_hsaco))
    mod_seq = _kernel_instr_seq(_disasm(objdump, mcpu, mod_hsaco))
    assert base_seq and mod_seq, "no instructions disassembled"
    assert sorted(base_seq) == sorted(mod_seq), "reorder must not add or remove instructions"
    assert base_seq != mod_seq, "scheduling pass did not change instruction order"
