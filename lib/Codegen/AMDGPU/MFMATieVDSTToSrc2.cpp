//===-- MFMATieVDSTToSrc2.cpp -----------------------------------------------===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// For MAI/MFMA (and SMFMAC) instructions whose vdst register class is exactly
// 128 bits wide, tie the vdst def to the src2 use at the MachineInstr level.
// This is the runtime-pass equivalent of changing
//
//   bit NoDstOverlap = !gt(DstVT.Size, 128);   // strictly >
//
// to
//
//   bit NoDstOverlap = !ge(DstVT.Size, 128);   // >=
//   bit HasEarlyClobberForm = !gt(DstVT.Size, 128);
//
// inside VOPProfileMAI in VOP3PInstructions.td.  TableGen's effect of the
// `>=` form is to add a Constraints = "$vdst = $src2" string on 128-bit-dst
// MFMAs (without giving them the early-clobber form reserved for >128-bit-dst
// variants).  We achieve the same regalloc outcome by setting the IsTied flag
// on the operands directly, which TwoAddressInstructionPass and
// RegisterCoalescer honor.
//
// Why a pass instead of editing TableGen: keeps FlyDSL's local LLVM fork
// non-existent — the .td edit would touch a hot upstream file and would
// require an LLVM source patch.  The pass is gated by
// -fly-amdgpu-tie-vdst-src2-for-mfma-128 (default true).
//
//===----------------------------------------------------------------------===//

#include "flydsl/Codegen/AMDGPU/MFMATieVDSTToSrc2.h"

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "fly-amdgpu-mfma-tie-vdst-src2"

static cl::opt<bool> EnableTieVDSTSrc2ForMFMA128(
    "fly-amdgpu-tie-vdst-src2-for-mfma-128",
    cl::desc("(FlyDSL) Tie vdst to src2 on 128-bit-dst MFMA instructions "
             "(equivalent to VOPProfileMAI NoDstOverlap >= 128 td edit)."),
    cl::init(true), cl::Hidden);

namespace {

class FlyAMDGPUMFMATieVDSTToSrc2Impl {
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  MachineRegisterInfo &MRI;

public:
  FlyAMDGPUMFMATieVDSTToSrc2Impl(const TargetSubtargetInfo &ST,
                                 MachineRegisterInfo &MRI)
      : TII(*ST.getInstrInfo()), TRI(*ST.getRegisterInfo()), MRI(MRI) {}

  bool run(MachineFunction &MF);

private:
  // True when MI is an MAI/MFMA instruction whose vdst register class is
  // exactly 128 bits.  We identify MFMA without SIInstrInfo::isMAI by
  // looking at the MCID name prefix — every MAI mnemonic in AMDGPU
  // TableGen begins with "V_MFMA_" or "V_SMFMAC_".
  bool isMFMAWith128BitDst(const MachineInstr &MI, int VDstIdx) const;
};

class FlyAMDGPUMFMATieVDSTToSrc2Legacy : public MachineFunctionPass {
public:
  static char ID;

  FlyAMDGPUMFMATieVDSTToSrc2Legacy() : MachineFunctionPass(ID) {
    initializeFlyAMDGPUMFMATieVDSTToSrc2LegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "FlyDSL AMDGPU MFMA Tie vdst to src2 (128-bit)";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(FlyAMDGPUMFMATieVDSTToSrc2Legacy, DEBUG_TYPE,
                      "FlyDSL AMDGPU MFMA Tie vdst to src2 (128-bit)", false,
                      false)
INITIALIZE_PASS_END(FlyAMDGPUMFMATieVDSTToSrc2Legacy, DEBUG_TYPE,
                    "FlyDSL AMDGPU MFMA Tie vdst to src2 (128-bit)", false,
                    false)

char FlyAMDGPUMFMATieVDSTToSrc2Legacy::ID = 0;

char &llvm::FlyAMDGPUMFMATieVDSTToSrc2LegacyID =
    FlyAMDGPUMFMATieVDSTToSrc2Legacy::ID;

bool FlyAMDGPUMFMATieVDSTToSrc2Legacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  return FlyAMDGPUMFMATieVDSTToSrc2Impl(MF.getSubtarget(), MF.getRegInfo())
      .run(MF);
}

PreservedAnalyses
FlyAMDGPUMFMATieVDSTToSrc2Pass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &MFAM) {
  FlyAMDGPUMFMATieVDSTToSrc2Impl(MF.getSubtarget(), MF.getRegInfo()).run(MF);
  return PreservedAnalyses::all();
}

bool FlyAMDGPUMFMATieVDSTToSrc2Impl::isMFMAWith128BitDst(const MachineInstr &MI,
                                                        int VDstIdx) const {
  // Identify MFMA / SMFMAC by mnemonic prefix.  We can't call SIInstrInfo::isMAI
  // (that header isn't installed by LLVM), but the AMDGPU TableGen mnemonics
  // for MAI instructions have used these prefixes since gfx908 and are stable.
  StringRef Name = TII.getName(MI.getOpcode());
  if (!Name.starts_with("V_MFMA_") && !Name.starts_with("V_SMFMAC_"))
    return false;

  // Public TII.getRegClass(MCID, OpNum) returns the operand's register
  // class even before vreg assignment, which is what we need at PreRegAlloc
  // time.  (Older LLVMs took (MCID, OpNum, TRI, MF); recent versions
  // simplified the signature.)
  const MCInstrDesc &MCID = MI.getDesc();
  const TargetRegisterClass *RC =
      TII.getRegClass(MCID, static_cast<unsigned>(VDstIdx));
  if (!RC)
    return false;
  return TRI.getRegSizeInBits(*RC) == 128;
}

// MFMA operand naming on AMDGPU: vdst is operand 0; src2 follows src0 / src1
// (and any modifier operands).  Rather than calling AMDGPU::getNamedOperandIdx
// (which lives in an internal header), we use the canonical operand layout for
// MAI: [vdst, src0, src1, src2, ...modifiers].  This matches the layout in
// VOP3PInstructions.td and has been stable for many years.
static int getVDstIdx(const MachineInstr &) { return 0; }
static int getSrc2IdxForMFMA(const MachineInstr &MI) {
  // Defs come first; the first 3 reg uses are src0, src1, src2.
  unsigned NumDefs = MI.getDesc().getNumDefs();
  unsigned UseStart = NumDefs;
  // src2 is the 3rd reg use slot.
  unsigned NumOperands = MI.getNumOperands();
  unsigned RegUseSeen = 0;
  for (unsigned I = UseStart; I < NumOperands; ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg())
      continue;
    if (RegUseSeen == 2)
      return static_cast<int>(I);
    ++RegUseSeen;
  }
  return -1;
}

bool FlyAMDGPUMFMATieVDSTToSrc2Impl::run(MachineFunction &MF) {
  if (!EnableTieVDSTSrc2ForMFMA128)
    return false;

  // MFMA / SMFMAC only exist on gfx90a+.  CPU-test (not feature-string-test)
  // because getFeatureString() only returns user-supplied -mattr= flags;
  // implicit features from FeatureISAVersionXxx_Common bundles wouldn't
  // appear there.  On older chips the loop below would no-op anyway, but
  // bailing out early skips the scan.
  StringRef CPU = MF.getSubtarget().getCPU();
  if (CPU != "gfx90a" && CPU != "gfx940" && CPU != "gfx941" &&
      CPU != "gfx942" && CPU != "gfx950")
    return false;

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      const int VDstIdx = getVDstIdx(MI);
      if (VDstIdx < 0)
        continue;
      if (!isMFMAWith128BitDst(MI, VDstIdx))
        continue;
      const int Src2Idx = getSrc2IdxForMFMA(MI);
      if (Src2Idx < 0)
        continue;

      MachineOperand &Src2 = MI.getOperand(Src2Idx);
      if (!Src2.isReg() || Src2.isTied())
        continue;

      MI.tieOperands(VDstIdx, Src2Idx);
      Changed = true;
    }
  }

  return Changed;
}
