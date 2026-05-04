//===-- PreferAGPRForDSRead.cpp ---------------------------------------------===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Forces DS load destinations marked NonTemporal (or all DS loads, if the
// -fly-amdgpu-prefer-agpr-for-ds-read cl::opt is on) to be allocated into
// AccVGPR (AGPR) physical registers, then propagates the constraint through
// COPY/PHI chains so that loop-carried iter_args also stay in AGPR across
// loop boundaries.
//
// Compared with the prior in-tree LLVM patch this version uses the public
// MachineRegisterInfo::constrainRegClass(VirtReg, AGPR_RC) API instead of the
// custom AMDGPURI::PreferAGPR hint type — the hint-type approach required a
// new enum value plus a new case in SIRegisterInfo::getRegAllocationHints,
// which would have meant editing LLVM source.  constrainRegClass tightens the
// virtual register's class so that RegAllocGreedy can only pick AGPR phys
// regs, achieving the same end without touching LLVM.
//
//===----------------------------------------------------------------------===//

#include "flydsl/Codegen/AMDGPU/PreferAGPRForDSRead.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "fly-amdgpu-prefer-agpr-for-ds-read"

static cl::opt<bool> EnablePreferAGPRForDSRead(
    "fly-amdgpu-prefer-agpr-for-ds-read",
    cl::desc("(FlyDSL) Set AGPR allocation hints on ALL DS read destinations, "
             "not just nontemporal ones."),
    cl::init(false), cl::Hidden);

namespace {

class FlyAMDGPUPreferAGPRForDSReadImpl {
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  MachineRegisterInfo &MRI;

  bool tryConstrainToAGPR(Register VirtReg, DenseSet<Register> &Constrained);

public:
  FlyAMDGPUPreferAGPRForDSReadImpl(const TargetSubtargetInfo &ST,
                                   MachineRegisterInfo &MRI)
      : TII(*ST.getInstrInfo()), TRI(*ST.getRegisterInfo()), MRI(MRI) {}

  bool run(MachineFunction &MF);
};

class FlyAMDGPUPreferAGPRForDSReadLegacy : public MachineFunctionPass {
public:
  static char ID;

  FlyAMDGPUPreferAGPRForDSReadLegacy() : MachineFunctionPass(ID) {
    initializeFlyAMDGPUPreferAGPRForDSReadLegacyPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "FlyDSL AMDGPU Prefer AGPR For DS Read";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(FlyAMDGPUPreferAGPRForDSReadLegacy, DEBUG_TYPE,
                      "FlyDSL AMDGPU Prefer AGPR For DS Read", false, false)
INITIALIZE_PASS_END(FlyAMDGPUPreferAGPRForDSReadLegacy, DEBUG_TYPE,
                    "FlyDSL AMDGPU Prefer AGPR For DS Read", false, false)

char FlyAMDGPUPreferAGPRForDSReadLegacy::ID = 0;

char &llvm::FlyAMDGPUPreferAGPRForDSReadLegacyID =
    FlyAMDGPUPreferAGPRForDSReadLegacy::ID;

bool FlyAMDGPUPreferAGPRForDSReadLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  return FlyAMDGPUPreferAGPRForDSReadImpl(MF.getSubtarget(), MF.getRegInfo())
      .run(MF);
}

PreservedAnalyses
FlyAMDGPUPreferAGPRForDSReadPass::run(MachineFunction &MF,
                                      MachineFunctionAnalysisManager &MFAM) {
  FlyAMDGPUPreferAGPRForDSReadImpl(MF.getSubtarget(), MF.getRegInfo()).run(MF);
  return PreservedAnalyses::all();
}

// ---------------------------------------------------------------------------
// AMDGPU-internal helpers we reach for via plain extern symbols rather than by
// pulling in lib/Target/AMDGPU/ headers (which aren't installed by LLVM).
//
// We only need:
//   - SIRegisterInfo::isAGPRClass  (static-ish helper, exposed via a C++ name)
//   - SIRegisterInfo::getEquivalentAGPRClass
//   - GCNSubtarget::hasGFX90AInsts
//
// To stay header-clean we duck-type via the generic TargetRegisterInfo /
// TargetSubtargetInfo APIs and feature-test using the subtarget's features
// string.  This costs a couple of extra string lookups per kernel-compile,
// which is negligible.
// ---------------------------------------------------------------------------

namespace {

/// Is `RC` an AGPR-class register class on AMDGPU?
/// We can't include SIRegisterInfo.h, but the AMDGPU TableGen reg-class
/// identifiers are stable: pure AGPR classes are named `AGPR_*` or `AReg_*`,
/// and the AGPR/VGPR mixed classes are named `AV_*` (which we explicitly do
/// NOT count — pinning to AV_* would let RegAllocGreedy fall back to VGPR).
bool isAGPRClass(const TargetRegisterInfo &TRI, const TargetRegisterClass *RC) {
  if (!RC)
    return false;
  StringRef Name = TRI.getRegClassName(RC);
  if (Name.starts_with("AV_"))
    return false;
  return Name.starts_with("AGPR_") || Name.starts_with("AReg_");
}

/// Find an AGPR-class equivalent of `VRC`, or nullptr if none exists.
/// Walks the TRI's register class list and picks the first AGPR class with
/// the same total bit width as VRC.  Equivalent in spirit to
/// SIRegisterInfo::getEquivalentAGPRClass without needing its header.
const TargetRegisterClass *
findEquivalentAGPRClass(const TargetRegisterInfo &TRI,
                        const TargetRegisterClass *VRC) {
  if (!VRC)
    return nullptr;
  if (isAGPRClass(TRI, VRC))
    return VRC;
  unsigned Bits = TRI.getRegSizeInBits(*VRC);
  for (unsigned I = 0, E = TRI.getNumRegClasses(); I < E; ++I) {
    const TargetRegisterClass *Candidate = TRI.getRegClass(I);
    if (!isAGPRClass(TRI, Candidate))
      continue;
    if (TRI.getRegSizeInBits(*Candidate) == Bits)
      return Candidate;
  }
  return nullptr;
}

/// gfx90a / gfx940 / gfx941 / gfx942 / gfx950 all expose AGPRs.  We CPU-test
/// instead of feature-testing because getFeatureString() only returns
/// user-supplied -mattr= flags, not the resolved feature bits, so
/// `find("gfx940-insts")` would miss CPUs that get the feature implicitly
/// from their FeatureISAVersionXxx_Common bundle.
bool subtargetHasAGPRs(const MachineFunction &MF) {
  StringRef CPU = MF.getSubtarget().getCPU();
  return CPU == "gfx90a" || CPU == "gfx940" || CPU == "gfx941" ||
         CPU == "gfx942" || CPU == "gfx950";
}

} // namespace

bool FlyAMDGPUPreferAGPRForDSReadImpl::tryConstrainToAGPR(
    Register VirtReg, DenseSet<Register> &Constrained) {
  if (Constrained.count(VirtReg))
    return false;

  const TargetRegisterClass *CurRC = MRI.getRegClass(VirtReg);
  if (isAGPRClass(TRI, CurRC)) {
    Constrained.insert(VirtReg);
    return false;
  }

  const TargetRegisterClass *ARC = findEquivalentAGPRClass(TRI, CurRC);
  if (!ARC)
    return false;

  const TargetRegisterClass *NewRC = MRI.constrainRegClass(VirtReg, ARC);
  if (!NewRC)
    return false;

  Constrained.insert(VirtReg);
  return true;
}

bool FlyAMDGPUPreferAGPRForDSReadImpl::run(MachineFunction &MF) {
  if (!subtargetHasAGPRs(MF))
    return false;

  DenseSet<Register> AGPRConstrained;
  bool Changed = false;

  // Phase 1: Constrain nontemporal DS load destinations to AGPR.
  // We identify "DS load" via the MCID memory-operand category — checking
  // mayLoad() and the memory operand's address space (3 = local) avoids
  // needing SIInstrInfo::isDS.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!MI.mayLoad())
        continue;

      bool IsDS = false;
      bool MarkedNonTemporal = false;
      for (const MachineMemOperand *MMO : MI.memoperands()) {
        if (MMO->getAddrSpace() == 3 /* AMDGPUAS::LOCAL_ADDRESS */)
          IsDS = true;
        if (MMO->isNonTemporal())
          MarkedNonTemporal = true;
      }
      if (!IsDS)
        continue;

      if (!MarkedNonTemporal && !EnablePreferAGPRForDSRead)
        continue;

      for (MachineOperand &MO : MI.defs()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        if (tryConstrainToAGPR(MO.getReg(), AGPRConstrained))
          Changed = true;
        break;
      }
    }
  }

  if (AGPRConstrained.empty())
    return Changed;

  // Phase 2: Propagate AGPR constraints through COPY and PHI chains so
  // loop-carried iter_args don't get widened back to AV_* by the coalescer.
  bool Propagated = true;
  while (Propagated) {
    Propagated = false;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!MI.isPHI() && !MI.isCopy())
          continue;

        Register DestReg = MI.getOperand(0).getReg();
        if (!DestReg.isVirtual())
          continue;

        SmallVector<Register, 8> SrcRegs;
        if (MI.isCopy()) {
          Register Src = MI.getOperand(1).getReg();
          if (Src.isVirtual())
            SrcRegs.push_back(Src);
        } else {
          for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
            Register Src = MI.getOperand(i).getReg();
            if (Src.isVirtual())
              SrcRegs.push_back(Src);
          }
        }

        bool AnyAGPR = AGPRConstrained.count(DestReg);
        for (Register Src : SrcRegs) {
          if (AGPRConstrained.count(Src)) {
            AnyAGPR = true;
            break;
          }
        }
        if (!AnyAGPR)
          continue;

        if (tryConstrainToAGPR(DestReg, AGPRConstrained)) {
          Propagated = true;
          Changed = true;
        }
        for (Register Src : SrcRegs) {
          if (tryConstrainToAGPR(Src, AGPRConstrained)) {
            Propagated = true;
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}
