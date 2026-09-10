// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Experimental explicit register allocation. Identity calls keep allocation
// boundaries visible through IR optimization. Immediately before ISel, convert
// them to stackmaps with real value operands. Consume these before scheduling;
// no stackmap records, debug carriers, inline assembly, or LLVM patches are used.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/RegisterTargetPassConfigCallback.h"
#include <cstdlib>
#include <memory>

using namespace llvm;

namespace {
constexpr StringLiteral MarkerPrefix = "__flydsl_register_value_";
// Shared only by the passes of one TargetMachine codegen invocation. Existing
// user stackmap IDs are excluded when allocating our IDs. No magic prefix or
// process-global placement state can accidentally claim an unrelated stackmap.
struct Carrier {
  unsigned ClassID;
  int64_t BitOffset;
  unsigned Words;
};
struct PlacementState {
  DenseMap<uint64_t, Carrier> Carriers;
  DenseSet<const Function *> Functions;
};
using SharedPlacementState = std::shared_ptr<PlacementState>;

const Carrier *getCarrier(const MachineInstr &MI, const PlacementState &State) {
  if (MI.getOpcode() != TargetOpcode::STACKMAP || MI.getNumOperands() < 2 ||
      !MI.getOperand(0).isImm())
    return nullptr;
  auto It = State.Carriers.find(uint64_t(MI.getOperand(0).getImm()));
  return It == State.Carriers.end() ? nullptr : &It->second;
}

// Decode LLVM's documented STACKMAP payload once for both machine passes.
// A null operand denotes a constant: no register exists at this boundary.
void forEachCarrierWord(const MachineInstr &MI, const Carrier &C,
                        function_ref<void(unsigned, const MachineOperand *)> Visit) {
  unsigned Word = 0;
  for (unsigned I = 2; I < MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg() && MO.isImplicit())
      continue;
    if (Word >= C.Words)
      report_fatal_error("FlyDSL register carrier contains too many words");
    if (MO.isImm() && MO.getImm() == StackMaps::ConstantOp) {
      if (++I >= MI.getNumOperands() || !MI.getOperand(I).isImm())
        report_fatal_error("malformed constant in FlyDSL register carrier");
      Visit(Word++, nullptr);
    } else {
      if (!MO.isReg() || !MO.getReg().isVirtual())
        report_fatal_error("FlyDSL register carrier lost its virtual register operand");
      Visit(Word++, &MO);
    }
  }
  if (Word != C.Words)
    report_fatal_error("FlyDSL register carrier lost words during instruction selection");
}

const TargetRegisterClass &resolveRegisterClass(const TargetRegisterInfo &TRI, StringRef Name) {
  for (const auto &RC : TRI.regclasses())
    if (Name == TRI.getRegClassName(&RC))
      return RC;
  report_fatal_error(Twine("unknown LLVM register class: ") + Name);
}

MCRegister registerAt(const TargetRegisterClass &RC, unsigned Index) {
  if (Index >= RC.getNumRegs())
    report_fatal_error("explicit allocation exceeds LLVM register class members");
  return RC.getRegister(Index);
}

// Match tuples through LLVM's subregister relationships, never target spelling.
MCRegister findRegisterTuple(const TargetRegisterInfo &TRI, const TargetRegisterClass &RC,
                             unsigned First, unsigned Bits) {
  unsigned UnitBits = TRI.getRegSizeInBits(RC);
  unsigned Count = divideCeil(Bits, UnitBits);
  MCRegister Leaf = registerAt(RC, First);
  for (MCRegister Candidate : TRI.superregs_inclusive(Leaf)) {
    const auto *CandidateRC = TRI.getMinimalPhysRegClass(Candidate);
    if (!CandidateRC || TRI.getRegSizeInBits(*CandidateRC) != Bits)
      continue;
    bool Matches = true;
    for (unsigned I = 0; I < Count; ++I) {
      MCRegister Part = registerAt(RC, First + I);
      if (Count == 1 && Candidate == Part)
        continue;
      unsigned Sub = TRI.getSubRegIndex(Candidate, Part);
      if (!Sub || TRI.getSubRegIdxOffset(Sub) != I * UnitBits ||
          TRI.getSubRegIdxSize(Sub) != UnitBits) {
        Matches = false;
        break;
      }
    }
    if (Matches)
      return Candidate;
  }
  report_fatal_error("no LLVM physical register tuple for requested class, index and width");
}

class PrepareRegisterPlacement : public ModulePass {
  TargetMachine &TM;
  SharedPlacementState State;

public:
  static char ID;
  PrepareRegisterPlacement(TargetMachine &TM, SharedPlacementState State)
      : ModulePass(ID), TM(TM), State(std::move(State)) {}
  StringRef getPassName() const override { return "FlyDSL prepare register placement"; }
  bool runOnModule(Module &M) override {
    State->Carriers.clear();
    State->Functions.clear();
    DenseSet<uint64_t> UsedIDs;
    SmallVector<CallInst *> Calls;
    for (Function &F : M)
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *CI = dyn_cast<CallInst>(&I)) {
            Function *Callee = CI->getCalledFunction();
            if (!Callee)
              continue;
            if (Callee->getIntrinsicID() == Intrinsic::experimental_stackmap)
              UsedIDs.insert(cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue());
            if (Callee->getName().starts_with(MarkerPrefix) &&
                Callee->hasFnAttribute("flydsl-register-class"))
              Calls.push_back(CI);
          }
    if (Calls.empty())
      return false;
    if (TM.getOptLevel() == CodeGenOptLevel::None || TM.Options.EnableGlobalISel ||
        TM.Options.EnableFastISel)
      report_fatal_error("explicit register placement requires optimized SelectionDAG codegen");

    auto *StackMap = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::experimental_stackmap);
    uint64_t NextID = 0;
    for (CallInst *CI : Calls) {
      Function &F = *CI->getFunction();
      State->Functions.insert(&F);
      Function &Marker = *CI->getCalledFunction();
      if (Marker.getFnAttribute("flydsl-register-target").getValueAsString() !=
          TM.getTargetTriple().getArchName())
        report_fatal_error("register class target does not match LLVM target architecture");
      const auto &TRI = *TM.getSubtargetImpl(F)->getRegisterInfo();
      StringRef Name = Marker.getFnAttribute("flydsl-register-class").getValueAsString();
      const auto &RC = resolveRegisterClass(TRI, Name);
      // The representation is generic; this backend currently supports these
      // three base classes only. Reject other classes instead of guessing.
      if (Name != "VGPR_32" && Name != "AGPR_32" && Name != "SGPR_32")
        report_fatal_error("unsupported LLVM register class for AMDGPU explicit placement");
      unsigned UnitBits = TRI.getRegSizeInBits(RC);
      uint64_t Start = cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
      uint64_t Offset = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      uint64_t StorageBits = cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue();
      uint64_t Bits = M.getDataLayout().getTypeSizeInBits(CI->getType());
      if (UnitBits != 32)
        report_fatal_error("unsupported register class width in FlyDSL carrier");
      if (Start >= RC.getNumRegs() || !StorageBits ||
          StorageBits > uint64_t(RC.getNumRegs() - Start) * UnitBits)
        report_fatal_error("explicit allocation exceeds LLVM register class members");
      if (!Bits || Bits % UnitBits || Offset % UnitBits || Offset > StorageBits ||
          Bits > StorageBits - Offset)
        report_fatal_error(
            "explicit register slices must cover whole 32-bit registers within storage");
      uint64_t BitOffset = Start * UnitBits + Offset;
      IRBuilder<> B(CI);
      Value *V = CI->getArgOperand(0);
      unsigned Count = Bits / 32;
      Type *WordsTy = Count == 1 ? static_cast<Type *>(B.getInt32Ty())
                                 : FixedVectorType::get(B.getInt32Ty(), Count);
      Value *Words = B.CreateBitCast(V, WordsTy);
      while (UsedIDs.contains(NextID))
        ++NextID;
      uint64_t ID = NextID++;
      UsedIDs.insert(ID);
      State->Carriers.try_emplace(ID, Carrier{RC.getID(), int64_t(BitOffset), Count});
      SmallVector<Value *> Args{B.getInt64(ID), B.getInt32(0)};
      for (unsigned I = 0; I < Count; ++I)
        Args.push_back(Count == 1 ? Words : B.CreateExtractElement(Words, B.getInt32(I)));
      B.CreateCall(StackMap, Args);
      CI->replaceAllUsesWith(V);
      CI->eraseFromParent();
    }
    return true;
  }
};
char PrepareRegisterPlacement::ID;

// Reserve before liveness/coalescing/RA, so every analysis and allocator sees
// the same register set. Never mutate reservedRegs after LiveIntervals exists.
class ReserveRegisterPlacement : public MachineFunctionPass {
  SharedPlacementState State;

public:
  static char ID;
  explicit ReserveRegisterPlacement(SharedPlacementState State)
      : MachineFunctionPass(ID), State(std::move(State)) {}
  StringRef getPassName() const override { return "FlyDSL reserve register storage"; }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!State->Functions.contains(&MF.getFunction()))
      return false;
    auto &MRI = MF.getRegInfo();
    const auto &TRI = *MF.getSubtarget().getRegisterInfo();
    if (!MRI.reservedRegsFrozen())
      MRI.freezeReservedRegs();
    const auto &TII = *MF.getSubtarget().getInstrInfo();
    bool OtherStackMaps = false;
    SmallSetVector<MCRegister, 32> Requested;
    for (auto &MBB : MF) {
      for (auto &MI : MBB) {
        if (MI.getOpcode() != TargetOpcode::STACKMAP)
          continue;
        const Carrier *C = getCarrier(MI, *State);
        if (!C) {
          OtherStackMaps = true;
          continue;
        }
        // Drop the synthetic call frame before liveness sees its SP operands.
        auto *Before = MI.getPrevNode();
        auto *After = MI.getNextNode();
        while (Before && Before->isDebugInstr())
          Before = Before->getPrevNode();
        while (After && After->isDebugInstr())
          After = After->getNextNode();
        if (!Before || !After || Before->getOpcode() != TII.getCallFrameSetupOpcode() ||
            After->getOpcode() != TII.getCallFrameDestroyOpcode())
          report_fatal_error("unexpected call-frame sequence around FlyDSL register stackmap");
        Before->eraseFromParent();
        After->eraseFromParent();
        unsigned First = C->BitOffset / 32;
        const auto &RC = *TRI.getRegClass(C->ClassID);
        forEachCarrierWord(MI, *C, [&](unsigned Word, const MachineOperand *MO) {
          if (!MO)
            return;
          MCRegister Phys = registerAt(RC, First + Word);
          if (MRI.isReserved(Phys))
            report_fatal_error(Twine("requested register is unavailable on this target: ") +
                               TRI.getName(Phys));
          Requested.insert(Phys);
        });
      }
    }
    // Existing precolored operands (including ABI inputs) are not covered by
    // virtual-register interference. Reject them conservatively; reservation
    // alone cannot prevent an already physical value from being overwritten.
    for (MCRegister Phys : Requested) {
      for (const auto &MBB : MF)
        for (const auto &MI : MBB)
          for (const MachineOperand &MO : MI.operands()) {
            if (MO.isReg() && MO.getReg().isPhysical() && TRI.regsOverlap(Phys, MO.getReg()))
              report_fatal_error(
                  Twine("explicit register conflicts with precolored machine operand: ") +
                  TRI.getName(Phys));
            if (MO.isRegMask() && MO.clobbersPhysReg(Phys))
              report_fatal_error(Twine("explicit register conflicts with a call clobber: ") +
                                 TRI.getName(Phys));
          }
      MRI.reserveReg(Phys, &TRI);
    }
    MF.getFrameInfo().setHasStackMap(OtherStackMaps);
    return true;
  }
};
char ReserveRegisterPlacement::ID;

struct Placement {
  unsigned ClassID;
  int64_t BitOffset;
};

class ApplyRegisterPlacement : public MachineFunctionPass {
  SharedPlacementState State;

public:
  static char ID;
  explicit ApplyRegisterPlacement(SharedPlacementState State)
      : MachineFunctionPass(ID), State(std::move(State)) {}
  StringRef getPassName() const override { return "FlyDSL apply register placement"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!State->Functions.contains(&MF.getFunction()))
      return false;
    auto &MRI = MF.getRegInfo();
    const auto &TRI = *MF.getSubtarget().getRegisterInfo();
    const auto &TII = *MF.getSubtarget().getInstrInfo();
    DenseMap<Register, Placement> Bindings;
    SmallVector<MachineInstr *> Markers;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        const Carrier *C = getCarrier(MI, *State);
        if (!C)
          continue;
        Markers.push_back(&MI);
        forEachCarrierWord(MI, *C, [&](unsigned Word, const MachineOperand *MO) {
          if (!MO)
            return;
          int64_t Offset = C->BitOffset + Word * 32;
          int64_t Base = Offset - (MO->getSubReg() ? TRI.getSubRegIdxOffset(MO->getSubReg()) : 0);
          auto [It, Inserted] = Bindings.try_emplace(MO->getReg(), Placement{C->ClassID, Base});
          if (!Inserted && (It->second.ClassID != C->ClassID || It->second.BitOffset != Base))
            report_fatal_error("conflicting FlyDSL register placements after machine optimization");
        });
      }
    }
    if (Markers.empty())
      report_fatal_error("FlyDSL register placement lost all carriers");
    if (Bindings.empty())
      report_fatal_error("explicit register placement has no materialized SSA values; "
                         "constant-only placement is unsupported");

    DenseMap<Register, MCRegister> Assignments;
    SmallSetVector<MCRegister, 32> FixedRegs;
    for (auto [VReg, P] : Bindings) {
      const auto &RC = *TRI.getRegClass(P.ClassID);
      unsigned UnitBits = TRI.getRegSizeInBits(RC);
      if (P.BitOffset < 0 || P.BitOffset % UnitBits)
        report_fatal_error("explicit register slices must start on a register boundary");
      const auto *OriginalRC = MRI.getRegClass(VReg);
      unsigned Bits = TRI.getRegSizeInBits(*OriginalRC);
      unsigned First = P.BitOffset / UnitBits;
      unsigned Count = divideCeil(Bits, UnitBits);
      MCRegister Phys = findRegisterTuple(TRI, RC, First, Bits);
      // SGPRs hold uniform wave values. Never force a divergent VGPR value
      // into an SGPR or remove constraints of a special scalar subclass.
      if (StringRef(TRI.getRegClassName(&RC)) == "SGPR_32" && !OriginalRC->contains(Phys))
        report_fatal_error("SGPR placement requires a compatible uniform LLVM register class");
      for (unsigned I = 0; I < Count; ++I) {
        MCRegister Leaf = registerAt(RC, First + I);
        if (!MRI.isReserved(Leaf))
          report_fatal_error(
              "machine coalescing extended a fixed value outside its reserved storage");
        FixedRegs.insert(Leaf);
      }
      Assignments[VReg] = Phys;
    }
    // Fixed placements are hard constraints, including between distinct
    // allocations. Do not silently overwrite a simultaneously live fixed value.
    auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    for (auto I = Assignments.begin(); I != Assignments.end(); ++I)
      for (auto J = std::next(I); J != Assignments.end(); ++J)
        if (TRI.regsOverlap(I->second, J->second) &&
            LIS.getInterval(I->first).overlaps(LIS.getInterval(J->first)))
          report_fatal_error("overlapping live intervals in explicit register storage");
    for (MachineInstr *MI : Markers)
      MI->eraseFromParent();
    DenseSet<MachineInstr *> ChangedInstructions;
    for (auto [VReg, Phys] : Assignments) {
      for (MachineOperand &MO : make_early_inc_range(MRI.reg_operands(VReg))) {
        ChangedInstructions.insert(MO.getParent());
        MO.substPhysReg(Phys, TRI);
        if (MO.isUse())
          MO.setIsKill(false);
        if (MO.isDef())
          MO.setIsDead(false);
      }
    }

    // MFMA has separate machine opcodes for its VGPR and AGPR C/D forms.
    // Select the matching encoding; general instruction legalization is outside
    // this experimental allocator's scope.
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!ChangedInstructions.contains(&MI))
          continue;
        StringRef Name = TII.getName(MI.getOpcode());
        if (!Name.starts_with("V_MFMA_") || MI.getNumOperands() == 0 || !MI.getOperand(0).isReg() ||
            !MI.getOperand(0).getReg().isPhysical())
          continue;
        const auto &VGPRClass = resolveRegisterClass(TRI, "VGPR_32");
        bool VGPR = llvm::any_of(TRI.subregs_inclusive(MI.getOperand(0).getReg()),
                                 [&](MCRegister Reg) { return VGPRClass.contains(Reg); });
        std::string Desired = Name.str();
        size_t Pos = Desired.find("_vgprcd");
        if (VGPR && Pos == std::string::npos) {
          Pos = Desired.rfind("_e64");
          if (Pos != std::string::npos)
            Desired.insert(Pos, "_vgprcd");
        } else if (!VGPR && Pos != std::string::npos)
          Desired.erase(Pos, 7);
        if (Desired != Name) {
          bool Found = false;
          for (unsigned I = 0; I < TII.getNumOpcodes(); ++I)
            if (Desired == TII.getName(I)) {
              MI.setDesc(TII.get(I));
              Found = true;
              break;
            }
          if (!Found)
            report_fatal_error(Twine("no MFMA encoding for explicit register placement: ") +
                               Desired);
        }
      }
    }
    // These newly physical values may cross loop/back edges. LLVM's generic
    // addLiveIns deliberately skips reserved registers, so add our explicit
    // storage live-ins ourselves and iterate to a fixed point.
    bool Changed;
    do {
      Changed = false;
      for (MachineBasicBlock &MBB : reverse(MF)) {
        LivePhysRegs Live;
        computeLiveIns(Live, MBB);
        for (MCRegister Reg : FixedRegs)
          if (Live.contains(Reg) && !MBB.isLiveIn(Reg)) {
            MBB.addLiveIn(Reg);
            Changed = true;
          }
        MBB.sortUniqueLiveIns();
      }
    } while (Changed);
    if (const char *Dir = std::getenv("FLYDSL_REGISTER_DUMP_DIR")) {
      sys::fs::create_directories(Dir);
      SmallString<256> Path(Dir);
      sys::path::append(Path, MF.getName() + ".registers.txt");
      std::error_code EC;
      raw_fd_ostream OS(Path, EC);
      if (EC)
        report_fatal_error(Twine("cannot write register placement MIR: ") + EC.message());
      MF.print(OS);
    }
    return true;
  }
};
char ApplyRegisterPlacement::ID;
} // namespace

namespace mlir::fly {
void registerRegisterPlacementCodegen() {
  static RegisterTargetPassConfigCallback Callback(
      [](TargetMachine &TM, legacy::PassManagerBase &PM, TargetPassConfig *Config) {
        if (!TM.getTargetTriple().isAMDGPU())
          return;
        auto State = std::make_shared<PlacementState>();
        PM.add(new PrepareRegisterPlacement(TM, State));
        Config->insertPass(&FinalizeISelID, new ReserveRegisterPlacement(State));
        Config->insertPass(&RenameIndependentSubregsID, new ApplyRegisterPlacement(State));
      });
}
} // namespace mlir::fly
