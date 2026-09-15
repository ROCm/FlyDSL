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
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/PassInfo.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/RegisterTargetPassConfigCallback.h"
#include <cstdlib>
#include <memory>
#include <optional>

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
  bool Automatic;
  uint64_t Alignment;
};
struct PendingCopy {
  Register Fixed;
  unsigned SubReg;
  Register Temporary;
};
struct AutomaticPlacement {
  unsigned ClassID;
  int64_t BitOffset;
  uint64_t Alignment;
  const TargetRegisterClass *RC;
};
struct DeferredPlacement {
  DenseMap<Register, MCRegister> Registers;
  DenseMap<Register, AutomaticPlacement> Automatic;
  SmallVector<PendingCopy> Copies;
  bool ScalarsVerified = false;
  DenseMap<Register, MCRegister> AutomaticAssignments;
  bool Verified = false;
};
struct PlacementState {
  DenseMap<uint64_t, Carrier> Carriers;
  DenseSet<const Function *> Functions;
  bool CanDefer = false;
  DenseMap<const Function *, DeferredPlacement> Deferred;
};
using SharedPlacementState = std::shared_ptr<PlacementState>;

// A registered pass ID creates a fresh checker at every greedy RA boundary.
// Keep its state in a per-pipeline immutable analysis, not in global storage.
class PlacementStateAnalysis : public ImmutablePass {
public:
  static char ID;
  SharedPlacementState State;
  explicit PlacementStateAnalysis(SharedPlacementState State = std::make_shared<PlacementState>())
      : ImmutablePass(ID), State(std::move(State)) {}
};
char PlacementStateAnalysis::ID;
static RegisterPass<PlacementStateAnalysis>
    PlacementStateRegistration("fly-register-placement-state", "FlyDSL register placement state",
                               false, true);

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

// Decode the first class member through LLVM's subregister model. This is
// also the physical index used for allocation-origin alignment checks.
std::optional<unsigned> registerIndex(const TargetRegisterInfo &TRI,
                                      const TargetRegisterClass &Bank, MCRegister Reg) {
  for (unsigned I = 0; I < Bank.getNumRegs(); ++I) {
    MCRegister Leaf = Bank.getRegister(I);
    if (Leaf == Reg)
      return I;
    unsigned Sub = TRI.getSubRegIndex(Reg, Leaf);
    if (Sub && TRI.getSubRegIdxOffset(Sub) == 0 &&
        TRI.getSubRegIdxSize(Sub) == TRI.getRegSizeInBits(Bank))
      return I;
  }
  return std::nullopt;
}

const TargetRegisterClass *findRegisterClass(const TargetRegisterInfo &TRI,
                                             const TargetRegisterClass &Bank, unsigned Bits) {
  const TargetRegisterClass *Best = nullptr;
  for (const auto &MC : TRI.regclasses()) {
    const auto *RC = TRI.getRegClass(MC.getID());
    if (!RC->isAllocatable() || TRI.getRegSizeInBits(*RC) != Bits ||
        (Best && RC->getNumRegs() <= Best->getNumRegs()))
      continue;
    bool Matches = llvm::all_of(RC->getRegisters(), [&](MCRegister Reg) {
      auto First = registerIndex(TRI, Bank, Reg);
      if (!First || *First + divideCeil(Bits, 32u) > Bank.getNumRegs())
        return false;
      for (unsigned I = 0; I < divideCeil(Bits, 32u); ++I) {
        MCRegister Leaf = Bank.getRegister(*First + I);
        if (Leaf == Reg && Bits == 32)
          continue;
        unsigned Sub = TRI.getSubRegIndex(Reg, Leaf);
        if (!Sub || TRI.getSubRegIdxOffset(Sub) != I * 32 || TRI.getSubRegIdxSize(Sub) != 32)
          return false;
      }
      return true;
    });
    if (Matches)
      Best = RC;
  }
  if (!Best)
    report_fatal_error("no LLVM allocatable register class for requested bank and width");
  return Best;
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
    State->Deferred.clear();
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
      int64_t Start = cast<ConstantInt>(CI->getArgOperand(1))->getSExtValue();
      bool Automatic = Start == -1;
      uint64_t Alignment = cast<ConstantInt>(CI->getArgOperand(4))->getZExtValue();
      if (!isPowerOf2_64(Alignment) || Alignment > uint64_t(INT64_MAX) || Start < -1 ||
          (!Automatic && uint64_t(Start) % Alignment))
        report_fatal_error("invalid register origin or alignment in FlyDSL carrier");
      uint64_t Offset = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      uint64_t StorageBits = cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue();
      uint64_t Bits = M.getDataLayout().getTypeSizeInBits(CI->getType());
      if (UnitBits != 32)
        report_fatal_error("unsupported register class width in FlyDSL carrier");
      if (!StorageBits ||
          (!Automatic && (uint64_t(Start) >= RC.getNumRegs() ||
                          StorageBits > uint64_t(RC.getNumRegs() - Start) * UnitBits)))
        report_fatal_error("explicit allocation exceeds LLVM register class members");
      if (!Bits || Bits % UnitBits || Offset % UnitBits || Offset > StorageBits ||
          Bits > StorageBits - Offset)
        report_fatal_error(
            "explicit register slices must cover whole 32-bit registers within storage");
      uint64_t BitOffset = (Automatic ? 0 : Start * UnitBits) + Offset;
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
      State->Carriers.try_emplace(
          ID, Carrier{RC.getID(), int64_t(BitOffset), Count, Automatic, Alignment});
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

// Detect class requests that need the target's normal allocation/rewrite
// pipeline. Follow only LLVM COPY/subregister relationships, never opcodes.
bool needsTargetRewrite(MachineFunction &MF, const PlacementState &State) {
  if (!State.CanDefer)
    return false;
  auto &MRI = MF.getRegInfo();
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();
  const auto &TII = *MF.getSubtarget().getInstrInfo();
  bool Needed = false;
  for (const auto &MBB : MF)
    for (const auto &MI : MBB) {
      const Carrier *C = getCarrier(MI, State);
      if (!C)
        continue;
      if (C->Automatic)
        return true;
      const auto &RC = *TRI.getRegClass(C->ClassID);
      if (StringRef(TRI.getRegClassName(&RC)) == "SGPR_32")
        continue;
      forEachCarrierWord(MI, *C, [&](unsigned Word, const MachineOperand *MO) {
        if (!MO)
          return;
        Register VReg = MO->getReg();
        int64_t Base = C->BitOffset + Word * 32 -
                       (MO->getSubReg() ? TRI.getSubRegIdxOffset(MO->getSubReg()) : 0);
        DenseSet<Register> Seen;
        MachineInstr *Def = nullptr;
        while (Seen.insert(VReg).second && (Def = MRI.getUniqueVRegDef(VReg)) && Def->isCopy() &&
               !Def->getOperand(0).getSubReg()) {
          const auto &Src = Def->getOperand(1);
          if (!Src.getReg().isVirtual() || Src.isUndef())
            break;
          Base -= Src.getSubReg() ? TRI.getSubRegIdxOffset(Src.getSubReg()) : 0;
          VReg = Src.getReg();
        }
        if (!Def || Base < 0 || Base % 32)
          return;
        unsigned Bits = TRI.getRegSizeInBits(*MRI.getRegClass(VReg));
        if (uint64_t(Base) + Bits > uint64_t(RC.getNumRegs()) * 32)
          return;
        MCRegister Phys = findRegisterTuple(TRI, RC, Base / 32, Bits);
        for (const MachineOperand &Op : Def->operands())
          if (Op.isReg() && Op.isDef() && Op.getReg() == VReg)
            if (const auto *Required = Def->getRegClassConstraint(Op.getOperandNo(), &TII, &TRI))
              Needed |=
                  !Required->contains(Op.getSubReg() ? TRI.getSubReg(Phys, Op.getSubReg()) : Phys);
      });
    }
  return Needed;
}

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
    bool Deferred = needsTargetRewrite(MF, *State);
    if (Deferred)
      State->Deferred.try_emplace(&MF.getFunction());
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
        if (C->Automatic) {
          if (!Deferred)
            report_fatal_error(
                "automatic register placement requires LLVM's registered allocation pipeline");
          continue;
        }
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
      // Deferred vector values participate in ordinary RA. Their requested
      // numbers are hints initially and hard postconditions after target
      // rewriting. Never assign a virtual register to a reserved register.
      if (!Deferred || resolveRegisterClass(TRI, "SGPR_32").contains(Phys))
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
  bool Automatic;
  uint64_t Alignment;
};

// The direct path requires compatible definitions. The deferred path expresses
// class boundaries with COPYs for LLVM's native rewriting, then requires all
// copies within fixed dataflow to become identity copies before emission.
// Only unconstrained consumers may retain transfers through read temporaries.
// This runs after coalescing/two-address rewriting, so inserted copies cannot
// be coalesced back into the incompatible fixed register.
class RegisterUseBridges {
  MachineRegisterInfo &MRI;
  const TargetRegisterInfo &TRI;
  const TargetInstrInfo &TII;
  const TargetRegisterClass &SGPR;
  DeferredPlacement *Deferred;

  MCRegister placement(const MachineOperand &Op) const {
    if (!Op.isReg())
      return MCRegister();
    Register Reg = Op.getReg();
    if (Deferred && Reg.isVirtual())
      if (auto It = Deferred->Automatic.find(Reg); It != Deferred->Automatic.end()) {
        MCRegister Representative = MRI.getRegClass(Reg)->getRegister(0);
        return Op.getSubReg() ? TRI.getSubReg(Representative, Op.getSubReg()) : Representative;
      }
    MCRegister Phys = Reg.isPhysical() ? MCRegister(Reg)
                      : Deferred       ? Deferred->Registers.lookup(Reg)
                                       : MCRegister();
    return Phys && Op.getSubReg() ? TRI.getSubReg(Phys, Op.getSubReg()) : Phys;
  }

  bool isScalar(Register Reg) const {
    if (Reg.isVirtual()) {
      const auto *RC = MRI.getRegClass(Reg);
      // A mixed class is not proof of a uniform value.
      return llvm::all_of(RC->getRegisters(), [&](MCRegister R) { return isScalar(R); });
    }
    return llvm::any_of(TRI.subregs_inclusive(Reg), [&](MCRegister R) { return SGPR.contains(R); });
  }

  [[noreturn]] void fail(const MachineInstr &MI, StringRef Reason) const {
    report_fatal_error(Twine("cannot bridge explicit register class for ") +
                       TII.getName(MI.getOpcode()) + ": " + Reason);
  }

public:
  RegisterUseBridges(MachineFunction &MF, DeferredPlacement *Deferred = nullptr)
      : MRI(MF.getRegInfo()), TRI(*MF.getSubtarget().getRegisterInfo()),
        TII(*MF.getSubtarget().getInstrInfo()), SGPR(resolveRegisterClass(TRI, "SGPR_32")),
        Deferred(Deferred) {}

  // Keep temporary classes as narrow as the original legal instruction,
  // instead of arbitrarily choosing an SGPR subclass of a mixed scalar/vector
  // operand constraint. Placement in vector storage does not by itself license
  // a lane-selecting transfer back to the original scalar class.
  using OperandClasses = DenseMap<unsigned, const TargetRegisterClass *>;

  void repair(MachineInstr &MI, const OperandClasses &OriginalClasses) {
    for (auto [Index, OriginalRC] : OriginalClasses) {
      const auto &Op = MI.getOperand(Index);
      if (Deferred || !Op.isReg() || !Op.isDef() || !Op.getReg().isPhysical())
        continue;
      const auto *Required = MI.getRegClassConstraint(Index, &TII, &TRI);
      if (Required && !Required->contains(Op.getReg()))
        report_fatal_error(
            Twine("explicit register definition cannot use ") + TRI.getName(Op.getReg()) + " for " +
            TII.getName(MI.getOpcode()) + " (requires " + TRI.getRegClassName(Required) +
            "); automatic write-back copies are disabled; choose a compatible register class "
            "or remove set_register");
    }
    // COPY has no descriptor register classes. It still must not turn a wave
    // value into a scalar by silently selecting a lane. The scalar placement
    // guard runs before substitution; this also covers pre-existing copies.
    if (MI.isCopy() && isScalar(MI.getOperand(0).getReg()) && !isScalar(MI.getOperand(1).getReg()))
      fail(MI, "vector-to-scalar transfer requires an explicit uniform conversion");
    DenseSet<unsigned> Done;
    for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
      if (!OriginalClasses.contains(I) || Done.contains(I))
        continue;
      auto &MO = MI.getOperand(I);
      if (!placement(MO) || MO.isImplicit())
        continue;
      SmallVector<unsigned, 2> Group{I};
      if (MO.isTied())
        Group.push_back(MI.findTiedOperandIdx(I));
      bool NeedsBridge = false;
      const TargetRegisterClass *TempRC = OriginalClasses.lookup(I);
      bool HasConstraint = TempRC != nullptr;
      for (unsigned Index : Group) {
        Done.insert(Index);
        auto &Op = MI.getOperand(Index);
        if (!Op.isReg() || Op.getReg() != MO.getReg() || Op.getSubReg() != MO.getSubReg())
          fail(MI, "tied operands must name the same complete physical register");
        const auto *Required = MI.getRegClassConstraint(Index, &TII, &TRI);
        if (Required) {
          if (Deferred && Deferred->Automatic.contains(Op.getReg())) {
            const auto *RC = MRI.getRegClass(Op.getReg());
            if (Op.getSubReg())
              RC = TRI.getSubRegisterClass(RC, Op.getSubReg());
            NeedsBridge |= !RC || !Required->hasSubClassEq(RC);
          } else {
            NeedsBridge |= !Required->contains(placement(Op));
          }
          TempRC = !HasConstraint ? Required
                   : TempRC       ? TRI.getCommonSubClass(TempRC, Required)
                                  : nullptr;
          HasConstraint = true;
        }
      }
      if (!NeedsBridge)
        continue;
      // Transfers are allowed only when leaving explicitly placed dataflow.
      // In particular, a fixed recurrence must not read through a temporary
      // on every iteration, even when its output has a compatible class.
      bool HasFixedDef = llvm::any_of(
          OriginalClasses, [&](const auto &Entry) { return MI.getOperand(Entry.first).isDef(); });
      if (!Deferred && HasFixedDef)
        fail(MI, "instructions producing explicit registers must also accept their fixed "
                 "inputs directly; choose compatible register classes or remove set_register");
      if (MI.isBundled() || MI.isTerminator() || MI.isInlineAsm())
        fail(MI, "operand needs a copy at an unsupported instruction boundary");
      TempRC = TRI.getAllocatableClass(TempRC);
      if (!TempRC)
        fail(MI, "no allocatable class satisfies the tied operand constraints");
      Register Temp = MRI.createVirtualRegister(TempRC);
      MachineBasicBlock &MBB = *MI.getParent();
      auto After = std::next(MI.getIterator());
      for (unsigned Index : Group) {
        auto &Op = MI.getOperand(Index);
        MCRegister Phys = placement(Op);
        Register Fixed = Op.getReg();
        unsigned FixedSub = Op.getSubReg();
        if (TRI.getRegSizeInBits(*TempRC) !=
            TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(Phys)))
          fail(MI, "copy would change the operand width");
        // Vector-to-scalar transfers are not bitwise copies of wave values.
        // Never silently pick a lane from a divergent value.
        bool ScalarDest = isScalar(Temp);
        bool ScalarSource = isScalar(Phys);
        if (Op.isUse() && !Op.isUndef()) {
          if (ScalarDest && !ScalarSource)
            fail(MI, "vector-to-scalar transfer requires an explicit uniform conversion");
          BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(TargetOpcode::COPY), Temp)
              .addReg(Fixed, {}, FixedSub);
          if (Deferred && HasFixedDef)
            Deferred->Copies.push_back({Fixed, FixedSub, Temp});
        }
        if (Op.isDef()) {
          if (ScalarSource && !ScalarDest)
            fail(MI, "cannot write a divergent result to scalar register storage");
          BuildMI(MBB, After, MI.getDebugLoc(), TII.get(TargetOpcode::COPY))
              .addReg(Fixed, RegState::Define | (Op.isUndef() ? RegState::Undef : RegState{}),
                      FixedSub)
              .addReg(Temp);
          Deferred->Copies.push_back({Fixed, FixedSub, Temp});
        }
        Op.setReg(Temp);
        Op.setSubReg(0);
        if (Op.isUse())
          Op.setIsKill(false);
        else
          Op.setIsDead(false);
      }
    }
    // Descriptor classes are not the only target constraints (e.g. AMDGPU's
    // constant bus). Fail here if the repaired instruction is still illegal.
    StringRef Reason;
    if (!TII.verifyInstruction(MI, Reason))
      fail(MI, Reason);
  }
};

// Use the full LLVM verifier, including virtual operand classes and liveness.
// Capture its diagnostic explicitly: release builds need useful failure text too.
void verifyPlacementFunction(MachineFunction &MF, const char *Banner) {
  std::string Diagnostic;
  raw_string_ostream OS(Diagnostic);
  if (!MF.verify(nullptr, Banner, &OS, /*AbortOnError=*/false))
    report_fatal_error(Twine(Banner) + ":\n" + Diagnostic);
}

class VerifyRegisterPlacement : public MachineFunctionPass {
  SharedPlacementState State;
  const char *Banner;

public:
  static char ID;
  VerifyRegisterPlacement(SharedPlacementState State, const char *Banner)
      : MachineFunctionPass(ID), State(std::move(State)), Banner(Banner) {}
  StringRef getPassName() const override { return "FlyDSL verify register placement"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (auto It = State->Deferred.find(&MF.getFunction());
        It != State->Deferred.end() && !It->second.Verified)
      report_fatal_error(
          "LLVM target register rewrite did not run; explicit placement cannot be enforced");
    if (State->Functions.contains(&MF.getFunction()))
      verifyPlacementFunction(MF, Banner);
    return false;
  }
};
char VerifyRegisterPlacement::ID;

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
    verifyPlacementFunction(MF, "Before FlyDSL register placement");
    auto DeferredIt = State->Deferred.find(&MF.getFunction());
    DeferredPlacement *Deferred =
        DeferredIt == State->Deferred.end() ? nullptr : &DeferredIt->second;
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
          Register Value = MO->getReg();
          // ISel can make a COPY solely to express a carrier word. Follow
          // bit-preserving virtual COPYs to the value used by real code,
          // retaining tuple offsets through LLVM's subregister model.
          DenseSet<Register> Seen;
          while (Seen.insert(Value).second &&
                 llvm::none_of(MRI.use_nodbg_operands(Value), [&](const MachineOperand &Use) {
                   return !getCarrier(*Use.getParent(), *State);
                 })) {
            MachineInstr *Def = MRI.getUniqueVRegDef(Value);
            if (!Def || !Def->isCopy() || Def->getOperand(0).getSubReg())
              break;
            const auto &Source = Def->getOperand(1);
            if (!Source.getReg().isVirtual() || Source.isUndef())
              break;
            unsigned SourceBits = Source.getSubReg()
                                      ? TRI.getSubRegIdxSize(Source.getSubReg())
                                      : TRI.getRegSizeInBits(*MRI.getRegClass(Source.getReg()));
            if (SourceBits != TRI.getRegSizeInBits(*MRI.getRegClass(Value)))
              break;
            Base -= Source.getSubReg() ? TRI.getSubRegIdxOffset(Source.getSubReg()) : 0;
            Value = Source.getReg();
          }
          auto [It, Inserted] =
              Bindings.try_emplace(Value, Placement{C->ClassID, Base, C->Automatic, C->Alignment});
          if (!Inserted) {
            auto &P = It->second;
            auto residue = [](int64_t Offset, uint64_t Alignment) {
              return uint64_t(Offset / 32) & (Alignment - 1);
            };
            if (P.ClassID != C->ClassID || (!P.Automatic && !C->Automatic && P.BitOffset != Base) ||
                (!P.Automatic && C->Automatic &&
                 residue(P.BitOffset, C->Alignment) != residue(Base, C->Alignment)) ||
                (P.Automatic && !C->Automatic &&
                 residue(P.BitOffset, P.Alignment) != residue(Base, P.Alignment)) ||
                residue(P.BitOffset, std::min(P.Alignment, C->Alignment)) !=
                    residue(Base, std::min(P.Alignment, C->Alignment)))
              report_fatal_error(
                  "conflicting FlyDSL register placements after machine optimization");
            if (!C->Automatic || (P.Automatic && C->Alignment > P.Alignment))
              P.BitOffset = Base;
            P.Automatic &= C->Automatic;
            P.Alignment = std::max(P.Alignment, C->Alignment);
          }
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
      // ISel may materialize a copy only for a carrier while real users keep
      // consuming the original value. Binding that dead copy would appear to
      // succeed and disappear later. Reject instead of promising placement.
      if (llvm::none_of(MRI.use_nodbg_operands(VReg), [&](const MachineOperand &MO) {
            return !getCarrier(*MO.getParent(), *State);
          })) {
        std::string Detail;
        raw_string_ostream OS(Detail);
        if (auto *Def = MRI.getUniqueVRegDef(VReg))
          Def->print(OS);
        report_fatal_error(Twine("explicit register carrier has no surviving machine uses; "
                                 "placement cannot be enforced: ") +
                           Detail);
      }
      const auto &RC = *TRI.getRegClass(P.ClassID);
      unsigned UnitBits = TRI.getRegSizeInBits(RC);
      if ((!P.Automatic && P.BitOffset < 0) || P.BitOffset % UnitBits)
        report_fatal_error("explicit register slices must start on a register boundary");
      const auto *OriginalRC = MRI.getRegClass(VReg);
      unsigned Bits = TRI.getRegSizeInBits(*OriginalRC);
      if (P.Automatic) {
        const auto *RequestedRC = findRegisterClass(TRI, RC, Bits);
        // Retain all existing compatible target restrictions. A class change
        // is expressed with COPY boundaries, never by rewriting an opcode.
        const auto *Common = TRI.getCommonSubClass(OriginalRC, RequestedRC);
        if (StringRef(TRI.getRegClassName(&RC)) == "SGPR_32" && !Common)
          report_fatal_error("SGPR placement requires a compatible uniform LLVM register class");
        if (Common)
          RequestedRC = Common;
        const auto &TII = *MF.getSubtarget().getInstrInfo();
        for (const auto &MO : MRI.reg_operands(VReg)) {
          const auto *Required =
              MO.getParent()->getRegClassConstraint(MO.getOperandNo(), &TII, &TRI);
          if (!Required)
            continue;
          const auto *Narrow =
              MO.getSubReg() ? TRI.getMatchingSuperRegClass(RequestedRC, Required, MO.getSubReg())
                             : TRI.getCommonSubClass(RequestedRC, Required);
          if (Narrow)
            RequestedRC = Narrow;
        }
        Deferred->Automatic[VReg] = {P.ClassID, P.BitOffset, P.Alignment, RequestedRC};
        continue;
      }
      unsigned First = P.BitOffset / UnitBits;
      unsigned Count = divideCeil(Bits, UnitBits);
      MCRegister Phys = findRegisterTuple(TRI, RC, First, Bits);
      // SGPRs hold uniform wave values. Never force a divergent VGPR value
      // into an SGPR or remove constraints of a special scalar subclass.
      if (StringRef(TRI.getRegClassName(&RC)) == "SGPR_32" && !OriginalRC->contains(Phys))
        report_fatal_error("SGPR placement requires a compatible uniform LLVM register class");
      for (unsigned I = 0; I < Count; ++I) {
        MCRegister Leaf = registerAt(RC, First + I);
        if (!Deferred && !MRI.isReserved(Leaf))
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
    DenseMap<MachineInstr *, RegisterUseBridges::OperandClasses> ChangedInstructions;
    for (auto [VReg, Phys] : Assignments) {
      bool KeepVirtual = Deferred && StringRef(TRI.getRegClassName(TRI.getRegClass(
                                         Bindings.lookup(VReg).ClassID))) != "SGPR_32";
      if (KeepVirtual)
        Deferred->Registers[VReg] = Phys;
      for (MachineOperand &MO : make_early_inc_range(MRI.reg_operands(VReg))) {
        const auto *OriginalRC = MRI.getRegClass(VReg);
        if (MO.getSubReg())
          OriginalRC = TRI.getSubRegisterClass(OriginalRC, MO.getSubReg());
        ChangedInstructions[MO.getParent()][MO.getOperandNo()] = OriginalRC;
        if (!KeepVirtual)
          MO.substPhysReg(Phys, TRI);
        if (MO.isUse())
          MO.setIsKill(false);
        if (MO.isDef())
          MO.setIsDead(false);
      }
    }

    if (Deferred)
      for (auto [VReg, Request] : Deferred->Automatic) {
        for (MachineOperand &MO : MRI.reg_operands(VReg)) {
          const auto *OriginalRC = MRI.getRegClass(VReg);
          if (MO.getSubReg())
            OriginalRC = TRI.getSubRegisterClass(OriginalRC, MO.getSubReg());
          ChangedInstructions[MO.getParent()][MO.getOperandNo()] = OriginalRC;
          if (MO.isUse())
            MO.setIsKill(false);
          if (MO.isDef())
            MO.setIsDead(false);
        }
        MRI.setRegClass(VReg, Request.RC);
      }
    if (Deferred)
      for (auto [VReg, Phys] : Deferred->Registers) {
        MRI.setRegClass(VReg, TRI.getMinimalPhysRegClass(Phys));
        MRI.setSimpleHint(VReg, Phys);
      }

    // Never select a target-specific encoding here. In deferred functions,
    // LLVM must rewrite the class boundaries before exact allocation commits.
    RegisterUseBridges Bridges(MF, Deferred);
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : make_early_inc_range(MBB))
        if (auto It = ChangedInstructions.find(&MI); It != ChangedInstructions.end())
          Bridges.repair(MI, It->second);
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
    // Do not pass this pass's cached LiveIntervals: substitution and bridging
    // invalidate it, and normal RA will recompute it after this pass returns.
    verifyPlacementFunction(MF, "After FlyDSL register placement");
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

// Choose physical numbers from LLVM's allocation order. Alignment constrains
// the origin modulo class members; it does not make independent SSA slices
// contiguous. Do not evict implicit values or introduce additional spills.
struct AutomaticAllocation {
  SmallVector<Register> Roots;
  DenseMap<Register, MCRegister> Previous;
};

AutomaticAllocation unassignAutomaticPlacement(MachineFunction &MF, DeferredPlacement &Pending,
                                               VirtRegMap &VRM, LiveIntervals &LIS,
                                               LiveRegMatrix &Matrix, bool Scalars) {
  auto &MRI = MF.getRegInfo();
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();
  const auto &SGPR = resolveRegisterClass(TRI, "SGPR_32");
  AutomaticAllocation Allocation;
  auto &Roots = Allocation.Roots;
  DenseSet<Register> Values;
  for (auto [VReg, Request] : Pending.Automatic) {
    if ((Request.ClassID == SGPR.getID()) != Scalars)
      continue;
    Roots.push_back(VReg);
    Values.insert(VReg);
  }
  if (Roots.empty())
    return Allocation;
  // Stable order makes allocation independent of DenseMap iteration order.
  llvm::sort(Roots, [](Register A, Register B) { return A.id() < B.id(); });
  for (const auto &Copy : Pending.Copies)
    if (llvm::is_contained(Roots, Copy.Fixed))
      Values.insert(Copy.Temporary);
  for (unsigned I = 0; I < MRI.getNumVirtRegs(); ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (VRM.getOriginal(VReg) != VReg && Values.contains(VRM.getOriginal(VReg)) &&
        !MRI.reg_nodbg_empty(VReg))
      report_fatal_error("LLVM split a class-constrained value; remove set_register");
  }
  auto &Previous = Allocation.Previous;
  for (Register VReg : Values) {
    if (MRI.reg_nodbg_empty(VReg) || !VRM.hasPhys(VReg) || !LIS.hasInterval(VReg))
      report_fatal_error("LLVM could not retain a class-constrained value for final allocation");
    Previous[VReg] = VRM.getPhys(VReg);
    Matrix.unassign(LIS.getInterval(VReg));
  }
  return Allocation;
}

void commitAutomaticPlacement(MachineFunction &MF, DeferredPlacement &Pending, LiveIntervals &LIS,
                              LiveRegMatrix &Matrix, const AutomaticAllocation &Allocation) {
  auto &MRI = MF.getRegInfo();
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();
  RegisterClassInfo Classes;
  Classes.runOnMachineFunction(MF);
  for (Register Root : Allocation.Roots) {
    const auto &Request = Pending.Automatic.find(Root)->second;
    const auto &Bank = *TRI.getRegClass(Request.ClassID);
    SmallVector<MCRegister> Candidates{Allocation.Previous.lookup(Root)};
    for (MCPhysReg Reg : Classes.getOrder(MRI.getRegClass(Root)))
      if (Reg != Candidates.front())
        Candidates.push_back(Reg);
    bool Assigned = false;
    bool Compatible = false;
    for (MCRegister Candidate : Candidates) {
      if (!Request.RC->contains(Candidate) || !MRI.getRegClass(Root)->contains(Candidate))
        continue;
      auto Index = registerIndex(TRI, Bank, Candidate);
      if (!Index || ((*Index - uint64_t(Request.BitOffset / 32)) & (Request.Alignment - 1)))
        continue;
      SmallVector<std::pair<Register, MCRegister>> Group{{Root, Candidate}};
      bool Valid = true;
      for (const auto &Copy : Pending.Copies) {
        if (Copy.Fixed != Root)
          continue;
        MCRegister Part = Copy.SubReg ? TRI.getSubReg(Candidate, Copy.SubReg) : Candidate;
        if (!Part || !MRI.getRegClass(Copy.Temporary)->contains(Part)) {
          Valid = false;
          break;
        }
        auto Entry = std::pair(Copy.Temporary, Part);
        if (!llvm::is_contained(Group, Entry))
          Group.push_back(Entry);
      }
      if (!Valid)
        continue;
      Compatible = true;
      SmallVector<Register> Committed;
      for (auto [VReg, Phys] : Group) {
        if (MRI.isReserved(Phys) ||
            Matrix.checkInterference(LIS.getInterval(VReg), Phys) != LiveRegMatrix::IK_Free) {
          Valid = false;
          break;
        }
        Matrix.assign(LIS.getInterval(VReg), Phys);
        Committed.push_back(VReg);
      }
      if (Valid) {
        Pending.AutomaticAssignments[Root] = Candidate;
        Assigned = true;
        break;
      }
      for (Register VReg : Committed)
        Matrix.unassign(LIS.getInterval(VReg));
    }
    if (!Assigned) {
      if (!Compatible)
        report_fatal_error(
            "LLVM could not satisfy the requested register class/alignment or "
            "eliminate its class transfers; automatic write-back copies are disabled");
      report_fatal_error("no non-interfering register assignment satisfies the requested "
                         "class/alignment; implicit values are not evicted");
    }
  }
}

// AMDGPU rewrites SGPR virtual registers before vector allocation. Check and
// commit automatic scalar requests at the registered greedy allocator boundary,
// while its ordinary VirtRegMap and LiveRegMatrix still exist.
class VerifyAutomaticSGPRPlacement : public MachineFunctionPass {
public:
  static char ID;
  VerifyAutomaticSGPRPlacement() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return "FlyDSL verify automatic scalar registers"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<PlacementStateAnalysis>();
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    auto State = getAnalysis<PlacementStateAnalysis>().State;
    auto It = State->Deferred.find(&MF.getFunction());
    if (It == State->Deferred.end() || It->second.ScalarsVerified)
      return false;
    auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
    auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    auto &Matrix = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
    auto Allocation = unassignAutomaticPlacement(MF, It->second, VRM, LIS, Matrix, true);
    commitAutomaticPlacement(MF, It->second, LIS, Matrix, Allocation);
    It->second.ScalarsVerified = true;
    return true;
  }
};
char VerifyAutomaticSGPRPlacement::ID;
static RegisterPass<VerifyAutomaticSGPRPlacement>
    VerifyScalarRegistration("fly-verify-automatic-sgpr-placement",
                             "FlyDSL verify automatic scalar registers", false, false);

// An allocation hint is not a placement guarantee. Accept the deferred path
// only after target rewriting enables every fixed operand class and the
// interference matrix permits the exact numbers without splitting or spilling.
class VerifyDeferredPlacement : public MachineFunctionPass {
  SharedPlacementState State;

public:
  static char ID;
  explicit VerifyDeferredPlacement(SharedPlacementState State)
      : MachineFunctionPass(ID), State(std::move(State)) {}
  StringRef getPassName() const override { return "FlyDSL verify target register rewrite"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<VirtRegMapWrapperLegacy>();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<LiveRegMatrixWrapperLegacy>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    auto It = State->Deferred.find(&MF.getFunction());
    if (It == State->Deferred.end())
      return false;
    auto &Pending = It->second;
    auto &MRI = MF.getRegInfo();
    const auto &TRI = *MF.getSubtarget().getRegisterInfo();
    auto &VRM = getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
    auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    auto &Matrix = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
    // Automatic requests may move to make room for exact requests. Detach
    // both kinds before committing numbers; never detach implicit values.
    auto Automatic = unassignAutomaticPlacement(MF, Pending, VRM, LIS, Matrix, false);
    DenseMap<Register, MCRegister> Assignments = Pending.Registers;
    for (const auto &Copy : Pending.Copies) {
      if (Pending.Automatic.contains(Copy.Fixed))
        continue;
      MCRegister Fixed =
          Copy.Fixed.isPhysical() ? MCRegister(Copy.Fixed) : Pending.Registers.lookup(Copy.Fixed);
      if (Fixed && Copy.SubReg)
        Fixed = TRI.getSubReg(Fixed, Copy.SubReg);
      if (!Fixed || !MRI.getRegClass(Copy.Temporary)->contains(Fixed))
        report_fatal_error(
            "explicit register definition cannot use requested placement: LLVM did not eliminate "
            "the register-class transfer; automatic write-back copies are disabled; choose a "
            "compatible register class or remove set_register");
      auto [Assigned, Inserted] = Assignments.try_emplace(Copy.Temporary, Fixed);
      if (!Inserted && Assigned->second != Fixed)
        report_fatal_error("LLVM combined values with different explicit register numbers");
    }
    for (unsigned I = 0; I < MRI.getNumVirtRegs(); ++I) {
      Register VReg = Register::index2VirtReg(I);
      Register Original = VRM.getOriginal(VReg);
      if (Original != VReg && Assignments.contains(Original) && !MRI.reg_nodbg_empty(VReg))
        report_fatal_error("LLVM split an explicitly placed value; remove set_register");
    }
    // LLVM owns class conversion; use its interference matrix only to commit
    // exact numbers after conversion. Remove all old assignments first so
    // permutations between fixed values do not look like false conflicts.
    // Never evict implicit values, change reservedRegs, or ignore interference.
    for (auto [VReg, Phys] : Assignments) {
      if (MRI.reg_nodbg_empty(VReg) || !VRM.hasPhys(VReg) || !MRI.getRegClass(VReg)->contains(Phys))
        report_fatal_error("LLVM could not retain an explicitly placed value for final allocation");
      Matrix.unassign(LIS.getInterval(VReg));
    }
    for (auto [VReg, Phys] : Assignments) {
      if (MRI.isReserved(Phys) ||
          Matrix.checkInterference(LIS.getInterval(VReg), Phys) != LiveRegMatrix::IK_Free)
        report_fatal_error(
            Twine("explicit register interferes with a live value after LLVM allocation: ") +
            TRI.getName(Phys) + "; choose another range or remove set_register");
      Matrix.assign(LIS.getInterval(VReg), Phys);
    }
    for (const auto &Copy : Pending.Copies) {
      if (Pending.Automatic.contains(Copy.Fixed))
        continue;
      MCRegister Fixed = Copy.Fixed.isPhysical()   ? MCRegister(Copy.Fixed)
                         : VRM.hasPhys(Copy.Fixed) ? VRM.getPhys(Copy.Fixed)
                                                   : MCRegister();
      if (Fixed && Copy.SubReg)
        Fixed = TRI.getSubReg(Fixed, Copy.SubReg);
      if (!Fixed || !VRM.hasPhys(Copy.Temporary) || VRM.getPhys(Copy.Temporary) != Fixed)
        report_fatal_error(
            "LLVM did not eliminate an explicit register-class transfer; automatic write-back "
            "copies are disabled; choose a compatible register class or remove set_register");
    }
    const auto &SGPR = resolveRegisterClass(TRI, "SGPR_32");
    if (!Pending.ScalarsVerified && llvm::any_of(Pending.Automatic, [&](const auto &Entry) {
          return Entry.second.ClassID == SGPR.getID();
        }))
      report_fatal_error("automatic SGPR placement requires LLVM's registered greedy allocator");
    commitAutomaticPlacement(MF, Pending, LIS, Matrix, Automatic);
    verifyPlacementFunction(MF, "After LLVM target register rewrite");
    Pending.Verified = true;
    if (const char *Dir = std::getenv("FLYDSL_REGISTER_DUMP_DIR")) {
      SmallString<256> Path(Dir);
      sys::path::append(Path, MF.getName() + ".rewritten-registers.txt");
      std::error_code EC;
      raw_fd_ostream OS(Path, EC);
      if (EC)
        report_fatal_error(Twine("cannot write target register rewrite MIR: ") + EC.message());
      MF.print(OS);
      VRM.print(OS);
      for (auto [VReg, Phys] : Pending.AutomaticAssignments) {
        const auto &Request = Pending.Automatic.find(VReg)->second;
        OS << "\n; FlyDSL automatic " << VReg.id() << " class "
           << TRI.getRegClassName(TRI.getRegClass(Request.ClassID)) << " bitOffset "
           << Request.BitOffset << " alignment " << Request.Alignment << " assigned "
           << TRI.getName(Phys) << " index "
           << *registerIndex(TRI, *TRI.getRegClass(Request.ClassID), Phys) << "\n";
      }
    }
    return true;
  }
};
char VerifyDeferredPlacement::ID;
} // namespace

namespace mlir::fly {
void registerRegisterPlacementCodegen() {
  static RegisterTargetPassConfigCallback Callback(
      [](TargetMachine &TM, legacy::PassManagerBase &PM, TargetPassConfig *Config) {
        if (!TM.getTargetTriple().isAMDGPU())
          return;
        auto State = std::make_shared<PlacementState>();
        // Use the target's registered pipeline pass through LLVM's public
        // registry. No target-private headers or opcode tables are required.
        // If it is unavailable, retain the existing direct-placement path.
        const auto *Rewrite = PassRegistry::getPassRegistry()->getPassInfo(
            StringRef("amdgpu-rewrite-agpr-copy-mfma"));
        State->CanDefer = Rewrite != nullptr;
        if (const auto *Greedy = PassRegistry::getPassRegistry()->getPassInfo(StringRef("greedy")))
          Config->insertPass(Greedy->getTypeInfo(), &VerifyAutomaticSGPRPlacement::ID);
        if (Rewrite)
          Config->insertPass(Rewrite->getTypeInfo(), new VerifyDeferredPlacement(State));
        PM.add(new PlacementStateAnalysis(State));
        PM.add(new PrepareRegisterPlacement(TM, State));
        Config->insertPass(&FinalizeISelID, new ReserveRegisterPlacement(State));
        Config->insertPass(&RenameIndependentSubregsID, new ApplyRegisterPlacement(State));
        // AMDGPU can run register allocation/rewriting more than once. The
        // standard post-RA expansion point runs after all those phases.
        Config->insertPass(&ExpandPostRAPseudosID,
                           new VerifyRegisterPlacement(State, "After post-RA pseudo expansion"));
      });
}
} // namespace mlir::fly
