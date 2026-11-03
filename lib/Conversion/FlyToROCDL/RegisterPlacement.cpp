// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Experimental explicit register allocation. Identity calls keep allocation
// boundaries visible through IR optimization. When codegen starts, convert
// them to stackmaps with real value operands. Consume these before scheduling;
// no stackmap records, debug carriers, inline assembly, or LLVM patches are used.
//
// A request binds every materialized value of the storage to the requested
// physical registers for that value's live range. The registers are not
// reserved: ordinary physical-register liveness protects them while the value
// is live, and other values may use them elsewhere.
//
// Rejected requests are RegisterPlacementDiagnostic errors when the
// serializer handles them; the function is then compiled without placement.
// Other drivers keep the fatal behavior.

#include "RegisterPlacementDiagnostic.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/RegisterTargetPassConfigCallback.h"
#include <memory>
#include <optional>

using namespace llvm;

int mlir::fly::getRegisterPlacementDiagnosticKind() {
  static const int Kind = getNextAvailablePluginDiagnosticKind();
  return Kind;
}

void mlir::fly::RegisterPlacementDiagnostic::print(DiagnosticPrinter &DP) const {
  DP << "explicit register placement in '" << Fn.getName() << "': " << Message;
}

namespace {

constexpr StringLiteral MarkerPrefix = "__flydsl_register_value_";
// Shared only by the passes of one TargetMachine codegen invocation. Existing
// user stackmap IDs are excluded when allocating our IDs. No magic prefix or
// process-global placement state can accidentally claim an unrelated stackmap.

struct Carrier {
  unsigned ClassID;
  int64_t BitOffset;
  unsigned Words;
  // Absolute bit range of the whole allocation.
  int64_t StorageBegin;
  int64_t StorageEnd;
  // Identifies the lowered fly.register_value for diagnostics.
  uint64_t Request;
};

struct PlacementState {
  DenseMap<uint64_t, Carrier> Carriers;
  DenseSet<const Function *> Functions;
  // Functions with a rejected request. Their carriers are removed and the
  // remaining passes compile them without placement.
  DenseSet<const Function *> Failed;
  // The serializer converts placement diagnostics to MLIR errors.
  bool Recoverable = false;
};

// Report a rejected request. Callers must leave valid machine code that no
// longer depends on the function's requests.
void reject(PlacementState &State, const Function &F, std::optional<uint64_t> Request,
            const Twine &Message) {
  if (!State.Recoverable)
    report_fatal_error(Message);

  F.getContext().diagnose(mlir::fly::RegisterPlacementDiagnostic(F, Request, Message.str()));
  State.Failed.insert(&F);
}

const Carrier *getCarrier(const MachineInstr &MI, const PlacementState &State) {
  if (MI.getOpcode() != TargetOpcode::STACKMAP || MI.getNumOperands() < 2 ||
      !MI.getOperand(0).isImm())
    return nullptr;
  auto It = State.Carriers.find(uint64_t(MI.getOperand(0).getImm()));
  return It == State.Carriers.end() ? nullptr : &It->second;
}

// Decode LLVM's documented STACKMAP payload once for all machine passes.
// A null operand denotes a constant: no register exists at this boundary.
// Returns an error when instruction selection did not keep the expected shape.
std::optional<std::string>
forEachCarrierWord(const MachineInstr &MI, const Carrier &C,
                   function_ref<void(unsigned, const MachineOperand *)> Visit) {
  unsigned Word = 0;
  for (unsigned I = 2; I < MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg() && MO.isImplicit())
      continue;
    if (Word >= C.Words)
      return std::string("register carrier contains too many words");
    if (MO.isImm() && MO.getImm() == StackMaps::ConstantOp) {
      if (++I >= MI.getNumOperands() || !MI.getOperand(I).isImm())
        return std::string("malformed constant in FlyDSL register carrier");
      Visit(Word++, nullptr);
    } else {
      if (!MO.isReg() || !MO.getReg().isVirtual())
        return std::string("register carrier lost its virtual register operand");
      Visit(Word++, &MO);
    }
  }
  if (Word != C.Words)
    return std::string("register carrier lost words during instruction selection");
  return std::nullopt;
}

const TargetRegisterClass *findNamedRegisterClass(const TargetRegisterInfo &TRI, StringRef Name) {
  for (const auto &RC : TRI.regclasses())
    if (Name == TRI.getRegClassName(&RC))
      return &RC;
  return nullptr;
}

// Classes named by this backend itself; a missing one is an LLVM change.
const TargetRegisterClass &resolveRegisterClass(const TargetRegisterInfo &TRI, StringRef Name) {
  if (const auto *RC = findNamedRegisterClass(TRI, Name))
    return *RC;
  report_fatal_error(Twine("unknown LLVM register class: ") + Name);
}

MCRegister registerAt(const TargetRegisterClass &RC, unsigned Index) {
  if (Index >= RC.getNumRegs())
    report_fatal_error("explicit allocation exceeds LLVM register class members");
  return RC.getRegister(Index);
}

// Match tuples through LLVM's subregister relationships, never target spelling.
// Returns no register when the target has no tuple at this index and width.
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
  return MCRegister();
}

class PrepareRegisterPlacement : public ModulePass {
  TargetMachine &TM;
  std::shared_ptr<PlacementState> State;

public:
  static char ID;

  PrepareRegisterPlacement(TargetMachine &TM, std::shared_ptr<PlacementState> State)
      : ModulePass(ID), TM(TM), State(std::move(State)) {}

  StringRef getPassName() const override { return "PrepareRegisterPlacement"; }

  bool runOnModule(Module &M) override {
    State->Carriers.clear();
    State->Functions.clear();
    State->Failed.clear();

    auto *Flag =
        mdconst::extract_or_null<ConstantInt>(M.getModuleFlag(mlir::fly::RegisterDiagnosticsFlag));
    State->Recoverable = Flag && !Flag->isZero();

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

    // The codegen pipeline is shared by every function in the module.
    if (TM.getOptLevel() == CodeGenOptLevel::None || TM.Options.EnableGlobalISel ||
        TM.Options.EnableFastISel) {
      reject(*State, *Calls.front()->getFunction(), std::nullopt,
             "explicit register placement requires optimized SelectionDAG codegen");
      for (CallInst *CI : Calls) {
        CI->replaceAllUsesWith(CI->getArgOperand(0));
        CI->eraseFromParent();
      }
      return true;
    }

    auto *StackMap = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::experimental_stackmap);
    uint64_t NextID = 0;
    for (CallInst *CI : Calls) {
      Function &F = *CI->getFunction();
      State->Functions.insert(&F);
      Value *V = CI->getArgOperand(0);
      // A rejected request leaves the value without a carrier.
      auto Drop = [&] {
        CI->replaceAllUsesWith(V);
        CI->eraseFromParent();
      };
      uint64_t Request = 0;
      if (CI->arg_size() != 4 ||
          !all_of(drop_begin(CI->args()), [](const Use &U) { return isa<ConstantInt>(U.get()); }) ||
          CI->getCalledFunction()
              ->getFnAttribute("flydsl-register-request")
              .getValueAsString()
              .getAsInteger(10, Request)) {
        reject(*State, F, std::nullopt, "malformed register carrier");
        Drop();
        continue;
      }
      auto Fail = [&](const Twine &Message) {
        reject(*State, F, Request, Message);
        Drop();
      };
      if (State->Failed.contains(&F)) {
        Drop();
        continue;
      }
      Function &Marker = *CI->getCalledFunction();
      if (Marker.getFnAttribute("flydsl-register-target").getValueAsString() !=
          TM.getTargetTriple().getArchName()) {
        Fail("register class target does not match LLVM target architecture");
        continue;
      }
      const auto &TRI = *TM.getSubtargetImpl(F)->getRegisterInfo();
      StringRef Name = Marker.getFnAttribute("flydsl-register-class").getValueAsString();
      const auto *RC = findNamedRegisterClass(TRI, Name);
      if (!RC) {
        Fail(Twine("unknown LLVM register class: ") + Name);
        continue;
      }

      // The representation is generic; this backend currently supports these
      // three base classes only. Reject other classes instead of guessing.
      if (Name != "VGPR_32" && Name != "AGPR_32" && Name != "SGPR_32") {
        Fail("unsupported LLVM register class for AMDGPU explicit placement");
        continue;
      }

      unsigned UnitBits = TRI.getRegSizeInBits(*RC);
      // Upper passes split register storage into members of the declared width.
      if (Marker.getFnAttribute("flydsl-register-bits").getValueAsString() !=
          std::to_string(UnitBits)) {
        Fail(Twine("register class width does not match LLVM: ") + Name + " members are " +
             Twine(UnitBits) + " bits");
        continue;
      }
      int64_t Start = cast<ConstantInt>(CI->getArgOperand(1))->getSExtValue();
      if (Start < 0) {
        Fail("invalid register origin in the carrier");
        continue;
      }
      uint64_t Offset = cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      uint64_t StorageBits = cast<ConstantInt>(CI->getArgOperand(3))->getZExtValue();
      uint64_t Bits = M.getDataLayout().getTypeSizeInBits(CI->getType());

      if (!StorageBits || uint64_t(Start) >= RC->getNumRegs() ||
          StorageBits > uint64_t(RC->getNumRegs() - Start) * UnitBits) {
        Fail("explicit allocation exceeds LLVM register class members");
        continue;
      }
      if (!Bits || Bits % UnitBits || Offset % UnitBits || Offset > StorageBits ||
          Bits > StorageBits - Offset) {
        Fail(Twine("explicit register slices must cover whole ") + Twine(UnitBits) +
             "-bit registers within storage");
        continue;
      }
      int64_t StorageBegin = Start * UnitBits;
      IRBuilder<> B(CI);
      unsigned Count = Bits / UnitBits;
      Type *UnitTy = B.getIntNTy(UnitBits);
      Type *WordsTy = Count == 1 ? UnitTy : FixedVectorType::get(UnitTy, Count);
      Value *Words = B.CreateBitCast(V, WordsTy);

      while (UsedIDs.contains(NextID))
        ++NextID;
      uint64_t ID = NextID++;
      UsedIDs.insert(ID);

      State->Carriers.try_emplace(ID, Carrier{RC->getID(), StorageBegin + int64_t(Offset), Count,
                                              StorageBegin, StorageBegin + int64_t(StorageBits),
                                              Request});
      SmallVector<Value *> Args{B.getInt64(ID), B.getInt32(0)};
      for (unsigned I = 0; I < Count; ++I)
        Args.push_back(Count == 1 ? Words : B.CreateExtractElement(Words, B.getInt32(I)));
      B.CreateCall(StackMap, Args);
      Drop();
    }
    return true;
  }
};

char PrepareRegisterPlacement::ID;

// Remove the carriers' synthetic call frames before liveness sees them, and
// check the requested numbers against registers that liveness cannot protect:
// target-reserved registers, precolored operands (including ABI inputs) and
// call clobbers. Nothing is reserved; a fixed value's live range protects its
// registers after the carriers are applied.
class ValidateRegisterPlacement : public MachineFunctionPass {
  std::shared_ptr<PlacementState> State;

public:
  static char ID;

  explicit ValidateRegisterPlacement(std::shared_ptr<PlacementState> State)
      : MachineFunctionPass(ID), State(std::move(State)) {}

  StringRef getPassName() const override { return "ValidateRegisterPlacement"; }

  bool runOnMachineFunction(MachineFunction &MF) override {
    const Function &F = MF.getFunction();
    if (!State->Functions.contains(&F))
      return false;
    auto &MRI = MF.getRegInfo();
    const auto &TRI = *MF.getSubtarget().getRegisterInfo();
    if (!MRI.reservedRegsFrozen())
      MRI.freezeReservedRegs();
    const auto &TII = *MF.getSubtarget().getInstrInfo();

    bool OtherStackMaps = false;
    std::optional<uint64_t> FrameError;
    SmallVector<MachineInstr *> Markers;
    for (auto &MBB : MF)
      for (auto &MI : MBB) {
        if (MI.getOpcode() != TargetOpcode::STACKMAP)
          continue;
        if (const Carrier *C = getCarrier(MI, *State)) {
          Markers.push_back(&MI);
          // Drop the synthetic call frame before liveness sees its SP operands.
          auto *Before = MI.getPrevNode();
          auto *After = MI.getNextNode();
          while (Before && Before->isDebugInstr())
            Before = Before->getPrevNode();
          while (After && After->isDebugInstr())
            After = After->getNextNode();
          if (!Before || !After || Before->getOpcode() != TII.getCallFrameSetupOpcode() ||
              After->getOpcode() != TII.getCallFrameDestroyOpcode()) {
            FrameError = C->Request;
            continue;
          }
          Before->eraseFromParent();
          After->eraseFromParent();
        } else {
          OtherStackMaps = true;
        }
      }

    MF.getFrameInfo().setHasStackMap(OtherStackMaps);
    // Placement is abandoned for this function; compile it without carriers.
    auto Strip = [&](std::optional<uint64_t> Request, const Twine &Message) {
      if (!State->Failed.contains(&F))
        reject(*State, F, Request, Message);
      for (MachineInstr *MI : Markers)
        MI->eraseFromParent();
      return true;
    };
    if (State->Failed.contains(&F))
      return Strip(std::nullopt, "");
    if (FrameError)
      return Strip(FrameError, "unexpected call-frame sequence around register stackmap");

    // Registers that instructions read or write implicitly (VCC, M0, EXEC,
    // ...) are compiler-managed: code inserted after this check could
    // overwrite a fixed value there. AMDGPU's SGPR_32 ends with VCC_LO/HI.
    std::optional<SmallSetVector<MCRegister, 16>> Implicit;
    auto implicitRegisters = [&]() -> const SmallSetVector<MCRegister, 16> & {
      if (!Implicit) {
        Implicit.emplace();
        for (unsigned Opcode = 0, E = TII.getNumOpcodes(); Opcode != E; ++Opcode) {
          const MCInstrDesc &Desc = TII.get(Opcode);
          for (MCPhysReg Reg : Desc.implicit_defs())
            Implicit->insert(Reg);
          for (MCPhysReg Reg : Desc.implicit_uses())
            Implicit->insert(Reg);
        }
      }
      return *Implicit;
    };

    MapVector<MCRegister, uint64_t> Requested;
    for (MachineInstr *MI : Markers) {
      const Carrier *C = getCarrier(*MI, *State);
      const auto &RC = *TRI.getRegClass(C->ClassID);
      unsigned First = C->BitOffset / TRI.getRegSizeInBits(RC);
      std::optional<std::string> Unavailable;
      auto Error = forEachCarrierWord(*MI, *C, [&](unsigned Word, const MachineOperand *MO) {
        if (!MO)
          return;
        MCRegister Phys = registerAt(RC, First + Word);
        if (MRI.isReserved(Phys) && !Unavailable)
          Unavailable =
              (Twine("requested register is unavailable on this target: ") + TRI.getName(Phys))
                  .str();
        for (MCRegister Reg : implicitRegisters())
          if (!Unavailable && TRI.regsOverlap(Phys, Reg))
            Unavailable = (Twine("requested register ") + TRI.getName(Phys) + " overlaps " +
                           TRI.getName(Reg) +
                           ", which instructions read or write implicitly; choose another number")
                              .str();
        Requested.insert({Phys, C->Request});
      });
      if (Error)
        return Strip(C->Request, *Error);
      if (Unavailable)
        return Strip(C->Request, *Unavailable);
    }

    // Existing precolored operands (including ABI inputs) are not covered by
    // virtual-register interference, and frame lowering may add uses of ABI
    // registers later. Reject any overlap conservatively. Index the requested
    // register units so the function is scanned once.
    DenseMap<MCRegUnit, std::pair<MCRegister, uint64_t>> Units;
    for (auto [Phys, Request] : Requested)
      for (MCRegUnit Unit : TRI.regunits(Phys))
        Units.try_emplace(Unit, Phys, Request);

    for (const auto &MBB : MF)
      for (const auto &MI : MBB)
        for (const MachineOperand &MO : MI.operands()) {
          if (MO.isReg() && MO.getReg().isPhysical())
            for (MCRegUnit Unit : TRI.regunits(MO.getReg().asMCReg()))
              if (auto It = Units.find(Unit); It != Units.end())
                return Strip(
                    It->second.second,
                    Twine("explicit register conflicts with precolored machine operand: ") +
                        TRI.getName(It->second.first));
          if (MO.isRegMask())
            for (auto [Phys, Request] : Requested)
              if (MO.clobbersPhysReg(Phys))
                return Strip(Request, Twine("explicit register conflicts with a call clobber: ") +
                                          TRI.getName(Phys));
        }
    return true;
  }
};

char ValidateRegisterPlacement::ID;

struct Placement {
  unsigned ClassID;
  int64_t BitOffset;
  uint64_t Request;
  // Allocations bound to this value, as absolute bit ranges.
  SmallVector<std::pair<int64_t, int64_t>, 1> Storage;
};

// Records the machine changes made while applying a function's requests, so
// a rejected request leaves the code as instruction selection produced it.
class UndoLog {
  struct OperandState {
    MachineInstr *MI;
    unsigned OpNo;
    Register Reg;
    unsigned SubReg;
    bool Kill;
    bool Dead;
    bool Undef;
  };
  SmallVector<OperandState> Operands;
  DenseSet<std::pair<MachineInstr *, unsigned>> SeenOperands;
  SmallVector<MachineInstr *> Inserted;
  SmallVector<std::pair<MachineBasicBlock *, MCRegister>> LiveIns;

public:
  void operand(MachineOperand &MO) {
    MachineInstr *MI = MO.getParent();
    unsigned OpNo = MO.getOperandNo();
    if (SeenOperands.insert({MI, OpNo}).second)
      Operands.push_back({MI, OpNo, MO.getReg(), MO.getSubReg(), MO.isUse() && MO.isKill(),
                          MO.isDef() && MO.isDead(), MO.isUndef()});
  }
  void inserted(MachineInstr *MI) { Inserted.push_back(MI); }

  void liveIn(MachineBasicBlock *MBB, MCRegister Reg) { LiveIns.push_back({MBB, Reg}); }

  void undo() {
    for (auto [MBB, Reg] : LiveIns)
      MBB->removeLiveIn(Reg);
    for (MachineInstr *MI : reverse(Inserted))
      MI->eraseFromParent();
    for (const auto &S : reverse(Operands)) {
      MachineOperand &MO = S.MI->getOperand(S.OpNo);
      MO.setReg(S.Reg);
      MO.setSubReg(S.SubReg);
      if (MO.isUse())
        MO.setIsKill(S.Kill);
      else
        MO.setIsDead(S.Dead);
      MO.setIsUndef(S.Undef);
    }
  }
};

// Placed values are physical: instructions producing them must accept the
// requested registers directly, inputs included. Only consumers that produce
// no placed value may read them through temporaries. This runs after
// coalescing/two-address rewriting, so inserted copies cannot be coalesced
// back into the incompatible register.
class RegisterUseBridges {
  MachineRegisterInfo &MRI;
  const TargetRegisterInfo &TRI;
  const TargetInstrInfo &TII;
  const TargetRegisterClass &SGPR;
  const TargetRegisterClass &AGPR;
  const TargetRegisterClass &VGPR;
  UndoLog &Log;
  std::string Error;

  MCRegister placement(const MachineOperand &Op) const {
    if (!Op.isReg())
      return MCRegister();
    Register Reg = Op.getReg();
    if (!Reg.isPhysical())
      return MCRegister();
    return Op.getSubReg() ? TRI.getSubReg(Reg, Op.getSubReg()) : MCRegister(Reg);
  }

  // The 32-bit bank a register or tuple is built from, if it is one of ours.
  const TargetRegisterClass *bankOf(MCRegister Reg) const {
    for (const TargetRegisterClass *Bank : {&AGPR, &VGPR, &SGPR})
      if (llvm::any_of(TRI.subregs_inclusive(Reg), [&](MCRegister R) { return Bank->contains(R); }))
        return Bank;
    return nullptr;
  }

  bool isScalar(Register Reg) const {
    if (Reg.isVirtual()) {
      const auto *RC = MRI.getRegClass(Reg);
      // A mixed class is not proof of a uniform value.
      return llvm::all_of(RC->getRegisters(), [&](MCRegister R) { return isScalar(R); });
    }
    return llvm::any_of(TRI.subregs_inclusive(Reg), [&](MCRegister R) { return SGPR.contains(R); });
  }

  bool fail(const MachineInstr &MI, const Twine &Reason) {
    Error = (Twine("cannot bridge explicit register class for ") + TII.getName(MI.getOpcode()) +
             ": " + Reason)
                .str();
    return false;
  }

public:
  RegisterUseBridges(MachineFunction &MF, UndoLog &Log)
      : MRI(MF.getRegInfo()), TRI(*MF.getSubtarget().getRegisterInfo()),
        TII(*MF.getSubtarget().getInstrInfo()), SGPR(resolveRegisterClass(TRI, "SGPR_32")),
        AGPR(resolveRegisterClass(TRI, "AGPR_32")), VGPR(resolveRegisterClass(TRI, "VGPR_32")),
        Log(Log) {}

  StringRef error() const { return Error; }

  // Keep temporary classes as narrow as the original legal instruction,
  // instead of arbitrarily choosing an SGPR subclass of a mixed scalar/vector
  // operand constraint. Placement in vector storage does not by itself license
  // a lane-selecting transfer back to the original scalar class.
  using OperandClasses = DenseMap<unsigned, const TargetRegisterClass *>;

  // Returns false with error() set; the undo log restores the instruction.
  bool repair(MachineInstr &MI, const OperandClasses &OriginalClasses) {
    bool HasFixedDef = false;
    for (auto [Index, OriginalRC] : OriginalClasses) {
      const auto &Op = MI.getOperand(Index);
      if (!Op.isReg() || !Op.isDef())
        continue;
      HasFixedDef = true;
      const auto *Required = MI.getRegClassConstraint(Index, &TII, &TRI);
      if (Required && !Required->contains(Op.getReg())) {
        // Compare banks through tuple leaves: a class such as AV_128_Align2
        // contains AGPR tuples, none of which is an AGPR_32 member itself.
        const TargetRegisterClass *Bank = bankOf(Op.getReg());
        StringRef Hint;
        if (Bank && llvm::any_of(*Required, [&](MCRegister R) { return bankOf(R) == Bank; }))
          Hint = "; the instruction accepts this register bank but not this start, for example "
                 "an odd tuple start on gfx90a and later: raise the storage alignment in bytes";
        else if (Bank == &AGPR)
          // An MFMA accumulator is selected in VGPR form on gfx90a and later;
          // only LLVM's MFMA rewriting could move it to AGPRs (planned).
          Hint = "; fixed AGPR placement of a result LLVM selects in VGPR form (such as an MFMA "
                 "accumulator) is not supported yet";
        Error =
            (Twine("explicit register definition cannot use ") + TRI.getName(Op.getReg()) +
             " for " + TII.getName(MI.getOpcode()) + " (requires " + TRI.getRegClassName(Required) +
             "); automatic write-back copies are disabled; choose a compatible register "
             "class or remove register_place" +
             Hint)
                .str();
        return false;
      }
    }
    // COPY has no descriptor register classes. It still must not turn a wave
    // value into a scalar by silently selecting a lane. The scalar placement
    // guard runs before substitution; this also covers pre-existing copies.
    if (MI.isCopy() && isScalar(MI.getOperand(0).getReg()) && !isScalar(MI.getOperand(1).getReg()))
      return fail(MI, "vector-to-scalar transfer requires an explicit uniform conversion");
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
          return fail(MI, "tied operands must name the same complete physical register");
        const auto *Required = MI.getRegClassConstraint(Index, &TII, &TRI);
        if (Required) {
          NeedsBridge |= !Required->contains(placement(Op));
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
      if (HasFixedDef)
        return fail(MI, "instructions producing explicit registers must also accept their fixed "
                        "inputs directly; choose compatible register classes or remove "
                        "register_place");
      if (MI.isBundled() || MI.isTerminator() || MI.isInlineAsm())
        return fail(MI, "operand needs a copy at an unsupported instruction boundary");
      TempRC = TRI.getAllocatableClass(TempRC);
      if (!TempRC)
        return fail(MI, "no allocatable class satisfies the tied operand constraints");
      Register Temp = MRI.createVirtualRegister(TempRC);
      MachineBasicBlock &MBB = *MI.getParent();
      // Only uses reach here; definitions of placed values failed above.
      for (unsigned Index : Group) {
        auto &Op = MI.getOperand(Index);
        MCRegister Phys = placement(Op);
        Register Fixed = Op.getReg();
        unsigned FixedSub = Op.getSubReg();
        if (TRI.getRegSizeInBits(*TempRC) !=
            TRI.getRegSizeInBits(*TRI.getMinimalPhysRegClass(Phys)))
          return fail(MI, "copy would change the operand width");
        // Vector-to-scalar transfers are not bitwise copies of wave values.
        // Never silently pick a lane from a divergent value.
        if (!Op.isUndef()) {
          if (isScalar(Temp) && !isScalar(Phys))
            return fail(MI, "vector-to-scalar transfer requires an explicit uniform conversion");
          Log.inserted(BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(TargetOpcode::COPY), Temp)
                           .addReg(Fixed, {}, FixedSub));
        }
        Log.operand(Op);
        Op.setReg(Temp);
        Op.setSubReg(0);
        Op.setIsKill(false);
      }
    }
    // Descriptor classes are not the only target constraints (e.g. AMDGPU's
    // constant bus). Reject if the repaired instruction is still illegal.
    StringRef Reason;
    if (!TII.verifyInstruction(MI, Reason))
      return fail(MI, Reason);
    return true;
  }
};

// Use the full LLVM verifier, including virtual operand classes and liveness.
// Capture its diagnostic explicitly: release builds need useful failure text too.
std::optional<std::string> verifyFunction(MachineFunction &MF, const char *Banner) {
  std::string Diagnostic;
  raw_string_ostream OS(Diagnostic);
  if (MF.verify(nullptr, Banner, &OS, /*AbortOnError=*/false))
    return std::nullopt;
  return (Twine(Banner) + ":\n" + Diagnostic).str();
}

// Verification failures of code this backend produced are internal errors.
void verifyPlacementFunction(MachineFunction &MF, const char *Banner) {
  if (auto Diagnostic = verifyFunction(MF, Banner))
    report_fatal_error(Twine(*Diagnostic));
}

class ApplyRegisterPlacement : public MachineFunctionPass {
private:
  std::shared_ptr<PlacementState> State;

public:
  static char ID;

  explicit ApplyRegisterPlacement(std::shared_ptr<PlacementState> State)
      : MachineFunctionPass(ID), State(std::move(State)) {}

  StringRef getPassName() const override { return "ApplyRegisterPlacement"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    const Function &F = MF.getFunction();
    if (!State->Functions.contains(&F) || State->Failed.contains(&F))
      return false;

    auto &MRI = MF.getRegInfo();
    const auto &TRI = *MF.getSubtarget().getRegisterInfo();
    verifyPlacementFunction(MF, "Before register placement");

    // Plan without changing machine code, so a rejection only drops carriers.
    DenseMap<Register, Placement> Bindings;
    SmallVector<MachineInstr *> Markers;
    auto Reject = [&](std::optional<uint64_t> Request, const Twine &Message) {
      reject(*State, F, Request, Message);
      for (MachineInstr *MI : Markers)
        MI->eraseFromParent();
      return true;
    };

    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB)
        if (getCarrier(MI, *State))
          Markers.push_back(&MI);
    if (Markers.empty())
      return Reject(std::nullopt, "register placement lost all carriers");

    for (MachineInstr *MI : Markers) {
      const Carrier *C = getCarrier(*MI, *State);
      unsigned UnitBits = TRI.getRegSizeInBits(*TRI.getRegClass(C->ClassID));
      std::optional<std::string> Conflict;
      auto Error = forEachCarrierWord(*MI, *C, [&](unsigned Word, const MachineOperand *MO) {
        if (!MO)
          return;
        int64_t Offset = C->BitOffset + Word * UnitBits;
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
            Bindings.try_emplace(Value, Placement{C->ClassID, Base, C->Request, {}});
        auto &P = It->second;
        if (!Inserted && (P.ClassID != C->ClassID || P.BitOffset != Base)) {
          if (!Conflict)
            Conflict = "conflicting register placements after machine optimization";
          return;
        }
        if (!llvm::is_contained(P.Storage, std::pair(C->StorageBegin, C->StorageEnd)))
          P.Storage.push_back({C->StorageBegin, C->StorageEnd});
      });
      if (Error)
        return Reject(C->Request, *Error);
      if (Conflict)
        return Reject(C->Request, *Conflict);
    }

    if (Bindings.empty())
      return Reject(getCarrier(*Markers.front(), *State)->Request,
                    "explicit register placement has no materialized SSA values; "
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
        return Reject(P.Request, Twine("explicit register carrier has no surviving machine uses; "
                                       "placement cannot be enforced: ") +
                                     Detail);
      }
      const auto &RC = *TRI.getRegClass(P.ClassID);
      unsigned UnitBits = TRI.getRegSizeInBits(RC);
      if (P.BitOffset < 0 || P.BitOffset % UnitBits)
        return Reject(P.Request, "explicit register slices must start on a register boundary");
      const auto *OriginalRC = MRI.getRegClass(VReg);
      unsigned Bits = TRI.getRegSizeInBits(*OriginalRC);
      unsigned First = P.BitOffset / UnitBits;
      unsigned Count = divideCeil(Bits, UnitBits);
      // Coalescing may widen the machine value beyond the requested slice.
      // Its other registers must still belong to a bound allocation.
      for (unsigned I = 0; I < Count; ++I) {
        int64_t Leaf = int64_t(First + I) * UnitBits;
        if (llvm::none_of(P.Storage, [&](const auto &Range) {
              return Range.first <= Leaf && Leaf + UnitBits <= Range.second;
            }))
          return Reject(P.Request,
                        "machine coalescing extended a fixed value outside its register storage");
      }
      MCRegister Phys = findRegisterTuple(TRI, RC, First, Bits);
      if (!Phys)
        return Reject(P.Request, Twine("no LLVM physical register tuple for ") +
                                     TRI.getRegClassName(&RC) + " index " + Twine(First) +
                                     " and width " + Twine(Bits));
      // SGPRs hold uniform wave values. Never force a divergent VGPR value
      // into an SGPR or remove constraints of a special scalar subclass.
      if (StringRef(TRI.getRegClassName(&RC)) == "SGPR_32" && !OriginalRC->contains(Phys))
        return Reject(P.Request,
                      "SGPR placement requires a compatible uniform LLVM register class");
      for (unsigned I = 0; I < Count; ++I)
        FixedRegs.insert(registerAt(RC, First + I));
      Assignments[VReg] = Phys;
    }
    // Fixed placements are hard constraints, including between distinct
    // allocations. Do not silently overwrite a simultaneously live fixed value.
    auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    for (auto I = Assignments.begin(); I != Assignments.end(); ++I)
      for (auto J = std::next(I); J != Assignments.end(); ++J)
        if (TRI.regsOverlap(I->second, J->second) &&
            LIS.getInterval(I->first).overlaps(LIS.getInterval(J->first)))
          return Reject(Bindings.lookup(I->first).Request,
                        "overlapping live intervals in explicit register storage");

    // Commit. Every change is logged; a rejection below restores the code.
    for (MachineInstr *MI : Markers)
      MI->eraseFromParent();
    Markers.clear();
    UndoLog Log;
    auto Undo = [&](std::optional<uint64_t> Request, const Twine &Message) {
      Log.undo();
      reject(*State, F, Request, Message);
      return true;
    };
    DenseMap<MachineInstr *, RegisterUseBridges::OperandClasses> ChangedInstructions;
    DenseMap<MachineInstr *, uint64_t> InstructionRequests;
    for (auto [VReg, Phys] : Assignments) {
      uint64_t Request = Bindings.lookup(VReg).Request;
      for (MachineOperand &MO : make_early_inc_range(MRI.reg_operands(VReg))) {
        const auto *OriginalRC = MRI.getRegClass(VReg);
        if (MO.getSubReg())
          OriginalRC = TRI.getSubRegisterClass(OriginalRC, MO.getSubReg());
        ChangedInstructions[MO.getParent()][MO.getOperandNo()] = OriginalRC;
        InstructionRequests.try_emplace(MO.getParent(), Request);
        Log.operand(MO);
        MO.substPhysReg(Phys, TRI);
        if (MO.isUse())
          MO.setIsKill(false);
        if (MO.isDef())
          MO.setIsDead(false);
      }
    }

    // Never select a target-specific encoding here.
    RegisterUseBridges Bridges(MF, Log);
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : make_early_inc_range(MBB))
        if (auto It = ChangedInstructions.find(&MI); It != ChangedInstructions.end())
          if (!Bridges.repair(MI, It->second))
            return Undo(InstructionRequests.lookup(&MI), Bridges.error());
    // These newly physical values may cross loop/back edges. LLVM's generic
    // live-in computation is not rerun here, so add the explicit storage
    // live-ins ourselves and iterate to a fixed point.
    bool Changed;
    do {
      Changed = false;
      for (MachineBasicBlock &MBB : reverse(MF)) {
        LivePhysRegs Live;
        computeLiveIns(Live, MBB);
        for (MCRegister Reg : FixedRegs)
          if (Live.contains(Reg) && !MBB.isLiveIn(Reg)) {
            MBB.addLiveIn(Reg);
            Log.liveIn(&MBB, Reg);
            Changed = true;
          }
        MBB.sortUniqueLiveIns();
      }
    } while (Changed);
    // Do not pass this pass's cached LiveIntervals: substitution and bridging
    // invalidate it, and normal RA will recompute it after this pass returns.
    if (auto Diagnostic = verifyFunction(MF, "After register placement"))
      return Undo(std::nullopt, *Diagnostic);
    return true;
  }
};

char ApplyRegisterPlacement::ID;
// Registered so LLVM's -print-after/-stop-after options can name this pass.
static RegisterPass<ApplyRegisterPlacement> ApplyRegistration("fly-apply-register-placement",
                                                              "FlyDSL apply register placement");

class VerifyRegisterPlacement : public MachineFunctionPass {
private:
  std::shared_ptr<PlacementState> State;
  const char *Banner;

public:
  static char ID;

  VerifyRegisterPlacement(std::shared_ptr<PlacementState> State, const char *Banner)
      : MachineFunctionPass(ID), State(std::move(State)), Banner(Banner) {}

  StringRef getPassName() const override { return "FlyDSL verify register placement"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    const Function &F = MF.getFunction();
    if (!State->Functions.contains(&F) || State->Failed.contains(&F))
      return false;
    verifyPlacementFunction(MF, Banner);
    return false;
  }
};

char VerifyRegisterPlacement::ID;

} // namespace

namespace mlir::fly {

void registerRegisterPlacementCodegen() {
  static RegisterTargetPassConfigCallback Callback(
      [](TargetMachine &TM, legacy::PassManagerBase &PM, TargetPassConfig *Config) {
        if (!TM.getTargetTriple().isAMDGPU())
          return;
        auto State = std::make_shared<PlacementState>();
        PM.add(new PrepareRegisterPlacement(TM, State));

        Config->insertPass(&FinalizeISelID, new ValidateRegisterPlacement(State));
        Config->insertPass(&RenameIndependentSubregsID, new ApplyRegisterPlacement(State));
        // AMDGPU can run register allocation/rewriting more than once. The
        // standard post-RA expansion point runs after all those phases.
        Config->insertPass(&ExpandPostRAPseudosID,
                           new VerifyRegisterPlacement(State, "After post-RA pseudo expansion"));
      });
}

} // namespace mlir::fly
