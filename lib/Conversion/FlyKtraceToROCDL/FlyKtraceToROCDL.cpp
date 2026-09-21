// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Expands fly_ktrace annotations into the IR that actually writes a trace record.
// This is where that expansion lives; the Python frontend only emits the
// annotations.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "flydsl/Conversion/FlyKtraceToROCDL/FlyKtraceToROCDL.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyKtrace/IR/Dialect.h"

namespace mlir {
#define GEN_PASS_DEF_FLYKTRACETOROCDLCONVERSIONPASS
#include "flydsl/Conversion/FlyKtraceToROCDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::fly_ktrace;

namespace {

// Record layout, 32 bytes. Must match python/flydsl/expr/experimental/ktrace.py, which decodes
// it with struct.unpack("<QQIIII").
constexpr int64_t kRecordBytes = 32;
constexpr int64_t kOffTimestamp = 0;
constexpr int64_t kOffPayload = 8;
constexpr int64_t kOffHwId = 16;
constexpr int64_t kOffXccId = 20;
constexpr int64_t kOffPacked = 24;
constexpr int64_t kOffStartSlot = 28;

// Event kinds, stored in the top byte of the packed dword. Must match
// python/flydsl/expr/experimental/ktrace.py.
constexpr uint32_t kKindMark = 0;
constexpr uint32_t kKindRangePush = 1;
constexpr uint32_t kKindRangePop = 2;
constexpr uint32_t kKindRangeStart = 3;
constexpr uint32_t kKindRangeEnd = 4;

// Slot id for a range whose START record was never written. Must be a value no
// real slot can take; matches ktrace.UNPAIRED_SLOT.
constexpr int64_t kUnpairedSlot = 0xFFFFFFFF;

// Used when the module carries no fly_ktrace.capacity_slots -- a hand-written test,
// or fly-opt run directly. Matches env.ktrace.buffer_bytes's default.
constexpr int64_t kDefaultCapacitySlots = (64 << 20) / kRecordBytes - 1;

/// Per-kernel state, built once and reused by every event in that kernel.
struct KernelTraceState {
  Value base;   // i32: first slot this wave owns
  Value bufPtr; // !llvm.ptr<1>: the trace buffer parameter
  Value hwId;   // i32
  Value xccId;  // i32
  // i1: wave-uniform FLYDSL_KTRACE_BLOCKS predicate, null when recording every
  // workgroup. Gates both the slot claim and every store -- see blockFilter.
  Value recording;
};

Value i32Const(OpBuilder &b, Location loc, int64_t v) {
  return b.create<LLVM::ConstantOp>(loc, b.getI32Type(), b.getI32IntegerAttr(v));
}

/// Byte-offset GEP on a global-address-space pointer.
Value gepBytes(OpBuilder &b, Location loc, Value ptr, Value byteOffset) {
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext(), /*addressSpace=*/1);
  return b.create<LLVM::GEPOp>(loc, ptrTy, b.getI8Type(), ptr, ValueRange{byteOffset},
                               LLVM::GEPNoWrapFlags::none);
}

/// `llvm.call_intrinsic` with no operand bundles.
Value callIntrinsic(OpBuilder &b, Location loc, StringRef name, Type resultTy,
                    ValueRange args = {}) {
  auto op = b.create<LLVM::CallIntrinsicOp>(loc, resultTy, b.getStringAttr(name), args);
  return op.getResult(0);
}

/// s.getreg's immarg: id | (size - 1) << 11, offset 0.
Value hwregImmarg(OpBuilder &b, Location loc, int64_t regId, int64_t size) {
  return i32Const(b, loc, regId | ((size - 1) << 11));
}

/// Index of the lowest active lane, as i32. ballot(true) is exec, so cttz of it
/// is the first active lane.
Value firstActiveLane(OpBuilder &b, Location loc) {
  Type i64 = b.getI64Type();
  Value one = b.create<LLVM::ConstantOp>(loc, b.getI1Type(), b.getIntegerAttr(b.getI1Type(), 1));
  Value mask = b.create<ROCDL::BallotOp>(loc, i64, one);
  // is_zero_poison: the mask is exec inside an executing region, so it is never
  // zero. Saying so lets the backend emit a bare s_ff1_i32_b64 instead of the
  // defined-on-zero form, which would add a compare-and-select to every event.
  Value cttz = b.create<LLVM::CountTrailingZerosOp>(loc, i64, mask,
                                                    /*is_zero_poison=*/true);
  return b.create<LLVM::TruncOp>(loc, b.getI32Type(), cttz);
}

/// Build the FLYDSL_KTRACE_BLOCKS predicate, or a null Value to record every
/// workgroup. Accepts the spellings FLYDSL_KTRACE_BLOCKS documents: "", "all",
/// "xcc:N", and "x,y,z".
///
/// Filtering on the device rather than in post-processing means a non-recording
/// wave pays one comparison at entry and writes nothing at all.
///
/// Returns failure (after emitting a diagnostic) on a malformed spec, rather than
/// asserting: a bad environment variable must not take down the compiler.
FailureOr<Value> blockFilter(OpBuilder &b, Location loc, StringRef spec, Value xccId,
                             Operation *diagOp) {
  spec = spec.trim();
  if (spec.empty() || spec == "all")
    return Value();

  if (spec.consume_front("xcc:")) {
    uint32_t want;
    if (spec.getAsInteger(10, want))
      return diagOp->emitError("ktrace: FLYDSL_KTRACE_BLOCKS 'xcc:N' needs an "
                               "integer N, got ")
             << spec;
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, xccId, i32Const(b, loc, want))
        .getResult();
  }

  SmallVector<StringRef, 4> parts;
  spec.split(parts, ',');
  if (parts.size() != 3)
    return diagOp->emitError("ktrace: FLYDSL_KTRACE_BLOCKS must be 'x,y,z', "
                             "'xcc:N' or 'all'; got ")
           << spec;

  static constexpr gpu::Dimension kDims[] = {gpu::Dimension::x, gpu::Dimension::y,
                                             gpu::Dimension::z};
  Type i32 = b.getI32Type();
  Value cond;
  for (auto [dim, part] : llvm::zip_equal(kDims, parts)) {
    uint32_t want;
    if (part.trim().getAsInteger(10, want))
      return diagOp->emitError("ktrace: FLYDSL_KTRACE_BLOCKS 'x,y,z' needs "
                               "integer coordinates, got ")
             << spec;
    // gpu.block_id is index-typed; convert-gpu-to-rocdl runs after this pass and
    // lowers it. The compare is i32 to match the rest of the record arithmetic.
    Value got = b.create<gpu::BlockIdOp>(loc, dim);
    got = b.create<arith::IndexCastOp>(loc, i32, got);
    Value term =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, got, i32Const(b, loc, want));
    cond = cond ? b.create<arith::AndIOp>(loc, cond, term).getResult() : term;
  }
  return cond;
}

/// Emit the once-per-wave prologue at the kernel's entry block.
///
/// Placed at the entry block rather than at the first event: the base, buffer
/// pointer and wave ids are used by every later event, so emitting them inside an
/// scf.for body would leave them failing to dominate uses outside that region.
FailureOr<KernelTraceState> emitPrologue(gpu::GPUFuncOp kernel, int64_t eventsPerWave,
                                         StringRef blocksSpec) {
  Block &entry = kernel.getBody().front();
  if (entry.getNumArguments() == 0)
    return kernel.emitError("ktrace: kernel has no arguments, so no trace-buffer parameter; the "
                            "kernel was traced without ktrace enabled");

  OpBuilder b(&entry, entry.begin());
  Location loc = kernel.getLoc();
  Type i32Ty = b.getI32Type();

  KernelTraceState st;
  // The buffer arrives as the trailing implicit kernel parameter, added during
  // tracing by kernel_function.py. A pass cannot add it here.
  //
  // It is a `!fly.ptr<i8, global>`, not a bare LLVM pointer -- the frontend
  // appends it as a PointerJitArg, whose IR type is Fly's. The atomics, GEPs and
  // stores below are all LLVM ops, so it has to be converted first.
  Value rawBuf = entry.getArgument(entry.getNumArguments() - 1);
  auto flyPtrTy = dyn_cast<fly::PointerType>(rawBuf.getType());
  if (!flyPtrTy)
    return kernel.emitError("ktrace: expected the trailing kernel argument to be a "
                            "!fly.ptr trace buffer, got ")
           << rawBuf.getType();

  auto llvmPtrTy = LLVM::LLVMPointerType::get(b.getContext(), /*addressSpace=*/1);
  st.bufPtr = b.create<fly::ToLLVMPtrOp>(loc, llvmPtrTy, rawBuf, b.getI32IntegerAttr(1));

  // HW_ID (id 4, whole 32-bit register) and XCC_ID (id 20, bits [3:0]) are
  // wave-invariant on CDNA -- waves are CU-resident -- so they are read once.
  st.hwId =
      callIntrinsic(b, loc, "llvm.amdgcn.s.getreg", b.getI32Type(), {hwregImmarg(b, loc, 4, 32)});
  st.xccId =
      callIntrinsic(b, loc, "llvm.amdgcn.s.getreg", b.getI32Type(), {hwregImmarg(b, loc, 20, 4)});

  // Computed before the claim because it gates it: with a filter in a large grid
  // every wave would still claim its range, and the cursor runs past the buffer
  // even though only a handful ever record -- the claim itself then overflows,
  // which is an illegal memory access rather than a truncated trace.
  FailureOr<Value> recording = blockFilter(b, loc, blocksSpec, st.xccId, kernel);
  if (failed(recording))
    return failure();
  st.recording = *recording;

  // One atomic per wave: claim a contiguous slot range up front. The cursor lives
  // in the buffer's reserved slot 0, so the host reads the same word the kernel
  // incremented.
  //
  // Confined to ONE lane and broadcast: unguarded, the atomic runs on all 64 lanes,
  // so a wave claims 64 ranges, the buffer is exhausted 64x early, and waves
  // claiming past the end store out of bounds -- an illegal access on device, not a
  // truncated trace. readfirstlane, because every lane needs the same base after.
  Value perWave = i32Const(b, loc, eventsPerWave);
  auto emitClaim = [&]() -> Value {
    Value lane = b.create<arith::IndexCastOp>(
        loc, i32Ty, b.create<gpu::LaneIdOp>(loc, /*upper_bound=*/nullptr));
    Value isLeader =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane, firstActiveLane(b, loc));
    auto claimIf = b.create<scf::IfOp>(loc, TypeRange{i32Ty}, isLeader,
                                       /*withElseRegion=*/true);
    b.setInsertionPointToStart(claimIf.thenBlock());
    Value claimed = b.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, st.bufPtr, perWave,
                                                LLVM::AtomicOrdering::monotonic);
    // The atomic returns the OLD cursor, so the first wave would get base 0 --
    // whose counter word is the global cursor itself. Shift past it.
    b.create<scf::YieldOp>(loc,
                           ValueRange{b.create<arith::AddIOp>(loc, claimed, i32Const(b, loc, 1))});
    b.setInsertionPointToStart(claimIf.elseBlock());
    b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, 0)});
    b.setInsertionPointAfter(claimIf);
    // Only the leader's lane holds the claimed base; the rest yielded 0.
    return b.create<ROCDL::ReadfirstlaneOp>(loc, i32Ty, claimIf.getResult(0));
  };

  if (!st.recording) {
    st.base = emitClaim();
    return st;
  }

  auto claimIf =
      b.create<scf::IfOp>(loc, TypeRange{b.getI32Type()}, st.recording, /*withElseRegion=*/true);
  b.setInsertionPointToStart(claimIf.thenBlock());
  b.create<scf::YieldOp>(loc, ValueRange{emitClaim()});
  b.setInsertionPointToStart(claimIf.elseBlock());
  // Non-recording waves never store, so any base works; 0 costs no atomic.
  b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, 0)});
  st.base = claimIf.getResult(0);
  return st;
}

/// What distinguishes the five ops. Everything else about the expansion -- the
/// leader guard, the slot claim, the bounds checks, the six stores -- is identical,
/// so they share one template and differ only in these fields.
struct EventShape {
  uint32_t kind;
  uint32_t eventId;  // 0 for range_pop, which takes its name from the push
  Value payload;     // null when the annotation carried none
  Value startSlot;   // only range_end pairs on a token; null otherwise
  bool producesSlot; // range_start hands its record's slot to its range_end
};

/// Expand one annotation in place. Returns the record's absolute slot when the op
/// produced one, so the caller can replace its result.
Value expandEvent(Operation *op, const EventShape &shape, const KernelTraceState &st,
                  int64_t eventsPerWave, int64_t capacitySlots) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  Type i32 = b.getI32Type();

  // One record per wave, not 64: the first ACTIVE lane writes it. Lane 0 may be
  // inactive inside a divergent region, and the event would vanish.
  Value lane =
      b.create<arith::IndexCastOp>(loc, i32, b.create<gpu::LaneIdOp>(loc, /*upper_bound=*/nullptr));
  Value leader = firstActiveLane(b, loc);
  Value isLeader = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane, leader);
  // A filtered-out wave took the zero base in the prologue, so it must not reach
  // a store: its slots would land on top of a recording wave's range.
  if (st.recording)
    isLeader = b.create<arith::AndIOp>(loc, isLeader, st.recording);

  // Both the counter bump and the store sit inside the leader guard: bumping from
  // all 64 lanes would consume 64 slots per event while only one lane records.
  // Owned: a TypeRange is a non-owning view, and the second guard below would
  // read it after its temporary died.
  SmallVector<Type, 1> guardResults;
  if (shape.producesSlot)
    guardResults.push_back(i32);
  auto leaderIf = b.create<scf::IfOp>(loc, guardResults, isLeader,
                                      /*withElseRegion=*/shape.producesSlot);
  b.setInsertionPointToStart(leaderIf.thenBlock());

  // A wave whose claim landed past the end of the buffer must not touch memory at
  // all, including its counter word at `base` -- that one is not covered by the
  // record bounds checks below, so an over-subscribed launch would fault on device
  // rather than being rejected by the host afterwards.
  Value baseInBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, st.base,
                                               i32Const(b, loc, capacitySlots));
  auto baseIf = b.create<scf::IfOp>(loc, guardResults, baseInBounds,
                                    /*withElseRegion=*/shape.producesSlot);
  b.setInsertionPointToStart(baseIf.thenBlock());

  // The counter is the wave's own first reserved word, so this atomic is
  // uncontended. It is a RUNTIME index: a call site inside a loop is traced once
  // but executes every iteration, and a compile-time constant would make every
  // iteration overwrite the same slot.
  Value ctrPtr = gepBytes(b, loc, st.bufPtr,
                          b.create<arith::MulIOp>(loc, st.base, i32Const(b, loc, kRecordBytes)));
  Value local = b.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, ctrPtr,
                                            i32Const(b, loc, 1), LLVM::AtomicOrdering::monotonic);
  Value slot = b.create<arith::AddIOp>(loc, st.base, local);

  // UNSIGNED compares. `local` and `slot` come from an i32 atomic add that is
  // never saturated, so past 2^31 a signed compare flips from "in bounds" to
  // "always pass" and the store GEPs to a negative byte offset. The host reads
  // the cursor as c_uint32, so unsigned is the shared reading.
  Value withinWave = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local,
                                             i32Const(b, loc, eventsPerWave - 1));
  Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, slot,
                                           i32Const(b, loc, capacitySlots));
  Value storeOk = b.create<arith::AndIOp>(loc, withinWave, inBounds);

  auto storeIf = b.create<scf::IfOp>(loc, guardResults, storeOk,
                                     /*withElseRegion=*/shape.producesSlot);
  b.setInsertionPointToStart(storeIf.thenBlock());

  // Slot 0 is the global cursor; slot `base` is this wave's counter word; records
  // follow at base+1+local. `slot` is already base+local, so one more skips the
  // counter.
  Value slot1 = b.create<arith::AddIOp>(loc, slot, i32Const(b, loc, 1));
  Value byteOff = b.create<arith::MulIOp>(loc, slot1, i32Const(b, loc, kRecordBytes));
  Value recPtr = gepBytes(b, loc, st.bufPtr, byteOff);

  Value ts = callIntrinsic(b, loc, "llvm.amdgcn.s.memrealtime", b.getI64Type());
  Value payload =
      shape.payload
          ? shape.payload
          : b.create<LLVM::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(0)).getResult();
  // A range_end's token operand is still !fly_ktrace.token here if its
  // range_start has not been expanded yet, but program order guarantees it has:
  // expandEvent replaced that result with the i32 slot the guards yield. Assert
  // rather than emit a store of the wrong type, which llvm.store rejects by
  // crashing in its builder rather than with a diagnostic.
  Value startSlot = i32Const(b, loc, 0);
  if (shape.startSlot) {
    assert(shape.startSlot.getType() == i32 &&
           "range_end's token must already be lowered to its i32 slot");
    startSlot = shape.startSlot;
  }
  uint32_t packed = (shape.eventId & 0xFFFFFF) | ((shape.kind & 0xFF) << 24);

  struct Field {
    int64_t offset;
    Value value;
  };
  Field fields[] = {
      {kOffTimestamp, ts},
      {kOffPayload, payload},
      {kOffHwId, st.hwId},
      {kOffXccId, st.xccId},
      {kOffPacked, i32Const(b, loc, static_cast<int64_t>(packed))},
      {kOffStartSlot, startSlot},
  };
  for (const Field &f : fields) {
    Value fieldPtr = gepBytes(b, loc, recPtr, i32Const(b, loc, f.offset));
    b.create<LLVM::StoreOp>(loc, f.value, fieldPtr);
  }

  if (!shape.producesSlot) {
    op->erase();
    return Value();
  }

  // The record's own slot is what a range_end pairs on, so it is yielded out of
  // both guards. A suppressed store -- and a non-leader lane -- yields
  // UNPAIRED_SLOT instead: yielding the un-offset slot would alias the record
  // written one event earlier, and its range_end would pop an unrelated range.
  b.create<scf::YieldOp>(loc, ValueRange{slot1});

  b.setInsertionPointToStart(storeIf.elseBlock());
  b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, kUnpairedSlot)});

  // Out through baseIf, then out through leaderIf: the slot has to cross both of
  // the guards it was computed inside.
  b.setInsertionPointAfter(storeIf);
  b.create<scf::YieldOp>(loc, ValueRange{storeIf.getResult(0)});

  b.setInsertionPointToStart(baseIf.elseBlock());
  b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, kUnpairedSlot)});

  b.setInsertionPointAfter(baseIf);
  b.create<scf::YieldOp>(loc, ValueRange{baseIf.getResult(0)});

  b.setInsertionPointToStart(leaderIf.elseBlock());
  b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, kUnpairedSlot)});

  Value result = leaderIf.getResult(0);
  op->getResult(0).replaceAllUsesWith(result);
  op->erase();
  return result;
}

/// s_memrealtime and HW_REG_XCC_ID exist on CDNA3 and CDNA4 only. The gate reads
/// the module's own target, which is the compile target rather than the host.
///
/// Refusing loudly matters more than it looks: on RDNA `s_getreg` does not error,
/// it silently retargets hwreg(4) and hwreg(20,0,4) to WAVE_STATE_PRIV /
/// WAVE_SCRATCH_BASE_LO, so an ungated compile produces a kernel that assembles
/// and writes plausible garbage.
LogicalResult checkTargetArch(gpu::GPUModuleOp mod, StringRef chipHint) {
  auto rejectChip = [&](StringRef chip) {
    return mod.emitError("ktrace: in-kernel timestamps are not supported on "
                         "target arch '")
           << chip
           << "'; supported: gfx942 (CDNA3) and gfx950 (CDNA4). s_memrealtime "
              "is absent on RDNA, and s_getreg silently reads a different "
              "register there rather than failing";
  };

  ArrayAttr targets = mod.getTargetsAttr();
  if (!targets || targets.empty()) {
    // An extern-linked kernel has its targets stripped before compile, because
    // rocdl-attach-target is then the sole source of them (jit_function.py). The
    // chip is still known, so the frontend publishes it and the gate uses that
    // rather than refusing to compile a kernel it could have checked.
    if (chipHint.empty())
      return mod.emitError("ktrace: gpu.module carries no target and no "
                           "fly_ktrace.chip; cannot verify that in-kernel "
                           "timestamps are supported");
    if (chipHint.starts_with("gfx942") || chipHint.starts_with("gfx950"))
      return success();
    return rejectChip(chipHint);
  }

  for (Attribute t : targets) {
    auto rocdlTarget = dyn_cast<ROCDL::ROCDLTargetAttr>(t);
    if (!rocdlTarget)
      continue;
    StringRef chip = rocdlTarget.getChip();
    if (chip.starts_with("gfx942") || chip.starts_with("gfx950"))
      return success();
    return rejectChip(chip);
  }
  return mod.emitError("ktrace: gpu.module has no ROCDL target");
}

/// Retypes an `arith.select` whose operands carry a token.
///
/// The SCF structural patterns cover for/if/while, but canonicalize turns an
/// `scf.if` that only picks between two existing tokens into a select, which they
/// do not touch. Nothing about the op depends on what the type means, so this is
/// the same mechanical retype they perform.
struct SelectTokenConversion : public OpConversionPattern<arith::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getType());
    if (!converted)
      return failure();
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, converted, adaptor.getCondition(),
                                                 adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

/// Retype every !fly_ktrace.token to i32 before anything is expanded.
///
/// expandEvent replaces a range_start's result in place, which retypes the *use*
/// but not the signature of any region carrying it: an scf.for's result and body
/// block argument stay !fly_ktrace.token, and the range_end inside then reads a
/// token where an i32 store is required. That is the loop-carried idiom the token
/// form exists for, so it is the common case, not a corner one. Rewriting region
/// signatures is what a dialect conversion does.
///
/// The degenerate `i32 to i32` casts it leaves behind are transparent; the
/// expansion reads through them and reconcile-unrealized-casts erases them.
LogicalResult legalizeTokenTypes(Operation *root) {
  MLIRContext *ctx = root->getContext();
  Type i32 = IntegerType::get(ctx, 32);

  TypeConverter converter;
  converter.addConversion([](Type t) { return t; });
  converter.addConversion([i32](fly_ktrace::TokenType) { return i32; });
  // The ktrace ops still carry token-typed values while this runs, so the
  // materialisations have to be buildable; an unrealized cast is folded away
  // once both sides agree.
  auto materialize = [](OpBuilder &b, Type resultType, ValueRange inputs, Location loc) -> Value {
    return b.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
  };
  converter.addSourceMaterialization(materialize);
  converter.addTargetMaterialization(materialize);

  ConversionTarget target(*ctx);
  // Only the structural carriers are under conversion. The ktrace ops declare
  // their token AnyType, so they are legal either way and are left for the
  // expansion walk.
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  scf::populateSCFStructuralTypeConversionTarget(converter, target);

  // arith.select carries a token the same way an scf.for iter_arg does, and it is
  // how canonicalize rewrites `scf.if` that only picks between two existing tokens.
  // Leaving it out let a legal kernel reach the expansion with a token-typed select
  // result, which asserts rather than diagnosing.
  target.addDynamicallyLegalOp<arith::SelectOp>(
      [&converter](arith::SelectOp op) { return converter.isLegal(op.getType()); });

  RewritePatternSet patterns(ctx);
  scf::populateSCFStructuralTypeConversions(converter, patterns);
  patterns.add<SelectTokenConversion>(converter, ctx);
  if (failed(applyPartialConversion(root, target, std::move(patterns))))
    return failure();

  // The conversion bridges each retyped value back to !fly_ktrace.token with an
  // unrealized cast, because the ktrace ops were left legal and still declare a
  // token operand. They take AnyType, so they can consume the i32 directly:
  // forward each cast's input to its users and drop it. Without this the
  // range_end inside a loop body still reads a token and the expansion aborts.
  SmallVector<UnrealizedConversionCastOp> dead;
  root->walk([&](UnrealizedConversionCastOp cast) {
    if (cast.getNumOperands() != 1 || cast.getNumResults() != 1)
      return;
    if (!cast.getOperand(0).getType().isInteger(32))
      return;
    if (!isa<fly_ktrace::TokenType>(cast.getResult(0).getType()))
      return;
    cast.getResult(0).replaceAllUsesWith(cast.getOperand(0));
    dead.push_back(cast);
  });
  for (UnrealizedConversionCastOp cast : dead)
    cast.erase();
  return success();
}

struct FlyKtraceToROCDLConversionPass
    : public impl::FlyKtraceToROCDLConversionPassBase<FlyKtraceToROCDLConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *root = getOperation();

    // Before any expansion: a token riding an scf.for iter_arg has to become an
    // i32 on the region signature too, not just at its uses.
    if (failed(legalizeTokenTypes(root))) {
      signalPassFailure();
      return;
    }

    // Event ids have to be unique across every module compiled in one process,
    // not merely within this one. The device writes only the integer, and records
    // from several kernels sit in one buffer -- they accumulate until a read --
    // so the host has nothing to disambiguate two kernels that both numbered from
    // 1. The frontend passes the highest id it has already handed out, and this
    // module numbers above it; the table is published as fly_ktrace.event_names.
    // Seeded with the bindings the process has already made, so a name that
    // appears in two modules keeps ONE id. Renumbering it per module would leave
    // the earlier module's records pointing at an id the host no longer maps,
    // and summarize would drop them without a word.
    llvm::StringMap<uint32_t> ids;
    uint32_t nextId = 1;
    if (auto known = root->getAttrOfType<DictionaryAttr>("fly_ktrace.known_event_names")) {
      for (NamedAttribute kv : known) {
        auto v = dyn_cast<IntegerAttr>(kv.getValue());
        if (!v)
          continue;
        uint32_t id = static_cast<uint32_t>(v.getInt());
        ids[kv.getName().strref()] = id;
        nextId = std::max(nextId, id + 1);
      }
    }

    auto idFor = [&ids, &nextId](StringRef name) {
      auto it = ids.find(name);
      if (it != ids.end())
        return it->second;
      uint32_t id = nextId++;
      ids[name] = id;
      return id;
    };

    WalkResult result = root->walk([&](gpu::GPUFuncOp kernel) {
      // Collect before rewriting: expandEvent erases as it goes, and range_start
      // replaces a result its range_end still refers to.
      SmallVector<Operation *> events;
      kernel.walk([&](Operation *op) {
        if (isa<MarkOp, RangePushOp, RangePopOp, RangeStartOp, RangeEndOp, SentinelTokenOp>(op))
          events.push_back(op);
      });
      if (events.empty())
        return WalkResult::advance();

      auto mod = kernel->getParentOfType<gpu::GPUModuleOp>();
      StringRef chipHint;
      if (auto attr = root->getAttrOfType<StringAttr>("fly_ktrace.chip"))
        chipHint = attr.getValue();
      if (!mod || failed(checkTargetArch(mod, chipHint)))
        return WalkResult::interrupt();

      // Slots each wave reserves. Fixed, not configurable: it used to be an env
      // var read independently by the emitter, the decoder and the cache key,
      // which is how a value changed between launch and collect() silently made
      // the decoder read counter words as records. Mirrors
      // ktrace.EVENTS_PER_WAVE; a test pins the two together.
      constexpr int64_t kEventsPerWave = 256;

      // The buffer size stays configurable -- it is the knob the overflow error
      // tells users to turn -- so the frontend publishes what it allocated and
      // the pass bakes that, rather than re-reading the environment and possibly
      // disagreeing with the host's allocation.
      int64_t capacitySlots = kDefaultCapacitySlots;
      if (auto attr = root->getAttrOfType<IntegerAttr>("fly_ktrace.capacity_slots"))
        capacitySlots = attr.getInt();

      // Which workgroups record. Published by the frontend beside
      // capacity_slots rather than re-read from the environment here, for the
      // same reason: one source of truth per compilation, and the binary ends
      // up self-describing.
      StringRef blocksSpec;
      if (auto attr = root->getAttrOfType<StringAttr>("fly_ktrace.blocks"))
        blocksSpec = attr.getValue();

      FailureOr<KernelTraceState> st = emitPrologue(kernel, kEventsPerWave, blocksSpec);
      if (failed(st))
        return WalkResult::interrupt();

      // One op at a time, in program order -- which kernel.walk already gives.
      // A range_end must read its token AFTER the range_start that produced it
      // has been expanded, because that expansion is what replaces the
      // !fly_ktrace.token result with the i32 slot llvm.store can write.
      // Reading every operand up front instead would leave the token at its
      // declared type and crash in the store builder.
      //
      // This in-place replacement is also why both ops declare the token
      // AnyType: a FlyKtrace_TokenType constraint makes TableGen emit a
      // cast<TypedValue<TokenType>> accessor that asserts the moment the
      // operand becomes an i32. See FlyKtrace_TokenType in TypeDefs.td.
      for (Operation *op : events) {
        // A sentinel writes no record: it only has to yield a value of the right
        // type for the iter_arg it seeds. UNPAIRED_SLOT rather than 0, so that a
        // token which somehow did reach a store could not alias slot 0 -- the
        // global cursor -- and be paired against it by the host.
        if (isa<SentinelTokenOp>(op)) {
          OpBuilder b(op);
          Value seed = i32Const(b, op->getLoc(), kUnpairedSlot);
          op->getResult(0).replaceAllUsesWith(seed);
          op->erase();
          continue;
        }

        EventShape shape;
        if (auto mark = dyn_cast<MarkOp>(op)) {
          shape = {kKindMark, idFor(mark.getEventName()), mark.getPayload(), Value(), false};
        } else if (auto push = dyn_cast<RangePushOp>(op)) {
          shape = {kKindRangePush, idFor(push.getEventName()), push.getPayload(), Value(), false};
        } else if (auto pop = dyn_cast<RangePopOp>(op)) {
          // No name: the host takes it from the matching push.
          shape = {kKindRangePop, 0, pop.getPayload(), Value(), false};
        } else if (auto start = dyn_cast<RangeStartOp>(op)) {
          shape = {kKindRangeStart, idFor(start.getEventName()), start.getPayload(), Value(), true};
        } else {
          auto end = cast<RangeEndOp>(op);
          shape = {kKindRangeEnd, idFor(end.getEventName()), end.getPayload(), end.getToken(),
                   false};
        }
        expandEvent(op, shape, *st, kEventsPerWave, capacitySlots);
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Publish the table the host needs to decode the trace. A record carries only
    // the 24-bit id, and these ids are assigned here, so without this attribute
    // the host has no way to map one back to a name -- summarize() would report
    // zero phases for a buffer full of records.
    //
    // It goes on the top-level module rather than per gpu.module because the ids
    // are module-global; see idFor above.
    if (!ids.empty()) {
      OpBuilder b(root->getContext());
      SmallVector<NamedAttribute> entries;
      entries.reserve(ids.size());
      for (const auto &kv : ids)
        entries.emplace_back(b.getStringAttr(kv.getKey()), b.getI32IntegerAttr(kv.getValue()));
      root->setAttr("fly_ktrace.event_names", b.getDictionaryAttr(entries));
    }
  }
};

} // namespace
