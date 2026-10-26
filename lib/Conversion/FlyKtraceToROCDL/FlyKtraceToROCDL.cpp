// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

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
#include "llvm/ADT/StringSwitch.h"

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

constexpr int64_t kRecordBytes = 32;
constexpr int64_t kOffTimestamp = 0;
constexpr int64_t kOffPayload = 8;
constexpr int64_t kOffHwId = 16;
constexpr int64_t kOffXccId = 20;
constexpr int64_t kOffPacked = 24;
constexpr int64_t kOffStartSlot = 28;

constexpr uint32_t kKindMark = 0;
constexpr uint32_t kKindRangePush = 1;
constexpr uint32_t kKindRangePop = 2;
constexpr uint32_t kKindRangeStart = 3;
constexpr uint32_t kKindRangeStop = 4;

constexpr int64_t kUnpairedSlot = 0xFFFFFFFF;

constexpr int64_t kDefaultCapacitySlots = (64 << 20) / kRecordBytes - 1;

struct KernelTraceState {
  Value base;
  Value bufPtr;
  Value hwId;
  Value xccId;
  Value recording;
};

Value i32Const(OpBuilder &b, Location loc, int64_t v) {
  return b.create<LLVM::ConstantOp>(loc, b.getI32Type(), b.getI32IntegerAttr(v));
}

Value gepBytes(OpBuilder &b, Location loc, Value ptr, Value byteOffset) {
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext(), 1);
  return b.create<LLVM::GEPOp>(loc, ptrTy, b.getI8Type(), ptr, ValueRange{byteOffset},
                               LLVM::GEPNoWrapFlags::none);
}

Value callIntrinsic(OpBuilder &b, Location loc, StringRef name, Type resultTy,
                    ValueRange args = {}) {
  auto op = b.create<LLVM::CallIntrinsicOp>(loc, resultTy, b.getStringAttr(name), args);
  return op.getResult(0);
}

Value hwregImmarg(OpBuilder &b, Location loc, int64_t regId, int64_t size) {
  return i32Const(b, loc, regId | ((size - 1) << 11));
}

Value firstActiveLane(OpBuilder &b, Location loc) {
  Type i64 = b.getI64Type();
  Value one = b.create<LLVM::ConstantOp>(loc, b.getI1Type(), b.getIntegerAttr(b.getI1Type(), 1));
  Value mask = b.create<ROCDL::BallotOp>(loc, i64, one);
  Value cttz = b.create<LLVM::CountTrailingZerosOp>(loc, i64, mask, true);
  return b.create<LLVM::TruncOp>(loc, b.getI32Type(), cttz);
}

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
    Value got = b.create<gpu::BlockIdOp>(loc, dim);
    got = b.create<arith::IndexCastOp>(loc, i32, got);
    Value term =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, got, i32Const(b, loc, want));
    cond = cond ? b.create<arith::AndIOp>(loc, cond, term).getResult() : term;
  }
  return cond;
}

FailureOr<KernelTraceState> emitPrologue(gpu::GPUFuncOp kernel, int64_t eventsPerWave,
                                         int64_t capacitySlots, StringRef blocksSpec) {
  Block &entry = kernel.getBody().front();
  if (entry.getNumArguments() == 0)
    return kernel.emitError("ktrace: kernel has no arguments, so no trace-buffer parameter; the "
                            "kernel was traced without ktrace enabled");

  OpBuilder b(&entry, entry.begin());
  Location loc = kernel.getLoc();
  Type i32Ty = b.getI32Type();

  KernelTraceState st;
  Value rawBuf = entry.getArgument(entry.getNumArguments() - 1);
  auto flyPtrTy = dyn_cast<fly::PointerType>(rawBuf.getType());
  if (!flyPtrTy)
    return kernel.emitError("ktrace: expected the trailing kernel argument to be a "
                            "!fly.ptr trace buffer, got ")
           << rawBuf.getType();

  auto llvmPtrTy = LLVM::LLVMPointerType::get(b.getContext(), 1);
  st.bufPtr = b.create<fly::ToLLVMPtrOp>(loc, llvmPtrTy, rawBuf, b.getI32IntegerAttr(1));

  Value firstThread;
  for (gpu::Dimension dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
    Value block = b.create<gpu::BlockIdOp>(loc, dim);
    Value thread = b.create<gpu::ThreadIdOp>(loc, dim);
    Value blockZero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, block,
                                              b.create<arith::ConstantIndexOp>(loc, 0));
    Value threadZero = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, thread,
                                               b.create<arith::ConstantIndexOp>(loc, 0));
    Value dimZero = b.create<arith::AndIOp>(loc, blockZero, threadZero);
    firstThread =
        firstThread ? b.create<arith::AndIOp>(loc, firstThread, dimZero).getResult() : dimZero;
  }
  auto launchIf = b.create<scf::IfOp>(loc, firstThread, false);
  b.setInsertionPointToStart(launchIf.thenBlock());
  Value launchPtr = gepBytes(b, loc, st.bufPtr, i32Const(b, loc, 4));
  b.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, launchPtr, i32Const(b, loc, 1),
                              LLVM::AtomicOrdering::monotonic);
  b.setInsertionPointAfter(launchIf);

  st.hwId =
      callIntrinsic(b, loc, "llvm.amdgcn.s.getreg", b.getI32Type(), {hwregImmarg(b, loc, 4, 32)});
  st.xccId =
      callIntrinsic(b, loc, "llvm.amdgcn.s.getreg", b.getI32Type(), {hwregImmarg(b, loc, 20, 4)});
  Value dispatchId = callIntrinsic(b, loc, "llvm.amdgcn.dispatch.id", b.getI64Type());
  auto queuePtrTy = LLVM::LLVMPointerType::get(b.getContext(), 4);
  Value queuePtr = callIntrinsic(b, loc, "llvm.amdgcn.queue.ptr", queuePtrTy);
  Value queueId = b.create<LLVM::PtrToIntOp>(loc, b.getI64Type(), queuePtr);

  FailureOr<Value> recording = blockFilter(b, loc, blocksSpec, st.xccId, kernel);
  if (failed(recording))
    return failure();
  st.recording = *recording;

  Value perWave = i32Const(b, loc, eventsPerWave);
  auto emitClaim = [&]() -> Value {
    Value lane = b.create<arith::IndexCastOp>(loc, i32Ty, b.create<gpu::LaneIdOp>(loc, nullptr));
    Value isLeader =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane, firstActiveLane(b, loc));
    auto claimIf = b.create<scf::IfOp>(loc, TypeRange{i32Ty}, isLeader, true);
    b.setInsertionPointToStart(claimIf.thenBlock());
    // Stop claiming once the cursor is past the buffer so the 32-bit cursor cannot wrap back
    // into range on long captures. A claim at exactly capacity still goes through, so the host
    // sees the cursor exceed capacity and reports the overflow.
    auto cursorLoad = b.create<LLVM::LoadOp>(loc, i32Ty, st.bufPtr);
    cursorLoad.setAlignment(4);
    cursorLoad.setOrdering(LLVM::AtomicOrdering::monotonic);
    Value cursor = cursorLoad;
    Value hasRoom = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ule, cursor,
                                            i32Const(b, loc, capacitySlots));
    auto roomIf = b.create<scf::IfOp>(loc, TypeRange{i32Ty}, hasRoom, true);
    b.setInsertionPointToStart(roomIf.thenBlock());
    Value fetched = b.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, st.bufPtr, perWave,
                                                LLVM::AtomicOrdering::monotonic);
    b.create<scf::YieldOp>(loc, ValueRange{fetched});
    b.setInsertionPointToStart(roomIf.elseBlock());
    b.create<scf::YieldOp>(loc, ValueRange{cursor});
    b.setInsertionPointAfter(roomIf);
    Value claimed = roomIf.getResult(0);
    Value base = b.create<arith::AddIOp>(loc, claimed, i32Const(b, loc, 1));
    Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, base,
                                             i32Const(b, loc, capacitySlots));
    auto metadataIf = b.create<scf::IfOp>(loc, inBounds, false);
    b.setInsertionPointToStart(metadataIf.thenBlock());
    Value byteOff = b.create<arith::AddIOp>(
        loc, b.create<arith::MulIOp>(loc, base, i32Const(b, loc, kRecordBytes)),
        i32Const(b, loc, 8));
    b.create<LLVM::StoreOp>(loc, dispatchId, gepBytes(b, loc, st.bufPtr, byteOff));
    Value queueByteOff = b.create<arith::AddIOp>(loc, byteOff, i32Const(b, loc, 8));
    b.create<LLVM::StoreOp>(loc, queueId, gepBytes(b, loc, st.bufPtr, queueByteOff));
    // Remaining header words: (block.z << 8 | wave in block) at +4, block.x at +24, block.y
    // at +28.
    auto blockId = [&](gpu::Dimension dim) -> Value {
      return b.create<arith::IndexCastOp>(loc, i32Ty, b.create<gpu::BlockIdOp>(loc, dim));
    };
    Value wave =
        b.create<arith::IndexCastOp>(loc, i32Ty, b.create<gpu::SubgroupIdOp>(loc, nullptr));
    Value waveWord = b.create<arith::OrIOp>(
        loc, b.create<arith::ShLIOp>(loc, blockId(gpu::Dimension::z), i32Const(b, loc, 8)), wave);
    Value headerOff = b.create<arith::SubIOp>(loc, byteOff, i32Const(b, loc, 4));
    b.create<LLVM::StoreOp>(loc, waveWord, gepBytes(b, loc, st.bufPtr, headerOff));
    Value blockXOff = b.create<arith::AddIOp>(loc, queueByteOff, i32Const(b, loc, 8));
    b.create<LLVM::StoreOp>(loc, blockId(gpu::Dimension::x),
                            gepBytes(b, loc, st.bufPtr, blockXOff));
    Value blockYOff = b.create<arith::AddIOp>(loc, blockXOff, i32Const(b, loc, 4));
    b.create<LLVM::StoreOp>(loc, blockId(gpu::Dimension::y),
                            gepBytes(b, loc, st.bufPtr, blockYOff));
    b.setInsertionPointAfter(metadataIf);
    b.create<scf::YieldOp>(loc, ValueRange{base});
    b.setInsertionPointToStart(claimIf.elseBlock());
    b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, 0)});
    b.setInsertionPointAfter(claimIf);
    return b.create<ROCDL::ReadfirstlaneOp>(loc, i32Ty, claimIf.getResult(0));
  };

  if (!st.recording) {
    st.base = emitClaim();
    return st;
  }

  auto claimIf = b.create<scf::IfOp>(loc, TypeRange{b.getI32Type()}, st.recording, true);
  b.setInsertionPointToStart(claimIf.thenBlock());
  b.create<scf::YieldOp>(loc, ValueRange{emitClaim()});
  b.setInsertionPointToStart(claimIf.elseBlock());
  b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, 0)});
  st.base = claimIf.getResult(0);
  return st;
}

struct EventShape {
  uint32_t kind;
  uint32_t eventId;
  Value payload;
  Value startSlot;
  bool producesSlot;
  uint32_t payloadType = 0;
};

Value expandEvent(Operation *op, const EventShape &shape, const KernelTraceState &st,
                  int64_t eventsPerWave, int64_t capacitySlots) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  Type i32 = b.getI32Type();

  Value lane = b.create<arith::IndexCastOp>(loc, i32, b.create<gpu::LaneIdOp>(loc, nullptr));
  Value leader = firstActiveLane(b, loc);
  Value isLeader = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lane, leader);
  if (st.recording)
    isLeader = b.create<arith::AndIOp>(loc, isLeader, st.recording);

  SmallVector<Type, 1> guardResults;
  if (shape.producesSlot)
    guardResults.push_back(i32);
  auto leaderIf = b.create<scf::IfOp>(loc, guardResults, isLeader, shape.producesSlot);
  b.setInsertionPointToStart(leaderIf.thenBlock());

  Value baseInBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, st.base,
                                               i32Const(b, loc, capacitySlots));
  auto baseIf = b.create<scf::IfOp>(loc, guardResults, baseInBounds, shape.producesSlot);
  b.setInsertionPointToStart(baseIf.thenBlock());

  // Read the clock before the slot atomic so its latency does not skew the timestamp.
  Value ts = callIntrinsic(b, loc, "llvm.amdgcn.s.memrealtime", b.getI64Type());
  Value ctrPtr = gepBytes(b, loc, st.bufPtr,
                          b.create<arith::MulIOp>(loc, st.base, i32Const(b, loc, kRecordBytes)));
  Value local = b.create<LLVM::AtomicRMWOp>(loc, LLVM::AtomicBinOp::add, ctrPtr,
                                            i32Const(b, loc, 1), LLVM::AtomicOrdering::monotonic);
  Value slot = b.create<arith::AddIOp>(loc, st.base, local);

  Value withinWave = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local,
                                             i32Const(b, loc, eventsPerWave - 1));
  Value inBounds = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, slot,
                                           i32Const(b, loc, capacitySlots));
  Value storeOk = b.create<arith::AndIOp>(loc, withinWave, inBounds);

  auto storeIf = b.create<scf::IfOp>(loc, guardResults, storeOk, shape.producesSlot);
  b.setInsertionPointToStart(storeIf.thenBlock());

  Value slot1 = b.create<arith::AddIOp>(loc, slot, i32Const(b, loc, 1));
  Value byteOff = b.create<arith::MulIOp>(loc, slot1, i32Const(b, loc, kRecordBytes));
  Value recPtr = gepBytes(b, loc, st.bufPtr, byteOff);

  Value payload =
      shape.payload
          ? shape.payload
          : b.create<LLVM::ConstantOp>(loc, b.getI64Type(), b.getI64IntegerAttr(0)).getResult();
  Value startSlot = i32Const(b, loc, 0);
  if (shape.startSlot) {
    assert(shape.startSlot.getType() == i32 &&
           "range_stop's token must already be lowered to its i32 slot");
    startSlot = shape.startSlot;
  }
  uint32_t packed = (shape.eventId & 0xFFFFFF) | ((shape.kind & 0x7) << 24) |
                    (shape.payloadType << 27) | (shape.payload ? (1u << 31) : 0);

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

  b.create<scf::YieldOp>(loc, ValueRange{slot1});

  b.setInsertionPointToStart(storeIf.elseBlock());
  b.create<scf::YieldOp>(loc, ValueRange{i32Const(b, loc, kUnpairedSlot)});

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
    if (chipHint.empty())
      return mod.emitError("ktrace: gpu.module carries no target and no "
                           "fly_ktrace.chip; cannot verify that in-kernel "
                           "timestamps are supported");
    if (chipHint.starts_with("gfx942") || chipHint.starts_with("gfx950"))
      return success();
    return rejectChip(chipHint);
  }

  bool sawRocdl = false;
  for (Attribute t : targets) {
    auto rocdlTarget = dyn_cast<ROCDL::ROCDLTargetAttr>(t);
    if (!rocdlTarget)
      continue;
    StringRef chip = rocdlTarget.getChip();
    if (!chip.starts_with("gfx942") && !chip.starts_with("gfx950"))
      return rejectChip(chip);
    sawRocdl = true;
  }
  if (!sawRocdl)
    return mod.emitError("ktrace: gpu.module has no ROCDL target");
  return success();
}

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

LogicalResult legalizeTokenTypes(Operation *root) {
  MLIRContext *ctx = root->getContext();
  Type i32 = IntegerType::get(ctx, 32);

  TypeConverter converter;
  converter.addConversion([](Type t) { return t; });
  converter.addConversion([i32](fly_ktrace::TokenType) { return i32; });
  auto materialize = [](OpBuilder &b, Type resultType, ValueRange inputs, Location loc) -> Value {
    return b.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
  };
  converter.addSourceMaterialization(materialize);
  converter.addTargetMaterialization(materialize);

  ConversionTarget target(*ctx);
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  scf::populateSCFStructuralTypeConversionTarget(converter, target);

  target.addDynamicallyLegalOp<arith::SelectOp>(
      [&converter](arith::SelectOp op) { return converter.isLegal(op.getType()); });

  RewritePatternSet patterns(ctx);
  scf::populateSCFStructuralTypeConversions(converter, patterns);
  patterns.add<SelectTokenConversion>(converter, ctx);
  if (failed(applyPartialConversion(root, target, std::move(patterns))))
    return failure();

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

    if (failed(legalizeTokenTypes(root))) {
      signalPassFailure();
      return;
    }

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
      SmallVector<Operation *> events;
      kernel.walk([&](Operation *op) {
        if (isa<MarkOp, RangePushOp, RangePopOp, RangeStartOp, RangeStopOp, InactiveTokenOp>(op))
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

      constexpr int64_t kEventsPerWave = 256;

      int64_t capacitySlots = kDefaultCapacitySlots;
      if (auto attr = root->getAttrOfType<IntegerAttr>("fly_ktrace.capacity_slots"))
        capacitySlots = attr.getInt();

      StringRef blocksSpec;
      if (auto attr = root->getAttrOfType<StringAttr>("fly_ktrace.blocks"))
        blocksSpec = attr.getValue();

      FailureOr<KernelTraceState> st =
          emitPrologue(kernel, kEventsPerWave, capacitySlots, blocksSpec);
      if (failed(st))
        return WalkResult::interrupt();

      for (Operation *op : events) {
        if (isa<InactiveTokenOp>(op)) {
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
          shape = {kKindRangePop, 0, pop.getPayload(), Value(), false};
        } else if (auto start = dyn_cast<RangeStartOp>(op)) {
          shape = {kKindRangeStart, idFor(start.getEventName()), start.getPayload(), Value(), true};
        } else {
          auto end = cast<RangeStopOp>(op);
          shape = {kKindRangeStop, idFor(end.getEventName()), end.getPayload(), end.getToken(),
                   false};
        }
        if (shape.eventId > 0xFFFFFF) {
          op->emitError("ktrace: more than 2^24 - 1 distinct event names");
          return WalkResult::interrupt();
        }
        if (auto type = op->getAttrOfType<StringAttr>("payload_type")) {
          int encoding = llvm::StringSwitch<int>(type.getValue())
                             .Case("uint64", 0)
                             .Case("int64", 1)
                             .Case("float32", 2)
                             .Case("float64", 3)
                             .Case("bool", 4)
                             .Default(-1);
          if (encoding < 0) {
            op->emitError("ktrace: unsupported payload_type ") << type.getValue();
            return WalkResult::interrupt();
          }
          shape.payloadType = encoding;
        }
        expandEvent(op, shape, *st, kEventsPerWave, capacitySlots);
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

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
