// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"

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

constexpr uint32_t kKindMark = 0;
constexpr uint32_t kKindRangePush = 1;
constexpr uint32_t kKindRangePop = 2;
constexpr uint32_t kKindRangeStart = 3;
constexpr uint32_t kKindRangeStop = 4;

// No slot is 0, because the buffer header comes first, so the highest token the lanes of a wave
// hold is the latest range any of them started, and it is this one only if none did.
constexpr int64_t kInactiveToken = 0;

constexpr int64_t kDefaultCapacitySlots = (64 << 20) / kRecordBytes - 1;

// A wave owns kEventsPerWave consecutive slots: its header, then its events.
constexpr int64_t kEventsPerWave = 256;
// Partition p's wave counter is at slot kCursorStrideSlots * (p + 1); keeping the counters 256
// bytes apart keeps their atomics from serializing on a shared line.
constexpr int64_t kCursorStrideSlots = 8;
constexpr int64_t kLaunchCountByte = 4;
// Raw-buffer descriptor word 3 on CDNA: 32-bit data format.
constexpr int64_t kRawBufferFlags = (7 << 12) | (4 << 15);

struct BufferLayout {
  int64_t partitions;
  int64_t headerSlots;
  // Slots after the header, cut into kEventsPerWave-slot wave regions; the last may be short.
  int64_t waveSlots;
};

struct KernelTraceState {
  // Buffer descriptor (four i32) for the wave's own slots. Its record count is 0 when the wave
  // does not record, so every store through it is dropped by the hardware bounds check instead
  // of being branched around.
  Value rsrc;
  // Absolute slot of the wave's header; a record's slot is this plus its index in the region.
  Value regionSlot;
  // Private i32 holding the number of events the wave has issued. LLVM promotes it to a
  // register, so events need no memory round trip to find their slot.
  Value counter;
  Value hwId;
  Value xccId;
};

Value i32Const(OpBuilder &b, Location loc, int64_t v) {
  return LLVM::ConstantOp::create(b, loc, b.getI32Type(), b.getI32IntegerAttr(v));
}

Value i64Const(OpBuilder &b, Location loc, int64_t v) {
  return LLVM::ConstantOp::create(b, loc, b.getI64Type(), b.getI64IntegerAttr(v));
}

Value gepBytes(OpBuilder &b, Location loc, Value ptr, Value byteOffset) {
  auto ptrTy = LLVM::LLVMPointerType::get(b.getContext(), 1);
  return LLVM::GEPOp::create(b, loc, ptrTy, b.getI8Type(), ptr, ValueRange{byteOffset},
                             LLVM::GEPNoWrapFlags::none);
}

Value callIntrinsic(OpBuilder &b, Location loc, StringRef name, Type resultTy,
                    ValueRange args = {}) {
  auto op = LLVM::CallIntrinsicOp::create(b, loc, resultTy, b.getStringAttr(name), args);
  return op.getResult(0);
}

Value hwregImmarg(OpBuilder &b, Location loc, int64_t regId, int64_t size) {
  return i32Const(b, loc, regId | ((size - 1) << 11));
}

Value firstActiveLane(OpBuilder &b, Location loc) {
  Type i64 = b.getI64Type();
  Value one = LLVM::ConstantOp::create(b, loc, b.getI1Type(), b.getIntegerAttr(b.getI1Type(), 1));
  Value mask = ROCDL::BallotOp::create(b, loc, i64, one);
  Value cttz = LLVM::CountTrailingZerosOp::create(b, loc, i64, mask, true);
  return LLVM::TruncOp::create(b, loc, b.getI32Type(), cttz);
}

// The highest value any active lane holds. For a wave-uniform value this is a copy.
Value waveMax(OpBuilder &b, Location loc, Value value) {
  Type i32 = b.getI32Type();
  Value max =
      callIntrinsic(b, loc, "llvm.amdgcn.wave.reduce.umax", i32, {value, i32Const(b, loc, 0)});
  return ROCDL::ReadfirstlaneOp::create(b, loc, i32, max);
}

std::pair<Value, Value> splitI64(OpBuilder &b, Location loc, Value v) {
  Type i32 = b.getI32Type();
  Value lo = arith::TruncIOp::create(b, loc, i32, v);
  Value hi =
      arith::TruncIOp::create(b, loc, i32, arith::ShRUIOp::create(b, loc, v, i64Const(b, loc, 32)));
  return {lo, hi};
}

Value packI32(OpBuilder &b, Location loc, ArrayRef<Value> elems) {
  auto vecTy = VectorType::get({static_cast<int64_t>(elems.size())}, b.getI32Type());
  Value vec = LLVM::PoisonOp::create(b, loc, vecTy);
  for (auto [i, elem] : llvm::enumerate(elems))
    vec = LLVM::InsertElementOp::create(b, loc, vec, elem, i32Const(b, loc, i));
  return vec;
}

// Every record word is wave-uniform, so records go out as scalar buffer stores: they hold no
// VGPRs, add nothing to vmcnt, and are bounds-checked like vector buffer stores. LLVM cannot
// emit scalar stores, hence the inline assembly; the wave must run s_dcache_wb before it ends.
void storeToRegion(OpBuilder &b, Location loc, const KernelTraceState &st, Value data,
                   Value byteOffset) {
  auto vecTy = dyn_cast<VectorType>(data.getType());
  int64_t dwords = vecTy ? vecTy.getNumElements() : 1;
  StringRef op = dwords == 4 ? "s_buffer_store_dwordx4" : "s_buffer_store_dword";
  assert((dwords == 4 || dwords == 1) && "ktrace stores one or four dwords");
  LLVM::InlineAsmOp::create(b, loc, Type(), ValueRange{data, st.rsrc, byteOffset},
                            (op + " $0, $1, $2").str(), "s,s,s", /*has_side_effects=*/true,
                            /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
                            LLVM::AsmDialectAttr::get(b.getContext(), LLVM::AsmDialect::AD_ATT),
                            ArrayAttr());
}

FailureOr<BufferLayout> bufferLayout(Operation *root, int64_t capacitySlots, int64_t partitions) {
  if (partitions < 1 || !llvm::isPowerOf2_64(partitions))
    return root->emitError("ktrace: fly_ktrace.partitions must be a power of two, got ")
           << partitions;
  int64_t headerSlots = kCursorStrideSlots * (partitions + 1);
  int64_t waveSlots = capacitySlots + 1 - headerSlots;
  if (waveSlots < 2)
    return root->emitError("ktrace: a ")
           << (capacitySlots + 1) * kRecordBytes << "-byte trace buffer cannot hold " << partitions
           << " partitions";
  return BufferLayout{partitions, headerSlots, waveSlots};
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
    return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, xccId, i32Const(b, loc, want))
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
    Value got = gpu::BlockIdOp::create(b, loc, dim);
    got = arith::IndexCastOp::create(b, loc, i32, got);
    Value term =
        arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, got, i32Const(b, loc, want));
    cond = cond ? arith::AndIOp::create(b, loc, cond, term).getResult() : term;
  }
  return cond;
}

// Consecutive waves of a row of workgroups take consecutive partitions, so a launch fills them
// evenly; odd multipliers scatter the rows, and the dispatch id rotates successive launches.
Value partitionOf(OpBuilder &b, Location loc, gpu::GPUFuncOp kernel, Value dispatchId,
                  int64_t partitions) {
  if (partitions == 1)
    return i32Const(b, loc, 0);
  Type i32 = b.getI32Type();
  auto toI32 = [&](Value index) -> Value { return arith::IndexCastOp::create(b, loc, i32, index); };
  Value wavesPerBlock;
  if (std::optional<ArrayRef<int32_t>> known = kernel.getKnownBlockSize()) {
    wavesPerBlock = i32Const(b, loc, llvm::divideCeil((*known)[0] * (*known)[1] * (*known)[2], 64));
  } else {
    Value threads = toI32(gpu::BlockDimOp::create(b, loc, gpu::Dimension::x));
    for (gpu::Dimension dim : {gpu::Dimension::y, gpu::Dimension::z})
      threads = arith::MulIOp::create(b, loc, threads, toI32(gpu::BlockDimOp::create(b, loc, dim)));
    wavesPerBlock = arith::ShRUIOp::create(
        b, loc, arith::AddIOp::create(b, loc, threads, i32Const(b, loc, 63)), i32Const(b, loc, 6));
  }
  auto blockId = [&](gpu::Dimension dim) -> Value {
    return toI32(gpu::BlockIdOp::create(b, loc, dim));
  };
  Value key = arith::MulIOp::create(b, loc, blockId(gpu::Dimension::z), i32Const(b, loc, 0x85EB));
  key = arith::AddIOp::create(b, loc, key, blockId(gpu::Dimension::y));
  key = arith::MulIOp::create(b, loc, key, i32Const(b, loc, 0x9E37));
  key = arith::AddIOp::create(
      b, loc, key, arith::MulIOp::create(b, loc, blockId(gpu::Dimension::x), wavesPerBlock));
  key = arith::AddIOp::create(b, loc, key, toI32(gpu::SubgroupIdOp::create(b, loc, nullptr)));
  Value dispatchLow = arith::TruncIOp::create(b, loc, i32, dispatchId);
  key = arith::AddIOp::create(
      b, loc, key,
      arith::MulIOp::create(b, loc, dispatchLow,
                            i32Const(b, loc, static_cast<int32_t>(0x9E3779B1u))));
  return arith::AndIOp::create(b, loc, key, i32Const(b, loc, partitions - 1));
}

FailureOr<KernelTraceState> emitPrologue(gpu::GPUFuncOp kernel, const BufferLayout &layout,
                                         StringRef blocksSpec) {
  Block &entry = kernel.getBody().front();
  if (entry.getNumArguments() == 0)
    return kernel.emitError("ktrace: kernel has no arguments, so no trace-buffer parameter; the "
                            "kernel was traced without ktrace enabled");

  OpBuilder b(&entry, entry.begin());
  Location loc = kernel.getLoc();
  MLIRContext *ctx = b.getContext();
  Type i32Ty = b.getI32Type();
  Type i64Ty = b.getI64Type();

  KernelTraceState st;
  Value rawBuf = entry.getArgument(entry.getNumArguments() - 1);
  auto flyPtrTy = dyn_cast<fly::PointerType>(rawBuf.getType());
  if (!flyPtrTy)
    return kernel.emitError("ktrace: expected the trailing kernel argument to be a "
                            "!fly.ptr trace buffer, got ")
           << rawBuf.getType();

  auto llvmPtrTy = LLVM::LLVMPointerType::get(ctx, 1);
  Value bufPtr = fly::ToLLVMPtrOp::create(b, loc, llvmPtrTy, rawBuf, b.getI32IntegerAttr(1));

  st.counter = LLVM::AllocaOp::create(b, loc, LLVM::LLVMPointerType::get(ctx, 5), i32Ty,
                                      i32Const(b, loc, 1), /*alignment=*/4);
  LLVM::StoreOp::create(b, loc, i32Const(b, loc, 0), st.counter);

  Value firstThread;
  for (gpu::Dimension dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
    Value block = gpu::BlockIdOp::create(b, loc, dim);
    Value thread = gpu::ThreadIdOp::create(b, loc, dim);
    Value blockZero = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, block,
                                            arith::ConstantIndexOp::create(b, loc, 0));
    Value threadZero = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, thread,
                                             arith::ConstantIndexOp::create(b, loc, 0));
    Value dimZero = arith::AndIOp::create(b, loc, blockZero, threadZero);
    firstThread =
        firstThread ? arith::AndIOp::create(b, loc, firstThread, dimZero).getResult() : dimZero;
  }
  auto launchIf = scf::IfOp::create(b, loc, firstThread, false);
  b.setInsertionPointToStart(launchIf.thenBlock());
  Value launchPtr = gepBytes(b, loc, bufPtr, i32Const(b, loc, kLaunchCountByte));
  LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::add, launchPtr, i32Const(b, loc, 1),
                            LLVM::AtomicOrdering::monotonic);
  b.setInsertionPointAfter(launchIf);

  st.hwId = callIntrinsic(b, loc, "llvm.amdgcn.s.getreg", i32Ty, {hwregImmarg(b, loc, 4, 32)});
  st.xccId = callIntrinsic(b, loc, "llvm.amdgcn.s.getreg", i32Ty, {hwregImmarg(b, loc, 20, 4)});
  Value dispatchId = callIntrinsic(b, loc, "llvm.amdgcn.dispatch.id", i64Ty);
  auto queuePtrTy = LLVM::LLVMPointerType::get(ctx, 4);
  Value queuePtr = callIntrinsic(b, loc, "llvm.amdgcn.queue.ptr", queuePtrTy);
  Value queueId = LLVM::PtrToIntOp::create(b, loc, i64Ty, queuePtr);

  FailureOr<Value> recording = blockFilter(b, loc, blocksSpec, st.xccId, kernel);
  if (failed(recording))
    return failure();

  Value partition = partitionOf(b, loc, kernel, dispatchId, layout.partitions);

  // One returning atomic per wave, on its partition's counter, numbers the wave within that
  // partition. The counter keeps counting past the partition's capacity, so the host sees the
  // overflow.
  Value lane = arith::IndexCastOp::create(b, loc, i32Ty, gpu::LaneIdOp::create(b, loc, nullptr));
  Value claims =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lane, firstActiveLane(b, loc));
  if (*recording)
    claims = arith::AndIOp::create(b, loc, claims, *recording);
  auto claimIf = scf::IfOp::create(b, loc, TypeRange{i32Ty}, claims, true);
  b.setInsertionPointToStart(claimIf.thenBlock());
  Value cursorByte =
      arith::MulIOp::create(b, loc, arith::AddIOp::create(b, loc, partition, i32Const(b, loc, 1)),
                            i32Const(b, loc, kCursorStrideSlots * kRecordBytes));
  Value claimed = LLVM::AtomicRMWOp::create(b, loc, LLVM::AtomicBinOp::add,
                                            gepBytes(b, loc, bufPtr, cursorByte),
                                            i32Const(b, loc, 1), LLVM::AtomicOrdering::monotonic);
  scf::YieldOp::create(b, loc, ValueRange{claimed});
  b.setInsertionPointToStart(claimIf.elseBlock());
  scf::YieldOp::create(b, loc, ValueRange{i32Const(b, loc, 0)});
  b.setInsertionPointAfter(claimIf);
  Value wave = ROCDL::ReadfirstlaneOp::create(b, loc, i32Ty, claimIf.getResult(0));

  // Wave k of partition p takes region k * partitions + p: the waves running at the same time
  // write neighbouring regions, which keeps their stores on few pages. The last region may be
  // short, and a wave numbered past the regions gets none.
  int64_t regions = llvm::divideCeil(layout.waveSlots, kEventsPerWave);
  Value interleaved = arith::AddIOp::create(
      b, loc,
      arith::MulIOp::create(b, loc, arith::MinUIOp::create(b, loc, wave, i32Const(b, loc, regions)),
                            i32Const(b, loc, layout.partitions)),
      partition);
  Value region = arith::MinUIOp::create(b, loc, interleaved, i32Const(b, loc, regions));
  Value regionStart = arith::MulIOp::create(b, loc, region, i32Const(b, loc, kEventsPerWave));
  Value hasRegion =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, region, i32Const(b, loc, regions));
  Value regionSlots = arith::SelectOp::create(
      b, loc, hasRegion,
      arith::MinUIOp::create(
          b, loc, arith::SubIOp::create(b, loc, i32Const(b, loc, layout.waveSlots), regionStart),
          i32Const(b, loc, kEventsPerWave)),
      i32Const(b, loc, 0));
  Value regionBytes = arith::MulIOp::create(b, loc, regionSlots, i32Const(b, loc, kRecordBytes));
  if (*recording)
    regionBytes = arith::SelectOp::create(b, loc, *recording, regionBytes, i32Const(b, loc, 0));
  // The scalar stores need every descriptor word in an SGPR.
  auto uniform = [&](Value v) -> Value { return ROCDL::ReadfirstlaneOp::create(b, loc, i32Ty, v); };
  st.regionSlot =
      uniform(arith::AddIOp::create(b, loc, i32Const(b, loc, layout.headerSlots), regionStart));
  Value regionAddr = LLVM::PtrToIntOp::create(
      b, loc, i64Ty,
      gepBytes(b, loc, bufPtr,
               arith::MulIOp::create(b, loc, arith::ExtUIOp::create(b, loc, i64Ty, st.regionSlot),
                                     i64Const(b, loc, kRecordBytes))));
  auto [addrLo, addrHi] = splitI64(b, loc, regionAddr);
  // Word 1 holds address bits 47:32 and a zero stride.
  st.rsrc = packI32(b, loc,
                    {uniform(addrLo),
                     uniform(arith::AndIOp::create(b, loc, addrHi, i32Const(b, loc, 0xFFFF))),
                     uniform(regionBytes), i32Const(b, loc, kRawBufferFlags)});

  // Header: the event count at +0, written again when the wave returns, then
  // (block.z << 8 | wave in block) and the dispatch id; the queue id and block.x, block.y at +16.
  auto blockId = [&](gpu::Dimension dim) -> Value {
    return arith::IndexCastOp::create(b, loc, i32Ty, gpu::BlockIdOp::create(b, loc, dim));
  };
  Value waveInBlock = uniform(
      arith::IndexCastOp::create(b, loc, i32Ty, gpu::SubgroupIdOp::create(b, loc, nullptr)));
  Value waveWord = arith::OrIOp::create(
      b, loc, arith::ShLIOp::create(b, loc, blockId(gpu::Dimension::z), i32Const(b, loc, 8)),
      waveInBlock);
  auto [dispatchLo, dispatchHi] = splitI64(b, loc, dispatchId);
  auto [queueLo, queueHi] = splitI64(b, loc, queueId);
  storeToRegion(b, loc, st,
                packI32(b, loc, {i32Const(b, loc, 0), waveWord, dispatchLo, dispatchHi}),
                i32Const(b, loc, 0));
  storeToRegion(
      b, loc, st,
      packI32(b, loc, {queueLo, queueHi, blockId(gpu::Dimension::x), blockId(gpu::Dimension::y)}),
      i32Const(b, loc, 16));
  return st;
}

// The scalar data cache is write-back, and nothing else flushes it: without the s_dcache_wb a
// wave's records can stay in it when the kernel completes.
void emitEventCounts(gpu::GPUFuncOp kernel, const KernelTraceState &st) {
  kernel.walk([&](gpu::ReturnOp ret) {
    OpBuilder b(ret);
    Location loc = ret.getLoc();
    Value issued = waveMax(b, loc, LLVM::LoadOp::create(b, loc, b.getI32Type(), st.counter));
    storeToRegion(b, loc, st, issued, i32Const(b, loc, 0));
    LLVM::CallIntrinsicOp::create(b, loc, b.getStringAttr("llvm.amdgcn.s.dcache.wb"), ValueRange{});
  });
}

bool recordsEvents(Region &region) {
  return region
      .walk([](Operation *op) {
        return isa<MarkOp, RangePushOp, RangePopOp, RangeStartOp, RangeStopOp>(op)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

// The lanes of a wave that diverge at an scf.if run both branches, one after the other, and a
// lane's count only advances in the branch it takes, so both branches would number their events
// from the same count. Such an if becomes two: the first runs the then branch, and the second
// forwards the first's results to those lanes and runs the else branch, so the wave can agree on
// its count in between (syncCountAfterRegions). The lanes that take the else branch never read
// the first if's results, so it yields them poison, which every result type has.
void splitIfsThatRecordOnBothSides(gpu::GPUFuncOp kernel) {
  SmallVector<scf::IfOp> ifs;
  kernel.walk([&](scf::IfOp op) {
    if (op.elseBlock() && recordsEvents(op.getThenRegion()) && recordsEvents(op.getElseRegion()))
      ifs.push_back(op);
  });
  for (scf::IfOp op : ifs) {
    OpBuilder b(op);
    Location loc = op.getLoc();
    auto first = scf::IfOp::create(b, loc, op.getResultTypes(), op.getCondition(),
                                   /*addThenBlock=*/true, /*addElseBlock=*/true);
    b.setInsertionPointToStart(first.elseBlock());
    SmallVector<Value> unread;
    for (Type type : op.getResultTypes())
      unread.push_back(ub::PoisonOp::create(b, loc, type));
    scf::YieldOp::create(b, loc, unread);
    first.getThenRegion().takeBody(op.getThenRegion());

    b.setInsertionPointAfter(first);
    auto second = scf::IfOp::create(b, loc, op.getResultTypes(), op.getCondition(),
                                    /*addThenBlock=*/true, /*addElseBlock=*/false);
    second.getElseRegion().takeBody(op.getElseRegion());
    b.setInsertionPointToStart(second.thenBlock());
    scf::YieldOp::create(b, loc, first.getResults());
    op.replaceAllUsesWith(second.getResults());
    op.erase();
  }
}

// A lane's count falls behind in a region where only some lanes of the wave record: one side of
// a divergent branch, or loop iterations that other lanes have left. After every region that
// records, the wave takes its highest count, so later events of any subset of its lanes start
// past every slot the region used. Under wave-uniform control flow this is a copy.
void syncCountAfterRegions(gpu::GPUFuncOp kernel, const KernelTraceState &st) {
  SmallVector<Operation *> ops;
  kernel.walk([&](Operation *op) {
    if (op != kernel.getOperation() && llvm::any_of(op->getRegions(), recordsEvents))
      ops.push_back(op);
  });
  for (Operation *op : ops) {
    OpBuilder b(op->getContext());
    b.setInsertionPointAfter(op);
    Location loc = op->getLoc();
    Value count = waveMax(b, loc, LLVM::LoadOp::create(b, loc, b.getI32Type(), st.counter));
    LLVM::StoreOp::create(b, loc, count, st.counter);
  }
}

struct EventShape {
  uint32_t kind;
  uint32_t eventId;
  Value payload;
  Value startSlot;
  bool producesSlot;
  uint32_t payloadType = 0;
};

// Straight-line code: the slot comes from the wave's count, and the record is stored by every
// active lane with wave-uniform data, so no lane guard or bounds branch splits the block.
void expandEvent(Operation *op, const EventShape &shape, const KernelTraceState &st) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  Type i32 = b.getI32Type();
  Type i64 = b.getI64Type();

  Value startSlot = i32Const(b, loc, 0);
  Value active;
  if (shape.startSlot) {
    assert(shape.startSlot.getType() == i32 &&
           "range_stop's token must already be lowered to its i32 slot");
    // The lanes that reach a stop can hold different tokens, and some none, as after a loop they
    // leave at different iterations. The stop closes the latest range they started; if none of
    // them started one, it takes no slot and its stores go out of bounds.
    startSlot = waveMax(b, loc, shape.startSlot);
    active = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne, startSlot,
                                   i32Const(b, loc, kInactiveToken));
  }

  Value issued = waveMax(b, loc, LLVM::LoadOp::create(b, loc, i32, st.counter));
  Value taken =
      active ? arith::ExtUIOp::create(b, loc, i32, active).getResult() : i32Const(b, loc, 1);
  LLVM::StoreOp::create(b, loc, arith::AddIOp::create(b, loc, issued, taken), st.counter);
  Value ts = callIntrinsic(b, loc, "llvm.amdgcn.s.memrealtime", i64);

  // Events past the per-wave limit all go one slot past the region, which the bound drops.
  Value slotInRegion = arith::AddIOp::create(
      b, loc, arith::MinUIOp::create(b, loc, issued, i32Const(b, loc, kEventsPerWave - 1)),
      i32Const(b, loc, 1));
  Value byteOffset = arith::MulIOp::create(b, loc, slotInRegion, i32Const(b, loc, kRecordBytes));
  if (active)
    byteOffset = arith::SelectOp::create(b, loc, active, byteOffset,
                                         i32Const(b, loc, kEventsPerWave * kRecordBytes));

  Value payload = shape.payload ? ROCDL::ReadfirstlaneOp::create(b, loc, i64, shape.payload)
                                : i64Const(b, loc, 0);
  uint32_t packed = (shape.eventId & 0xFFFFFF) | ((shape.kind & 0x7) << 24) |
                    (shape.payloadType << 27) | (shape.payload ? (1u << 31) : 0);

  auto [tsLo, tsHi] = splitI64(b, loc, ts);
  auto [payloadLo, payloadHi] = splitI64(b, loc, payload);
  storeToRegion(b, loc, st, packI32(b, loc, {tsLo, tsHi, payloadLo, payloadHi}), byteOffset);
  storeToRegion(
      b, loc, st,
      packI32(b, loc,
              {st.hwId, st.xccId, i32Const(b, loc, static_cast<int32_t>(packed)), startSlot}),
      arith::AddIOp::create(b, loc, byteOffset, i32Const(b, loc, 16)));

  if (shape.producesSlot)
    op->getResult(0).replaceAllUsesWith(
        arith::AddIOp::create(b, loc, st.regionSlot, slotInRegion).getResult());
  op->erase();
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
    return UnrealizedConversionCastOp::create(b, loc, resultType, inputs).getResult(0);
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

      int64_t capacitySlots = kDefaultCapacitySlots;
      if (auto attr = root->getAttrOfType<IntegerAttr>("fly_ktrace.capacity_slots"))
        capacitySlots = attr.getInt();
      int64_t partitions = 1;
      if (auto attr = root->getAttrOfType<IntegerAttr>("fly_ktrace.partitions"))
        partitions = attr.getInt();
      FailureOr<BufferLayout> layout = bufferLayout(kernel, capacitySlots, partitions);
      if (failed(layout))
        return WalkResult::interrupt();

      StringRef blocksSpec;
      if (auto attr = root->getAttrOfType<StringAttr>("fly_ktrace.blocks"))
        blocksSpec = attr.getValue();

      FailureOr<KernelTraceState> st = emitPrologue(kernel, *layout, blocksSpec);
      if (failed(st))
        return WalkResult::interrupt();
      splitIfsThatRecordOnBothSides(kernel);
      syncCountAfterRegions(kernel, *st);
      emitEventCounts(kernel, *st);

      for (Operation *op : events) {
        if (isa<InactiveTokenOp>(op)) {
          OpBuilder b(op);
          Value seed = i32Const(b, op->getLoc(), kInactiveToken);
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
        expandEvent(op, shape, *st);
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
