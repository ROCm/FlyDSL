// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

#include "./BufferFatPtr.h"

namespace mlir {
#define GEN_PASS_DEF_FLYTOROCDLCONVERSIONPASS
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::fly;

namespace {

unsigned mapToLLVMAddressSpace(AddressSpace addrSpace) {
  switch (addrSpace) {
  case AddressSpace::Global:
    return 1;
  case AddressSpace::Shared:
    return 3;
  case AddressSpace::Register:
    return 5;
  case AddressSpace::BufferDesc:
    return 8;
  default:
    assert(false && "Unsupported address space");
    return 0;
  }
}

Value applySwizzleOnPtr(OpBuilder &b, Location loc, Value ptr, SwizzleAttr swizzle) {
  if (swizzle.isTrivialSwizzle())
    return ptr;
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  auto i64Ty = b.getI64Type();
  Value ptrInt = LLVM::PtrToIntOp::create(b, loc, i64Ty, ptr);
  int64_t bitMaskValue = ((int64_t{1} << swizzle.getMask()) - 1)
                         << (swizzle.getBase() + swizzle.getShift());
  Value bitMask = arith::ConstantIntOp::create(b, loc, i64Ty, bitMaskValue);
  Value shiftAmt = arith::ConstantIntOp::create(b, loc, i64Ty, swizzle.getShift());
  Value masked = arith::AndIOp::create(b, loc, ptrInt, bitMask);
  Value shifted = arith::ShRUIOp::create(b, loc, masked, shiftAmt);
  Value swizzled = arith::XOrIOp::create(b, loc, ptrInt, shifted);
  return LLVM::IntToPtrOp::create(b, loc, ptrTy, swizzled);
}

class MakePtrOpLowering : public OpConversionPattern<MakePtrOp> {
public:
  MakePtrOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<MakePtrOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(MakePtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getResult().getType());
    if (!flyPtrTy)
      return failure();

    Location loc = op.getLoc();
    AddressSpace addrSpace = flyPtrTy.getAddressSpace().getValue();

    if (addrSpace == AddressSpace::Register) {
      auto dictAttrs = op.getDictAttrs();
      if (!dictAttrs)
        return rewriter.notifyMatchFailure(op, "register make_ptr requires dictAttrs");
      auto allocSize = dictAttrs->getAs<IntegerAttr>("allocaSize");
      if (!allocSize)
        return rewriter.notifyMatchFailure(op, "register make_ptr requires allocSize in ptrAttrs");
      unsigned llvmAS = mapToLLVMAddressSpace(AddressSpace::Register);
      auto llvmPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), llvmAS);
      Value nElems = arith::ConstantIntOp::create(rewriter, loc, allocSize.getInt(), 64);
      Value ptr = LLVM::AllocaOp::create(rewriter, loc, llvmPtrTy, flyPtrTy.getElemTy(), nElems, 0);
      rewriter.replaceOp(op, ptr);
      return success();
    } else if (addrSpace == AddressSpace::BufferDesc) {
      auto args = adaptor.getArgs();
      if (args.size() != 4)
        return rewriter.notifyMatchFailure(
            op, "buffer_rsrc make_ptr expects 4 args: base, stride, numRecords, flags");

      Value base = args[0];
      Value stride = args[1];
      Value numRecords = args[2];
      Value flags = args[3];

      auto rsrcPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                                  mapToLLVMAddressSpace(AddressSpace::BufferDesc));
      Value bufferRsrc = ROCDL::MakeBufferRsrcOp::create(rewriter, loc, rsrcPtrTy, base, stride,
                                                         numRecords, flags);
      rewriter.replaceOp(op, BufferFatPtr::pack(rewriter, loc, bufferRsrc));
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported make_ptr address space");
  }
};

class GetDynSharedOpLowering : public OpConversionPattern<GetDynSharedOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GetDynSharedOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto flyPtrTy = cast<fly::PointerType>(op.getResult().getType());
    unsigned addrSpace = mapToLLVMAddressSpace(flyPtrTy.getAddressSpace().getValue());

    auto moduleOp = op->getParentOfType<gpu::GPUModuleOp>();
    if (!moduleOp)
      return op->emitError("get_dyn_shared must be inside a gpu.module");

    LLVM::GlobalOp sharedGlobal = getOrCreateDynSharedGlobal(rewriter, moduleOp, loc, addrSpace);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto basePtr = LLVM::AddressOfOp::create(rewriter, loc, sharedGlobal);
    Type ptrType = basePtr->getResultTypes()[0];

    auto i8Ty = IntegerType::get(rewriter.getContext(), 8);
    Value sharedPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrType, i8Ty, basePtr, ArrayRef<LLVM::GEPArg>{0});

    rewriter.replaceOp(op, sharedPtr);
    return success();
  }

private:
  static LLVM::GlobalOp getOrCreateDynSharedGlobal(ConversionPatternRewriter &rewriter,
                                                   gpu::GPUModuleOp moduleOp, Location loc,
                                                   unsigned addrSpace) {
    llvm::StringSet<> existingNames;
    for (auto globalOp : moduleOp.getBody()->getOps<LLVM::GlobalOp>()) {
      existingNames.insert(globalOp.getSymName());
      if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(globalOp.getType())) {
        if (globalOp.getAddrSpace() == addrSpace && arrayType.getNumElements() == 0 &&
            globalOp.getAlignment().value_or(0) == 1024)
          return globalOp;
      }
    }

    unsigned counter = 0;
    SmallString<128> symName = SymbolTable::generateSymbolName<128>(
        "__dynamic_shared_", [&](StringRef candidate) { return existingNames.contains(candidate); },
        counter);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto zeroArrayTy = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), 0);

    auto globalOp = LLVM::GlobalOp::create(rewriter, loc, zeroArrayTy,
                                           /*isConstant=*/false, LLVM::Linkage::External, symName,
                                           /*value=*/Attribute(),
                                           /*alignment=*/1024, addrSpace);
    globalOp.setDsoLocal(true);
    return globalOp;
  }
};

class IntToPtrOpLowering : public OpConversionPattern<IntToPtrOp> {
public:
  IntToPtrOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<IntToPtrOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(IntToPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getResult().getType());
    if (!flyPtrTy)
      return failure();

    auto resultTy = dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(flyPtrTy));
    if (!resultTy)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(op, resultTy, adaptor.getSrc());
    return success();
  }
};

class PtrToIntOpLowering : public OpConversionPattern<PtrToIntOp> {
public:
  PtrToIntOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<PtrToIntOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(PtrToIntOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, resultTy, adaptor.getPtr());
    return success();
  }
};

class GetIterOpLowering : public OpConversionPattern<GetIterOp> {
public:
  GetIterOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<GetIterOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(GetIterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getMemref());
    return success();
  }
};

class ApplySwizzleOpLowering : public OpConversionPattern<ApplySwizzleOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ApplySwizzleOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

class RecastIterOpLowering : public OpConversionPattern<RecastIterOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(RecastIterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

class AddOffsetOpLowering : public OpConversionPattern<AddOffsetOp> {
public:
  AddOffsetOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<AddOffsetOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(AddOffsetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value base = adaptor.getPtr();
    Value offset = adaptor.getOffset();

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    auto offsetTy = dyn_cast<fly::IntTupleType>(offset.getType());
    IntTupleAttr offsetAttr = offsetTy.getAttr();
    if (!offsetAttr.isLeaf())
      return rewriter.notifyMatchFailure(op, "offset must be a leaf int tuple");

    Value offsetVal;
    auto offsetInt = offsetAttr.extractIntFromLeaf();
    if (offsetInt.isStatic()) {
      offsetVal = arith::ConstantIntOp::create(rewriter, loc, offsetInt.getValue(), 32);
    } else {
      Operation *defOp = offset.getDefiningOp();
      offsetVal = defOp->getOperand(0);
    }

    if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc) {
      BufferFatPtr bp(flyPtrTy, base);
      rewriter.replaceOp(op, bp.addOffset(rewriter, loc, offsetVal));
      return success();
    }

    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(base.getType());
    if (!ptrTy)
      return failure();

    Type elemTy = flyPtrTy.getElemTy();
    Value gep = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, base, ValueRange{offsetVal});
    rewriter.replaceOp(op, gep);
    return success();
  }
};

class MakeViewOpLowering : public OpConversionPattern<MakeViewOp> {
public:
  MakeViewOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<MakeViewOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(MakeViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (isa<fly::CoordTensorType>(op.getResult().getType())) {
      if (!op.getResult().use_empty())
        return rewriter.notifyMatchFailure(op, "coord_tensor result should have no uses");
      rewriter.eraseOp(op);
      return success();
    } else {
      Value base = adaptor.getIter();
      rewriter.replaceOp(op, base);
      return success();
    }
  }
};

class MemRefLoadVecOpLowering : public OpConversionPattern<MemRefLoadVecOp> {
public:
  using OpConversionPattern<MemRefLoadVecOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MemRefLoadVecOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value input = adaptor.getMemref();

    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(input.getType());
    if (!ptrTy)
      return failure();

    auto resVecTy = dyn_cast<VectorType>(op.getResult().getType());
    if (!resVecTy)
      return failure();

    Value loaded = LLVM::LoadOp::create(rewriter, loc, resVecTy, input);
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

class MemRefStoreVecOpLowering : public OpConversionPattern<MemRefStoreVecOp> {
public:
  using OpConversionPattern<MemRefStoreVecOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MemRefStoreVecOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value dest = adaptor.getMemref();
    Value valueToStore = adaptor.getVector();

    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(dest.getType());
    if (!ptrTy)
      return failure();

    auto vecTy = dyn_cast<VectorType>(valueToStore.getType());
    if (!vecTy)
      return failure();

    LLVM::StoreOp::create(rewriter, loc, valueToStore, dest);
    rewriter.eraseOp(op);
    return success();
  }
};

class PtrLoadOpLowering : public OpConversionPattern<PtrLoadOp> {
public:
  using OpConversionPattern<PtrLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PtrLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value ptr = adaptor.getPtr();

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    Type elemTy = flyPtrTy.getElemTy();

    if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc) {
      BufferFatPtr bp(flyPtrTy, ptr);
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      ArrayAttr noAttrs;
      Value loaded = ROCDL::RawPtrBufferLoadOp::create(
          rewriter, loc, elemTy, bp.bufferRsrc(rewriter, loc), bp.swizzleByteOffset(rewriter, loc),
          zero, zero, noAttrs, noAttrs, noAttrs);
      rewriter.replaceOp(op, loaded);
      return success();
    } else {
      ptr = applySwizzleOnPtr(rewriter, loc, ptr, flyPtrTy.getSwizzle());
      Value loaded = LLVM::LoadOp::create(rewriter, loc, elemTy, ptr);
      rewriter.replaceOp(op, loaded);
      return success();
    }
  }
};

class PtrStoreOpLowering : public OpConversionPattern<PtrStoreOp> {
public:
  using OpConversionPattern<PtrStoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PtrStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value ptr = adaptor.getPtr();
    Value value = adaptor.getValue();

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc) {
      BufferFatPtr bp(flyPtrTy, ptr);
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
      ArrayAttr noAttrs;
      ROCDL::RawPtrBufferStoreOp::create(rewriter, loc, value, bp.bufferRsrc(rewriter, loc),
                                         bp.swizzleByteOffset(rewriter, loc), zero, zero, noAttrs,
                                         noAttrs, noAttrs);
      rewriter.eraseOp(op);
      return success();
    } else {
      ptr = applySwizzleOnPtr(rewriter, loc, ptr, flyPtrTy.getSwizzle());
      LLVM::StoreOp::create(rewriter, loc, value, ptr);
      rewriter.eraseOp(op);
      return success();
    }
  }
};

class CopyAtomCallLowering : public OpConversionPattern<CopyAtomCall> {
public:
  using OpConversionPattern<CopyAtomCall>::OpConversionPattern;

  LogicalResult matchAndRewrite(CopyAtomCall op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type copyAtomType = op.getCopyAtom().getType();
    auto copyAtom = dyn_cast<CopyAtomType>(copyAtomType);
    if (!copyAtom)
      return rewriter.notifyMatchFailure(op, "copyAtom is not CopyAtomType");

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();
    Value pred = adaptor.getPred();

    auto srcFlyTy = dyn_cast<fly::MemRefType>(op.getSrc().getType());
    auto dstFlyTy = dyn_cast<fly::MemRefType>(op.getDst().getType());
    if (!srcFlyTy || !dstFlyTy)
      return rewriter.notifyMatchFailure(op, "expected Fly memref types on original op");

    if (srcFlyTy.getElemTy() != dstFlyTy.getElemTy())
      return rewriter.notifyMatchFailure(op, "src/dst element types mismatch");

    Location loc = op.getLoc();
    Type copyOpType = copyAtom.getCopyOp();

    auto emitCopyBody = [&](ConversionPatternRewriter &rewriter) -> LogicalResult {
      if (isa<CopyOpUniversalCopyType>(copyOpType)) {
        if (!isa<LLVM::LLVMPointerType>(src.getType()) ||
            !isa<LLVM::LLVMPointerType>(dst.getType()))
          return rewriter.notifyMatchFailure(op, "src/dst are not llvm.ptr for universal copy");
        return lowerUniversalCopy(op, rewriter, loc, copyAtom, srcFlyTy, dstFlyTy, src, dst);
      } else if (isa<fly_rocdl::CopyOpCDNA3BufferCopyType>(copyOpType))
        return lowerCDNA3BufferCopy(op, rewriter, loc, copyAtom, srcFlyTy, dstFlyTy, src, dst);
      return rewriter.notifyMatchFailure(op, "unsupported CopyOp type");
    };

    if (pred) {
      auto predFlyTy = dyn_cast<fly::MemRefType>(op.getPred().getType());
      if (!predFlyTy)
        return rewriter.notifyMatchFailure(op, "pred is not a Fly memref type");
      Type predElemTy = predFlyTy.getElemTy();
      Value predVal = LLVM::LoadOp::create(rewriter, loc, predElemTy, pred);
      auto ifOp = scf::IfOp::create(rewriter, loc, TypeRange{}, predVal, false);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      return emitCopyBody(rewriter);
    } else {
      return emitCopyBody(rewriter);
    }
  }

private:
  LogicalResult lowerUniversalCopy(CopyAtomCall op, ConversionPatternRewriter &rewriter,
                                   Location loc, CopyAtomType copyAtomTy, fly::MemRefType srcFlyTy,
                                   fly::MemRefType dstFlyTy, Value src, Value dst) const {
    LayoutBuilder<LayoutAttr> attrBuilder(rewriter.getContext());

    auto thrValLayoutSrc = dyn_cast<LayoutAttr>(copyAtomTy.getThrValLayoutSrc());
    if (!thrValLayoutSrc)
      return rewriter.notifyMatchFailure(op, "getThrValLayoutSrc returned null or non-LayoutAttr");
    int32_t numValSrc =
        intTupleProduct(attrBuilder, thrValLayoutSrc.getShape().at(1)).getLeafAsInt().getValue();

    Type elemTy = srcFlyTy.getElemTy();
    int32_t elemBits = elemTy.getIntOrFloatBitWidth();
    Value srcPtr = applySwizzleOnPtr(rewriter, loc, src, srcFlyTy.getSwizzle());
    Value dstPtr = applySwizzleOnPtr(rewriter, loc, dst, dstFlyTy.getSwizzle());
    int32_t copyBytes = numValSrc * elemBits / 8;
    Value len = arith::ConstantIntOp::create(rewriter, loc, copyBytes, /*width=*/32).getResult();
    LLVM::MemcpyOp::create(rewriter, loc, dstPtr, srcPtr, len, /*isVolatile=*/false);

    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerCDNA3BufferCopy(CopyAtomCall op, ConversionPatternRewriter &rewriter,
                                     Location loc, CopyAtomType copyAtomTy,
                                     fly::MemRefType srcFlyTy, fly::MemRefType dstFlyTy, Value src,
                                     Value dst) const {
    LayoutBuilder<LayoutAttr> attrBuilder(rewriter.getContext());

    auto thrValLayoutSrc = dyn_cast<LayoutAttr>(copyAtomTy.getThrValLayoutSrc());
    if (!thrValLayoutSrc)
      return rewriter.notifyMatchFailure(op, "getThrValLayoutSrc returned null");
    int32_t numValSrc =
        intTupleProduct(attrBuilder, thrValLayoutSrc.getShape().at(1)).getLeafAsInt().getValue();

    Type elemTy = srcFlyTy.getElemTy();
    int32_t vecWidth = numValSrc;
    Type vecTy = vecWidth == 1 ? elemTy : VectorType::get({vecWidth}, elemTy);

    AddressSpace srcAS = srcFlyTy.getAddressSpace().getValue();
    AddressSpace dstAS = dstFlyTy.getAddressSpace().getValue();

    bool srcIsBuffer = (srcAS == AddressSpace::BufferDesc);
    bool dstIsBuffer = (dstAS == AddressSpace::BufferDesc);

    if (srcIsBuffer == dstIsBuffer)
      return rewriter.notifyMatchFailure(
          op, "CDNA3 buffer copy requires exactly one side to be BufferDesc");

    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    ArrayAttr noAttrs;

    auto unpackBuffer = [&](Value val, fly::MemRefType flyTy) -> std::pair<Value, Value> {
      BufferFatPtr bp(flyTy.getPointerType(), val);
      return {bp.bufferRsrc(rewriter, loc), bp.swizzleByteOffset(rewriter, loc)};
    };

    if (srcIsBuffer) {
      auto [srcRsrc, srcOff] = unpackBuffer(src, srcFlyTy);
      Value loaded = ROCDL::RawPtrBufferLoadOp::create(rewriter, loc, vecTy, srcRsrc, srcOff, zero,
                                                       zero, noAttrs, noAttrs, noAttrs);
      LLVM::StoreOp::create(rewriter, loc, loaded, dst);
    } else {
      auto [dstRsrc, dstOff] = unpackBuffer(dst, dstFlyTy);
      Value loaded = LLVM::LoadOp::create(rewriter, loc, vecTy, src);
      ROCDL::RawPtrBufferStoreOp::create(rewriter, loc, loaded, dstRsrc, dstOff, zero, zero,
                                         noAttrs, noAttrs, noAttrs);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class MmaAtomCallLowering : public OpConversionPattern<MmaAtomCall> {
public:
  using OpConversionPattern<MmaAtomCall>::OpConversionPattern;

  LogicalResult matchAndRewrite(MmaAtomCall op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type mmaAtomType = op.getMmaAtom().getType();
    if (!isa<MmaAtomTypeInterface>(mmaAtomType))
      return rewriter.notifyMatchFailure(op,
                                         "expected MmaAtomTypeInterface type for mmaAtom operand");

    Location loc = op.getLoc();

    Value dPtr = adaptor.getD();
    Value aPtr = adaptor.getA();
    Value bPtr = adaptor.getB();
    Value cPtr = adaptor.getC();

    if (!isa<LLVM::LLVMPointerType>(dPtr.getType()) ||
        !isa<LLVM::LLVMPointerType>(aPtr.getType()) ||
        !isa<LLVM::LLVMPointerType>(bPtr.getType()) || !isa<LLVM::LLVMPointerType>(cPtr.getType()))
      return rewriter.notifyMatchFailure(op, "expected llvm.ptr operands after type conversion");

    if (auto universalFma = dyn_cast<MmaAtomUniversalFMAType>(mmaAtomType))
      return lowerUniversalFMA(op, rewriter, loc, universalFma, dPtr, aPtr, bPtr, cPtr);
    else if (auto cdna3Mfma = dyn_cast<fly_rocdl::MmaAtomCDNA3_MFMAType>(mmaAtomType))
      return lowerCDNA3MFMA(op, rewriter, loc, cdna3Mfma, dPtr, aPtr, bPtr, cPtr);

    return rewriter.notifyMatchFailure(op, "unsupported MmaAtom type");
  }

private:
  LogicalResult lowerUniversalFMA(MmaAtomCall op, ConversionPatternRewriter &rewriter, Location loc,
                                  MmaAtomUniversalFMAType atomTy, Value dPtr, Value aPtr,
                                  Value bPtr, Value cPtr) const {
    Type elemTy = atomTy.getElemTy();

    Value a = LLVM::LoadOp::create(rewriter, loc, elemTy, aPtr);
    Value b = LLVM::LoadOp::create(rewriter, loc, elemTy, bPtr);
    Value c = LLVM::LoadOp::create(rewriter, loc, elemTy, cPtr);

    Value mul = LLVM::FMulOp::create(rewriter, loc, elemTy, a, b);
    Value res = LLVM::FAddOp::create(rewriter, loc, elemTy, mul, c);

    LLVM::StoreOp::create(rewriter, loc, res, dPtr);
    rewriter.eraseOp(op);
    return success();
  }

  static bool isFP8(Type ty) { return isa<Float8E4M3FNUZType>(ty); }
  static bool isBF8(Type ty) { return isa<Float8E5M2FNUZType>(ty); }
  static bool isF8(Type ty) { return isFP8(ty) || isBF8(ty); }

  static Type getMfmaABType(MLIRContext *ctx, Type elemTy) {
    if (elemTy.isF32())
      return Float32Type::get(ctx);
    if (elemTy.isF16())
      return VectorType::get({4}, Float16Type::get(ctx));
    if (elemTy.isBF16())
      return VectorType::get({2}, IntegerType::get(ctx, 16));
    if (isF8(elemTy))
      return IntegerType::get(ctx, 64);
    return nullptr;
  }

  static int64_t getMfmaAccVecSize(int32_t m, int32_t k, Type elemTyA) {
    if (elemTyA.isF32()) {
      if (m == 32 && k == 1)
        return 32;
      if (m == 32 && k == 2)
        return 16;
      if (m == 16 && k == 1)
        return 16;
      if (m == 16 && k == 4)
        return 4;
      if (m == 4 && k == 1)
        return 4;
    }
    if (elemTyA.isF16()) {
      if (m == 32 && k == 4)
        return 32;
      if (m == 32 && k == 8)
        return 16;
      if (m == 16 && k == 4)
        return 16;
      if (m == 16 && k == 16)
        return 4;
      if (m == 4 && k == 4)
        return 4;
    }
    if (elemTyA.isBF16()) {
      if (m == 32 && k == 2)
        return 32;
      if (m == 32 && k == 4)
        return 16;
      if (m == 16 && k == 2)
        return 16;
      if (m == 16 && k == 8)
        return 4;
      if (m == 4 && k == 2)
        return 4;
    }
    if (isF8(elemTyA)) {
      if (m == 16 && k == 32)
        return 4;
      if (m == 32 && k == 16)
        return 16;
    }
    return 0;
  }

  template <typename MfmaOp>
  LogicalResult emitMfma(MmaAtomCall op, ConversionPatternRewriter &rewriter, Location loc,
                         Type abTyA, Type abTyB, VectorType accTy, Value aPtr, Value bPtr,
                         Value cPtr, Value dPtr) const {
    Value a = LLVM::LoadOp::create(rewriter, loc, abTyA, aPtr);
    Value b = LLVM::LoadOp::create(rewriter, loc, abTyB, bPtr);
    Value c = LLVM::LoadOp::create(rewriter, loc, accTy, cPtr);
    auto zeroAttr = rewriter.getI32IntegerAttr(0);
    Value res = MfmaOp::create(rewriter, loc, accTy, a, b, c, zeroAttr, zeroAttr, zeroAttr);
    LLVM::StoreOp::create(rewriter, loc, res, dPtr);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerCDNA3MFMA(MmaAtomCall op, ConversionPatternRewriter &rewriter, Location loc,
                               fly_rocdl::MmaAtomCDNA3_MFMAType atomTy, Value dPtr, Value aPtr,
                               Value bPtr, Value cPtr) const {
    int32_t m = atomTy.getM();
    int32_t n = atomTy.getN();
    int32_t k = atomTy.getK();
    Type elemTyA = atomTy.getElemTyA();
    Type elemTyB = atomTy.getElemTyB();
    MLIRContext *ctx = rewriter.getContext();

    Type abTyA = getMfmaABType(ctx, elemTyA);
    Type abTyB = getMfmaABType(ctx, elemTyB);
    if (!abTyA || !abTyB)
      return rewriter.notifyMatchFailure(op, "unsupported element type for MFMA");

    int64_t accVecSize = getMfmaAccVecSize(m, k, elemTyA);
    if (accVecSize == 0)
      return rewriter.notifyMatchFailure(op, "unsupported MNK combination for MFMA");

    Type accElemTy = atomTy.getElemTyAcc();
    VectorType accTy = VectorType::get({accVecSize}, accElemTy);

#define DISPATCH_MFMA(M_, K_, PRED, OP)                                                            \
  if (m == M_ && n == M_ && k == K_ && (PRED))                                                     \
    return emitMfma<ROCDL::OP>(op, rewriter, loc, abTyA, abTyB, accTy, aPtr, bPtr, cPtr, dPtr);

    DISPATCH_MFMA(32, 1, elemTyA.isF32(), mfma_f32_32x32x1f32)
    DISPATCH_MFMA(16, 1, elemTyA.isF32(), mfma_f32_16x16x1f32)
    DISPATCH_MFMA(4, 1, elemTyA.isF32(), mfma_f32_4x4x1f32)
    DISPATCH_MFMA(32, 2, elemTyA.isF32(), mfma_f32_32x32x2f32)
    DISPATCH_MFMA(16, 4, elemTyA.isF32(), mfma_f32_16x16x4f32)

    DISPATCH_MFMA(32, 4, elemTyA.isF16(), mfma_f32_32x32x4f16)
    DISPATCH_MFMA(16, 4, elemTyA.isF16(), mfma_f32_16x16x4f16)
    DISPATCH_MFMA(4, 4, elemTyA.isF16(), mfma_f32_4x4x4f16)
    DISPATCH_MFMA(32, 8, elemTyA.isF16(), mfma_f32_32x32x8f16)
    DISPATCH_MFMA(16, 16, elemTyA.isF16(), mfma_f32_16x16x16f16)

    DISPATCH_MFMA(32, 2, elemTyA.isBF16(), mfma_f32_32x32x2bf16)
    DISPATCH_MFMA(16, 2, elemTyA.isBF16(), mfma_f32_16x16x2bf16)
    DISPATCH_MFMA(4, 2, elemTyA.isBF16(), mfma_f32_4x4x2bf16)
    DISPATCH_MFMA(32, 4, elemTyA.isBF16(), mfma_f32_32x32x4bf16)
    DISPATCH_MFMA(16, 8, elemTyA.isBF16(), mfma_f32_16x16x8bf16)

    DISPATCH_MFMA(16, 32, isFP8(elemTyA) && isFP8(elemTyB), mfma_f32_16x16x32_fp8_fp8)
    DISPATCH_MFMA(16, 32, isFP8(elemTyA) && isBF8(elemTyB), mfma_f32_16x16x32_fp8_bf8)
    DISPATCH_MFMA(16, 32, isBF8(elemTyA) && isFP8(elemTyB), mfma_f32_16x16x32_bf8_fp8)
    DISPATCH_MFMA(16, 32, isBF8(elemTyA) && isBF8(elemTyB), mfma_f32_16x16x32_bf8_bf8)
    DISPATCH_MFMA(32, 16, isFP8(elemTyA) && isFP8(elemTyB), mfma_f32_32x32x16_fp8_fp8)
    DISPATCH_MFMA(32, 16, isFP8(elemTyA) && isBF8(elemTyB), mfma_f32_32x32x16_fp8_bf8)
    DISPATCH_MFMA(32, 16, isBF8(elemTyA) && isFP8(elemTyB), mfma_f32_32x32x16_bf8_fp8)
    DISPATCH_MFMA(32, 16, isBF8(elemTyA) && isBF8(elemTyB), mfma_f32_32x32x16_bf8_bf8)

#undef DISPATCH_MFMA

    return rewriter.notifyMatchFailure(op, "no matching ROCDL MFMA intrinsic");
  }
};

/// Lower `gpu.launch_func` kernel operands so that any `!fly.memref` values are
/// replaced by their type-converted builtin `memref` values. This prevents
/// `unrealized_conversion_cast` materializations from remaining live after
/// partial conversion (e.g., when the surrounding `func.func` signature has
/// been converted to builtin memrefs).
class GpuLaunchFuncOpLowering : public OpConversionPattern<gpu::LaunchFuncOp> {
public:
  using OpConversionPattern<gpu::LaunchFuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto kernelRef = adaptor.getKernel();

    auto grid =
        gpu::KernelDim3{adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ()};
    auto block =
        gpu::KernelDim3{adaptor.getBlockSizeX(), adaptor.getBlockSizeY(), adaptor.getBlockSizeZ()};

    std::optional<gpu::KernelDim3> clusterSize = std::nullopt;
    if (adaptor.getClusterSizeX() && adaptor.getClusterSizeY() && adaptor.getClusterSizeZ()) {
      clusterSize = gpu::KernelDim3{adaptor.getClusterSizeX(), adaptor.getClusterSizeY(),
                                    adaptor.getClusterSizeZ()};
    }

    // Preserve async token result type when present.
    Type asyncTokenType = nullptr;
    if (Value tok = op.getAsyncToken())
      asyncTokenType = tok.getType();

    // There are two relevant builder signatures in this MLIR:
    // - (kernel, ..., asyncTokenType, asyncDependencies, clusterSize)
    // - (kernel, ..., asyncObject, clusterSize)
    // Pick the one that matches the original op structure.
    if (Value asyncObj = adaptor.getAsyncObject()) {
      if (!adaptor.getAsyncDependencies().empty())
        return rewriter.notifyMatchFailure(
            op, "launch_func has both asyncObject and asyncDependencies");

      rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
          op, kernelRef, grid, block, adaptor.getDynamicSharedMemorySize(),
          adaptor.getKernelOperands(), asyncObj, clusterSize);
      return success();
    }

    rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
        op, kernelRef, grid, block, adaptor.getDynamicSharedMemorySize(),
        adaptor.getKernelOperands(), asyncTokenType, adaptor.getAsyncDependencies(), clusterSize);
    return success();
  }
};

class FlyTypeConverter : public TypeConverter {
public:
  FlyTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion([&](fly::MemRefType flyMemRefTy) -> Type {
      if (flyMemRefTy.getAddressSpace().getValue() == AddressSpace::BufferDesc)
        return BufferFatPtr::getType(flyMemRefTy.getContext());
      unsigned as = mapToLLVMAddressSpace(flyMemRefTy.getAddressSpace().getValue());
      return LLVM::LLVMPointerType::get(flyMemRefTy.getContext(), as);
    });
    addConversion([&](fly::PointerType flyPtrTy) -> Type {
      if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc)
        return BufferFatPtr::getType(flyPtrTy.getContext());
      unsigned as = mapToLLVMAddressSpace(flyPtrTy.getAddressSpace().getValue());
      return LLVM::LLVMPointerType::get(flyPtrTy.getContext(), as);
    });
  }
};

class ExtractAlignedPointerAsIndexLowering
    : public OpConversionPattern<ExtractAlignedPointerAsIndexOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ExtractAlignedPointerAsIndexOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // fly.memref is a bare pointer; after type conversion the operand is llvm.ptr<AS>.
    // Cast to the result type (e.g. llvm.ptr<0>) if address spaces differ.
    Value src = adaptor.getSource();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      resultType = op.getResult().getType();
    if (src.getType() != resultType)
      src = LLVM::AddrSpaceCastOp::create(rewriter, op.getLoc(), resultType, src);
    rewriter.replaceOp(op, src);
    return success();
  }
};

class FlyToROCDLConversionPass
    : public mlir::impl::FlyToROCDLConversionPassBase<FlyToROCDLConversionPass> {
public:
  using mlir::impl::FlyToROCDLConversionPassBase<
      FlyToROCDLConversionPass>::FlyToROCDLConversionPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect,
                           gpu::GPUDialect, func::FuncDialect, LLVM::LLVMDialect,
                           ROCDL::ROCDLDialect>();
    target.addIllegalDialect<fly::FlyDialect, fly_rocdl::FlyROCDLDialect>();

    // Constructors
    target.addLegalOp<StaticOp, MakeIntTupleOp, MakeLayoutOp, MakeTileOp, MakeComposedLayoutOp>();
    target.addLegalOp<MakeMmaAtomOp, MakeCopyAtomOp, MakeTiledCopyOp, MakeTiledMmaOp>();

    FlyTypeConverter typeConverter;

    // Ensure function signatures are type-converted; otherwise conversions may rely on
    // inserted unrealized casts that remain live.
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>(
        [&](gpu::GPUFuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });

    // IMPORTANT: `gpu.launch_func` itself is in a legal dialect, but its kernel operands may
    // still carry illegal `!fly.memref` types. If we don't mark it dynamically illegal in that
    // case, partial conversion won't try to rewrite it, leaving `unrealized_conversion_cast`
    // users alive and causing legalization failure.
    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>([&](gpu::LaunchFuncOp op) {
      auto isValueLegal = [&](Value v) {
        if (!v)
          return true;
        return typeConverter.isLegal(v.getType());
      };

      for (Value v : op.getKernelOperands())
        if (!isValueLegal(v))
          return false;

      if (!isValueLegal(op.getDynamicSharedMemorySize()))
        return false;

      // Async operands are part of the operand list; keep them consistent as well.
      for (Value dep : op.getAsyncDependencies())
        if (!isValueLegal(dep))
          return false;
      if (!isValueLegal(op.getAsyncObject()))
        return false;

      // Dimensions are typically index and already legal; no need to special-case.
      return true;
    });

    patterns.add<MakePtrOpLowering, GetDynSharedOpLowering>(typeConverter, context);
    patterns.add<IntToPtrOpLowering, PtrToIntOpLowering>(typeConverter, context);
    patterns.add<GetIterOpLowering, ApplySwizzleOpLowering, RecastIterOpLowering>(typeConverter,
                                                                                  context);
    patterns.add<AddOffsetOpLowering>(typeConverter, context);
    patterns.add<MakeViewOpLowering>(typeConverter, context);
    patterns.add<MemRefLoadVecOpLowering, MemRefStoreVecOpLowering>(typeConverter, context);
    patterns.add<PtrLoadOpLowering, PtrStoreOpLowering>(typeConverter, context);
    patterns.add<CopyAtomCallLowering, MmaAtomCallLowering>(typeConverter, context);
    patterns.add<GpuLaunchFuncOpLowering>(typeConverter, context);

    // TODO: deprecated in the future
    patterns.add<ExtractAlignedPointerAsIndexLowering>(typeConverter, context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateFunctionOpInterfaceTypeConversionPattern<gpu::GPUFuncOp>(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace impl {

std::unique_ptr<::mlir::Pass> createFlyToROCDLConversionPass() {
  return std::make_unique<FlyToROCDLConversionPass>();
}

} // namespace impl
