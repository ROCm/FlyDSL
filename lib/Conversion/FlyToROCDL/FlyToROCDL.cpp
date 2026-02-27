
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

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

namespace mlir {
#define GEN_PASS_DEF_FLYTOROCDLCONVERSIONPASS
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::fly;

namespace {

static unsigned mapAddressSpace(AddressSpace space) {
  switch (space) {
  case AddressSpace::Global:
    return 1;
  case AddressSpace::Shared:
    return 3;
  case AddressSpace::Register:
    return 5;
  case AddressSpace::BufferDesc:
    return 8;
  }
  return 0;
}

static LLVM::LLVMStructType getBufferFatPtrType(MLIRContext *ctx) {
  return LLVM::LLVMStructType::getLiteral(
      ctx, {LLVM::LLVMPointerType::get(ctx, 8), IntegerType::get(ctx, 32)});
}

static bool isBufferFatPtr(Type ty) {
  auto st = dyn_cast<LLVM::LLVMStructType>(ty);
  if (!st || st.getBody().size() != 2)
    return false;
  auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(st.getBody()[0]);
  return ptrTy && ptrTy.getAddressSpace() == 8 && st.getBody()[1].isInteger(32);
}

static Value extractBufferRsrc(OpBuilder &b, Location loc, Value fatPtr) {
  return LLVM::ExtractValueOp::create(b, loc, fatPtr, ArrayRef<int64_t>{0});
}

static Value extractBufferOffset(OpBuilder &b, Location loc, Value fatPtr) {
  return LLVM::ExtractValueOp::create(b, loc, fatPtr, ArrayRef<int64_t>{1});
}

static Value createBufferFatPtr(OpBuilder &b, Location loc, MLIRContext *ctx, Value rsrc,
                                Value byteOffset) {
  auto structTy = getBufferFatPtrType(ctx);
  Value undef = LLVM::UndefOp::create(b, loc, structTy);
  Value withRsrc = LLVM::InsertValueOp::create(b, loc, undef, rsrc, ArrayRef<int64_t>{0});
  return LLVM::InsertValueOp::create(b, loc, withRsrc, byteOffset, ArrayRef<int64_t>{1});
}

static int64_t getElemByteWidth(Type elemTy) {
  if (auto ft = dyn_cast<FloatType>(elemTy))
    return ft.getWidth() / 8;
  if (auto it = dyn_cast<IntegerType>(elemTy))
    return it.getWidth() / 8;
  return 0;
}

static FailureOr<Value> toI32(Value v, Location loc, ConversionPatternRewriter &rewriter) {
  Type i32Ty = rewriter.getI32Type();
  if (v.getType() == i32Ty)
    return v;
  if (v.getType().isIndex())
    return arith::IndexCastOp::create(rewriter, loc, i32Ty, v).getResult();
  if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
    if (intTy.getWidth() < 32)
      return arith::ExtSIOp::create(rewriter, loc, i32Ty, v).getResult();
    if (intTy.getWidth() > 32)
      return arith::TruncIOp::create(rewriter, loc, i32Ty, v).getResult();
  }
  return failure();
}

static FailureOr<Value> toI64(Value v, Location loc, ConversionPatternRewriter &rewriter) {
  Type i64Ty = rewriter.getI64Type();
  if (v.getType() == i64Ty)
    return v;
  if (v.getType().isIndex())
    return arith::IndexCastOp::create(rewriter, loc, i64Ty, v).getResult();
  if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
    if (intTy.getWidth() < 64)
      return arith::ExtSIOp::create(rewriter, loc, i64Ty, v).getResult();
    if (intTy.getWidth() > 64)
      return arith::TruncIOp::create(rewriter, loc, i64Ty, v).getResult();
  }
  return failure();
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
    auto args = adaptor.getArgs();

    if (addrSpace == AddressSpace::BufferDesc) {
      if (args.size() != 4)
        return rewriter.notifyMatchFailure(
            op, "buffer_rsrc make_ptr expects 4 args: base, stride, numRecords, flags");

      Value base = args[0];
      Value stride = args[1];
      Value numRecords = args[2];
      Value flags = args[3];

      auto rsrcPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 8);
      Value rsrc = ROCDL::MakeBufferRsrcOp::create(rewriter, loc, rsrcPtrTy, base, stride,
                                                   numRecords, flags);
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32).getResult();
      Value fatPtr = createBufferFatPtr(rewriter, loc, rewriter.getContext(), rsrc, zero);
      rewriter.replaceOp(op, fatPtr);
      return success();
    }

    auto resultTy = dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(flyPtrTy));
    if (!resultTy)
      return failure();

    if (args.size() == 1) {
      Value src = args[0];
      if (src.getType() == resultTy) {
        rewriter.replaceOp(op, src);
        return success();
      }
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultTy, src);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported make_ptr operand count");
  }
};

class MemRefAllocOpLowering : public OpConversionPattern<MemRefAllocaOp> {
public:
  MemRefAllocOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<MemRefAllocaOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(MemRefAllocaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyMemRefTy = dyn_cast<fly::MemRefType>(op.getResult().getType());
    if (!flyMemRefTy)
      return failure();

    LayoutAttr layoutAttr = flyMemRefTy.getLayout();
    auto elemTy = flyMemRefTy.getElemTy();

    LayoutBuilder<LayoutAttr> builder(rewriter.getContext());
    IntTupleAttr totalSize = layoutCosize(builder, layoutAttr);

    assert(totalSize.isStatic() && totalSize.isLeaf());

    auto convertedPtrTy =
        dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(flyMemRefTy));
    if (!convertedPtrTy)
      return failure();

    auto loc = op.getLoc();

    // Alloca array size is i64.
    Value nElems = arith::ConstantIntOp::create(rewriter, loc, totalSize.getLeafAsInt().getValue(),
                                                /*width=*/64)
                       .getResult();

    // `llvm.alloca` takes element type and array size. Keep alignment unspecified.
    Value ptr = LLVM::AllocaOp::create(rewriter, loc, convertedPtrTy, elemTy, nElems,
                                       /*alignment=*/0);
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

/// Materialize a scalar index from a non-array `!fly.int_tuple` value.
/// This is used for pointer/memref offset computations.
static FailureOr<Value> materializeScalarIndex(Value intTuple, Location loc,
                                               ConversionPatternRewriter &rewriter) {
  auto tupleTy = dyn_cast<fly::IntTupleType>(intTuple.getType());
  if (!tupleTy)
    return failure();

  IntTupleAttr profile = tupleTy.getAttr();
  if (!profile.isLeaf())
    return failure();

  // Static scalar.
  if (auto intAttr = dyn_cast<IntAttr>(profile.getValue())) {
    if (intAttr.isStatic()) {
      Value c = arith::ConstantIndexOp::create(rewriter, loc, intAttr.getValue());
      return c;
    }
  }
  if (profile.getLeafAsInt().isNone()) {
    Value c = arith::ConstantIndexOp::create(rewriter, loc, 0);
    return c;
  }

  // Dynamic scalar: expect it comes from fly.make_int_tuple with exactly one operand.
  if (Operation *defOp = intTuple.getDefiningOp()) {
    if (defOp->getName().getStringRef() == "fly.make_int_tuple" && defOp->getNumOperands() == 1) {
      Value v = defOp->getOperand(0);
      if (v.getType().isIndex())
        return v;
      // Most Fly scalars are i32; cast to index when needed.
      if (v.getType().isSignlessInteger())
        return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), v).getResult();
    }
  }

  return failure();
}

class GetIterOpLowering : public OpConversionPattern<GetIterOp> {
public:
  GetIterOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<GetIterOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(GetIterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value mem = adaptor.getMemref();
    Type resTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resTy)
      return failure();
    assert(mem.getType() == resTy);
    rewriter.replaceOp(op, mem);
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

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    auto offsetIdx = materializeScalarIndex(op.getOffset(), loc, rewriter);
    if (failed(offsetIdx))
      return failure();

    if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc) {
      if (!isBufferFatPtr(base.getType()))
        return failure();
      Type elemTy = flyPtrTy.getElemTy();
      int64_t elemBytes = getElemByteWidth(elemTy);
      if (elemBytes <= 0)
        return failure();

      FailureOr<Value> offsetI32 = toI32(*offsetIdx, loc, rewriter);
      if (failed(offsetI32))
        return failure();

      Value byteOffsetDelta = *offsetI32;
      if (elemBytes > 1) {
        Value scale = arith::ConstantIntOp::create(rewriter, loc, elemBytes, 32).getResult();
        byteOffsetDelta = arith::MulIOp::create(rewriter, loc, byteOffsetDelta, scale);
      }

      Value oldOffset = extractBufferOffset(rewriter, loc, base);
      Value newOffset = arith::AddIOp::create(rewriter, loc, oldOffset, byteOffsetDelta);
      Value rsrc = extractBufferRsrc(rewriter, loc, base);
      Value result = createBufferFatPtr(rewriter, loc, rewriter.getContext(), rsrc, newOffset);
      rewriter.replaceOp(op, result);
      return success();
    }

    auto basePtrTy = dyn_cast<LLVM::LLVMPointerType>(base.getType());
    if (!basePtrTy)
      return failure();

    auto resultTy =
        dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultTy)
      return failure();

    FailureOr<Value> offsetI64 = toI64(*offsetIdx, loc, rewriter);
    if (failed(offsetI64))
      return failure();

    Type elemTy = flyPtrTy.getElemTy();
    Value gep = LLVM::GEPOp::create(rewriter, loc, resultTy, elemTy, base, ValueRange{*offsetI64});
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
    Value base = adaptor.getIter();
    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultTy)
      return failure();
    if (base.getType() == resultTy) {
      rewriter.replaceOp(op, base);
      return success();
    }
    if (isa<LLVM::LLVMPointerType>(base.getType()) && isa<LLVM::LLVMPointerType>(resultTy)) {
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultTy, base);
      return success();
    }
    return failure();
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

    // Opaque pointers: we can directly load a vector from the base address.
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

class MemRefLoadOpLowering : public OpConversionPattern<MemRefLoadOp> {
public:
  using OpConversionPattern<MemRefLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MemRefLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value mem = adaptor.getMemref();
    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(mem.getType());
    if (!ptrTy)
      return failure();

    // `fly.memref.load` takes a scalar int_tuple offset.
    auto idxVal = materializeScalarIndex(op.getIndices(), op.getLoc(), rewriter);
    if (failed(idxVal))
      return failure();
    FailureOr<Value> idxI64 = toI64(*idxVal, op.getLoc(), rewriter);
    if (failed(idxI64))
      return failure();

    auto flyMemRefTy = dyn_cast<fly::MemRefType>(op.getMemref().getType());
    if (!flyMemRefTy)
      return failure();
    Type elemTy = flyMemRefTy.getElemTy();
    Value gep = LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, elemTy, mem, ValueRange{*idxI64});
    Value loaded = LLVM::LoadOp::create(rewriter, op.getLoc(), op.getResult().getType(), gep);
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

class MemRefStoreOpLowering : public OpConversionPattern<MemRefStoreOp> {
public:
  using OpConversionPattern<MemRefStoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MemRefStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value mem = adaptor.getMemref();
    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(mem.getType());
    if (!ptrTy)
      return failure();

    auto idxVal = materializeScalarIndex(op.getIndices(), op.getLoc(), rewriter);
    if (failed(idxVal))
      return failure();
    FailureOr<Value> idxI64 = toI64(*idxVal, op.getLoc(), rewriter);
    if (failed(idxI64))
      return failure();
    auto flyMemRefTy = dyn_cast<fly::MemRefType>(op.getMemref().getType());
    if (!flyMemRefTy)
      return failure();
    Type elemTy = flyMemRefTy.getElemTy();
    Value gep = LLVM::GEPOp::create(rewriter, op.getLoc(), ptrTy, elemTy, mem, ValueRange{*idxI64});
    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(), gep);
    rewriter.eraseOp(op);
    return success();
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

    auto srcFlyTy = dyn_cast<fly::MemRefType>(op.getSrc().getType());
    auto dstFlyTy = dyn_cast<fly::MemRefType>(op.getDst().getType());
    if (!srcFlyTy || !dstFlyTy)
      return rewriter.notifyMatchFailure(op, "expected Fly memref types on original op");

    if (srcFlyTy.getElemTy() != dstFlyTy.getElemTy())
      return rewriter.notifyMatchFailure(op, "src/dst element types mismatch");

    Location loc = op.getLoc();
    Type copyOpType = copyAtom.getCopyOp();

    if (isa<CopyOpUniversalCopyType>(copyOpType)) {
      if (!isa<LLVM::LLVMPointerType>(src.getType()) || !isa<LLVM::LLVMPointerType>(dst.getType()))
        return rewriter.notifyMatchFailure(op, "src/dst are not llvm.ptr for universal copy");
      return lowerUniversalCopy(op, rewriter, loc, copyAtom, srcFlyTy, src, dst);
    } else if (isa<fly_rocdl::CopyOpCDNA3BufferLDSTType>(copyOpType))
      return lowerCDNA3BufferLDST(op, rewriter, loc, copyAtom, srcFlyTy, dstFlyTy, src, dst);

    return rewriter.notifyMatchFailure(op, "unsupported CopyOp type");
  }

private:
  LogicalResult lowerUniversalCopy(CopyAtomCall op, ConversionPatternRewriter &rewriter,
                                   Location loc, CopyAtomType copyAtomTy, fly::MemRefType srcFlyTy,
                                   Value src, Value dst) const {
    LayoutBuilder<LayoutAttr> attrBuilder(rewriter.getContext());

    auto thrValLayoutSrc = dyn_cast<LayoutAttr>(copyAtomTy.getThrValLayoutSrc());
    if (!thrValLayoutSrc)
      return rewriter.notifyMatchFailure(op, "getThrValLayoutSrc returned null or non-LayoutAttr");
    IntAttr numValSrcAttr = intTupleProductImpl(attrBuilder, thrValLayoutSrc.getShape().at(1));
    if (!numValSrcAttr.isStatic())
      return rewriter.notifyMatchFailure(op, "NumValSrc is not static");
    int64_t numValSrc = numValSrcAttr.getValue();

    Type elemTy = srcFlyTy.getElemTy();
    int64_t elemBits = 0;
    if (auto ft = dyn_cast<FloatType>(elemTy))
      elemBits = ft.getWidth();
    else if (auto it = dyn_cast<IntegerType>(elemTy))
      elemBits = it.getWidth();
    else
      return rewriter.notifyMatchFailure(op, "unsupported element type for memcpy sizing");
    if (elemBits <= 0)
      return rewriter.notifyMatchFailure(op, "invalid element bit width");

    int64_t copyBytes = numValSrc * elemBits / 8;
    Value len = arith::ConstantIntOp::create(rewriter, loc, copyBytes, /*width=*/64).getResult();
    LLVM::MemcpyOp::create(rewriter, loc, dst, src, len, /*isVolatile=*/false);

    rewriter.eraseOp(op);
    return success();
  }

  static int64_t getElemBitWidth(Type elemTy) {
    if (auto ft = dyn_cast<FloatType>(elemTy))
      return ft.getWidth();
    if (auto it = dyn_cast<IntegerType>(elemTy))
      return it.getWidth();
    return 0;
  }

  LogicalResult lowerCDNA3BufferLDST(CopyAtomCall op, ConversionPatternRewriter &rewriter,
                                     Location loc, CopyAtomType copyAtomTy,
                                     fly::MemRefType srcFlyTy, fly::MemRefType dstFlyTy, Value src,
                                     Value dst) const {
    LayoutBuilder<LayoutAttr> attrBuilder(rewriter.getContext());

    auto thrValLayoutSrc = dyn_cast<LayoutAttr>(copyAtomTy.getThrValLayoutSrc());
    if (!thrValLayoutSrc)
      return rewriter.notifyMatchFailure(op, "getThrValLayoutSrc returned null");
    IntAttr numValSrcAttr = intTupleProductImpl(attrBuilder, thrValLayoutSrc.getShape().at(1));
    if (!numValSrcAttr.isStatic())
      return rewriter.notifyMatchFailure(op, "NumValSrc is not static");
    int64_t numValSrc = numValSrcAttr.getValue();

    Type elemTy = srcFlyTy.getElemTy();
    int64_t elemBits = getElemBitWidth(elemTy);
    if (elemBits <= 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    int64_t vecWidth = numValSrc;
    Type vecTy = vecWidth == 1 ? elemTy : VectorType::get({vecWidth}, elemTy);

    AddressSpace srcAS = srcFlyTy.getAddressSpace().getValue();
    AddressSpace dstAS = dstFlyTy.getAddressSpace().getValue();

    bool srcIsBuffer = (srcAS == AddressSpace::BufferDesc);
    bool dstIsBuffer = (dstAS == AddressSpace::BufferDesc);

    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 32).getResult();
    ArrayAttr noAttrs;

    auto unpackBuffer = [&](Value val) -> std::pair<Value, Value> {
      if (isBufferFatPtr(val.getType()))
        return {extractBufferRsrc(rewriter, loc, val), extractBufferOffset(rewriter, loc, val)};
      return {val, zero};
    };

    if (srcIsBuffer && !dstIsBuffer) {
      auto [srcRsrc, srcOff] = unpackBuffer(src);
      Value loaded = ROCDL::RawPtrBufferLoadOp::create(rewriter, loc, vecTy, srcRsrc, srcOff, zero,
                                                       zero, noAttrs, noAttrs, noAttrs);
      LLVM::StoreOp::create(rewriter, loc, loaded, dst);
    } else if (!srcIsBuffer && dstIsBuffer) {
      auto [dstRsrc, dstOff] = unpackBuffer(dst);
      Value loaded = LLVM::LoadOp::create(rewriter, loc, vecTy, src);
      ROCDL::RawPtrBufferStoreOp::create(rewriter, loc, loaded, dstRsrc, dstOff, zero, zero,
                                         noAttrs, noAttrs, noAttrs);
    } else if (srcIsBuffer && dstIsBuffer) {
      auto [srcRsrc, srcOff] = unpackBuffer(src);
      auto [dstRsrc, dstOff] = unpackBuffer(dst);
      Value loaded = ROCDL::RawPtrBufferLoadOp::create(rewriter, loc, vecTy, srcRsrc, srcOff, zero,
                                                       zero, noAttrs, noAttrs, noAttrs);
      ROCDL::RawPtrBufferStoreOp::create(rewriter, loc, loaded, dstRsrc, dstOff, zero, zero,
                                         noAttrs, noAttrs, noAttrs);
    } else {
      int64_t copyBytes = numValSrc * elemBits / 8;
      Value len = arith::ConstantIntOp::create(rewriter, loc, copyBytes, 64).getResult();
      LLVM::MemcpyOp::create(rewriter, loc, dst, src, len, false);
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
    Value res =
        MfmaOp::create(rewriter, loc, accTy, a, b, c, zeroAttr, zeroAttr, zeroAttr).getResult();
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
        return getBufferFatPtrType(flyMemRefTy.getContext());
      unsigned as = mapAddressSpace(flyMemRefTy.getAddressSpace().getValue());
      return LLVM::LLVMPointerType::get(flyMemRefTy.getContext(), as);
    });
    addConversion([&](fly::PointerType flyPtrTy) -> Type {
      if (flyPtrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc)
        return getBufferFatPtrType(flyPtrTy.getContext());
      unsigned as = mapAddressSpace(flyPtrTy.getAddressSpace().getValue());
      return LLVM::LLVMPointerType::get(flyPtrTy.getContext(), as);
    });
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
                           ROCDL::ROCDLDialect, fly_rocdl::FlyROCDLDialect>();
    target.addIllegalDialect<fly::FlyDialect>();

    target.addLegalOp<MakeIntTupleOp, MakeLayoutOp, MakeTileOp>();
    target.addLegalOp<MakeMmaAtomOp, MakeCopyAtomOp>();

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

    patterns.add<MakePtrOpLowering>(typeConverter, context);
    patterns.add<MemRefAllocOpLowering>(typeConverter, context);
    patterns.add<GetIterOpLowering>(typeConverter, context);
    patterns.add<AddOffsetOpLowering>(typeConverter, context);
    patterns.add<MakeViewOpLowering>(typeConverter, context);
    patterns.add<MemRefLoadVecOpLowering>(typeConverter, context);
    patterns.add<MemRefStoreVecOpLowering>(typeConverter, context);
    patterns.add<MemRefLoadOpLowering>(typeConverter, context);
    patterns.add<MemRefStoreOpLowering>(typeConverter, context);
    patterns.add<CopyAtomCallLowering>(typeConverter, context);
    patterns.add<MmaAtomCallLowering>(typeConverter, context);
    patterns.add<GpuLaunchFuncOpLowering>(typeConverter, context);

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
