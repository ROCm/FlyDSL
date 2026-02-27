
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
  // - Global   -> 1 (global)
  // - Shared   -> 3 (local/LDS/workgroup)
  // - Register -> 5 (private)
  // Fallback to 0 (generic).
  switch (space) {
  case AddressSpace::Global:
    return 1;
  case AddressSpace::Shared:
    return 3;
  case AddressSpace::Register:
    return 5;
  }
  return 0;
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
    // After type conversion, Fly memref is already a `!llvm.ptr`.
    Value mem = adaptor.getMemref();
    auto resTy =
        dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(op.getResult().getType()));
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
    auto baseTy = dyn_cast<LLVM::LLVMPointerType>(base.getType());
    if (!baseTy)
      return failure();

    auto offsetIdx = materializeScalarIndex(op.getOffset(), loc, rewriter);
    if (failed(offsetIdx))
      return failure();

    auto resultTy =
        dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultTy)
      return failure();

    FailureOr<Value> offsetI64 = toI64(*offsetIdx, loc, rewriter);
    if (failed(offsetI64))
      return failure();

    // Pointer arithmetic: gep by element offset.
    // Note: GEP element type is the (pointee) element type of the pointer.
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
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
    auto baseTy = dyn_cast<LLVM::LLVMPointerType>(base.getType());
    if (!baseTy)
      return failure();

    auto resultTy =
        dyn_cast<LLVM::LLVMPointerType>(getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultTy)
      return failure();
    if (base.getType() == resultTy) {
      rewriter.replaceOp(op, base);
      return success();
    }
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultTy, base);
    return success();
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
    auto copyAtomTy = dyn_cast<CopyAtomTypeInterface>(op.getCopyAtom().getType());
    if (!copyAtomTy)
      return rewriter.notifyMatchFailure(op, "copyAtom does not implement CopyAtomTypeInterface");

    Value src = adaptor.getSrc();
    Value dst = adaptor.getDst();

    auto srcPtrTy = dyn_cast<LLVM::LLVMPointerType>(src.getType());
    auto dstPtrTy = dyn_cast<LLVM::LLVMPointerType>(dst.getType());
    if (!srcPtrTy || !dstPtrTy)
      return rewriter.notifyMatchFailure(op, "src/dst are not llvm.ptr after conversion");

    auto srcFlyTy = dyn_cast<fly::MemRefType>(op.getSrc().getType());
    auto dstFlyTy = dyn_cast<fly::MemRefType>(op.getDst().getType());
    if (!srcFlyTy || !dstFlyTy)
      return rewriter.notifyMatchFailure(op, "expected Fly memref types on original op");

    if (srcFlyTy.getElemTy() != dstFlyTy.getElemTy())
      return rewriter.notifyMatchFailure(op, "src/dst element types mismatch");

    LayoutBuilder<LayoutAttr> attrBuilder(rewriter.getContext());

    auto thrValLayoutSrc = dyn_cast<LayoutAttr>(copyAtomTy.getThrValLayoutSrc());
    if (!thrValLayoutSrc)
      return rewriter.notifyMatchFailure(op, "getThrValLayoutSrc returned null or non-LayoutAttr");
    IntAttr numValSrcAttr = intTupleProductImpl(attrBuilder, thrValLayoutSrc.getShape().at(1));
    if (!numValSrcAttr.isStatic())
      return rewriter.notifyMatchFailure(op, "NumValSrc is not static");
    int64_t numValSrc = numValSrcAttr.getValue();

    Location loc = op.getLoc();
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

    // TODO: using copyAtom bitSize when layout_recast is ready.
    int64_t copyBytes = numValSrc * elemBits / 8;
    Value len = arith::ConstantIntOp::create(rewriter, loc, copyBytes, /*width=*/64).getResult();
    LLVM::MemcpyOp::create(rewriter, loc, dst, src, len, /*isVolatile=*/false);

    rewriter.eraseOp(op);
    return success();
  }
};

class MmaAtomCallLowering : public OpConversionPattern<MmaAtomCall> {
public:
  using OpConversionPattern<MmaAtomCall>::OpConversionPattern;

  LogicalResult matchAndRewrite(MmaAtomCall op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Only handle MFMA F32 16x16x4 F32 atom for now.

    Type mmaAtomType = op.getMmaAtom().getType();
    if (!isa<MmaAtomTypeInterface>(mmaAtomType))
      return rewriter.notifyMatchFailure(op,
                                         "expected MmaAtomTypeInterface type for mmaAtom operand");

    Location loc = op.getLoc();

    // After type conversion, memrefs are lowered to `!llvm.ptr<addrspace>`.
    Value dPtr = adaptor.getD();
    Value aPtr = adaptor.getA();
    Value bPtr = adaptor.getB();
    Value cPtr = adaptor.getC();

    auto dPtrTy = dyn_cast<LLVM::LLVMPointerType>(dPtr.getType());
    auto aPtrTy = dyn_cast<LLVM::LLVMPointerType>(aPtr.getType());
    auto bPtrTy = dyn_cast<LLVM::LLVMPointerType>(bPtr.getType());
    auto cPtrTy = dyn_cast<LLVM::LLVMPointerType>(cPtr.getType());

    if (!dPtrTy || !aPtrTy || !bPtrTy || !cPtrTy)
      return rewriter.notifyMatchFailure(op, "expected llvm.ptr operands after type conversion");

    Type f32Ty = rewriter.getF32Type();
    VectorType accTy = VectorType::get({4}, f32Ty);

    // Load A/B scalars and C accumulator vector from the provided pointers.
    Value a = LLVM::LoadOp::create(rewriter, loc, f32Ty, aPtr);
    Value b = LLVM::LoadOp::create(rewriter, loc, f32Ty, bPtr);
    Value c = LLVM::LoadOp::create(rewriter, loc, accTy, cPtr);

    // MFMA control attributes (cbsz, abid, blgp). Default to 0.
    // Note: These are I32Attr attributes, not Value operands!
    auto zeroAttr = rewriter.getI32IntegerAttr(0);

    // rocdl.mfma.f32.16x16x4f32 : (f32, f32, vector<4xf32>) -> vector<4xf32>
    // with attributes: cbsz, abid, blgp
    Value res = ROCDL::mfma_f32_16x16x4f32::create(rewriter, loc, accTy, a, b, c, zeroAttr,
                                                   zeroAttr, zeroAttr)
                    .getResult();

    // Store result back to D pointer.
    LLVM::StoreOp::create(rewriter, loc, res, dPtr);
    rewriter.eraseOp(op);
    return success();
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

    // Convert Fly memref/pointer to a raw LLVM pointer.
    addConversion([&](fly::MemRefType flyMemRefTy) -> Type {
      unsigned as = mapAddressSpace(flyMemRefTy.getAddressSpace().getValue());
      return LLVM::LLVMPointerType::get(flyMemRefTy.getContext(), as);
    });
    addConversion([&](fly::PointerType flyPtrTy) -> Type {
      unsigned as = mapAddressSpace(flyPtrTy.getAddressSpace().getValue());
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
      src = rewriter.create<LLVM::AddrSpaceCastOp>(op.getLoc(), resultType, src);
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
                           ROCDL::ROCDLDialect, fly_rocdl::FlyROCDLDialect>();
    target.addIllegalDialect<fly::FlyDialect>();

    target.addLegalOp<MakeIntTupleOp, MakeLayoutOp, MakeTileOp>();
    target.addLegalOp<MakeAtomOp>();

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
