// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/Transforms/StructuralTypeConversions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/bit.h"

#include <numeric>

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/CoreBridge.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Utils/BufferFatPtr.h"

namespace mlir {
#define GEN_PASS_DEF_FLYTOROCDLCONVERSIONPASS
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_rocdl;

namespace {

unsigned mapToLLVMAddressSpace(AddressSpace addrSpace) {
  switch (addrSpace) {
  case AddressSpace::Generic:
    return 0;
  case AddressSpace::Global:
    return ROCDL::ROCDLDialect::kGlobalMemoryAddressSpace;
  case AddressSpace::Shared:
    return ROCDL::ROCDLDialect::kSharedMemoryAddressSpace;
  case AddressSpace::Register:
    return ROCDL::ROCDLDialect::kPrivateMemoryAddressSpace;
  }
  llvm_unreachable("unsupported address space");
}

unsigned mapAttrToLLVMAddressSpace(Attribute attr) {
  if (auto e = dyn_cast<AddressSpaceAttr>(attr))
    return mapToLLVMAddressSpace(e.getValue());
  if (isTargetAddressSpace<BufferDescAddressAttr>(attr))
    return BufferFatPtr::getRsrcPtrType(attr.getContext()).getAddressSpace();
  return 0; // default to generic address space
}

/// Byte alignment that still holds after `applySwizzleOnPtr`.
///
/// The swizzle XORs address bits at and above `base`, so only the low `base`
/// bits of the address survive; whatever the pointer type promises above that
/// no longer holds. `llvm::Align` additionally requires a power of two, which
/// the divisibility arithmetic behind `AlignAttr` does not guarantee, so what
/// is left is the greatest power-of-two divisor.
static unsigned getLLVMAlignment(fly::PointerType ptrTy) {
  int32_t align = ptrTy.getAlignment().getAlignment();
  if (align <= 0)
    return 0;
  auto swizzle = ptrTy.getSwizzle();
  if (!swizzle.isTrivialSwizzle())
    align = std::gcd(align, int32_t{1} << swizzle.getBase());
  return 1u << llvm::countr_zero(static_cast<unsigned>(align));
}

//===----------------------------------------------------------------------===//
// Lowered Fly values
//
// A Fly algebra value lowers to core::get_llvm_type of its Core description,
// exactly as at function boundaries: the packed struct of its dynamic scalars,
// nested per structural component, or no value at all when it is fully static.
// A memref lowers to its pointer followed by its layout. Depth-first, the
// scalars appear in Core's binding order.
//===----------------------------------------------------------------------===//

/// `fn` applied to `type` cast to its Fly algebra type, a type that lowers to the
/// packed struct of its Core description; `otherwise` for any other type.
template <class Result, class Fn> Result visitAlgebraType(Type type, Result otherwise, Fn &&fn) {
  return llvm::TypeSwitch<Type, Result>(type)
      .template Case<IntTupleType, LayoutType, ComposedLayoutType, CoordTensorType, TileType,
                     SwizzleType, CoordSwizzleType>([&](auto value) -> Result { return fn(value); })
      .Default([&](Type) -> Result { return otherwise; });
}

/// The lowered struct of a Fly algebra type; null when the value is fully
/// static, std::nullopt for other types.
std::optional<LLVM::LLVMStructType> getLoweredType(Type type) {
  return visitAlgebraType<std::optional<LLVM::LLVMStructType>>(type, std::nullopt, [&](auto value) {
    return core::get_llvm_type(value.getRef(), *type.getContext());
  });
}

/// The value a single-carrier operand lowered to, or null for a static value.
Value loweredValue(ValueRange lowered) { return lowered.empty() ? Value() : lowered.front(); }

LLVM::LLVMStructType emptyStruct(MLIRContext *ctx) {
  return LLVM::LLVMStructType::getLiteral(ctx, {});
}

/// The atom of a tiled copy or MMA; any other type as is.
Type getAtomType(Type type) {
  if (auto tiledCopy = dyn_cast<TiledCopyType>(type))
    return tiledCopy.getCopyAtom();
  if (auto tiledMma = dyn_cast<TiledMmaType>(type))
    return tiledMma.getMmaAtom();
  return type;
}

/// The op type of a copy or MMA atom; null for any other type.
Type getAtomOpType(Type atomTy) {
  if (auto copyAtom = dyn_cast<CopyAtomType>(atomTy))
    return copyAtom.getCopyOp();
  if (auto mmaAtom = dyn_cast<MmaAtomType>(atomTy))
    return mmaAtom.getMmaOp();
  return {};
}

/// The atom value an emitter receives: a static atom carries no state.
Value loweredAtom(OpBuilder &builder, Location loc, ValueRange lowered) {
  if (!lowered.empty())
    return lowered.front();
  return LLVM::UndefOp::create(builder, loc, emptyStruct(builder.getContext()));
}

SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> result;
  for (ValueRange value : values)
    llvm::append_range(result, value);
  return result;
}

/// The dynamic scalars of a lowered value of Fly type `type`, in Core's binding order.
FailureOr<SmallVector<Value, 8>> dynamicScalars(OpBuilder &builder, Location loc, Type type,
                                                ValueRange lowered) {
  core::EmitContext ctx(builder, loc);
  // A memref's pointer carrier precedes its layout.
  if (auto memref = dyn_cast<fly::MemRefType>(type))
    return core::collect_llvm_scalars(memref.getRef().layout(), loweredValue(lowered.drop_front()),
                                      ctx);
  return visitAlgebraType<FailureOr<SmallVector<Value, 8>>>(type, failure(), [&](auto value) {
    return core::collect_llvm_scalars(value.getRef(), loweredValue(lowered), ctx);
  });
}

/// A lowered int tuple bound to its Core profile. A static leaf needs no carrier
/// and materializes as a constant.
FailureOr<core::IntTuple<core::EmitOp>> bindLoweredTuple(IntTupleType type, ValueRange lowered,
                                                         core::EmitContext &ctx) {
  return core::bind_llvm_value(type.getRef(), loweredValue(lowered), ctx);
}

/// A struct of type `type` holding `fields`.
FailureOr<Value> packFields(OpBuilder &builder, Location loc, LLVM::LLVMStructType type,
                            ValueRange fields) {
  if (TypeRange(fields) != type.getBody())
    return failure();
  Value result = LLVM::UndefOp::create(builder, loc, type);
  for (auto [i, field] : llvm::enumerate(fields))
    result = LLVM::InsertValueOp::create(builder, loc, result, field,
                                         ArrayRef<int64_t>{static_cast<int64_t>(i)});
  return result;
}

/// Core's lowered values, with the magic numbers computed the cheap way for
/// where they run: a host function keeps Core's integer division, while a
/// kernel computes an i32 multiplier in FP64, which CDNA runs at full rate.
class CDNA_EmitHooks : public core::EmitHooks {
public:
  Value makeFastDivMod(core::EmitContext &ctx, Value divisor) const override {
    Block *block = ctx.builder.getInsertionBlock();
    Operation *parent = block ? block->getParentOp() : nullptr;
    bool onDevice =
        parent && (isa<gpu::GPUModuleOp>(parent) || parent->getParentOfType<gpu::GPUModuleOp>());
    if (onDevice && divisor.getType().isInteger(32))
      return core::emit_op::makeFastDivModF64(ctx, divisor);
    return core::EmitHooks::makeFastDivMod(ctx, divisor);
  }
};

/// Constructors lower to the lowered values of their operands, in the
/// component order of the result.
class ConstructorLowering : public ConversionPattern {
public:
  ConstructorLowering(StringRef name, const TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, name, 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = op->getResult(0).getType();
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertType(resultType, types)))
      return failure();
    SmallVector<Value> fields;
    if (isa<MakeComposedLayoutOp>(op)) {
      for (int i : {2, 1, 0})
        llvm::append_range(fields, operands[i]);
    } else {
      for (ValueRange operand : operands)
        llvm::append_range(fields, operand);
    }
    // A memref keeps its carriers separate; other values pack theirs.
    if (isa<fly::MemRefType>(resultType) || types.empty()) {
      if (TypeRange(fields) != TypeRange(types))
        return rewriter.notifyMatchFailure(op, "unexpected lowered operand types");
      rewriter.replaceOpWithMultiple(op, {fields});
      return success();
    }
    auto structType = dyn_cast<LLVM::LLVMStructType>(types.front());
    if (types.size() != 1 || !structType)
      return failure();
    // A tile's operands are its dynamic scalars, which Core nests per layout;
    // the other constructors' operands are already the struct's fields.
    core::EmitContext ctx(rewriter, op->getLoc());
    FailureOr<Value> packed = failure();
    if (isa<MakeTileOp>(op)) {
      core::EmitBindings bindings;
      llvm::append_range(bindings.scalars, fields);
      if (auto tile = core::bind(cast<TileType>(resultType).getRef(), bindings, ctx);
          succeeded(tile))
        packed = core::build_llvm_value(*tile, ctx);
    } else if (TypeRange(fields) == structType.getBody()) {
      packed = core::emit_op::buildLLVMStruct(ctx, fields);
    }
    if (failed(packed) || !*packed || (*packed).getType() != structType)
      return rewriter.notifyMatchFailure(op, "unexpected lowered operand types");
    rewriter.replaceOp(op, *packed);
    return success();
  }
};

/// Extractors select from the lowered value of their operand.
class ExtractorLowering : public ConversionPattern {
public:
  ExtractorLowering(StringRef name, const TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, name, 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type sourceType = op->getOperand(0).getType();
    ValueRange source = operands[0];
    if (isa<GetLeavesOp>(op)) {
      auto results = dynamicScalars(rewriter, loc, sourceType, source);
      SmallVector<Type> resultTypes;
      if (failed(results) ||
          failed(getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)) ||
          TypeRange(*results) != TypeRange(resultTypes))
        return failure();
      rewriter.replaceOp(op, *results);
      return success();
    }
    if (isa<GetScalarOp>(op)) {
      // A static scalar has no lowered value; it materializes as a constant.
      core::EmitContext ctx(rewriter, loc);
      auto tuple = bindLoweredTuple(cast<IntTupleType>(sourceType), source, ctx);
      if (failed(tuple))
        return failure();
      auto ref = tuple->asRef();
      while (!ref.isLeaf() && ref.rank() == 1)
        ref = ref.at(0);
      if (!ref.isLeaf() || ref.getLeaf().scale().isRatio())
        return failure();
      auto scalar = ref.getLeaf().scale().materialize(ctx);
      if (failed(scalar) ||
          (*scalar).getType() != getTypeConverter()->convertType(op->getResult(0).getType()))
        return failure();
      rewriter.replaceOp(op, *scalar);
      return success();
    }
    // get_iter selects a memref's pointer carrier; fly-layout-lowering expanded
    // a coordinate tensor's.
    if (!isa<fly::MemRefType>(sourceType))
      return failure();
    rewriter.replaceOpWithMultiple(op, {source.take_front()});
    return success();
  }
};

/// Ops that lower to the lowered value of their first operand: a swizzle or a
/// recast lives on in the pointer type, and a tiled copy or MMA is its atom.
class ForwardOperandLowering : public ConversionPattern {
public:
  ForwardOperandLowering(StringRef name, const TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, name, 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithMultiple(op, {operands.front()});
    return success();
  }
};

/// A fast-divmod handle lowers to Core's lowered handle, and the fast division
/// ops to Core's default emission. make_fast_divmod is where the magic numbers
/// are computed: on the host for a host function, on the GPU for a kernel.
class MakeFastDivModOpLowering : public OpConversionPattern<MakeFastDivModOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MakeFastDivModOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    static const CDNA_EmitHooks hooks;
    core::EmitContext ctx(rewriter, op.getLoc(), &hooks);
    rewriter.replaceOp(op, ctx.getHooks().makeFastDivMod(ctx, adaptor.getDivisor()));
    return success();
  }
};

class FastDivModOpLowering : public OpConversionPattern<FastDivModOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(FastDivModOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    core::EmitContext ctx(rewriter, op.getLoc());
    auto [quotient, remainder] =
        ctx.getHooks().fastDivMod(ctx, adaptor.getDividend(), adaptor.getDivisor());
    rewriter.replaceOp(op, ValueRange{quotient, remainder});
    return success();
  }
};

class FastDivModGetDivisorOpLowering : public OpConversionPattern<FastDivModGetDivisorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(FastDivModGetDivisorOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    core::EmitContext ctx(rewriter, op.getLoc());
    rewriter.replaceOp(op, ctx.getHooks().getFastDivModDivisor(ctx, adaptor.getDivisor()));
    return success();
  }
};

// Create a freshly named LDS global of `[nbytes x i8]` in `addrSpace`, inserted
// at the start of `moduleOp`. The symbol name is `prefix` followed by a counter
// chosen to avoid collisions with existing globals in the module. Shared by
// the static-LDS `make_ptr` and dynamic-LDS `get_dyn_shared` lowerings.
static LLVM::GlobalOp createLDSGlobal(ConversionPatternRewriter &rewriter,
                                      gpu::GPUModuleOp moduleOp, Location loc, StringRef prefix,
                                      int64_t nbytes, int64_t align, unsigned addrSpace) {
  llvm::StringSet<> existingNames;
  for (auto globalOp : moduleOp.getBody()->getOps<LLVM::GlobalOp>())
    existingNames.insert(globalOp.getSymName());

  unsigned counter = 0;
  SmallString<128> symName = SymbolTable::generateSymbolName<128>(
      prefix, [&](StringRef candidate) { return existingNames.contains(candidate); }, counter);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto arrayTy = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), nbytes);
  auto globalOp = LLVM::GlobalOp::create(rewriter, loc, arrayTy,
                                         /*isConstant=*/false, LLVM::Linkage::External, symName,
                                         /*value=*/Attribute(),
                                         /*alignment=*/align, addrSpace);
  // LDS (addrspace 3) symbols are per-workgroup hardware and can never resolve across DSO.
  globalOp.setDsoLocal(true);
  return globalOp;
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
    Attribute addrSpaceAttr = flyPtrTy.getAddressSpace();

    if (isGenericAddressSpace<AddressSpace::Register>(addrSpaceAttr)) {
      auto dictAttrs = op.getDictAttrs();
      if (!dictAttrs)
        return rewriter.notifyMatchFailure(op, "register make_ptr requires dictAttrs");
      auto allocSize = dictAttrs->getAs<IntegerAttr>("allocSize");
      if (!allocSize)
        return rewriter.notifyMatchFailure(op, "register make_ptr requires allocSize in ptrAttrs");
      unsigned llvmAS = mapToLLVMAddressSpace(AddressSpace::Register);
      auto llvmPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), llvmAS);
      Value nElems = arith::ConstantIntOp::create(rewriter, loc, allocSize.getInt(), 64);
      Type elemTy = projectToLLVMCompatibleElemTy(flyPtrTy.getElemTy());
      Value ptr = LLVM::AllocaOp::create(rewriter, loc, llvmPtrTy, elemTy, nElems, 0);
      rewriter.replaceOp(op, ptr);
      return success();
    } else if (isGenericAddressSpace<AddressSpace::Shared>(addrSpaceAttr)) {
      // Static LDS sub-allocation. Each op lowers to a freshly named
      // `@__shared_alloc_<n>` LDS global.
      auto dictAttrs = op.getDictAttrs();
      if (!dictAttrs)
        return rewriter.notifyMatchFailure(
            op, "shared make_ptr requires dictAttrs={allocBytes, allocAlign}");
      auto allocBytesAttr = dictAttrs->getAs<IntegerAttr>("allocBytes");
      auto allocAlignAttr = dictAttrs->getAs<IntegerAttr>("allocAlign");
      if (!allocBytesAttr || !allocAlignAttr)
        return rewriter.notifyMatchFailure(
            op, "shared make_ptr requires allocBytes, allocAlign in dictAttrs");

      auto moduleOp = op->getParentOfType<gpu::GPUModuleOp>();
      if (!moduleOp)
        return op->emitError("shared make_ptr must be inside a gpu.module");

      LLVM::GlobalOp global =
          createLDSGlobal(rewriter, moduleOp, loc, "__shared_alloc", allocBytesAttr.getInt(),
                          allocAlignAttr.getInt(), mapToLLVMAddressSpace(AddressSpace::Shared));
      rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(op, global);
      return success();
    } else if (isTargetAddressSpace<BufferDescAddressAttr>(addrSpaceAttr)) {
      auto args = adaptor.getArgs();
      if (args.size() != 4)
        return rewriter.notifyMatchFailure(
            op, "buffer_rsrc make_ptr expects 4 args: base, stride, numRecords, flags");

      Value base = args[0];
      Value stride = args[1];
      Value numRecords = args[2];
      Value flags = args[3];

      unsigned llvmAS = mapAttrToLLVMAddressSpace(addrSpaceAttr);
      auto rsrcPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), llvmAS);
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
    unsigned addrSpace = mapAttrToLLVMAddressSpace(flyPtrTy.getAddressSpace());

    auto moduleOp = op->getParentOfType<gpu::GPUModuleOp>();
    if (!moduleOp)
      return op->emitError("get_dyn_shared must be inside a gpu.module");

    // Dynamic shared memory has a single logical region per kernel; reuse an
    // existing `[0 x i8]` global if one is already present so multiple
    // `get_dyn_shared` ops resolve to the same base symbol.
    LLVM::GlobalOp sharedGlobal;
    for (auto globalOp : moduleOp.getBody()->getOps<LLVM::GlobalOp>()) {
      if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(globalOp.getType())) {
        if (globalOp.getAddrSpace() == addrSpace && arrayType.getNumElements() == 0 &&
            globalOp.getAlignment().value_or(0) == 1024) {
          sharedGlobal = globalOp;
          break;
        }
      }
    }
    if (!sharedGlobal) {
      sharedGlobal = createLDSGlobal(rewriter, moduleOp, loc, "__dynamic_shared",
                                     /*nbytes=*/0, /*align=*/1024, addrSpace);
    }

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

class ToLLVMPtrOpLowering : public OpConversionPattern<ToLLVMPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ToLLVMPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto convertedPtrTy = dyn_cast<LLVM::LLVMPointerType>(adaptor.getPtr().getType());
    if (!convertedPtrTy)
      return op.emitError("pointer address space does not lower to a bare ROCDL LLVM pointer: ")
             << cast<fly::PointerType>(op.getPtr().getType()).getAddressSpace();

    auto requestedPtrTy = dyn_cast<LLVM::LLVMPointerType>(op.getResult().getType());
    if (!requestedPtrTy)
      return op.emitError("expected an LLVM pointer result, got ") << op.getResult().getType();
    if (requestedPtrTy.getAddressSpace() != convertedPtrTy.getAddressSpace())
      return op.emitError("requested LLVM address space ")
             << requestedPtrTy.getAddressSpace() << " but ROCDL conversion requires "
             << convertedPtrTy.getAddressSpace();

    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

/// A recast keeps the address. A buffer pointer carries an element offset, so a
/// recast to another storage width rescales it; other pointers pass through.
class RecastIterOpLowering : public OpConversionPattern<RecastIterOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(RecastIterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto srcTy = cast<fly::PointerType>(op.getSrc().getType());
    auto dstTy = cast<fly::PointerType>(op.getResult().getType());
    int64_t srcBits = srcTy.getRef().storageElementBitWidth();
    int64_t dstBits = dstTy.getRef().storageElementBitWidth();
    if (!isTargetAddressSpace<BufferDescAddressAttr>(srcTy.getAddressSpace()) ||
        srcBits == dstBits) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
    auto loc = op.getLoc();
    BufferFatPtr bp(srcTy, adaptor.getSrc());
    Value offset = bp.valOffset(rewriter, loc);
    auto constant = [&](int64_t value) {
      return arith::ConstantIntOp::create(rewriter, loc, value, 32);
    };
    if (srcBits % dstBits == 0) {
      offset = arith::MulIOp::create(rewriter, loc, offset, constant(srcBits / dstBits));
    } else {
      offset = arith::MulIOp::create(rewriter, loc, offset, constant(srcBits));
      offset = arith::DivUIOp::create(rewriter, loc, offset, constant(dstBits));
    }
    rewriter.replaceOp(op, BufferFatPtr::pack(rewriter, loc, bp.bufferRsrc(rewriter, loc), offset));
    return success();
  }
};

class GetBufferRsrcOpLowering : public OpConversionPattern<fly_rocdl::GetBufferRsrcOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(fly_rocdl::GetBufferRsrcOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy || !isTargetAddressSpace<BufferDescAddressAttr>(flyPtrTy.getAddressSpace()))
      return rewriter.notifyMatchFailure(op, "expected a buffer_desc address space pointer");

    // adaptor.getPtr() is the already-converted BufferFatPtr struct; extract the
    // buffer resource (element 0, an !llvm.ptr<8>).
    BufferFatPtr bp(flyPtrTy, adaptor.getPtr());
    rewriter.replaceOp(op, bp.bufferRsrc(rewriter, op.getLoc()));
    return success();
  }
};

class AddOffsetOpLowering : public OpConversionPattern<AddOffsetOp> {
public:
  AddOffsetOpLowering(const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern<AddOffsetOp>(typeConverter, context) {}

  LogicalResult matchAndRewrite(AddOffsetOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value base = loweredValue(adaptor.getPtr());

    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();

    auto offsetTy = dyn_cast<fly::IntTupleType>(op.getOffset().getType());
    IntTupleAttr offsetAttr = offsetTy.getAttr();
    if (!offsetAttr.isLeaf())
      return rewriter.notifyMatchFailure(op, "offset must be a leaf int tuple");

    core::EmitContext ctx(rewriter, loc);
    auto offset = bindLoweredTuple(offsetTy, adaptor.getOffset(), ctx);
    if (failed(offset))
      return failure();
    auto offsetInt = offset->asRef().getLeaf().scale();
    Value offsetVal;
    if (offsetInt.isStatic() && llvm::isInt<32>(offsetInt.staticValue())) {
      offsetVal = arith::ConstantIntOp::create(rewriter, loc, offsetInt.staticValue(), 32);
    } else {
      auto offsetScalar = offsetInt.materialize(ctx);
      if (failed(offsetScalar))
        return failure();
      offsetVal = *offsetScalar;
    }

    if (isTargetAddressSpace<BufferDescAddressAttr>(flyPtrTy.getAddressSpace())) {
      BufferFatPtr bp(flyPtrTy, base);
      rewriter.replaceOp(op, bp.addOffset(rewriter, loc, offsetVal));
      return success();
    }

    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(base.getType());
    if (!ptrTy)
      return failure();

    Type elemTy = projectToLLVMCompatibleElemTy(flyPtrTy.getElemTy());
    Value gep = LLVM::GEPOp::create(rewriter, loc, ptrTy, elemTy, base, ValueRange{offsetVal});
    rewriter.replaceOp(op, gep);
    return success();
  }
};

/// Whether a vector access of `accessTy` through `ptrTy` stays within one
/// swizzle granule, which the swizzle moves as a whole.
bool fitsSwizzleGranule(fly::PointerType ptrTy, Type accessTy) {
  auto vecTy = dyn_cast<VectorType>(accessTy);
  auto swizzle = ptrTy.getSwizzle();
  if (!vecTy || swizzle.isTrivialSwizzle())
    return true;
  int64_t vecBytes = vecTy.getNumElements() * vecTy.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t baseBytes = int64_t{1} << swizzle.getBase();
  return baseBytes % vecBytes == 0;
}

/// Loads `type` through the lowered Fly pointer `ptr`: a raw buffer load through a
/// buffer descriptor, an aligned load through the swizzled LLVM pointer otherwise.
Value loadThroughPointer(OpBuilder &builder, Location loc, fly::PointerType ptrTy, Value ptr,
                         Type type) {
  if (isTargetAddressSpace<BufferDescAddressAttr>(ptrTy.getAddressSpace())) {
    BufferFatPtr bp(ptrTy, ptr);
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    auto auxAttr = builder.getI32IntegerAttr(0);
    ArrayAttr noAttrs;
    return ROCDL::RawPtrBufferLoadOp::create(builder, loc, type, bp.bufferRsrc(builder, loc),
                                             bp.swizzleByteOffset(builder, loc), zero, auxAttr,
                                             noAttrs, noAttrs, noAttrs);
  }
  ptr = applySwizzleOnPtr(builder, loc, cast<TypedValue<LLVM::LLVMPointerType>>(ptr),
                          ptrTy.getSwizzle());
  return LLVM::LoadOp::create(builder, loc, type, ptr, getLLVMAlignment(ptrTy));
}

/// Stores `value` through the lowered Fly pointer `ptr`, as `loadThroughPointer` loads.
void storeThroughPointer(OpBuilder &builder, Location loc, fly::PointerType ptrTy, Value ptr,
                         Value value) {
  if (isTargetAddressSpace<BufferDescAddressAttr>(ptrTy.getAddressSpace())) {
    BufferFatPtr bp(ptrTy, ptr);
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    auto auxAttr = builder.getI32IntegerAttr(0);
    ArrayAttr noAttrs;
    ROCDL::RawPtrBufferStoreOp::create(builder, loc, value, bp.bufferRsrc(builder, loc),
                                       bp.swizzleByteOffset(builder, loc), zero, auxAttr, noAttrs,
                                       noAttrs, noAttrs);
    return;
  }
  ptr = applySwizzleOnPtr(builder, loc, cast<TypedValue<LLVM::LLVMPointerType>>(ptr),
                          ptrTy.getSwizzle());
  LLVM::StoreOp::create(builder, loc, value, ptr, getLLVMAlignment(ptrTy));
}

class PtrLoadOpLowering : public OpConversionPattern<PtrLoadOp> {
public:
  using OpConversionPattern<PtrLoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PtrLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();
    Type loadTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!loadTy)
      return rewriter.notifyMatchFailure(op, "failed to convert ptr.load result type");
    if (!fitsSwizzleGranule(flyPtrTy, loadTy))
      return rewriter.notifyMatchFailure(
          op, "vector ptr.load byte size must divide swizzle base granularity");
    rewriter.replaceOp(
        op, loadThroughPointer(rewriter, op.getLoc(), flyPtrTy, adaptor.getPtr(), loadTy));
    return success();
  }
};

class PtrStoreOpLowering : public OpConversionPattern<PtrStoreOp> {
public:
  using OpConversionPattern<PtrStoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(PtrStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto flyPtrTy = dyn_cast<fly::PointerType>(op.getPtr().getType());
    if (!flyPtrTy)
      return failure();
    if (!fitsSwizzleGranule(flyPtrTy, adaptor.getValue().getType()))
      return rewriter.notifyMatchFailure(
          op, "vector ptr.store byte size must divide swizzle base granularity");
    storeThroughPointer(rewriter, op.getLoc(), flyPtrTy, adaptor.getPtr(), adaptor.getValue());
    rewriter.eraseOp(op);
    return success();
  }
};

/// An atom lowers to its initial state: nothing for a static atom, the state its
/// op type builds from the construction arguments for a stateful one, and an
/// empty struct otherwise.
template <class OpT> class MakeAtomOpLowering : public OpConversionPattern<OpT> {
public:
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<OpT>::OpAdaptor;

  LogicalResult matchAndRewrite(OpT op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type atomTy = op.getResult().getType();
    ValueRange args = adaptor.getOperands();
    if (cast<MayStaticTypeInterface>(atomTy).isStatic() && args.empty()) {
      rewriter.replaceOpWithMultiple(op, {ValueRange{}});
      return success();
    }
    if (auto stateful = dyn_cast<StatefulOpTypeInterface>(getAtomOpType(atomTy))) {
      Value state = stateful.getAtomState(rewriter, op.getLoc(), args);
      if (!state)
        return failure();
      rewriter.replaceOp(op, state);
      return success();
    }
    if (!args.empty())
      return rewriter.notifyMatchFailure(op, "stateless atom takes no construction arguments");
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, this->getTypeConverter()->convertType(atomTy));
    return success();
  }
};

/// An int tuple field value reaches `setAtomState` as a struct of all its
/// integer leaves: static leaves as constants, dynamic leaves as their scalars.
FailureOr<Value> tupleFieldValue(OpBuilder &builder, Location loc, IntTupleType type,
                                 ValueRange lowered) {
  core::EmitContext ctx(builder, loc);
  auto tuple = bindLoweredTuple(type, lowered, ctx);
  if (failed(tuple))
    return failure();
  SmallVector<Value> leaves;
  for (auto leaf : tuple->asRef().leaves()) {
    auto scale = leaf.scale();
    if (leaf.isBasis() || scale.isRatio() || scale.isNone())
      return failure();
    auto value = scale.materialize(ctx);
    if (failed(value))
      return failure();
    leaves.push_back(*value);
  }
  auto structType =
      LLVM::LLVMStructType::getLiteral(builder.getContext(), llvm::to_vector(TypeRange(leaves)));
  return packFields(builder, loc, structType, leaves);
}

class AtomSetValueOpLowering : public OpConversionPattern<AtomSetValueOp> {
public:
  using OpConversionPattern<AtomSetValueOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(AtomSetValueOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto stateful = dyn_cast_if_present<StatefulOpTypeInterface>(
        getAtomOpType(getAtomType(op.getAtom().getType())));
    if (!stateful)
      return rewriter.notifyMatchFailure(op, "atom is not stateful");
    StringAttr fieldAttr = op.getFieldAttr();
    Location loc = op.getLoc();

    Value structVal = loweredValue(adaptor.getAtom());
    Value fieldVal = loweredValue(adaptor.getValue());
    if (auto tupleTy = dyn_cast<IntTupleType>(op.getValue().getType())) {
      auto tuple = tupleFieldValue(rewriter, loc, tupleTy, adaptor.getValue());
      if (failed(tuple))
        return op.emitOpError("expected an int tuple field value of integer leaves");
      fieldVal = *tuple;
    }
    Value result = stateful.setAtomState(rewriter, loc, structVal, fieldAttr, fieldVal);
    if (!result)
      return rewriter.notifyMatchFailure(op, "setAtomState failed");

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Replaces an SSA atom call by its result, or erases one that has none.
void replaceOrErase(ConversionPatternRewriter &rewriter, Operation *op, Value result) {
  if (op->getNumResults())
    rewriter.replaceOp(op, result);
  else
    rewriter.eraseOp(op);
}

class CopyAtomCallLowering : public OpConversionPattern<CopyAtomCall> {
public:
  using OpConversionPattern<CopyAtomCall>::OpConversionPattern;

  LogicalResult matchAndRewrite(CopyAtomCall op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type copyAtomType = op.getCopyAtom().getType();
    auto copyAtom = dyn_cast<CopyAtomType>(copyAtomType);
    if (!copyAtom)
      return rewriter.notifyMatchFailure(op, "copyAtom is not CopyAtomType");

    Value copyAtomVal = loweredAtom(rewriter, op.getLoc(), adaptor.getCopyAtom());
    Value src = loweredValue(adaptor.getSrc());
    Value dst = loweredValue(adaptor.getDst());
    Value pred = loweredValue(adaptor.getPred());

    Type srcTy = op.getSrc().getType();
    Type dstTy = op.getDst().getType();
    auto srcMemTy = dyn_cast<fly::MemRefType>(srcTy);
    auto dstMemTy = dyn_cast<fly::MemRefType>(dstTy);
    if (!srcMemTy && !isa<fly::CoordTensorType>(srcTy))
      return rewriter.notifyMatchFailure(op, "src is neither a MemRef nor a coord tensor");
    if (!dstMemTy && !isa<fly::CoordTensorType>(dstTy))
      return rewriter.notifyMatchFailure(op, "dst is neither a MemRef nor a coord tensor");
    if (srcMemTy && dstMemTy && srcMemTy.getElemTy() != dstMemTy.getElemTy())
      return rewriter.notifyMatchFailure(op, "src/dst element types mismatch");

    Location loc = op.getLoc();

    Type predMemTy = nullptr;
    if (pred) {
      predMemTy = dyn_cast<fly::MemRefType>(op.getPred().getType());
      if (!predMemTy)
        return rewriter.notifyMatchFailure(op, "pred is not a MemRef type");
    }

    if (pred) {
      if (failed(copyAtom.emitAtomCall(rewriter, loc, copyAtomType, srcTy, dstTy, predMemTy,
                                       copyAtomVal, src, dst, pred)))
        return failure();
    } else {
      if (failed(copyAtom.emitAtomCall(rewriter, loc, copyAtomType, srcTy, dstTy, copyAtomVal, src,
                                       dst)))
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class CopyAtomCallSSALowering : public OpConversionPattern<CopyAtomCallSSA> {
public:
  using OpConversionPattern<CopyAtomCallSSA>::OpConversionPattern;

  LogicalResult matchAndRewrite(CopyAtomCallSSA op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type copyAtomType = op.getCopyAtom().getType();
    auto copyAtom = dyn_cast<CopyAtomType>(copyAtomType);
    if (!copyAtom)
      return rewriter.notifyMatchFailure(op, "copyAtom is not CopyAtomType");

    Location loc = op.getLoc();
    bool hasResult = op.getResults().size() > 0;
    Type srcTy = op.getSrc().getType();
    Value pred = loweredValue(adaptor.getPred());

    Type resultTy = hasResult ? op.getResult(0).getType() : Type{};
    Type dstTy = op.getDst() ? op.getDst().getType() : Type{};

    // SSA-form copies may retain a register memref predicate. Normalize it
    // here so every SSA emitter receives a scalar i1 condition.
    if (pred) {
      if (auto predMemTy = dyn_cast<fly::MemRefType>(op.getPred().getType())) {
        if (!isGenericAddressSpace<AddressSpace::Register>(predMemTy.getAddressSpace()) ||
            !predMemTy.getElemTy().isInteger(1) || !isa<LLVM::LLVMPointerType>(pred.getType()))
          return op.emitOpError(
              "expected an i1 register memref predicate with a lowered LLVM pointer");
        auto predPtr = applySwizzleOnPtr(
            rewriter, loc, cast<TypedValue<LLVM::LLVMPointerType>>(pred), predMemTy.getSwizzle());
        pred = LLVM::LoadOp::create(rewriter, loc, rewriter.getI1Type(), predPtr);
      }
      if (!pred.getType().isInteger(1))
        return op.emitOpError("expected a scalar i1 predicate after SSA lowering");
    }

    Value atom = loweredAtom(rewriter, loc, adaptor.getCopyAtom());
    Value src = loweredValue(adaptor.getSrc());
    Value dst = loweredValue(adaptor.getDst());
    FailureOr<Value> result =
        pred ? copyAtom.emitAtomCallSSA(rewriter, loc, resultTy, copyAtomType, srcTy, dstTy,
                                        pred.getType(), atom, src, dst, pred)
             : copyAtom.emitAtomCallSSA(rewriter, loc, resultTy, copyAtomType, srcTy, dstTy, atom,
                                        src, dst);
    if (failed(result))
      return failure();

    replaceOrErase(rewriter, op, *result);
    return success();
  }
};

/// The lowered pointers of a group of memref operands.
SmallVector<Value> loweredValues(ArrayRef<ValueRange> lowered) {
  return llvm::map_to_vector(lowered, [](ValueRange value) { return loweredValue(value); });
}

class MmaAtomCallLowering : public OpConversionPattern<MmaAtomCall> {
public:
  using OpConversionPattern<MmaAtomCall>::OpConversionPattern;

  LogicalResult matchAndRewrite(MmaAtomCall op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto mmaAtomTy = dyn_cast<MmaAtomType>(op.getMmaAtom().getType());
    if (!mmaAtomTy)
      return rewriter.notifyMatchFailure(op, "expected MmaAtomType for mmaAtom operand");

    Location loc = op.getLoc();
    Value d = loweredValue(adaptor.getD());
    Value c = loweredValue(adaptor.getC());
    auto a = loweredValues(adaptor.getA());
    auto b = loweredValues(adaptor.getB());

    auto isPointer = [](Value value) {
      return value && isa<LLVM::LLVMPointerType>(value.getType());
    };
    if (!isPointer(d) || !llvm::all_of(a, isPointer) || !llvm::all_of(b, isPointer) ||
        !isPointer(c))
      return rewriter.notifyMatchFailure(op, "expected llvm.ptr operands after type conversion");

    if (failed(
            mmaAtomTy.emitAtomCall(rewriter, loc, mmaAtomTy, op.getD().getType(),
                                   op.getA().getTypes(), op.getB().getTypes(), op.getC().getType(),
                                   loweredAtom(rewriter, loc, adaptor.getMmaAtom()), d, a, b, c)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class MmaAtomCallSSALowering : public OpConversionPattern<MmaAtomCallSSA> {
public:
  using OpConversionPattern<MmaAtomCallSSA>::OpConversionPattern;

  LogicalResult matchAndRewrite(MmaAtomCallSSA op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto mmaAtomTy = dyn_cast<MmaAtomType>(op.getMmaAtom().getType());
    if (!mmaAtomTy)
      return rewriter.notifyMatchFailure(op, "expected MmaAtomType for mmaAtom operand");

    Location loc = op.getLoc();
    auto a = loweredValues(adaptor.getA());
    auto b = loweredValues(adaptor.getB());

    bool hasResult = op.getResults().size() > 0;

    Type resultTy = hasResult ? op.getResult(0).getType() : Type{};
    Type dTy = op.getD() ? op.getD().getType() : Type{};
    Value dPtr = hasResult ? Value{} : loweredValue(adaptor.getD());

    auto result = mmaAtomTy.emitAtomCallSSA(
        rewriter, loc, resultTy, mmaAtomTy, dTy, op.getA().getTypes(), op.getB().getTypes(),
        op.getC().getType(), loweredAtom(rewriter, loc, adaptor.getMmaAtom()), dPtr, a, b,
        loweredValue(adaptor.getC()));
    if (failed(result))
      return failure();

    replaceOrErase(rewriter, op, *result);
    return success();
  }
};

/// Lower `gpu.launch_func` kernel operands to their lowered carriers: a static
/// Fly value passes nothing, a memref passes its pointer and its layout.
class GpuLaunchFuncOpLowering : public OpConversionPattern<gpu::LaunchFuncOp> {
public:
  using OpConversionPattern<gpu::LaunchFuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto kernelRef = op.getKernel();

    auto grid =
        gpu::KernelDim3{loweredValue(adaptor.getGridSizeX()), loweredValue(adaptor.getGridSizeY()),
                        loweredValue(adaptor.getGridSizeZ())};
    auto block = gpu::KernelDim3{loweredValue(adaptor.getBlockSizeX()),
                                 loweredValue(adaptor.getBlockSizeY()),
                                 loweredValue(adaptor.getBlockSizeZ())};

    std::optional<gpu::KernelDim3> clusterSize = std::nullopt;
    if (op.getClusterSizeX() && op.getClusterSizeY() && op.getClusterSizeZ()) {
      clusterSize = gpu::KernelDim3{loweredValue(adaptor.getClusterSizeX()),
                                    loweredValue(adaptor.getClusterSizeY()),
                                    loweredValue(adaptor.getClusterSizeZ())};
    }
    Value sharedMemorySize = loweredValue(adaptor.getDynamicSharedMemorySize());
    auto kernelOperands = flattenValues(adaptor.getKernelOperands());

    // Preserve async token result type when present.
    Type asyncTokenType = nullptr;
    if (Value tok = op.getAsyncToken())
      asyncTokenType = tok.getType();

    rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
        op, kernelRef, grid, block, sharedMemorySize, kernelOperands, asyncTokenType,
        flattenValues(adaptor.getAsyncDependencies()), loweredValue(adaptor.getAsyncObject()),
        clusterSize);
    return success();
  }
};

class FlyTypeConverter : public TypeConverter {
public:
  FlyTypeConverter() {
    addConversion([](Type type) { return type; });

    addConversion([&](FloatType floatTy) -> std::optional<Type> {
      if (floatTy.getWidth() < 16)
        return IntegerType::get(floatTy.getContext(), floatTy.getWidth());
      return std::nullopt;
    });
    addConversion([&](VectorType vecTy) -> std::optional<Type> {
      Type convertedElem = convertType(vecTy.getElementType());
      if (!convertedElem || convertedElem == vecTy.getElementType())
        return std::nullopt;
      return VectorType::get(vecTy.getShape(), convertedElem, vecTy.getScalableDims());
    });
    // Small floats become same-width integers, but ops that stay legal keep the float type. Bridge
    // exactly those pairs with a bitcast; any other mismatch is still a conversion error.
    auto bitcastMaterialization = [&](OpBuilder &builder, Type resultType, ValueRange inputs,
                                      Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();
      Type inputType = inputs.front().getType();
      bool isConvertedPair =
          convertType(inputType) == resultType || convertType(resultType) == inputType;
      if (!isConvertedPair ||
          !arith::BitcastOp::areCastCompatible(TypeRange{inputType}, TypeRange{resultType}))
        return Value();
      return arith::BitcastOp::create(builder, loc, resultType, inputs.front());
    };
    addSourceMaterialization(bitcastMaterialization);
    addTargetMaterialization(bitcastMaterialization);
    // A memref lowers to its pointer, followed by its layout unless that is static.
    addConversion([&](fly::MemRefType type, SmallVectorImpl<Type> &results) {
      results.push_back(convertType(PointerType::get(type.getElemTy(), type.getAddressSpace(),
                                                     type.getAlignment(), type.getSwizzle())));
      if (auto layout = core::get_llvm_type(type.getRef().layout(), *type.getContext()))
        results.push_back(layout);
      return success();
    });
    // A fully static algebra value lowers to nothing.
    addConversion([&](Type type, SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      auto lowered = getLoweredType(type);
      if (!lowered)
        return std::nullopt;
      if (*lowered)
        results.push_back(*lowered);
      return success();
    });
    addConversion([](FastDivModType type) -> Type {
      return core::emit_op::getFastDivModType(*type.getContext(), type.getDivisorType().getWidth());
    });
    addConversion([&](fly::PointerType flyPtrTy) -> Type {
      if (isTargetAddressSpace<BufferDescAddressAttr>(flyPtrTy.getAddressSpace()))
        return BufferFatPtr::getType(flyPtrTy.getContext());
      unsigned as = mapAttrToLLVMAddressSpace(flyPtrTy.getAddressSpace());
      return LLVM::LLVMPointerType::get(flyPtrTy.getContext(), as);
    });
    // A static atom lowers to nothing and a stateful atom to its state; a tiled
    // copy or MMA lowers to its atom.
    addConversion([](Type type, SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
      Type atomTy = getAtomType(type);
      Type opType = getAtomOpType(atomTy);
      if (!opType)
        return std::nullopt;
      if (!cast<MayStaticTypeInterface>(atomTy).isStatic()) {
        auto stateful = dyn_cast<StatefulOpTypeInterface>(opType);
        results.push_back(stateful ? stateful.getConvertedType(type.getContext())
                                   : emptyStruct(type.getContext()));
      }
      return success();
    });
  }
};

class ExtractAlignedPointerAsIndexLowering
    : public OpConversionPattern<ExtractAlignedPointerAsIndexOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ExtractAlignedPointerAsIndexOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // A memref's first carrier is its pointer, e.g. llvm.ptr<AS>. Cast to the
    // result type (e.g. llvm.ptr<0>) if address spaces differ.
    Value src = loweredValue(adaptor.getSource());
    if (!src)
      return failure();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      resultType = op.getResult().getType();
    if (src.getType() != resultType)
      src = LLVM::AddrSpaceCastOp::create(rewriter, op.getLoc(), resultType, src);
    rewriter.replaceOp(op, src);
    return success();
  }
};

/// Select is type-agnostic: select each lowered carrier of a Fly value.
class SelectOpTypeConversion : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::SelectOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value condition = loweredValue(adaptor.getCondition());
    ValueRange trueValues = adaptor.getTrueValue();
    ValueRange falseValues = adaptor.getFalseValue();
    if (trueValues.size() != falseValues.size())
      return failure();
    SmallVector<Value> results;
    for (auto [trueValue, falseValue] : llvm::zip(trueValues, falseValues)) {
      auto select =
          arith::SelectOp::create(rewriter, op.getLoc(), condition, trueValue, falseValue);
      select->setDiscardableAttrs(op->getDiscardableAttrDictionary());
      results.push_back(select);
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

/// Calls pass and return lowered carriers; discardable attributes are kept.
class CallOpTypeConversion : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(func::CallOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    SmallVector<unsigned> resultCounts;
    for (Type type : op.getResultTypes()) {
      size_t before = resultTypes.size();
      if (failed(getTypeConverter()->convertType(type, resultTypes)))
        return failure();
      resultCounts.push_back(resultTypes.size() - before);
    }
    auto call = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(), resultTypes,
                                     flattenValues(adaptor.getOperands()));
    call->setDiscardableAttrs(op->getDiscardableAttrDictionary());
    SmallVector<ValueRange> replacements;
    unsigned offset = 0;
    for (unsigned count : resultCounts) {
      replacements.push_back(call.getResults().slice(offset, count));
      offset += count;
    }
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

class GpuReturnTypeConversion : public OpConversionPattern<gpu::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(gpu::ReturnOp op, OneToNOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<gpu::ReturnOp>(op, flattenValues(adaptor.getOperands()));
    return success();
  }
};

/// Converts a func.func or gpu.func signature to the lowered carriers. The
/// attributes of a value move to its first carrier; workgroup and private
/// attributions after the arguments of a gpu.func keep their types.
class FunctionSignatureConversion : public ConversionPattern {
public:
  FunctionSignatureConversion(StringRef name, const TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, name, 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<ValueRange>,
                                ConversionPatternRewriter &rewriter) const override {
    auto function = cast<FunctionOpInterface>(op);
    auto type = dyn_cast<FunctionType>(function.getFunctionType());
    if (!type)
      return failure();
    const TypeConverter &converter = *getTypeConverter();
    MLIRContext *ctx = op->getContext();
    auto emptyAttrs = DictionaryAttr::get(ctx);

    TypeConverter::SignatureConversion inputs(type.getNumInputs());
    if (failed(converter.convertSignatureArgs(type.getInputs(), inputs)))
      return failure();
    SmallVector<Attribute> argAttrs(inputs.getConvertedTypes().size(), emptyAttrs);
    for (unsigned i = 0, e = type.getNumInputs(); i < e; ++i)
      if (auto mapping = inputs.getInputMapping(i); mapping && mapping->size > 0)
        argAttrs[mapping->inputNo] =
            function.getArgAttrDict(i) ? function.getArgAttrDict(i) : emptyAttrs;

    SmallVector<Type> results;
    SmallVector<Attribute> resultAttrs;
    for (auto [i, result] : llvm::enumerate(type.getResults())) {
      size_t before = results.size();
      if (failed(converter.convertType(result, results)))
        return failure();
      for (size_t j = before; j < results.size(); ++j) {
        DictionaryAttr attrs = j == before ? function.getResultAttrDict(i) : DictionaryAttr();
        resultAttrs.push_back(attrs ? attrs : emptyAttrs);
      }
    }

    if (!function.isExternal()) {
      Block &entry = function.getFunctionBody().front();
      TypeConverter::SignatureConversion block(entry.getNumArguments());
      if (failed(converter.convertSignatureArgs(type.getInputs(), block)))
        return failure();
      for (unsigned i = type.getNumInputs(); i < entry.getNumArguments(); ++i)
        block.addInputs(i, entry.getArgument(i).getType());
      rewriter.applySignatureConversion(&entry, block, &converter);
    }

    rewriter.modifyOpInPlace(op, [&] {
      function.setFunctionTypeAttr(
          TypeAttr::get(FunctionType::get(ctx, inputs.getConvertedTypes(), results)));
      function.setAllArgAttrs(argAttrs);
      function.setAllResultAttrs(resultAttrs);
    });
    return success();
  }
};

/// Whether `type` is or contains a Fly type.
bool containsFlyType(Type type) {
  bool found = false;
  type.walk([&](Type nested) {
    StringRef name = nested.getDialect().getNamespace();
    found |= name == fly::FlyDialect::getDialectNamespace() ||
             name == fly_rocdl::FlyROCDLDialect::getDialectNamespace();
  });
  return found;
}

/// An operation is legal once no operand, result, or block argument has a Fly type.
bool hasNoFlyTypes(Operation *op) {
  auto none = [](TypeRange types) { return llvm::none_of(types, containsFlyType); };
  return none(op->getOperandTypes()) && none(op->getResultTypes()) &&
         llvm::all_of(op->getRegions(), [&](Region &region) {
           return llvm::all_of(region,
                               [&](Block &block) { return none(block.getArgumentTypes()); });
         });
}

class FlyToROCDLConversionPass
    : public mlir::impl::FlyToROCDLConversionPassBase<FlyToROCDLConversionPass> {
public:
  using mlir::impl::FlyToROCDLConversionPassBase<
      FlyToROCDLConversionPass>::FlyToROCDLConversionPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    ConversionTarget target(getContext());

    // Every remaining operation is legal unless it still carries a Fly type.
    target.markUnknownOpDynamicallyLegal(hasNoFlyTypes);
    target.addDynamicallyLegalDialect<arith::ArithDialect, scf::SCFDialect, vector::VectorDialect,
                                      gpu::GPUDialect, func::FuncDialect, LLVM::LLVMDialect,
                                      ROCDL::ROCDLDialect>(hasNoFlyTypes);
    target.addIllegalDialect<fly::FlyDialect, fly_rocdl::FlyROCDLDialect>();

    FlyTypeConverter typeConverter;

    // Convert every control-flow boundary along with its users.
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
    cf::populateCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);

    target.addDynamicallyLegalOp<arith::SelectOp>(
        [&](arith::SelectOp op) { return typeConverter.isLegal(op.getOperation()); });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp, gpu::ReturnOp>(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    // Function signatures carry the lowered values of their Fly arguments and results.
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
    patterns.add<ToLLVMPtrOpLowering>(typeConverter, context);
    patterns.add<GetBufferRsrcOpLowering>(typeConverter, context);
    patterns.add<AddOffsetOpLowering>(typeConverter, context);

    for (auto name : {StaticOp::getOperationName(), MakeIntTupleOp::getOperationName(),
                      MakeShapeOp::getOperationName(), MakeStrideOp::getOperationName(),
                      MakeCoordOp::getOperationName(), MakeLayoutOp::getOperationName(),
                      MakeComposedLayoutOp::getOperationName(), MakeTileOp::getOperationName(),
                      MakeViewOp::getOperationName()})
      patterns.add<ConstructorLowering>(name, typeConverter, context);
    for (auto name : {GetScalarOp::getOperationName(), GetLeavesOp::getOperationName(),
                      GetIterOp::getOperationName()})
      patterns.add<ExtractorLowering>(name, typeConverter, context);
    patterns.add<RecastIterOpLowering>(typeConverter, context);
    for (auto name : {ApplySwizzleOp::getOperationName(), MakeTiledCopyOp::getOperationName(),
                      MakeTiledMmaOp::getOperationName(), GetCopyAtomOp::getOperationName(),
                      GetMmaAtomOp::getOperationName()})
      patterns.add<ForwardOperandLowering>(name, typeConverter, context);
    patterns.add<MakeFastDivModOpLowering, FastDivModOpLowering, FastDivModGetDivisorOpLowering>(
        typeConverter, context);
    patterns.add<PtrLoadOpLowering, PtrStoreOpLowering>(typeConverter, context);
    patterns.add<MakeAtomOpLowering<MakeCopyAtomOp>, MakeAtomOpLowering<MakeMmaAtomOp>>(
        typeConverter, context);
    patterns.add<AtomSetValueOpLowering>(typeConverter, context);
    patterns.add<CopyAtomCallLowering, MmaAtomCallLowering>(typeConverter, context);
    patterns.add<CopyAtomCallSSALowering, MmaAtomCallSSALowering>(typeConverter, context);
    patterns.add<GpuLaunchFuncOpLowering>(typeConverter, context);
    patterns.add<SelectOpTypeConversion, GpuReturnTypeConversion>(typeConverter, context);

    // TODO: deprecated in the future
    patterns.add<ExtractAlignedPointerAsIndexLowering>(typeConverter, context);

    for (auto name : {func::FuncOp::getOperationName(), gpu::GPUFuncOp::getOperationName()})
      patterns.add<FunctionSignatureConversion>(name, typeConverter, context);
    patterns.add<CallOpTypeConversion>(typeConverter, context);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
