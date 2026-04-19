// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"

namespace mlir::fly {

TypedValue<LLVM::LLVMPointerType> applySwizzleOnPtr(OpBuilder &b, Location loc,
                                                    TypedValue<LLVM::LLVMPointerType> ptr,
                                                    SwizzleAttr swizzle) {
  if (swizzle.isTrivialSwizzle())
    return ptr;
  auto ptrTy = ptr.getType();
  auto i64Ty = b.getI64Type();
  Value ptrInt = LLVM::PtrToIntOp::create(b, loc, i64Ty, ptr);
  int64_t bitMaskValue = ((int64_t{1} << swizzle.getMask()) - 1)
                         << (swizzle.getBase() + swizzle.getShift());
  Value bitMask = arith::ConstantIntOp::create(b, loc, i64Ty, bitMaskValue);
  Value shiftAmt = arith::ConstantIntOp::create(b, loc, i64Ty, swizzle.getShift());
  Value masked = arith::AndIOp::create(b, loc, ptrInt, bitMask);
  Value shifted = arith::ShRUIOp::create(b, loc, masked, shiftAmt);
  Value swizzled = arith::XOrIOp::create(b, loc, ptrInt, shifted);
  return cast<TypedValue<LLVM::LLVMPointerType>>(
      LLVM::IntToPtrOp::create(b, loc, ptrTy, swizzled).getResult());
}

Type RegMem2SSAType(fly::MemRefType memRefTy) {
  if (memRefTy.getAddressSpace().getValue() != AddressSpace::Register)
    return Type();
  LayoutBuilder<LayoutAttr> builder(memRefTy.getContext());
  auto layoutAttr = cast<LayoutAttr>(memRefTy.getLayout());
  int32_t cosize = layoutCosize(builder, layoutAttr).getLeafAsInt().getValue();
  Type elemTy = memRefTy.getElemTy();
  if (cosize == 1)
    return elemTy;
  return VectorType::get({cosize}, elemTy);
}

} // namespace mlir::fly
