#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::fly {

bool intTupleHasNone(IntTupleAttr attr) {
  if (attr.isLeaf()) {
    return attr.isLeafNone();
  }
  for (int i = 0; i < attr.rank(); ++i) {
    if (intTupleHasNone(attr.at(i))) {
      return true;
    }
  }
  return false;
}
bool intTupleAllNone(IntTupleAttr attr) {
  if (attr.isLeaf()) {
    return attr.isLeafNone();
  }
  for (int i = 0; i < attr.rank(); ++i) {
    if (!intTupleAllNone(attr.at(i))) {
      return false;
    }
  }
  return true;
}

bool intTupleIsCongruent(IntTupleAttr lhs, IntTupleAttr rhs) {
  if (lhs.isLeaf() && rhs.isLeaf()) {
    return true;
  }
  if (lhs.isLeaf() != rhs.isLeaf()) {
    return false;
  }
  if (lhs.rank() != rhs.rank()) {
    return false;
  }
  for (int i = 0; i < lhs.rank(); ++i) {
    if (!intTupleIsCongruent(lhs.at(i), rhs.at(i))) {
      return false;
    }
  }
  return true;
}
bool intTupleIsWeaklyCongruent(IntTupleAttr lhs, IntTupleAttr rhs) {
  if (lhs.isLeaf()) {
    return true;
  }
  if (rhs.isLeaf()) {
    return false;
  }
  if (lhs.rank() != rhs.rank()) {
    return false;
  }
  for (int i = 0; i < lhs.rank(); ++i) {
    if (!intTupleIsWeaklyCongruent(lhs.at(i), rhs.at(i))) {
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// IntTupleBuilder<IntTupleAttr>
//===----------------------------------------------------------------------===//

IntTupleAttr IntTupleBuilder<IntTupleAttr>::add(IntTupleAttr lhs, IntTupleAttr rhs) const {
  if (lhs.isLeafInt() && rhs.isLeafInt()) {
    return IntTupleAttr::get(lhs.getLeafAsInt() + rhs.getLeafAsInt());
  } else {
    assert(lhs.isLeafBasis() || lhs.isLeafStaticValue(0) || !lhs.isLeaf());
    assert(rhs.isLeafBasis() || rhs.isLeafStaticValue(0) || !rhs.isLeaf());
    if (lhs.isLeafStaticValue(0)) {
      return rhs;
    }
    if (rhs.isLeafStaticValue(0)) {
      return lhs;
    }
    auto lhsTuple = lhs.isLeafBasis() ? intTupleBasis2Tuple(*this, lhs) : lhs;
    auto rhsTuple = rhs.isLeafBasis() ? intTupleBasis2Tuple(*this, rhs) : rhs;
    return intTupleAdd(*this, lhsTuple, rhsTuple);
  }
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::sub(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() - rhs.getLeafAsInt());
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::mul(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeaf() && rhs.isLeaf());
  assert(lhs.isLeafInt() || rhs.isLeafInt());

  if (lhs.isLeafInt() && rhs.isLeafInt()) {
    return IntTupleAttr::get(lhs.getLeafAsInt() * rhs.getLeafAsInt());
  } else if (lhs.isLeafInt()) {
    auto rhsBasis = rhs.getLeafAsBasis();
    return IntTupleAttr::get(lhs.getLeafAsInt() * rhsBasis.getValue(), rhsBasis.getModes());
  } else {
    auto lhsBasis = lhs.getLeafAsBasis();
    return IntTupleAttr::get(lhsBasis.getValue() * rhs.getLeafAsInt(), lhsBasis.getModes());
  }
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::div(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() / rhs.getLeafAsInt());
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::mod(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() % rhs.getLeafAsInt());
}

IntTupleAttr IntTupleBuilder<IntTupleAttr>::logicalAnd(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() && rhs.getLeafAsInt());
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::logicalOr(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() || rhs.getLeafAsInt());
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::logicalNot(IntTupleAttr val) const {
  assert(val.isLeafInt());
  return IntTupleAttr::get(!val.getLeafAsInt());
}

IntTupleAttr IntTupleBuilder<IntTupleAttr>::lt(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() < rhs.getLeafAsInt());
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::le(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() <= rhs.getLeafAsInt());
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::gt(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() > rhs.getLeafAsInt());
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::ge(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() >= rhs.getLeafAsInt());
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::eq(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() == rhs.getLeafAsInt());
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::ne(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(lhs.getLeafAsInt() != rhs.getLeafAsInt());
}

IntTupleAttr IntTupleBuilder<IntTupleAttr>::min(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(intMin(lhs.getLeafAsInt(), rhs.getLeafAsInt()));
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::max(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(intMax(lhs.getLeafAsInt(), rhs.getLeafAsInt()));
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::safeDiv(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeaf() && rhs.isLeafInt());
  if (lhs.isLeafInt()) {
    return IntTupleAttr::get(intSafeDiv(lhs.getLeafAsInt(), rhs.getLeafAsInt()));
  } else {
    return IntTupleAttr::get(intSafeDiv(lhs.getLeafAsBasis(), rhs.getLeafAsInt()));
  }
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::ceilDiv(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeaf() && rhs.isLeafInt());
  if (lhs.isLeafInt()) {
    return IntTupleAttr::get(intCeilDiv(lhs.getLeafAsInt(), rhs.getLeafAsInt()));
  } else {
    return IntTupleAttr::get(intCeilDiv(lhs.getLeafAsBasis(), rhs.getLeafAsInt()));
  }
}
IntTupleAttr IntTupleBuilder<IntTupleAttr>::shapeDiv(IntTupleAttr lhs, IntTupleAttr rhs) const {
  assert(lhs.isLeafInt() && rhs.isLeafInt());
  return IntTupleAttr::get(intShapeDiv(lhs.getLeafAsInt(), rhs.getLeafAsInt()));
}

IntTupleAttr IntTupleBuilder<IntTupleAttr>::applySwizzle(IntTupleAttr v,
                                                         SwizzleAttr swizzle) const {
  assert(v.isLeafInt() && "applySwizzle only supports leafInt IntTupleAttr");
  return IntTupleAttr::get(intApplySwizzle(v.getLeafAsInt(), swizzle));
}

//===----------------------------------------------------------------------===//
// IntTupleBuilder<IntTupleValueAdaptor>
//===----------------------------------------------------------------------===//

Type IntTupleBuilder<IntTupleValueAdaptor>::getIntType(IntTupleAttr t) const {
  assert(t.isLeaf());
  auto attr = t.extractIntFromLeaf();
  assert((attr.getWidth() == 64 || attr.getWidth() == 32 || attr.getWidth() == 0) &&
         "Invalid width");
  return attr.getWidth() == 64 ? builder.getI64Type() : builder.getI32Type();
}
Type IntTupleBuilder<IntTupleValueAdaptor>::getCommonIntType(IntTupleAttr lhs,
                                                             IntTupleAttr rhs) const {
  assert(lhs.isLeaf() && rhs.isLeaf());
  auto lhsAttr = lhs.extractIntFromLeaf();
  auto rhsAttr = rhs.extractIntFromLeaf();
  assert((lhsAttr.getWidth() == 64 || lhsAttr.getWidth() == 32 || lhsAttr.getWidth() == 0) &&
         "Invalid width");
  assert((rhsAttr.getWidth() == 64 || rhsAttr.getWidth() == 32 || rhsAttr.getWidth() == 0) &&
         "Invalid width");
  return lhsAttr.getWidth() == 64 || rhsAttr.getWidth() == 64 ? builder.getI64Type()
                                                              : builder.getI32Type();
}
Value IntTupleBuilder<IntTupleValueAdaptor>::extendToIntType(Value input, Type intType) const {
  if (input.getType() != intType) {
    input = arith::ExtSIOp::create(builder, loc, intType, input);
  }
  return input;
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::add(IntTupleValueAdaptor lhs,
                                                                IntTupleValueAdaptor rhs) const {
  if (lhs.isLeafInt() && rhs.isLeafInt()) {
    auto retAttr = attrBuilder.add(lhs.attr, rhs.attr);
    auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
    return IntTupleValueAdaptor{arith::AddIOp::create(builder, loc,
                                                      extendToIntType(lhs.value, cmpType),
                                                      extendToIntType(rhs.value, cmpType))
                                    .getResult(),
                                retAttr};
  } else {
    assert(lhs.isLeafBasis() || lhs.isLeafStaticValue(0) || !lhs.isLeaf());
    assert(rhs.isLeafBasis() || rhs.isLeafStaticValue(0) || !rhs.isLeaf());
    if (lhs.isLeafStaticValue(0)) {
      return rhs;
    }
    if (rhs.isLeafStaticValue(0)) {
      return lhs;
    }
    auto lhsTuple = lhs.isLeafBasis() ? intTupleBasis2Tuple(*this, lhs) : lhs;
    auto rhsTuple = rhs.isLeafBasis() ? intTupleBasis2Tuple(*this, rhs) : rhs;
    return intTupleAdd(*this, lhsTuple, rhsTuple);
  }
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::sub(IntTupleValueAdaptor lhs,
                                                                IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.sub(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return IntTupleValueAdaptor{arith::SubIOp::create(builder, loc,
                                                    extendToIntType(lhs.value, cmpType),
                                                    extendToIntType(rhs.value, cmpType))
                                  .getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::mul(IntTupleValueAdaptor lhs,
                                                                IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.mul(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return IntTupleValueAdaptor{arith::MulIOp::create(builder, loc,
                                                    extendToIntType(lhs.value, cmpType),
                                                    extendToIntType(rhs.value, cmpType))
                                  .getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::div(IntTupleValueAdaptor lhs,
                                                                IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.div(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return IntTupleValueAdaptor{arith::DivSIOp::create(builder, loc,
                                                     extendToIntType(lhs.value, cmpType),
                                                     extendToIntType(rhs.value, cmpType))
                                  .getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::mod(IntTupleValueAdaptor lhs,
                                                                IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.mod(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return IntTupleValueAdaptor{arith::RemSIOp::create(builder, loc,
                                                     extendToIntType(lhs.value, cmpType),
                                                     extendToIntType(rhs.value, cmpType))
                                  .getResult(),
                              retAttr};
}

IntTupleValueAdaptor
IntTupleBuilder<IntTupleValueAdaptor>::logicalAnd(IntTupleValueAdaptor lhs,
                                                  IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.logicalAnd(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  // (lhs != 0) && (rhs != 0)
  auto lhsBool = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, lhs.value,
      arith::ConstantIntOp::create(builder, loc, getIntType(lhs.attr), 0).getResult());
  auto rhsBool = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, rhs.value,
      arith::ConstantIntOp::create(builder, loc, getIntType(rhs.attr), 0).getResult());
  auto result = arith::AndIOp::create(builder, loc, lhsBool, rhsBool);
  return IntTupleValueAdaptor{arith::ExtUIOp::create(builder, loc, retType, result).getResult(),
                              retAttr};
}

IntTupleValueAdaptor
IntTupleBuilder<IntTupleValueAdaptor>::logicalOr(IntTupleValueAdaptor lhs,
                                                 IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.logicalOr(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  // (lhs != 0) || (rhs != 0)
  auto lhsBool = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, lhs.value,
      arith::ConstantIntOp::create(builder, loc, getIntType(lhs.attr), 0).getResult());
  auto rhsBool = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, rhs.value,
      arith::ConstantIntOp::create(builder, loc, getIntType(rhs.attr), 0).getResult());
  auto result = arith::OrIOp::create(builder, loc, lhsBool, rhsBool);
  return IntTupleValueAdaptor{arith::ExtUIOp::create(builder, loc, retType, result).getResult(),
                              retAttr};
}

IntTupleValueAdaptor
IntTupleBuilder<IntTupleValueAdaptor>::logicalNot(IntTupleValueAdaptor val) const {
  auto retAttr = attrBuilder.logicalNot(val.attr);
  auto retType = getIntType(retAttr);
  auto zero = arith::ConstantIntOp::create(builder, loc, getIntType(val.attr), 0).getResult();
  // !(val) == (val == 0)
  auto result = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, val.value, zero);
  return IntTupleValueAdaptor{arith::ExtUIOp::create(builder, loc, retType, result).getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::lt(IntTupleValueAdaptor lhs,
                                                               IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.lt(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return IntTupleValueAdaptor{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::le(IntTupleValueAdaptor lhs,
                                                               IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.le(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return IntTupleValueAdaptor{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::gt(IntTupleValueAdaptor lhs,
                                                               IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.gt(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return IntTupleValueAdaptor{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::ge(IntTupleValueAdaptor lhs,
                                                               IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.ge(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return IntTupleValueAdaptor{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::eq(IntTupleValueAdaptor lhs,
                                                               IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.eq(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return IntTupleValueAdaptor{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::ne(IntTupleValueAdaptor lhs,
                                                               IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.ne(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return IntTupleValueAdaptor{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::min(IntTupleValueAdaptor lhs,
                                                                IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.min(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return IntTupleValueAdaptor{arith::MinSIOp::create(builder, loc,
                                                     extendToIntType(lhs.value, cmpType),
                                                     extendToIntType(rhs.value, cmpType))
                                  .getResult(),
                              retAttr};
}

IntTupleValueAdaptor IntTupleBuilder<IntTupleValueAdaptor>::max(IntTupleValueAdaptor lhs,
                                                                IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.max(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return IntTupleValueAdaptor{arith::MaxSIOp::create(builder, loc,
                                                     extendToIntType(lhs.value, cmpType),
                                                     extendToIntType(rhs.value, cmpType))
                                  .getResult(),
                              retAttr};
}

IntTupleValueAdaptor
IntTupleBuilder<IntTupleValueAdaptor>::safeDiv(IntTupleValueAdaptor lhs,
                                               IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.safeDiv(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return IntTupleValueAdaptor{arith::DivSIOp::create(builder, loc,
                                                     extendToIntType(lhs.value, cmpType),
                                                     extendToIntType(rhs.value, cmpType))
                                  .getResult(),
                              retAttr};
}

IntTupleValueAdaptor
IntTupleBuilder<IntTupleValueAdaptor>::ceilDiv(IntTupleValueAdaptor lhs,
                                               IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.ceilDiv(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return IntTupleValueAdaptor{arith::CeilDivSIOp::create(builder, loc,
                                                         extendToIntType(lhs.value, cmpType),
                                                         extendToIntType(rhs.value, cmpType))
                                  .getResult(),
                              retAttr};
}

IntTupleValueAdaptor
IntTupleBuilder<IntTupleValueAdaptor>::shapeDiv(IntTupleValueAdaptor lhs,
                                                IntTupleValueAdaptor rhs) const {
  auto retAttr = attrBuilder.shapeDiv(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return IntTupleValueAdaptor{arith::CeilDivSIOp::create(builder, loc,
                                                         extendToIntType(lhs.value, cmpType),
                                                         extendToIntType(rhs.value, cmpType))
                                  .getResult(),
                              retAttr};
}

IntTupleValueAdaptor
IntTupleBuilder<IntTupleValueAdaptor>::applySwizzle(IntTupleValueAdaptor v,
                                                    SwizzleAttr swizzle) const {
  assert(v.isLeafInt() && "applySwizzle only supports leaf IntTupleValueAdaptor");

  auto retAttr = attrBuilder.applySwizzle(v.attr, swizzle);

  // shortcut for trivial swizzle and static value
  if (swizzle.isTrivialSwizzle()) {
    return IntTupleValueAdaptor{v.value, retAttr};
  }
  if (retAttr.isStatic()) {
    return materializeConstantLeaf(retAttr.getLeafAsInt());
  }

  auto intType =
      v.attr.getLeafAsInt().getWidth() == 64 ? builder.getI64Type() : builder.getI32Type();
  auto input = extendToIntType(v.value, intType);
  int64_t bitMaskValue = ((int64_t{1} << swizzle.getMask()) - 1)
                         << (swizzle.getBase() + swizzle.getShift());
  auto bitMask = arith::ConstantIntOp::create(builder, loc, intType, bitMaskValue).getResult();
  auto shiftAmount =
      arith::ConstantIntOp::create(builder, loc, intType, swizzle.getShift()).getResult();
  auto masked = arith::AndIOp::create(builder, loc, input, bitMask).getResult();
  auto shifted = arith::ShRUIOp::create(builder, loc, masked, shiftAmount).getResult();
  auto result = arith::XOrIOp::create(builder, loc, input, shifted).getResult();
  return IntTupleValueAdaptor{result, retAttr};
}

IntTupleAttr intTupleWrap(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr) {
  if (attr.isLeaf()) {
    SmallVector<Attribute> elements;
    elements.push_back(attr);
    return IntTupleAttr::get(ArrayAttr::get(attr.getContext(), elements));
  }
  return attr;
}
IntTupleAttr intTupleUnwrap(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr) {
  if (!attr.isLeaf()) {
    if (attr.rank() == 1) {
      return intTupleUnwrap(builder, attr.at(0));
    }
    return attr;
  }
  return attr;
}

namespace detail {

std::pair<IntTupleAttr, ArrayRef<IntTupleAttr>>
intTupleUnflattenImpl(ArrayRef<IntTupleAttr> flatElements, IntTupleAttr profile) {
  if (profile.isLeaf()) {
    return {flatElements[0], flatElements.drop_front()};
  }
  SmallVector<Attribute> resultElements;
  auto remaining = flatElements;
  for (int i = 0; i < profile.rank(); ++i) {
    auto [subResult, subRemaining] = intTupleUnflattenImpl(remaining, profile.at(i));
    resultElements.push_back(subResult);
    remaining = subRemaining;
  }
  return std::pair{IntTupleAttr::get(ArrayAttr::get(profile.getContext(), resultElements)),
                   remaining};
}

} // end namespace detail

IntTupleAttr intTupleUnflatten(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr,
                               IntTupleAttr profile) {
  if (attr.isLeaf()) {
    return attr;
  }
  SmallVector<IntTupleAttr> flatElements;
  for (int i = 0; i < attr.rank(); ++i) {
    flatElements.push_back(attr.at(i));
  }
  auto [result, remaining] = detail::intTupleUnflattenImpl(flatElements, profile);
  assert(remaining.empty() && "flat tuple has more elements than profile requires");
  return result;
}
IntTupleAttr intTupleExpand(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr,
                            ArrayRef<int32_t> indices) {
  if (attr.isLeaf() || indices.empty()) {
    return attr;
  }
  SmallVector<Attribute> elements;
  for (int i = 0; i < attr.rank(); ++i) {
    bool shouldExpand = false;
    for (int32_t idx : indices) {
      if (idx == i) {
        shouldExpand = true;
        break;
      }
    }
    if (shouldExpand && !attr.at(i).isLeaf()) {
      for (int j = 0; j < attr.at(i).rank(); ++j) {
        elements.push_back(attr.at(i).at(j));
      }
    } else {
      elements.push_back(attr.at(i));
    }
  }
  if (elements.size() == 1) {
    return cast<IntTupleAttr>(elements[0]);
  }
  return IntTupleAttr::get(ArrayAttr::get(attr.getContext(), elements));
}
IntTupleAttr intTupleGroup(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr,
                           int32_t begin, int32_t end) {
  if (attr.isLeaf()) {
    return attr;
  }
  if (end == -1) {
    end = attr.rank();
  }
  assert(begin >= 0 && begin <= end && "begin must be <= end");

  SmallVector<Attribute> result;
  for (int i = 0; i < begin; ++i) {
    result.push_back(attr.at(i));
  }
  if (begin < end) {
    SmallVector<Attribute> grouped;
    for (int i = begin; i < end; ++i) {
      grouped.push_back(attr.at(i));
    }
    result.push_back(IntTupleAttr::get(ArrayAttr::get(attr.getContext(), grouped)));
  }
  for (int i = end; i < attr.rank(); ++i) {
    result.push_back(attr.at(i));
  }
  return IntTupleAttr::get(ArrayAttr::get(attr.getContext(), result));
}

//===----------------------------------------------------------------------===//
// Basis operations
//===----------------------------------------------------------------------===//

IntTupleAttr intTupleBasis2Tuple(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr) {
  auto *ctx = attr.getContext();

  assert(attr.isLeafBasis() && "attr must be a basis");
  BasisAttr basis = attr.getLeafAsBasis();
  ArrayRef<int32_t> modes = basis.getModes();
  assert(!modes.empty() && "modes must not be empty");

  auto zero = IntTupleAttr::get(IntAttr::getStatic(ctx, 0));
  IntTupleAttr result = IntTupleAttr::get(basis.getValue());
  for (auto it = modes.rbegin(); it != modes.rend(); ++it) {
    int32_t n = *it;
    SmallVector<Attribute> elements;
    for (int32_t i = 0; i < n; ++i) {
      elements.push_back(zero);
    }
    elements.push_back(result);
    result = IntTupleAttr::get(ArrayAttr::get(ctx, elements));
  }
  return result;
}

IntTupleValueAdaptor intTupleBasis2Tuple(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                         IntTupleValueAdaptor basis) {
  assert(basis.isLeafBasis());
  IntTupleAttr attr = builder.getAttr(basis);
  IntTupleAttr newAttr = intTupleBasis2Tuple(builder.getAttrBuilder(), attr);
  return IntTupleValueAdaptor::create(builder, basis.getValue(), newAttr);
}

static IntTupleAttr intTupleMakeBasisTupleLikeImpl(MLIRContext *ctx, IntTupleAttr profile,
                                                   SmallVector<int32_t, 4> &modes) {
  if (profile.isLeaf()) {
    return IntTupleAttr::get(BasisAttr::get(IntAttr::getStatic(ctx, 1), modes));
  }
  SmallVector<Attribute> elements;
  for (int32_t i = 0; i < profile.rank(); ++i) {
    modes.push_back(i);
    elements.push_back(intTupleMakeBasisTupleLikeImpl(ctx, profile.at(i), modes));
    modes.pop_back();
  }
  return IntTupleAttr::get(ArrayAttr::get(ctx, elements));
}

IntTupleAttr intTupleMakeBasisTupleLike(IntTupleAttr profile) {
  auto *ctx = profile.getContext();
  SmallVector<int32_t, 4> modes;
  assert(!profile.isLeaf() && "intTupleMakeBasisTupleLike expects a non-leaf IntTupleAttr");
  return intTupleMakeBasisTupleLikeImpl(ctx, profile, modes);
}

} // namespace mlir::fly
