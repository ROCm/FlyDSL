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

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::add(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.add(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::AddIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                          extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}
IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::sub(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.sub(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::SubIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                          extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}
IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::mul(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.mul(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::MulIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                          extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}
IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::div(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.div(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::DivSIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                           extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}
IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::mod(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.mod(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::RemSIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                           extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::min(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.min(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::MinSIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                           extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::max(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.max(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::MaxSIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                           extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::ceilDiv(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.ceilDiv(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::CeilDivSIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                               extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::shapeDiv(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.shapeDiv(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::CeilDivSIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                               extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::logicalAnd(ArithValue lhs, ArithValue rhs) const {
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
  return ArithValue{arith::ExtUIOp::create(builder, loc, retType, result).getResult(), retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::logicalOr(ArithValue lhs, ArithValue rhs) const {
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
  return ArithValue{arith::ExtUIOp::create(builder, loc, retType, result).getResult(), retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::logicalNot(ArithValue val) const {
  auto retAttr = attrBuilder.logicalNot(val.attr);
  auto retType = getIntType(retAttr);
  auto zero = arith::ConstantIntOp::create(builder, loc, getIntType(val.attr), 0).getResult();
  // !(val) == (val == 0)
  auto result = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, val.value, zero);
  return ArithValue{arith::ExtUIOp::create(builder, loc, retType, result).getResult(), retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::lt(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.lt(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return ArithValue{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(), retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::le(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.le(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sle,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return ArithValue{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(), retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::gt(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.gt(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sgt,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return ArithValue{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(), retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::ge(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.ge(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return ArithValue{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(), retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::eq(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.eq(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return ArithValue{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(), retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::ne(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.ne(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  auto retType = getIntType(retAttr);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                   extendToIntType(lhs.value, cmpType),
                                   extendToIntType(rhs.value, cmpType));
  return ArithValue{arith::ExtUIOp::create(builder, loc, retType, cmp).getResult(), retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::bitwiseXor(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.bitwiseXor(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::XOrIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                          extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::bitwiseAnd(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.bitwiseAnd(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::AndIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                          extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
}

IntTupleBuilder<IntTupleValueAdaptor>::ArithValue
IntTupleBuilder<IntTupleValueAdaptor>::shiftRight(ArithValue lhs, ArithValue rhs) const {
  auto retAttr = attrBuilder.shiftRight(lhs.attr, rhs.attr);
  auto cmpType = getCommonIntType(lhs.attr, rhs.attr);
  return ArithValue{arith::ShRSIOp::create(builder, loc, extendToIntType(lhs.value, cmpType),
                                           extendToIntType(rhs.value, cmpType))
                        .getResult(),
                    retAttr};
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

IntTupleAttr intTupleExpandBasis(BasisAttr attr) {
  auto *ctx = attr.getContext();
  ArrayRef<int32_t> modes = attr.getModes();

  if (modes.empty()) {
    return IntTupleAttr::get(attr.getValue());
  }

  auto zero = IntTupleAttr::get(IntAttr::getStatic(ctx, 0));
  IntTupleAttr result = IntTupleAttr::get(attr.getValue());

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

namespace {

IntTupleAttr intTupleMakeBasisLikeImpl(MLIRContext *ctx, IntTupleAttr profile,
                                       SmallVector<int32_t, 4> &modes) {
  if (profile.isLeaf()) {
    auto one = IntAttr::getStatic(ctx, 1);
    return IntTupleAttr::get(BasisAttr::get(ctx, one, modes));
  }

  SmallVector<Attribute> elements;
  for (int32_t i = 0; i < profile.rank(); ++i) {
    modes.push_back(i);
    elements.push_back(intTupleMakeBasisLikeImpl(ctx, profile.at(i), modes));
    modes.pop_back();
  }
  return IntTupleAttr::get(ArrayAttr::get(ctx, elements));
}

} // namespace

IntTupleAttr intTupleMakeBasisLike(IntTupleAttr profile) {
  auto *ctx = profile.getContext();
  SmallVector<int32_t, 4> modes;
  assert(!profile.isLeaf() && "intTupleMakeBasisLike expects a non-leaf IntTupleAttr");
  return intTupleMakeBasisLikeImpl(ctx, profile, modes);
}

IntTupleAttr operator+(BasisAttr lhs, BasisAttr rhs) {
  IntTupleBuilder<IntTupleAttr> builder(lhs.getContext());
  return intTupleAdd(builder, intTupleExpandBasis(lhs), intTupleExpandBasis(rhs));
}
IntTupleAttr operator+(BasisAttr lhs, IntTupleAttr rhs) {
  IntTupleBuilder<IntTupleAttr> builder(lhs.getContext());
  return intTupleAdd(builder, intTupleExpandBasis(lhs), rhs);
}
IntTupleAttr operator+(IntTupleAttr lhs, BasisAttr rhs) {
  IntTupleBuilder<IntTupleAttr> builder(lhs.getContext());
  return intTupleAdd(builder, lhs, intTupleExpandBasis(rhs));
}

BasisAttr operator*(BasisAttr lhs, IntAttr rhs) {
  return BasisAttr::get(lhs.getContext(), cast<IntAttr>(lhs.getValue()) * rhs, lhs.getModes());
}
BasisAttr operator*(IntAttr lhs, BasisAttr rhs) {
  return BasisAttr::get(rhs.getContext(), lhs * cast<IntAttr>(rhs.getValue()), rhs.getModes());
}
BasisAttr operator/(BasisAttr lhs, IntAttr rhs) {
  return BasisAttr::get(lhs.getContext(), cast<IntAttr>(lhs.getValue()) / rhs, lhs.getModes());
}

BasisAttr basisSafeDiv(BasisAttr lhs, IntAttr rhs) {
  return BasisAttr::get(lhs.getContext(), intSafeDiv(cast<IntAttr>(lhs.getValue()), rhs),
                        lhs.getModes());
}
BasisAttr basisCeilDiv(BasisAttr lhs, IntAttr rhs) {
  return BasisAttr::get(lhs.getContext(), intCeilDiv(cast<IntAttr>(lhs.getValue()), rhs),
                        lhs.getModes());
}

} // namespace mlir::fly
