// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_UTILS_INTTUPLEUTILS_H
#define FLYDSL_DIALECT_FLY_UTILS_INTTUPLEUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntUtils.h"

//===----------------------------------------------------------------------===//
// IntTupleAttr utilities
//===----------------------------------------------------------------------===//

namespace mlir::fly {

bool intTupleHasNone(IntTupleAttr attr);
bool intTupleAllNone(IntTupleAttr attr);

bool intTupleIsCongruent(IntTupleAttr lhs, IntTupleAttr rhs);
bool intTupleIsWeaklyCongruent(IntTupleAttr lhs, IntTupleAttr rhs);

} // namespace mlir::fly

//===----------------------------------------------------------------------===//
// Universal IntTuple utilities
//===----------------------------------------------------------------------===//

namespace mlir::fly {

template <class IntTuple> class IntTupleBuilder;

/// Adapts a normal-form IntTuple SSA value while preserving the structural
/// information from `attr`.
///
/// Invariants:
/// - If `attr` is a leaf (`IntAttr` or `BasisAttr`), `value` is the scalar SSA
///   value for that leaf instead of a `MakeIntTupleOp` result.
/// - If `attr` is not a leaf, `value` is the `MakeIntTupleOp` result for the
///   current subtree, and `[dyncIdxStart, dyncIdxEnd)` identifies which dynamic
///   leaf operands belong to this subtree.
class IntTupleValueAdaptor {
private:
  int32_t prefixSumDyncElems(int32_t idx) const {
    int dyncOffset = 0;
    for (int32_t i = 0; i < idx; ++i) {
      dyncOffset += attr.at(i).dyncLeafCount();
    }
    return dyncOffset;
  }

  IntTupleValueAdaptor(Value value, IntTupleAttr attr, int32_t dyncIdxStart = 0,
                       int32_t dyncIdxEnd = -1)
      : value(value), attr(attr), dyncIdxStart(dyncIdxStart), dyncIdxEnd(dyncIdxEnd) {}

  Value value = nullptr;
  IntTupleAttr attr = nullptr;
  int32_t dyncIdxStart = 0, dyncIdxEnd = -1;

public:
  IntTupleValueAdaptor() = default;

  template <class Builder>
  static IntTupleValueAdaptor create(Builder &builder, Value value, IntTupleAttr attr) {
    // Rebuild the adaptor from a normal-form IntTuple value while re-establishing
    // the leaf/non-leaf invariant above.
    auto defOp = value.getDefiningOp<MakeIntTupleOp>();
    assert(defOp && "Value must be a MakeIntTupleOp");
    if (attr.isLeaf()) {
      if (attr.isStatic()) {
        if (attr.isLeafInt()) {
          return builder.materializeConstantLeaf(attr.getLeafAsInt());
        } else {
          return builder.materializeConstantLeaf(attr.getLeafAsBasis());
        }
      } else {
        return IntTupleValueAdaptor(value.getDefiningOp()->getOperand(0), attr);
      }
    } else {
      return IntTupleValueAdaptor(value, attr);
    }
  }

  bool isLeaf() const { return attr.isLeaf(); }
  bool isLeafNone() const { return attr.isLeafNone(); }
  bool isLeafInt() const { return attr.isLeafInt(); }
  bool isLeafBasis() const { return attr.isLeafBasis(); }
  bool isLeafStaticValue(int32_t value) const { return attr.isLeafStaticValue(value); }
  bool isStatic() const { return attr.isStatic(); }
  int32_t rank() const { return attr.rank(); }

  Value getValue() const { return value; }
  IntTupleAttr getAttr() const { return attr; }

  friend class IntTupleBuilder<IntTupleValueAdaptor>;
};

template <class IntTuple>
IntTuple intTupleAdd(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs);

IntTupleAttr intTupleBasis2Tuple(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr basis);

IntTupleValueAdaptor intTupleBasis2Tuple(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                         IntTupleValueAdaptor basis);

template <> class IntTupleBuilder<IntTupleAttr> {
protected:
  MLIRContext *ctx;

public:
  IntTupleBuilder(MLIRContext *ctx) : ctx(ctx) {}

  struct ElemCollector {
    SmallVector<Attribute> attrCollector;

    void push_back(Attribute attr) { attrCollector.push_back(attr); }
    size_t size() const { return attrCollector.size(); }
    bool empty() const { return attrCollector.empty(); }
    void reverse() { std::reverse(attrCollector.begin(), attrCollector.end()); }
  };

  IntTupleAttr add(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr sub(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr mul(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr div(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr mod(IntTupleAttr lhs, IntTupleAttr rhs) const;

  IntTupleAttr logicalAnd(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr logicalOr(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr logicalNot(IntTupleAttr val) const;
  IntTupleAttr lt(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr le(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr gt(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr ge(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr eq(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr ne(IntTupleAttr lhs, IntTupleAttr rhs) const;

  IntTupleAttr min(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr max(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr safeDiv(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr ceilDiv(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr shapeDiv(IntTupleAttr lhs, IntTupleAttr rhs) const;
  IntTupleAttr applySwizzle(IntTupleAttr v, SwizzleAttr swizzle) const;

  IntTupleAttr getAttr(IntTupleAttr attr) const { return attr; }

  int32_t getStaticValue(IntTupleAttr attr) const {
    assert(attr.isLeaf() && attr.isStatic() && "getStaticValue requires a static leaf");
    auto intAttr = attr.extractIntFromLeaf();
    return intAttr.getValue();
  }

  IntTupleAttr materializeConstantLeaf(int32_t value, ArrayRef<int32_t> modes = {}) const {
    if (modes.empty()) {
      return IntTupleAttr::get(IntAttr::getStatic(ctx, value));
    } else {
      return IntTupleAttr::get(BasisAttr::get(IntAttr::getStatic(ctx, value), modes));
    }
  }
  IntTupleAttr materializeConstantLeaf(IntAttr value, ArrayRef<int32_t> modes = {}) const {
    assert(value.isStatic() && "Value must be static");
    if (modes.empty()) {
      return IntTupleAttr::get(value);
    } else {
      return IntTupleAttr::get(BasisAttr::get(value, modes));
    }
  }
  IntTupleAttr materializeConstantLeaf(BasisAttr value) const {
    assert(value.isStatic() && "Value must be static");
    return IntTupleAttr::get(value);
  }
  IntTupleAttr materializeConstantTuple(IntTupleAttr attr) const {
    assert(attr.isStatic() && "Tuple must be static");
    return attr;
  }

  IntTupleAttr at(IntTupleAttr attr, int32_t idx) const { return attr.at(idx); }
  IntTupleAttr makeTuple(const ElemCollector &collector) const {
    return IntTupleAttr::get(ArrayAttr::get(ctx, collector.attrCollector));
  }

  const IntTupleBuilder<IntTupleAttr> &getAttrBuilder() const { return *this; }
};

template <> class IntTupleBuilder<IntTupleValueAdaptor> {
protected:
  PatternRewriter &builder;
  Location loc;
  IntTupleBuilder<IntTupleAttr> attrBuilder;

public:
  IntTupleBuilder(PatternRewriter &builder, Location loc)
      : builder(builder), loc(loc), attrBuilder(builder.getContext()) {}

  struct ElemCollector {
    typename IntTupleBuilder<IntTupleAttr>::ElemCollector attrCollector;
    SmallVector<Value> dyncElems;

    void push_back(const IntTupleValueAdaptor &element) {
      // A leaf contributes at most one scalar dynamic value. A non-leaf forwards
      // the contiguous operand slice that represents its subtree.
      auto elemAttr = element.attr;
      attrCollector.push_back(elemAttr);
      if (elemAttr.isLeaf()) {
        if (!elemAttr.isStatic()) {
          dyncElems.push_back(element.value);
        }
      } else {
        // Handle dyncIdxEnd == -1 (meaning "to the end")
        int32_t dyncIdxEnd = element.dyncIdxEnd == -1
                                 ? element.value.getDefiningOp()->getOperands().size()
                                 : element.dyncIdxEnd;
        dyncElems.append(element.value.getDefiningOp()->getOperands().begin() +
                             element.dyncIdxStart,
                         element.value.getDefiningOp()->getOperands().begin() + dyncIdxEnd);
      }
    }
    size_t size() const { return attrCollector.size(); }
    bool empty() const { return attrCollector.empty(); }
    void reverse() {
      attrCollector.reverse();
      std::reverse(dyncElems.begin(), dyncElems.end());
    }
  };

  Type getIntType(IntTupleAttr t) const;
  Type getCommonIntType(IntTupleAttr lhs, IntTupleAttr rhs) const;
  Value extendToIntType(Value input, Type intType) const;

  IntTupleValueAdaptor add(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor sub(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor mul(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor div(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor mod(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;

  IntTupleValueAdaptor logicalAnd(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor logicalOr(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor logicalNot(IntTupleValueAdaptor val) const;
  IntTupleValueAdaptor lt(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor le(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor gt(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor ge(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor eq(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor ne(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;

  IntTupleValueAdaptor min(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor max(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor safeDiv(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor ceilDiv(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor shapeDiv(IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const;
  IntTupleValueAdaptor applySwizzle(IntTupleValueAdaptor v, SwizzleAttr swizzle) const;

  IntTupleAttr getAttr(IntTupleValueAdaptor adaptor) const { return adaptor.attr; }

  int32_t getStaticValue(IntTupleValueAdaptor adaptor) const {
    return attrBuilder.getStaticValue(adaptor.attr);
  }

  IntTupleValueAdaptor materializeConstantLeaf(int32_t value, ArrayRef<int32_t> modes = {}) const {
    auto attr = attrBuilder.materializeConstantLeaf(value, modes);
    return IntTupleValueAdaptor(arith::ConstantIntOp::create(builder, loc, value, 32).getResult(),
                                attr);
  }
  IntTupleValueAdaptor materializeConstantLeaf(IntAttr value, ArrayRef<int32_t> modes = {}) const {
    assert(value.isStatic() && "Value must be static");
    auto attr = attrBuilder.materializeConstantLeaf(value, modes);
    return IntTupleValueAdaptor(
        arith::ConstantIntOp::create(builder, loc, value.getValue(), value.getWidth()).getResult(),
        attr);
  }
  IntTupleValueAdaptor materializeConstantLeaf(BasisAttr value) const {
    assert(value.isStatic() && "Value must be static");
    auto attr = attrBuilder.materializeConstantLeaf(value);
    return IntTupleValueAdaptor(arith::ConstantIntOp::create(builder, loc,
                                                             value.getValue().getValue(),
                                                             value.getValue().getWidth())
                                    .getResult(),
                                attr);
  }
  IntTupleValueAdaptor materializeConstantTuple(IntTupleAttr attr) const {
    assert(attr.isStatic() && "Tuple must be static");
    if (attr.isLeaf()) {
      if (attr.isLeafInt()) {
        return materializeConstantLeaf(attr.getLeafAsInt());
      } else {
        return materializeConstantLeaf(attr.getLeafAsBasis());
      }
    } else {
      return IntTupleValueAdaptor{
          MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr), {}).getResult(),
          attrBuilder.materializeConstantTuple(attr)};
    }
  }

  IntTupleValueAdaptor at(IntTupleValueAdaptor adaptor, int32_t idx) const {
    auto childAttr = adaptor.attr.at(idx);
    // Leaf children become scalar SSA values
    if (childAttr.isLeaf()) {
      if (childAttr.isStatic()) {
        return this->materializeConstantTuple(childAttr);
      } else {
        return IntTupleValueAdaptor(adaptor.value.getDefiningOp()->getOperand(
                                        adaptor.dyncIdxStart + adaptor.prefixSumDyncElems(idx)),
                                    childAttr);
      }
    } else {
      // Tuple children keep sharing the parent tuple value with a narrowed
      // dynamic-leaf slice.
      int32_t dyncOffset = adaptor.prefixSumDyncElems(idx);
      return IntTupleValueAdaptor(adaptor.value, childAttr, adaptor.dyncIdxStart + dyncOffset,
                                  adaptor.dyncIdxStart + dyncOffset + childAttr.dyncLeafCount());
    }
  }
  IntTupleValueAdaptor makeTuple(const ElemCollector &collector) const {
    auto TupleAttr = attrBuilder.makeTuple(collector.attrCollector);
    return IntTupleValueAdaptor(
        MakeIntTupleOp::create(builder, loc, IntTupleType::get(TupleAttr), collector.dyncElems)
            .getResult(),
        TupleAttr);
  }

  const IntTupleBuilder<IntTupleAttr> &getAttrBuilder() const { return attrBuilder; }

  TypedValue<IntTupleType> finalize(IntTupleValueAdaptor adaptor,
                                    std::optional<IntTupleAttr> destAttr = std::nullopt) const {
    // Convert the adaptor back to the normal IntTuple representation.
    // Result is always a MakeIntTupleOp, including leaf tuples.
    auto finalAttr = destAttr.value_or(adaptor.attr);
    auto Ty = IntTupleType::get(finalAttr);

    assert(adaptor.attr.dyncLeafCount() == finalAttr.dyncLeafCount() &&
           "Adaptor and finalAttr must have the same dyncLeafCount");
    if (adaptor.attr.isStatic()) {
      return MakeIntTupleOp::create(builder, loc, Ty, {}).getResult();
    }
    if (adaptor.isLeaf()) {
      return MakeIntTupleOp::create(builder, loc, Ty, {adaptor.value}).getResult();
    } else {
      int32_t dyncIdxEnd = adaptor.dyncIdxEnd == -1
                               ? adaptor.value.getDefiningOp()->getOperands().size()
                               : adaptor.dyncIdxEnd;
      int32_t dyncCount = dyncIdxEnd - adaptor.dyncIdxStart;
      assert(dyncCount == finalAttr.dyncLeafCount());
      return MakeIntTupleOp::create(builder, loc, Ty,
                                    adaptor.value.getDefiningOp()->getOperands().slice(
                                        adaptor.dyncIdxStart, dyncCount))
          .getResult();
    }
  }
};

template <class BinaryOp, class IntTuple>
IntTuple intTupleBinaryOp(const IntTupleBuilder<IntTuple> &builder, BinaryOp &&binaryOp,
                          IntTuple lhs, IntTuple rhs) {
  if (lhs.isLeaf()) {
    assert(rhs.isLeaf());
    return binaryOp(lhs, rhs);
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  const int minRank = std::min(lhs.rank(), rhs.rank());
  for (int i = 0; i < minRank; ++i) {
    collector.push_back(
        intTupleBinaryOp(builder, binaryOp, builder.at(lhs, i), builder.at(rhs, i)));
  }
  for (int i = minRank; i < lhs.rank(); ++i) {
    collector.push_back(builder.at(lhs, i));
  }
  for (int i = minRank; i < rhs.rank(); ++i) {
    collector.push_back(builder.at(rhs, i));
  }
  return builder.makeTuple(collector);
}

template <class IntTuple>
IntTuple intTupleAdd(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return intTupleBinaryOp(
      builder, [&](IntTuple a, IntTuple b) { return builder.add(a, b); }, lhs, rhs);
}
template <class IntTuple>
IntTuple intTupleSub(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return intTupleBinaryOp(
      builder, [&](IntTuple a, IntTuple b) { return builder.sub(a, b); }, lhs, rhs);
}
template <class IntTuple>
IntTuple intTupleMul(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return intTupleBinaryOp(
      builder, [&](IntTuple a, IntTuple b) { return builder.mul(a, b); }, lhs, rhs);
}
template <class IntTuple>
IntTuple intTupleDiv(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return intTupleBinaryOp(
      builder, [&](IntTuple a, IntTuple b) { return builder.div(a, b); }, lhs, rhs);
}
template <class IntTuple>
IntTuple intTupleMod(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return intTupleBinaryOp(
      builder, [&](IntTuple a, IntTuple b) { return builder.mod(a, b); }, lhs, rhs);
}
template <class IntTuple>
IntTuple intTupleMin(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return intTupleBinaryOp(
      builder, [&](IntTuple a, IntTuple b) { return builder.min(a, b); }, lhs, rhs);
}
template <class IntTuple>
IntTuple intTupleMax(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return intTupleBinaryOp(
      builder, [&](IntTuple a, IntTuple b) { return builder.max(a, b); }, lhs, rhs);
}

namespace detail {

template <class IntTuple>
std::pair<IntTuple, IntTuple> intTupleCeilDivFoldImpl(IntTupleBuilder<IntTuple> &builder,
                                                      IntTuple a, IntTuple b) {
  if (a.isLeaf()) {
    auto result = builder.ceilDiv(a, b);
    auto remainder = builder.ceilDiv(b, a);
    return {result, remainder};
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  IntTuple remaining = b;
  for (int i = 0; i < a.rank(); ++i) {
    auto [res, rem] = intTupleCeilDivFoldImpl(builder, builder.at(a, i), remaining);
    collector.push_back(res);
    remaining = rem;
  }
  return {builder.makeTuple(collector), remaining};
}

template <class IntTuple>
std::pair<IntTuple, IntTuple> intTupleShapeDivFoldImpl(IntTupleBuilder<IntTuple> &builder,
                                                       IntTuple a, IntTuple b) {
  if (a.isLeaf()) {
    auto result = builder.shapeDiv(a, b);
    auto remainder = builder.shapeDiv(b, a);
    return {result, remainder};
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  IntTuple remaining = b;
  for (int i = 0; i < a.rank(); ++i) {
    auto [res, rem] = intTupleShapeDivFoldImpl(builder, builder.at(a, i), remaining);
    collector.push_back(res);
    remaining = rem;
  }
  return {builder.makeTuple(collector), remaining};
}

} // namespace detail

template <class IntTuple> IntTuple intTupleSum(IntTupleBuilder<IntTuple> &builder, IntTuple t) {
  if (t.isLeaf()) {
    return t;
  }
  IntTuple result = intTupleSum(builder, builder.at(t, 0));
  for (int i = 1; i < t.rank(); ++i) {
    result = builder.add(result, intTupleSum(builder, builder.at(t, i)));
  }
  return result;
}

template <class IntTuple> IntTuple intTupleProduct(IntTupleBuilder<IntTuple> &builder, IntTuple t) {
  if (t.isLeaf()) {
    return t;
  }
  IntTuple result = intTupleProduct(builder, builder.at(t, 0));
  for (int i = 1; i < t.rank(); ++i) {
    result = builder.mul(result, intTupleProduct(builder, builder.at(t, i)));
  }
  return result;
}

template <class IntTuple>
IntTuple intTupleInnerProduct(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  if (lhs.isLeaf() && rhs.isLeaf()) {
    return builder.mul(lhs, rhs);
  }
  assert(lhs.rank() == rhs.rank() && "Mismatched ranks");
  IntTuple result = intTupleInnerProduct(builder, builder.at(lhs, 0), builder.at(rhs, 0));
  for (int i = 1; i < lhs.rank(); ++i) {
    result =
        builder.add(result, intTupleInnerProduct(builder, builder.at(lhs, i), builder.at(rhs, i)));
  }
  return result;
}

template <class IntTuple>
IntTuple intTupleCeilDiv(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  if (lhs.isLeaf()) {
    if (rhs.isLeaf()) {
      return builder.ceilDiv(lhs, rhs);
    }
    auto rhsProduct = intTupleProduct(builder, rhs);
    return builder.ceilDiv(lhs, rhsProduct);
  }
  if (rhs.isLeaf()) {
    auto [result, rest] = detail::intTupleCeilDivFoldImpl(builder, lhs, rhs);
    return result;
  }
  const int divRank = std::min(lhs.rank(), rhs.rank());
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < divRank; ++i) {
    collector.push_back(intTupleCeilDiv(builder, builder.at(lhs, i), builder.at(rhs, i)));
  }
  for (int i = divRank; i < lhs.rank(); ++i) {
    collector.push_back(builder.at(lhs, i));
  }
  return builder.makeTuple(collector);
}

template <class IntTuple>
IntTuple intTupleShapeDiv(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  if (lhs.isLeaf()) {
    if (rhs.isLeaf()) {
      return builder.shapeDiv(lhs, rhs);
    }
    auto rhsProduct = intTupleProduct(builder, rhs);
    return builder.shapeDiv(lhs, rhsProduct);
  }
  if (rhs.isLeaf()) {
    auto [result, rest] = detail::intTupleShapeDivFoldImpl(builder, lhs, rhs);
    return result;
  }
  const int divRank = std::min(lhs.rank(), rhs.rank());
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < divRank; ++i) {
    collector.push_back(intTupleShapeDiv(builder, builder.at(lhs, i), builder.at(rhs, i)));
  }
  for (int i = divRank; i < lhs.rank(); ++i) {
    collector.push_back(builder.at(lhs, i));
  }
  return builder.makeTuple(collector);
}

template <class IntTuple>
IntTuple intTupleProductEach(IntTupleBuilder<IntTuple> &builder, IntTuple val) {
  if (val.isLeaf()) {
    return val;
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < val.rank(); ++i) {
    collector.push_back(intTupleProduct(builder, builder.at(val, i)));
  }
  return builder.makeTuple(collector);
}

template <class IntTuple>
IntTuple intTupleProductLike(IntTupleBuilder<IntTuple> &builder, IntTuple tuple, IntTuple guide) {
  if (guide.isLeaf()) {
    return intTupleProduct(builder, tuple);
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < guide.rank(); ++i) {
    collector.push_back(intTupleProductLike(builder, builder.at(tuple, i), builder.at(guide, i)));
  }
  return builder.makeTuple(collector);
}

//===----------------------------------------------------------------------===//
// Attribute manipulation
//===----------------------------------------------------------------------===//

IntTupleAttr intTupleWrap(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr);
IntTupleAttr intTupleUnwrap(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr);

IntTupleAttr intTupleUnflatten(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr,
                               IntTupleAttr profile);

IntTupleAttr intTupleExpand(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr,
                            ArrayRef<int32_t> indices);
IntTupleAttr intTupleGroup(const IntTupleBuilder<IntTupleAttr> &builder, IntTupleAttr attr,
                           int32_t begin, int32_t end);

inline IntTupleValueAdaptor intTupleWrap(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                         IntTupleValueAdaptor adaptor) {
  IntTupleAttr newAttr = intTupleWrap(builder.getAttrBuilder(), builder.getAttr(adaptor));
  return IntTupleValueAdaptor::create(builder, builder.finalize(adaptor, newAttr), newAttr);
}
inline IntTupleValueAdaptor intTupleUnwrap(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                           IntTupleValueAdaptor adaptor) {
  IntTupleAttr newAttr = intTupleUnwrap(builder.getAttrBuilder(), builder.getAttr(adaptor));
  return IntTupleValueAdaptor::create(builder, builder.finalize(adaptor, newAttr), newAttr);
}
inline IntTupleValueAdaptor intTupleUnflatten(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                              IntTupleValueAdaptor adaptor, IntTupleAttr profile) {
  IntTupleAttr newAttr =
      intTupleUnflatten(builder.getAttrBuilder(), builder.getAttr(adaptor), profile);
  return IntTupleValueAdaptor::create(builder, builder.finalize(adaptor, newAttr), newAttr);
}
inline IntTupleValueAdaptor intTupleExpand(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                           IntTupleValueAdaptor adaptor,
                                           ArrayRef<int32_t> indices) {
  IntTupleAttr newAttr =
      intTupleExpand(builder.getAttrBuilder(), builder.getAttr(adaptor), indices);
  return IntTupleValueAdaptor::create(builder, builder.finalize(adaptor, newAttr), newAttr);
}
inline IntTupleValueAdaptor intTupleGroup(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                          IntTupleValueAdaptor adaptor, int32_t begin,
                                          int32_t end) {
  IntTupleAttr newAttr =
      intTupleGroup(builder.getAttrBuilder(), builder.getAttr(adaptor), begin, end);
  return IntTupleValueAdaptor::create(builder, builder.finalize(adaptor, newAttr), newAttr);
}
template <class IntTuple, class Collector>
void intTupleFlattenToVector(const IntTupleBuilder<IntTuple> &builder, IntTuple t,
                             Collector &result) {
  if (t.isLeaf()) {
    result.push_back(t);
  } else {
    for (int i = 0; i < t.rank(); ++i) {
      intTupleFlattenToVector(builder, builder.at(t, i), result);
    }
  }
}
template <class IntTuple>
IntTuple intTupleFlatten(const IntTupleBuilder<IntTuple> &builder, IntTuple t) {
  if (t.isLeaf()) {
    return t;
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  intTupleFlattenToVector(builder, t, collector);
  return builder.makeTuple(collector);
}

//===----------------------------------------------------------------------===//
// Transformation operations
//===----------------------------------------------------------------------===//

template <class IntTuple, class F>
IntTuple intTupleTransform(const IntTupleBuilder<IntTuple> &builder, F &&fn, IntTuple t0) {
  if (t0.isLeaf()) {
    return fn(t0);
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < t0.rank(); ++i) {
    collector.push_back(fn(builder.at(t0, i)));
  }
  return builder.makeTuple(collector);
}
template <class IntTuple, class F>
IntTuple intTupleTransform(const IntTupleBuilder<IntTuple> &builder, F &&fn, IntTuple t0,
                           IntTuple t1) {
  if (t0.isLeaf()) {
    return fn(t0, t1);
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < t0.rank(); ++i) {
    collector.push_back(fn(builder.at(t0, i), builder.at(t1, i)));
  }
  return builder.makeTuple(collector);
}
template <class IntTuple, class F>
IntTuple intTupleTransform(const IntTupleBuilder<IntTuple> &builder, F &&fn, IntTuple t0,
                           IntTuple t1, IntTuple t2) {
  if (t0.isLeaf()) {
    return fn(t0, t1, t2);
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < t0.rank(); ++i) {
    collector.push_back(fn(builder.at(t0, i), builder.at(t1, i), builder.at(t2, i)));
  }
  return builder.makeTuple(collector);
}

template <class IntTuple, class F>
IntTuple intTupleTransformLeaf(const IntTupleBuilder<IntTuple> &builder, F &&fn, IntTuple t0) {
  if (t0.isLeaf()) {
    return fn(t0);
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < t0.rank(); ++i) {
    collector.push_back(intTupleTransformLeaf(builder, fn, builder.at(t0, i)));
  }
  return builder.makeTuple(collector);
}
template <class IntTuple, class F>
IntTuple intTupleTransformLeaf(const IntTupleBuilder<IntTuple> &builder, F &&fn, IntTuple t0,
                               IntTuple t1) {
  if (t0.isLeaf()) {
    return fn(t0, t1);
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < t0.rank(); ++i) {
    collector.push_back(intTupleTransformLeaf(builder, fn, builder.at(t0, i), builder.at(t1, i)));
  }
  return builder.makeTuple(collector);
}
template <class IntTuple, class F>
IntTuple intTupleTransformLeaf(const IntTupleBuilder<IntTuple> &builder, F &&fn, IntTuple t0,
                               IntTuple t1, IntTuple t2) {
  if (t0.isLeaf()) {
    return fn(t0, t1, t2);
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < t0.rank(); ++i) {
    collector.push_back(intTupleTransformLeaf(builder, fn, builder.at(t0, i), builder.at(t1, i),
                                              builder.at(t2, i)));
  }
  return builder.makeTuple(collector);
}

template <class IntTuple>
IntTuple intTupleTake(const IntTupleBuilder<IntTuple> &builder, IntTuple val, int32_t begin,
                      int32_t end) {
  assert(!val.isLeaf() && "intTupleTake expects a non-leaf tuple");
  if (end == -1)
    end = val.rank();
  assert(begin >= 0 && end <= val.rank() && begin <= end);
  if (end - begin == 1)
    return builder.at(val, begin);
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int32_t i = begin; i < end; ++i)
    collector.push_back(builder.at(val, i));
  return builder.makeTuple(collector);
}

template <class IntTuple>
IntTuple intTupleSelect(const IntTupleBuilder<IntTuple> &builder, IntTuple val,
                        ArrayRef<int32_t> indices) {
  assert(!val.isLeaf() && "intTupleSelect expects a non-leaf tuple");
  if (indices.size() == 1) {
    return builder.at(val, indices[0]);
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int32_t idx : indices) {
    collector.push_back(builder.at(val, idx));
  }
  return builder.makeTuple(collector);
}

/// If n == -1, appends a single element.
template <class IntTuple>
IntTuple intTupleAppend(const IntTupleBuilder<IntTuple> &builder, IntTuple val, IntTuple elem,
                        int32_t n = -1) {
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  if (val.isLeaf()) {
    collector.push_back(val);
    if (n == -1) {
      collector.push_back(elem);
    } else {
      int32_t currentRank = 1;
      while (currentRank < n) {
        collector.push_back(elem);
        ++currentRank;
      }
    }
  } else {
    for (int i = 0; i < val.rank(); ++i) {
      collector.push_back(builder.at(val, i));
    }
    if (n == -1) {
      collector.push_back(elem);
    } else {
      int32_t currentRank = val.rank();
      assert(currentRank <= n && "intTupleAppend expects n >= current rank");
      while (currentRank < n) {
        collector.push_back(elem);
        ++currentRank;
      }
    }
  }
  return builder.makeTuple(collector);
}
/// If n == -1, prepends a single element.
template <class IntTuple>
IntTuple intTuplePrepend(const IntTupleBuilder<IntTuple> &builder, IntTuple val, IntTuple elem,
                         int32_t n = -1) {
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  if (val.isLeaf()) {
    if (n == -1) {
      collector.push_back(elem);
    } else {
      int32_t targetAppend = n - 1;
      for (int32_t i = 0; i < targetAppend; ++i) {
        collector.push_back(elem);
      }
    }
    collector.push_back(val);
  } else {
    if (n == -1) {
      collector.push_back(elem);
    } else {
      assert(n >= val.rank() && "intTuplePrepend expects n >= current rank");
      int32_t numToPrepend = n - val.rank();
      for (int32_t i = 0; i < numToPrepend; ++i) {
        collector.push_back(elem);
      }
    }
    for (int i = 0; i < val.rank(); ++i) {
      collector.push_back(builder.at(val, i));
    }
  }
  return builder.makeTuple(collector);
}

template <class IntTuple>
IntTuple intTupleZip(const IntTupleBuilder<IntTuple> &builder, IntTuple attr) {
  using Collector = typename IntTupleBuilder<IntTuple>::ElemCollector;
  if (attr.isLeaf()) {
    return attr;
  } else {
    auto firstChild = builder.at(attr, 0);
    if (firstChild.isLeaf()) {
      return attr;
    } else {
      int32_t innerRank = firstChild.rank();
      Collector result;
      for (int j = 0; j < innerRank; ++j) {
        Collector zipped;
        for (int i = 0; i < attr.rank(); ++i) {
          zipped.push_back(builder.at(builder.at(attr, i), j));
        }
        result.push_back(builder.makeTuple(zipped));
      }
      return builder.makeTuple(result);
    }
  }
}
template <class IntTuple>
IntTuple intTupleZip(const IntTupleBuilder<IntTuple> &builder, IntTuple t0, IntTuple t1) {
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  collector.push_back(t0);
  collector.push_back(t1);
  return intTupleZip(builder, builder.makeTuple(collector));
}
template <class IntTuple>
IntTuple intTupleZip(const IntTupleBuilder<IntTuple> &builder, IntTuple t0, IntTuple t1,
                     IntTuple t2) {
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  collector.push_back(t0);
  collector.push_back(t1);
  collector.push_back(t2);
  return intTupleZip(builder, builder.makeTuple(collector));
}

namespace detail {

template <class IntTuple>
std::pair<IntTuple, IntTuple> intTupleZip2ByImpl(const IntTupleBuilder<IntTuple> &builder,
                                                 IntTuple t, IntTupleAttr guide) {
  using Collector = typename IntTupleBuilder<IntTuple>::ElemCollector;
  if (guide.isLeaf()) {
    assert(t.rank() == 2 && "intTupleZip2By expects rank-2 tuple at terminal");
    return {builder.at(t, 0), builder.at(t, 1)};
  }
  // Canonicalize singleton guide wrappers so 1D profiles behave as leaf guides.
  // This keeps zip2By robust after singleton unwrapping in product/divide type canonicalization.
  if (guide.rank() == 1) {
    return intTupleZip2ByImpl(builder, t, guide.at(0));
  }
  Collector firsts;
  Collector seconds;

  int32_t guideRank = guide.rank();
  int32_t tRank = t.rank();
  assert(tRank >= guideRank && "Mismatched ranks in intTupleZip2By");
  for (int i = 0; i < guideRank; ++i) {
    auto [first, second] = intTupleZip2ByImpl(builder, builder.at(t, i), guide.at(i));
    firsts.push_back(first);
    seconds.push_back(second);
  }
  for (int i = guideRank; i < tRank; ++i) {
    seconds.push_back(builder.at(t, i));
  }
  return {builder.makeTuple(firsts), builder.makeTuple(seconds)};
}

} // namespace detail

template <class IntTuple>
IntTuple intTupleZip2By(const IntTupleBuilder<IntTuple> &builder, IntTuple t, IntTupleAttr guide) {
  if (guide.isLeaf()) {
    assert(t.rank() == 2 && "intTupleZip2By expects rank-2 tuple at terminal");
    return t;
  } else {
    using Collector = typename IntTupleBuilder<IntTuple>::ElemCollector;
    auto [first, second] = detail::intTupleZip2ByImpl(builder, t, guide);
    Collector collector;
    collector.push_back(first);
    collector.push_back(second);
    return builder.makeTuple(collector);
  }
}

namespace detail {

template <class IntTuple>
void intTupleSliceImpl(const IntTupleBuilder<IntTuple> &builder, IntTuple tuple, IntTupleAttr coord,
                       typename IntTupleBuilder<IntTuple>::ElemCollector &result) {
  if (coord.isLeaf()) {
    if (coord.isLeafNone()) {
      result.push_back(tuple);
    }
  } else {
    assert(coord.rank() == tuple.rank() && "Mismatched ranks in slice");
    for (int i = 0; i < coord.rank(); ++i) {
      intTupleSliceImpl(builder, builder.at(tuple, i), coord.at(i), result);
    }
  }
}
template <class IntTuple>
void intTupleDiceImpl(const IntTupleBuilder<IntTuple> &builder, IntTuple tuple, IntTupleAttr coord,
                      typename IntTupleBuilder<IntTuple>::ElemCollector &result) {
  if (coord.isLeaf()) {
    if (!coord.isLeafNone()) {
      result.push_back(tuple);
    }
  } else {
    assert(coord.rank() == tuple.rank() && "Mismatched ranks in dice");
    for (int i = 0; i < coord.rank(); ++i) {
      intTupleDiceImpl(builder, builder.at(tuple, i), coord.at(i), result);
    }
  }
}

} // namespace detail

template <class IntTuple>
IntTuple intTupleSlice(const IntTupleBuilder<IntTuple> &builder, IntTuple tuple,
                       IntTupleAttr coord) {
  if (coord.isLeaf()) {
    if (coord.isLeafNone()) {
      return tuple;
    }
    llvm_unreachable("not support empty IntTuple");
  } else {
    typename IntTupleBuilder<IntTuple>::ElemCollector collector;
    assert(coord.rank() == tuple.rank() && "Mismatched ranks in slice");
    for (int i = 0; i < coord.rank(); ++i) {
      detail::intTupleSliceImpl(builder, builder.at(tuple, i), coord.at(i), collector);
    }
    assert(!collector.empty() && "not support empty IntTuple");
    return intTupleUnwrap(builder, builder.makeTuple(collector));
  }
}
template <class IntTuple>
IntTuple intTupleDice(const IntTupleBuilder<IntTuple> &builder, IntTuple tuple,
                      IntTupleAttr coord) {
  if (coord.isLeaf()) {
    if (!coord.isLeafNone()) {
      return tuple;
    }
    llvm_unreachable("not support empty IntTuple");
  } else {
    typename IntTupleBuilder<IntTuple>::ElemCollector collector;
    assert(coord.rank() == tuple.rank() && "Mismatched ranks in dice");
    for (int i = 0; i < coord.rank(); ++i) {
      detail::intTupleDiceImpl(builder, builder.at(tuple, i), coord.at(i), collector);
    }
    assert(!collector.empty() && "not support empty IntTuple");
    return intTupleUnwrap(builder, builder.makeTuple(collector));
  }
}

template <class IntTuple>
IntTuple intTupleFilterZero(IntTupleBuilder<IntTuple> &builder, IntTupleAttr guide, IntTuple val) {
  using Collector = typename IntTupleBuilder<IntTuple>::ElemCollector;
  if (guide.isLeaf()) {
    if (guide.isLeafStaticValue(0)) {
      return intTupleTransformLeaf(
          builder, [&](auto) { return builder.materializeConstantLeaf(1); }, val);
    }
    return val;
  }
  assert(guide.rank() == val.rank() && "Mismatched ranks in intTupleFilterZero");
  Collector collector;
  for (int i = 0; i < guide.rank(); ++i) {
    collector.push_back(intTupleFilterZero(builder, guide.at(i), builder.at(val, i)));
  }
  return builder.makeTuple(collector);
}
template <class IntTuple>
IntTuple intTupleFilterZero(IntTupleBuilder<IntTuple> &builder, IntTuple val) {
  return intTupleFilterZero(builder, builder.getAttr(val), val);
}

//===----------------------------------------------------------------------===//
// Element-wise comparison
//===----------------------------------------------------------------------===//

namespace detail {

template <class IntTuple>
IntTuple intTupleElemLessImpl(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs,
                              IntTuple rhs) {
  if (lhs.isLeaf() && rhs.isLeaf()) {
    return builder.lt(lhs, rhs);
  }
  if (lhs.rank() > rhs.rank()) {
    return builder.materializeConstantLeaf(0);
  }
  IntTuple result = intTupleElemLessImpl(builder, builder.at(lhs, 0), builder.at(rhs, 0));
  for (int i = 1; i < lhs.rank(); ++i) {
    IntTuple ri = intTupleElemLessImpl(builder, builder.at(lhs, i), builder.at(rhs, i));
    result = builder.logicalAnd(result, ri);
  }
  return result;
}

} // namespace detail

template <class IntTuple>
IntTuple intTupleElemLess(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return detail::intTupleElemLessImpl(builder, lhs, rhs);
}
template <class IntTuple>
IntTuple intTupleElemLessEqual(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs,
                               IntTuple rhs) {
  return builder.logicalNot(detail::intTupleElemLessImpl(builder, rhs, lhs));
}
template <class IntTuple>
IntTuple intTupleElemGreater(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return detail::intTupleElemLessImpl(builder, rhs, lhs);
}
template <class IntTuple>
IntTuple intTupleElemGreaterEqual(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs,
                                  IntTuple rhs) {
  return builder.logicalNot(detail::intTupleElemLessImpl(builder, lhs, rhs));
}

template <class IntTuple>
IntTuple intTupleEqual(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  if (lhs.isLeaf() && rhs.isLeaf()) {
    return builder.eq(lhs, rhs);
  }
  if (lhs.isLeaf() != rhs.isLeaf() || lhs.rank() != rhs.rank()) {
    return builder.materializeConstantLeaf(0);
  }
  IntTuple result = intTupleEqual(builder, builder.at(lhs, 0), builder.at(rhs, 0));
  for (int i = 1; i < lhs.rank(); ++i) {
    result =
        builder.logicalAnd(result, intTupleEqual(builder, builder.at(lhs, i), builder.at(rhs, i)));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Compact stride generation
//===----------------------------------------------------------------------===//

namespace detail {

template <class IntTuple>
std::pair<IntTuple, IntTuple> intTupleCompactColMajorImpl(IntTupleBuilder<IntTuple> &builder,
                                                          IntTuple shape, IntTuple current) {
  if (shape.isLeaf()) {
    IntTuple nextCurrent = builder.mul(current, shape);
    return {current, nextCurrent};
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  IntTuple running = current;
  for (int i = 0; i < shape.rank(); ++i) {
    auto [childStride, nextRunning] =
        intTupleCompactColMajorImpl(builder, builder.at(shape, i), running);
    collector.push_back(childStride);
    running = nextRunning;
  }
  return {builder.makeTuple(collector), running};
}

} // namespace detail

template <class IntTuple>
IntTuple intTupleCompactColMajor(IntTupleBuilder<IntTuple> &builder, IntTuple shape,
                                 IntTuple current) {
  auto [stride, finalProduct] = detail::intTupleCompactColMajorImpl(builder, shape, current);
  return stride;
}

template <class IntTuple>
IntTuple intTupleCompactColMajor(IntTupleBuilder<IntTuple> &builder, IntTuple shape) {
  return intTupleCompactColMajor(builder, shape, builder.materializeConstantLeaf(1));
}

IntTupleAttr intTupleMakeBasisTupleLike(IntTupleAttr profile);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_INTTUPLEUTILS_H
