#ifndef FLYDSL_DIALECT_UTILS_INTTUPLEUTILS_H
#define FLYDSL_DIALECT_UTILS_INTTUPLEUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntUtils.h"

//===----------------------------------------------------------------------===//
// IntTupleAttr constexpr utilities
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

  Value value;
  IntTupleAttr attr;
  int32_t dyncIdxStart, dyncIdxEnd;

public:
  template <class Builder>
  static IntTupleValueAdaptor create(Builder &builder, Value value, IntTupleAttr attr) {
    auto defOp = value.getDefiningOp<MakeIntTupleOp>();
    assert(defOp && "Value must be a MakeIntTupleOp");
    if (attr.isLeaf()) {
      if (attr.isStatic()) {
        return IntTupleValueAdaptor(
            builder.materializeConstantArith(attr.getLeafAsInt().getValue()).value, attr);
      } else {
        return IntTupleValueAdaptor(value.getDefiningOp()->getOperand(0), attr);
      }
    } else {
      return IntTupleValueAdaptor(value, attr);
    }
  }

  bool isLeaf() const { return attr.isLeaf(); }
  int32_t rank() const { return attr.rank(); }
  int32_t depth() const { return attr.depth(); }

  friend class IntTupleBuilder<IntTupleValueAdaptor>;
};

template <> class IntTupleBuilder<IntTupleAttr> {
protected:
  MLIRContext *ctx;

public:
  IntTupleBuilder(MLIRContext *ctx) : ctx(ctx) {}

  using ArithValue = IntAttr;
  struct ElemCollector {
    SmallVector<Attribute> attrCollector;

    void push_back(Attribute attr) { attrCollector.push_back(attr); }
    size_t size() const { return attrCollector.size(); }
    bool empty() const { return attrCollector.empty(); }
    void reverse() { std::reverse(attrCollector.begin(), attrCollector.end()); }
  };

  ArithValue add(ArithValue lhs, ArithValue rhs) const { return lhs + rhs; }
  ArithValue sub(ArithValue lhs, ArithValue rhs) const { return lhs - rhs; }
  ArithValue mul(ArithValue lhs, ArithValue rhs) const { return lhs * rhs; }
  ArithValue div(ArithValue lhs, ArithValue rhs) const { return lhs / rhs; }
  ArithValue mod(ArithValue lhs, ArithValue rhs) const { return lhs % rhs; }

  ArithValue logicalAnd(ArithValue lhs, ArithValue rhs) const { return lhs && rhs; }
  ArithValue logicalOr(ArithValue lhs, ArithValue rhs) const { return lhs || rhs; }
  ArithValue logicalNot(ArithValue val) const { return !val; }
  ArithValue lt(ArithValue lhs, ArithValue rhs) const { return lhs < rhs; }
  ArithValue le(ArithValue lhs, ArithValue rhs) const { return lhs <= rhs; }
  ArithValue gt(ArithValue lhs, ArithValue rhs) const { return lhs > rhs; }
  ArithValue ge(ArithValue lhs, ArithValue rhs) const { return lhs >= rhs; }
  ArithValue eq(ArithValue lhs, ArithValue rhs) const { return lhs == rhs; }
  ArithValue ne(ArithValue lhs, ArithValue rhs) const { return lhs != rhs; }

  ArithValue min(ArithValue lhs, ArithValue rhs) const { return intMin(lhs, rhs); }
  ArithValue max(ArithValue lhs, ArithValue rhs) const { return intMax(lhs, rhs); }
  ArithValue safeDiv(ArithValue lhs, ArithValue rhs) const { return intSafeDiv(lhs, rhs); }
  ArithValue ceilDiv(ArithValue lhs, ArithValue rhs) const { return intCeilDiv(lhs, rhs); }
  ArithValue shapeDiv(ArithValue lhs, ArithValue rhs) const { return intShapeDiv(lhs, rhs); }

  IntTupleAttr getAttr(IntTupleAttr attr) const { return attr; }
  ArithValue getArithValue(IntTupleAttr attr) const { return attr.getLeafAsInt(); }

  ArithValue materializeConstantArith(int32_t value) const {
    return IntAttr::getStatic(ctx, value);
  }
  ArithValue materializeConstantArith(int64_t value) const;

  ArithValue materializeConstantArith(IntAttr value) const {
    assert(value.isStatic() && "Value must be static");
    return value;
  }

  IntTupleAttr materializeConstantTuple(IntTupleAttr attr) const {
    assert(attr.isStatic() && "Tuple must be static");
    return attr;
  }

  bool isNone(ArithValue val) const { return val.isNone(); }
  bool isStatic(ArithValue val) const { return val.isStatic(); }
  bool isStaticValue(ArithValue val, int32_t v) const { return val.isStaticValue(v); }
  int32_t getStaticValue(ArithValue val) const { return val.getValue(); }

  IntTupleAttr at(IntTupleAttr attr, int32_t idx) const { return attr.at(idx); }
  IntTupleAttr makeInt(ArithValue value) const { return IntTupleAttr::get(value); }
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

  struct ArithValue {
    Value value;
    IntAttr attr;
  };
  struct ElemCollector {
    typename IntTupleBuilder<IntTupleAttr>::ElemCollector attrCollector;
    SmallVector<Value> dyncElems;

    void push_back(const IntTupleValueAdaptor &element) {
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

  Type getIntType(IntAttr attr) const {
    assert((attr.getWidth() == 64 || attr.getWidth() == 32) && "Invalid width");
    return attr.getWidth() == 64 ? builder.getI64Type() : builder.getI32Type();
  }
  Type getCommonIntType(IntAttr lhs, IntAttr rhs) const {
    assert((lhs.getWidth() == 64 || lhs.getWidth() == 32) && "Invalid width");
    assert((rhs.getWidth() == 64 || rhs.getWidth() == 32) && "Invalid width");
    return lhs.getWidth() == 64 || rhs.getWidth() == 64 ? builder.getI64Type()
                                                        : builder.getI32Type();
  }
  Value extendToIntType(Value input, Type intType) const {
    if (input.getType() != intType) {
      input = arith::ExtSIOp::create(builder, loc, intType, input);
    }
    return input;
  }

  ArithValue add(ArithValue lhs, ArithValue rhs) const;
  ArithValue sub(ArithValue lhs, ArithValue rhs) const;
  ArithValue mul(ArithValue lhs, ArithValue rhs) const;
  ArithValue div(ArithValue lhs, ArithValue rhs) const;
  ArithValue mod(ArithValue lhs, ArithValue rhs) const;

  ArithValue logicalAnd(ArithValue lhs, ArithValue rhs) const;
  ArithValue logicalOr(ArithValue lhs, ArithValue rhs) const;
  ArithValue logicalNot(ArithValue val) const;
  ArithValue lt(ArithValue lhs, ArithValue rhs) const;
  ArithValue le(ArithValue lhs, ArithValue rhs) const;
  ArithValue gt(ArithValue lhs, ArithValue rhs) const;
  ArithValue ge(ArithValue lhs, ArithValue rhs) const;
  ArithValue eq(ArithValue lhs, ArithValue rhs) const;
  ArithValue ne(ArithValue lhs, ArithValue rhs) const;

  ArithValue min(ArithValue lhs, ArithValue rhs) const;
  ArithValue max(ArithValue lhs, ArithValue rhs) const;
  ArithValue safeDiv(ArithValue lhs, ArithValue rhs) const { return div(lhs, rhs); }
  ArithValue ceilDiv(ArithValue lhs, ArithValue rhs) const;
  ArithValue shapeDiv(ArithValue lhs, ArithValue rhs) const;

  IntTupleAttr getAttr(IntTupleValueAdaptor adaptor) const { return adaptor.attr; }

  ArithValue getArithValue(IntTupleValueAdaptor adaptor) const {
    assert(adaptor.attr.isLeaf() && "Adaptor must be a leaf");
    return ArithValue{adaptor.value, attrBuilder.getArithValue(adaptor.attr)};
  }

  ArithValue materializeConstantArith(int32_t value) const {
    return ArithValue{arith::ConstantIntOp::create(builder, loc, value, 32).getResult(),
                      attrBuilder.materializeConstantArith(value)};
  }
  ArithValue materializeConstantArith(int64_t value) const;

  ArithValue materializeConstantArith(IntAttr value) const {
    assert(value.isStatic() && "Value must be static");
    return ArithValue{
        arith::ConstantIntOp::create(builder, loc, value.getValue(), value.getWidth()).getResult(),
        value};
  }

  IntTupleValueAdaptor materializeConstantTuple(IntTupleAttr attr) const {
    assert(attr.isStatic() && "Tuple must be static");
    if (attr.isLeaf()) {
      return IntTupleValueAdaptor{
          arith::ConstantIntOp::create(builder, loc, attr.getLeafAsInt().getValue(), 32)
              .getResult(),
          attrBuilder.materializeConstantTuple(attr)};
    } else {
      return IntTupleValueAdaptor{
          MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr), {}).getResult(),
          attrBuilder.materializeConstantTuple(attr)};
    }
  }

  bool isNone(ArithValue val) const { return attrBuilder.isNone(val.attr); }
  bool isStatic(ArithValue val) const { return attrBuilder.isStatic(val.attr); }
  bool isStaticValue(ArithValue val, int32_t v) const {
    return attrBuilder.isStaticValue(val.attr, v);
  }
  int32_t getStaticValue(ArithValue val) const { return attrBuilder.getStaticValue(val.attr); }

  IntTupleValueAdaptor at(IntTupleValueAdaptor adaptor, int32_t idx) const {
    auto childAttr = adaptor.attr.at(idx);
    if (childAttr.isLeaf()) {
      if (childAttr.isStatic()) {
        return makeInt(this->materializeConstantArith(childAttr.getLeafAsInt().getValue()));
      } else {
        return IntTupleValueAdaptor(adaptor.value.getDefiningOp()->getOperand(
                                        adaptor.dyncIdxStart + adaptor.prefixSumDyncElems(idx)),
                                    childAttr);
      }
    } else {
      int32_t dyncOffset = adaptor.prefixSumDyncElems(idx);
      return IntTupleValueAdaptor(adaptor.value, childAttr, adaptor.dyncIdxStart + dyncOffset,
                                  adaptor.dyncIdxStart + dyncOffset + childAttr.dyncLeafCount());
    }
  }
  IntTupleValueAdaptor makeInt(ArithValue value) const {
    return IntTupleValueAdaptor(value.value, IntTupleAttr::get(value.attr));
  }
  IntTupleValueAdaptor makeTuple(const ElemCollector &collector) const {
    auto TupleAttr = attrBuilder.makeTuple(collector.attrCollector);
    return IntTupleValueAdaptor(
        MakeIntTupleOp::create(builder, loc, IntTupleType::get(TupleAttr), collector.dyncElems)
            .getResult(),
        TupleAttr);
  }
  const IntTupleBuilder<IntTupleAttr> &getAttrBuilder() const { return attrBuilder; }

  //===----------------------------------------------------------------------===//
  // IntTupleValueAdaptor only interface
  //===----------------------------------------------------------------------===//

  TypedValue<IntTupleType> finalize(IntTupleValueAdaptor adaptor) const {
    auto Ty = IntTupleType::get(adaptor.attr);
    if (adaptor.isLeaf()) {
      if (adaptor.attr.isStatic()) {
        return MakeIntTupleOp::create(builder, loc, Ty, {}).getResult();
      } else {
        return MakeIntTupleOp::create(builder, loc, Ty, adaptor.value).getResult();
      }
    } else if (adaptor.dyncIdxStart == 0 && adaptor.dyncIdxEnd == -1) {
      return cast<TypedValue<IntTupleType>>(adaptor.value);
    } else {
      int32_t dyncIdxEnd = adaptor.dyncIdxEnd == -1
                               ? adaptor.value.getDefiningOp()->getOperands().size()
                               : adaptor.dyncIdxEnd;
      return MakeIntTupleOp::create(builder, loc, Ty,
                                    adaptor.value.getDefiningOp()->getOperands().slice(
                                        adaptor.dyncIdxStart, dyncIdxEnd - adaptor.dyncIdxStart))
          .getResult();
    }
  }
  TypedValue<IntTupleType> reprofile(TypedValue<IntTupleType> value,
                                     IntTupleAttr newProfile) const {
    return MakeIntTupleOp::create(builder, value.getLoc(), IntTupleType::get(newProfile),
                                  value.getDefiningOp()->getOperands())
        .getResult();
  }
};

template <class BinaryOp, class IntTuple>
IntTuple intTupleBinaryOp(IntTupleBuilder<IntTuple> &builder, BinaryOp &&binaryOp, IntTuple lhs,
                          IntTuple rhs) {
  if (lhs.isLeaf()) {
    assert(rhs.isLeaf() && "Mismatched structure");
    return builder.makeInt(binaryOp(builder.getArithValue(lhs), builder.getArithValue(rhs)));
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
IntTuple intTupleAdd(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  return intTupleBinaryOp(
      builder, [&](ArithValue a, ArithValue b) { return builder.add(a, b); }, lhs, rhs);
}

template <class IntTuple>
IntTuple intTupleSub(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  return intTupleBinaryOp(
      builder, [&](ArithValue a, ArithValue b) { return builder.sub(a, b); }, lhs, rhs);
}

template <class IntTuple>
IntTuple intTupleMul(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  return intTupleBinaryOp(
      builder, [&](ArithValue a, ArithValue b) { return builder.mul(a, b); }, lhs, rhs);
}

template <class IntTuple>
IntTuple intTupleDiv(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  return intTupleBinaryOp(
      builder, [&](ArithValue a, ArithValue b) { return builder.div(a, b); }, lhs, rhs);
}

template <class IntTuple>
IntTuple intTupleMod(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  return intTupleBinaryOp(
      builder, [&](ArithValue a, ArithValue b) { return builder.mod(a, b); }, lhs, rhs);
}

template <class IntTuple>
IntTuple intTupleMin(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  return intTupleBinaryOp(
      builder, [&](ArithValue a, ArithValue b) { return builder.min(a, b); }, lhs, rhs);
}

template <class IntTuple>
IntTuple intTupleMax(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  return intTupleBinaryOp(
      builder, [&](ArithValue a, ArithValue b) { return builder.max(a, b); }, lhs, rhs);
}

template <class IntTuple>
typename IntTupleBuilder<IntTuple>::ArithValue intTupleSumImpl(IntTupleBuilder<IntTuple> &builder,
                                                               IntTuple t) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  if (t.isLeaf()) {
    return builder.getArithValue(t);
  }
  ArithValue result = intTupleSumImpl(builder, builder.at(t, 0));
  for (int i = 1; i < t.rank(); ++i) {
    result = builder.add(result, intTupleSumImpl(builder, builder.at(t, i)));
  }
  return result;
}

template <class IntTuple> IntTuple intTupleSum(IntTupleBuilder<IntTuple> &builder, IntTuple t) {
  return builder.makeInt(intTupleSumImpl(builder, t));
}

template <class IntTuple>
typename IntTupleBuilder<IntTuple>::ArithValue
intTupleProductImpl(IntTupleBuilder<IntTuple> &builder, IntTuple t) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  if (t.isLeaf()) {
    return builder.getArithValue(t);
  }
  ArithValue result = intTupleProductImpl(builder, builder.at(t, 0));
  for (int i = 1; i < t.rank(); ++i) {
    result = builder.mul(result, intTupleProductImpl(builder, builder.at(t, i)));
  }
  return result;
}

template <class IntTuple> IntTuple intTupleProduct(IntTupleBuilder<IntTuple> &builder, IntTuple t) {
  return builder.makeInt(intTupleProductImpl(builder, t));
}

template <class IntTuple>
typename IntTupleBuilder<IntTuple>::ArithValue
intTupleInnerProductImpl(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  if (lhs.isLeaf() && rhs.isLeaf()) {
    return builder.mul(builder.getArithValue(lhs), builder.getArithValue(rhs));
  }
  assert(lhs.rank() == rhs.rank() && "Mismatched ranks");
  ArithValue result = intTupleInnerProductImpl(builder, builder.at(lhs, 0), builder.at(rhs, 0));
  for (int i = 1; i < lhs.rank(); ++i) {
    result = builder.add(result,
                         intTupleInnerProductImpl(builder, builder.at(lhs, i), builder.at(rhs, i)));
  }
  return result;
}

template <class IntTuple>
IntTuple intTupleInnerProduct(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return builder.makeInt(intTupleInnerProductImpl(builder, lhs, rhs));
}

template <class IntTuple>
std::pair<IntTuple, typename IntTupleBuilder<IntTuple>::ArithValue>
intTupleCeilDivFoldImpl(IntTupleBuilder<IntTuple> &builder, IntTuple a,
                        typename IntTupleBuilder<IntTuple>::ArithValue b) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  if (a.isLeaf()) {
    auto aVal = builder.getArithValue(a);
    auto result = builder.ceilDiv(aVal, b);
    auto remainder = builder.ceilDiv(b, aVal);
    return {builder.makeInt(result), remainder};
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  ArithValue remaining = b;
  for (int i = 0; i < a.rank(); ++i) {
    auto [res, rem] = intTupleCeilDivFoldImpl(builder, builder.at(a, i), remaining);
    collector.push_back(res);
    remaining = rem;
  }
  return {builder.makeTuple(collector), remaining};
}

template <class IntTuple>
IntTuple intTupleCeilDiv(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  if (lhs.isLeaf()) {
    if (rhs.isLeaf()) {
      return builder.makeInt(
          builder.ceilDiv(builder.getArithValue(lhs), builder.getArithValue(rhs)));
    }
    auto rhsProduct = intTupleProductImpl(builder, rhs);
    return builder.makeInt(builder.ceilDiv(builder.getArithValue(lhs), rhsProduct));
  }
  if (rhs.isLeaf()) {
    auto [result, rest] = intTupleCeilDivFoldImpl(builder, lhs, builder.getArithValue(rhs));
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
std::pair<IntTuple, typename IntTupleBuilder<IntTuple>::ArithValue>
intTupleShapeDivFoldImpl(IntTupleBuilder<IntTuple> &builder, IntTuple a,
                         typename IntTupleBuilder<IntTuple>::ArithValue b) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  if (a.isLeaf()) {
    auto aVal = builder.getArithValue(a);
    auto result = builder.shapeDiv(aVal, b);
    auto remainder = builder.shapeDiv(b, aVal);
    return {builder.makeInt(result), remainder};
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  ArithValue remaining = b;
  for (int i = 0; i < a.rank(); ++i) {
    auto [res, rem] = intTupleShapeDivFoldImpl(builder, builder.at(a, i), remaining);
    collector.push_back(res);
    remaining = rem;
  }
  return {builder.makeTuple(collector), remaining};
}

template <class IntTuple>
IntTuple intTupleShapeDiv(IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  if (lhs.isLeaf()) {
    if (rhs.isLeaf()) {
      return builder.makeInt(
          builder.shapeDiv(builder.getArithValue(lhs), builder.getArithValue(rhs)));
    }
    auto rhsProduct = intTupleProductImpl(builder, rhs);
    return builder.makeInt(builder.shapeDiv(builder.getArithValue(lhs), rhsProduct));
  }
  if (rhs.isLeaf()) {
    auto [result, rest] = intTupleShapeDivFoldImpl(builder, lhs, builder.getArithValue(rhs));
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
  return IntTupleValueAdaptor::create(builder, builder.finalize(adaptor), newAttr);
}
inline IntTupleValueAdaptor intTupleUnwrap(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                           IntTupleValueAdaptor adaptor) {
  IntTupleAttr newAttr = intTupleUnwrap(builder.getAttrBuilder(), builder.getAttr(adaptor));
  return IntTupleValueAdaptor::create(
      builder, builder.reprofile(builder.finalize(adaptor), newAttr), newAttr);
}
inline IntTupleValueAdaptor intTupleUnflatten(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                              IntTupleValueAdaptor adaptor, IntTupleAttr profile) {
  IntTupleAttr newAttr =
      intTupleUnflatten(builder.getAttrBuilder(), builder.getAttr(adaptor), profile);
  return IntTupleValueAdaptor::create(
      builder, builder.reprofile(builder.finalize(adaptor), newAttr), newAttr);
}
inline IntTupleValueAdaptor intTupleExpand(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                           IntTupleValueAdaptor adaptor,
                                           ArrayRef<int32_t> indices) {
  IntTupleAttr newAttr =
      intTupleExpand(builder.getAttrBuilder(), builder.getAttr(adaptor), indices);
  return IntTupleValueAdaptor::create(
      builder, builder.reprofile(builder.finalize(adaptor), newAttr), newAttr);
}
inline IntTupleValueAdaptor intTupleGroup(const IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                          IntTupleValueAdaptor adaptor, int32_t begin,
                                          int32_t end) {
  IntTupleAttr newAttr =
      intTupleGroup(builder.getAttrBuilder(), builder.getAttr(adaptor), begin, end);
  return IntTupleValueAdaptor::create(
      builder, builder.reprofile(builder.finalize(adaptor), newAttr), newAttr);
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
IntTuple intTupleSelect(const IntTupleBuilder<IntTuple> &builder, IntTuple val,
                        ArrayRef<int32_t> indices) {
  assert(!val.isLeaf() && "intTupleSelect expects a non-leaf tuple");
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
  using Collector = typename IntTupleBuilder<IntTuple>::ElemCollector;
  auto [first, second] = detail::intTupleZip2ByImpl(builder, t, guide);
  Collector collector;
  collector.push_back(first);
  collector.push_back(second);
  return builder.makeTuple(collector);
}

namespace detail {

template <class IntTuple>
void intTupleSliceImpl(const IntTupleBuilder<IntTuple> &builder, IntTuple tuple, IntTupleAttr coord,
                       typename IntTupleBuilder<IntTuple>::ElemCollector &result) {
  if (coord.isLeaf()) {
    if (coord.isLeafNone()) {
      result.push_back(tuple);
    }
    return;
  }
  assert(coord.rank() == tuple.rank() && "Mismatched ranks in slice");
  for (int i = 0; i < coord.rank(); ++i) {
    intTupleSliceImpl(builder, builder.at(tuple, i), coord.at(i), result);
  }
}
template <class IntTuple>
void intTupleDiceImpl(const IntTupleBuilder<IntTuple> &builder, IntTuple tuple, IntTupleAttr coord,
                      typename IntTupleBuilder<IntTuple>::ElemCollector &result) {
  if (coord.isLeaf()) {
    if (!coord.isLeafNone()) {
      result.push_back(tuple);
    }
    return;
  }
  assert(coord.rank() == tuple.rank() && "Mismatched ranks in dice");
  for (int i = 0; i < coord.rank(); ++i) {
    intTupleDiceImpl(builder, builder.at(tuple, i), coord.at(i), result);
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
          builder, [&](auto) { return builder.makeInt(builder.materializeConstantArith(1)); }, val);
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
typename IntTupleBuilder<IntTuple>::ArithValue
intTupleElemLessImpl(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  if (lhs.isLeaf() && rhs.isLeaf()) {
    return builder.lt(builder.getArithValue(lhs), builder.getArithValue(rhs));
  }
  if (lhs.rank() > rhs.rank()) {
    return builder.materializeConstantArith(0);
  }
  ArithValue result = intTupleElemLessImpl(builder, builder.at(lhs, 0), builder.at(rhs, 0));
  for (int i = 1; i < lhs.rank(); ++i) {
    ArithValue ri = intTupleElemLessImpl(builder, builder.at(lhs, i), builder.at(rhs, i));
    result = builder.logicalAnd(result, ri);
  }
  return result;
}

} // namespace detail

template <class IntTuple>
IntTuple intTupleElemLess(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return builder.makeInt(detail::intTupleElemLessImpl(builder, lhs, rhs));
}
template <class IntTuple>
IntTuple intTupleElemLessEqual(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs,
                               IntTuple rhs) {
  return builder.makeInt(builder.logicalNot(detail::intTupleElemLessImpl(builder, rhs, lhs)));
}
template <class IntTuple>
IntTuple intTupleElemGreater(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs, IntTuple rhs) {
  return builder.makeInt(detail::intTupleElemLessImpl(builder, rhs, lhs));
}
template <class IntTuple>
IntTuple intTupleElemGreaterEqual(const IntTupleBuilder<IntTuple> &builder, IntTuple lhs,
                                  IntTuple rhs) {
  return builder.makeInt(builder.logicalNot(detail::intTupleElemLessImpl(builder, lhs, rhs)));
}

//===----------------------------------------------------------------------===//
// Basis arithmetic operations
//===----------------------------------------------------------------------===//

IntTupleAttr intTupleExpandBasis(BasisAttr attr);
IntTupleAttr intTupleMakeBasisLike(IntTupleAttr profile);

IntTupleAttr operator+(BasisAttr lhs, BasisAttr rhs);
IntTupleAttr operator+(BasisAttr lhs, IntTupleAttr rhs);
IntTupleAttr operator+(IntTupleAttr lhs, BasisAttr rhs);
BasisAttr operator*(BasisAttr lhs, IntAttr rhs);
BasisAttr operator*(IntAttr lhs, BasisAttr rhs);
BasisAttr operator/(BasisAttr lhs, IntAttr rhs);

BasisAttr basisSafeDiv(BasisAttr lhs, IntAttr rhs);
BasisAttr basisCeilDiv(BasisAttr lhs, IntAttr rhs);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_UTILS_INTTUPLEUTILS_H
